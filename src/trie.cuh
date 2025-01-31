#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>

#include <cstdint>
#include <bit>

#include "environment.cuh"
#include "intersect.cuh"

template<typename T, unsigned int div>
__device__ __host__ __forceinline__ T _ceil(T val) {
	return (val / div) + ((val % div) > 0);
}

template<typename T>
__device__ __host__ __forceinline__ T _ceil(T val, T div) {
	return (val / div) + ((val % div) > 0);
}

struct BasicAdd
{
	CUB_RUNTIME_FUNCTION __forceinline__
		unsigned int operator()(const unsigned int& a, const unsigned int& b) const {
		return a + b;
	}
};

#define MAX_LCOUNT 32
#define BUCKET_SIZE 8u
#define CT_GRID_DIM 1024u
#define CT_BLOCK_DIM 512u
#define CT_WARP_SIZE 32u
#define CT_WARP_PER_BLOCK (CT_BLOCK_DIM/CT_WARP_SIZE)
#define CT_C_SIZE MAX_LCOUNT * 2 * 2 * 2

struct CT_Bucket
{
	unsigned int data[BUCKET_SIZE];
};

__constant__ unsigned int CT_C[CT_C_SIZE];

__host__ __device__ unsigned int CIdx(
	unsigned int label_index,
	const unsigned int const_index
) {
	return 2 * label_index + const_index;
}

__device__ unsigned int CVal(
	unsigned int label_index,
	const unsigned int const_index
) {
	return CT_C[CIdx(label_index, const_index)];
}

struct VertexPair {
	unsigned int vertices, pointer;
};

struct VertexRow {
	unsigned int start, end;
};

template<unsigned int Size>
struct VertexRows {
	VertexRow rows[Size];
	__device__ __forceinline__
		VertexRow operator [](int i) const { return rows[i]; }
	__device__ __forceinline__
		VertexRow& operator [](int i) { return rows[i]; }
};

template<unsigned int Size>
struct LostJob {
	VertexRow rows[Size];
	__device__ __forceinline__
		VertexRow operator [](int i) const { return rows[i]; }
	__device__ __forceinline__
		VertexRow& operator [](int i) { return rows[i]; }
};

struct LabelEdge {
	unsigned int left, right, label;

	bool operator< (const LabelEdge& c) {

		if (label != c.label) {
			return label < c.label;
		}
		else if (left != c.left) {
			return left < c.left;
		}
		else {
			return right < c.right;
		}
	}
};

struct LabelEdge_decomposer
{
	__host__ __device__::cuda::std::tuple<unsigned int&, unsigned int&, unsigned int&> operator()(LabelEdge& key) const
	{
		return { key.label, key.left , key.right };
	}
};

struct TempCSRGraph {
	unsigned int* edgeCount;
	unsigned int* vertexLabels;

	LabelEdge* edges;
	LabelEdge* sortedEdges;
};

enum GraphType {
	Trie,
	Cuckoo
};

#define DefaultGPUGraph BaseGPUGraph<Trie>

#ifdef CTGraph_Format
#define GPUGraph BaseGPUGraph<Cuckoo>
#else
#define GPUGraph DefaultGPUGraph
#endif

template<GraphType gType>
struct BaseGPUGraph {
	int edgeCount, vertexCount;

	VertexPair* vertexPairs;
	unsigned int* neighbourOffsets;
	unsigned int* neighbourData;

	unsigned int graphLabel;
	unsigned int overflow;

	CT_Bucket* vertices;

	unsigned int bucketCount;
	unsigned int* _neighbourOffsets;
	unsigned int* _neighbourData;

	//Werid bug present if not padded on 8 word boundary
	unsigned int padding[3];


	__device__ __forceinline__
		VertexPair getPair(unsigned int vertex) {
		return vertexPairs[vertex / 32];
	}

	__device__ __forceinline__
		unsigned int getMask(VertexPair pair, unsigned int vertex) {
		unsigned int mod = vertex % 32;
		unsigned int highBit = 1 << mod;

		return pair.vertices & highBit;
	}

	__device__ __forceinline__
		unsigned int getLocalOffset(unsigned int vertices, unsigned int mask) {
		unsigned int remMask = mask - 1;
		return __popc(vertices & remMask);
	}

	//Hacking CT onto our format for profiling reasons!
#ifdef CTGraph_Format

	__device__ __forceinline__
		unsigned int getOffsetLoc(unsigned int bucket) {
		return bucket * BUCKET_SIZE;
	}

	__device__ __forceinline__
		unsigned int getHash(unsigned int vertex) {
		unsigned int hash_value = ((CVal(graphLabel, 0) ^ vertex) + CVal(graphLabel, 1)) % bucketCount;
		//printf("%lu vs %lu vs %lu vs %lu \n", vertex, hash_value, (CVal(graphLabel, 0) ^ vertex), CVal(graphLabel, 1));
		return hash_value;
	}

	__device__
		bool insert(unsigned int vertex) {

		unsigned int hash_value = getHash(vertex);

		CT_Bucket* bucket = vertices + hash_value;

		for (unsigned int j = 0; j < BUCKET_SIZE; j++)
		{
			/*
			if (atomicAdd(bucket->data + j, 0) == vertex) {
				printf("Bucket raced with vertex %lu hash %lu and label %lu %p for itr %lu\n", vertex, hash_value, graphLabel, bucket, j);
				return false;
			}*/

			if (atomicCAS(bucket->data + j, 0xffffffff, vertex) == 0xffffffff)
			{
				return true;
			}
		}

		//printf("Bucket overflowed with vertex %lu hash %lu and label %lu %p \n", vertex, hash_value, graphLabel, bucket);

		return false;
	}

#endif

	__device__ __forceinline__
		unsigned int getOffset(unsigned int vertex) {
		unsigned int neighourOffset = 0;

		if constexpr (gType == Trie) {
			VertexPair pair = getPair(vertex);
			unsigned int mask = getMask(pair, vertex);

			if (mask) {
				unsigned int localOffset = getLocalOffset(pair.vertices, mask);
				neighourOffset = pair.pointer + localOffset;
			}
		}
#ifdef CTGraph_Format
		if constexpr (gType == Cuckoo) {
			unsigned int hash_value = getHash(vertex);
			CT_Bucket bucket = vertices[hash_value];
			for (unsigned int j = 0; j < BUCKET_SIZE; j++)
			{
				if (bucket.data[j] == vertex)
				{
					return getOffsetLoc(hash_value) + j;
				}
			}
			return 0xffffffff;
		}
#endif 
		return neighourOffset;
	}


	__device__ __forceinline__
		VertexRow neighbours(unsigned int vertex) {

		if constexpr (gType == Trie) {
			VertexPair pair = getPair(vertex);
			unsigned int mask = getMask(pair, vertex);

			if (mask) {

				unsigned int localOffset = getLocalOffset(pair.vertices, mask);
				unsigned int* localNeighbourOffset = neighbourOffsets + (pair.pointer + localOffset);
				//printf("\nOffset %u", localOffset);
				unsigned int neighbourStart = localNeighbourOffset[0];
				unsigned int neighbourEnd = localNeighbourOffset[1];

				/*
				if (vertex == 3) {
					printf("VertexOffset Trie vertex %lu offsets ptr %p offsetLoc %p start %lu - %lu label %lu\n", vertex, neighbourOffsets, localNeighbourOffset, neighbourStart, neighbourEnd, graphLabel);
				}*/

				return { neighbourStart, neighbourEnd };
			}
		}
#ifdef CTGraph_Format
		if constexpr (gType == Cuckoo) {
			unsigned int neighbourOffset = getOffset(vertex);

			if (neighbourOffset != 0xffffffff) {

				unsigned int neighbourStart = _neighbourOffsets[neighbourOffset];
				unsigned int neighbourEnd = _neighbourOffsets[neighbourOffset + 1];

				/*
				if (vertex == 3) {
					printf("VertexOffset CT vertex %lu offsets ptr %p offsetLoc %lu start %lu - %lu label %lu\n", vertex, _neighbourOffsets, neighbourOffset, neighbourStart, neighbourEnd, graphLabel);
				}*/

				return { neighbourStart, neighbourEnd };
			}

		}
#endif 

		return {};
	}
};

//This is bad but required in some cases to get well formed debug prints, almost definitely there is a better way to do this.
__device__
void printRow(VertexRow row, unsigned int label, unsigned int vertex, unsigned int* neighbourData) {

	unsigned int delta = row.end - row.start;

	switch (delta) {
	case 1:
		printf("%u: Label %u (Start %u -> End %u): %u\n", vertex, label, row.start, row.end, neighbourData[row.start]);
		break;
	case 2:
		printf("%u: Label %u (Start %u -> End %u): %u, %u\n", vertex, label, row.start, row.end, neighbourData[row.start], neighbourData[row.start + 1]);
		break;
	case 3:
		printf("%u: Label %u (Start %u -> End %u): %u, %u, %u\n", vertex, label, row.start, row.end, neighbourData[row.start], neighbourData[row.start + 1], neighbourData[row.start + 2]);
		break;
	case 4:
		printf("%u: Label %u (Start %u -> End %u): %u, %u, %u, %u\n", vertex, label, row.start, row.end, neighbourData[row.start], neighbourData[row.start + 1], neighbourData[row.start + 2], neighbourData[row.start + 3]);
		break;
	case 5:
		printf("%u: Label %u (Start %u -> End %u): %u, %u, %u, %u, %u\n", vertex, label, row.start, row.end, neighbourData[row.start], neighbourData[row.start + 1], neighbourData[row.start + 2], neighbourData[row.start + 3], neighbourData[row.start + 4]);
		break;
	case 6:
		printf("%u: Label %u (Start %u -> End %u): %u, %u, %u, %u, %u, %u\n", vertex, label, row.start, row.end, neighbourData[row.start], neighbourData[row.start + 1], neighbourData[row.start + 2], neighbourData[row.start + 3], neighbourData[row.start + 4], neighbourData[row.start + 5]);
		break;
	case 7:
		printf("%u: Label %u (Start %u -> End %u): %u, %u, %u, %u, %u, %u, %u\n", vertex, label, row.start, row.end, neighbourData[row.start], neighbourData[row.start + 1], neighbourData[row.start + 2], neighbourData[row.start + 3], neighbourData[row.start + 4], neighbourData[row.start + 5], neighbourData[row.start + 6]);
		break;
	case 0:
		printf("%u: Label %u (Start %u -> End %u)\n", vertex, label, row.start, row.end);
		break;
	default:
		printf("%u: Label %u (Start %u -> End %u):\n", vertex, label, row.start, row.end);
		for (unsigned int p = row.start; p < row.end; p++) {
			printf("%u, ", neighbourData[p]);
		}
		printf("\n");
	}
}

template<GraphType gType>
__global__
void printTrie(BaseGPUGraph<gType>* triegraphs, int labelCount) {

	unsigned int counter = 0;

	for (unsigned int i = 1; i <= labelCount; i++) {
		BaseGPUGraph<gType> graph = triegraphs[i];
		for (unsigned int i2 = 0; i2 < graph.vertexCount; i2++) {
			VertexRow row = graph.neighbours(i2);
			printRow(row, i, i2, graph.neighbourData);
			counter++;
		}
	}

	/*
	for (int i = 1; i <= labelCount; i++) {
		GPUGraph triegraph = triegraphs[i];
		printf("\nGraph Offsets %i: ", i);
		for (int i2 = 0; i2 < 68; i2++) {
			printf("%u, ", triegraph.neighbourOffsets[i2]);
		}
	}

	for (int i = 1; i <= labelCount; i++) {
		GPUGraph triegraph = triegraphs[i];
		printf("\nGraph Data %i: ", i);
		for (int i2 = 0; i2 < 68; i2++) {
			printf("%u, ", triegraph.neighbourData[i2]);
		}
	}*/

	/*
	printf("\nTest: ");
	for (int i = 1; i <= labelCount; i++) {
		GPUGraph triegraph = triegraphs[i];
		printf("\nGraph label %i: ", i);
		for (int i2 = 0; i2 < 10; i2++) {
			printf("(%u, %u), ", triegraph.vertexPairs[i2].vertices, triegraph.vertexPairs[i2].pointer);
		}
	}*/
	__nanosleep(1'000'000);
}

#ifdef CTGraph_Format
__host__
void buildConstantTable(unsigned int labelCount) {

	unsigned int CT_C_[CT_C_SIZE];

	std::mt19937 gen(0);
	std::uniform_int_distribution<uint32_t> distrib(0u, UINT32_MAX);
	const uint32_t PRIME = 4294967291u;

	for (unsigned int index = 0; index < labelCount; index++)
	{

		/*

		cudaErrorCheck(cudaMemset(
			vertices,
			UINT8_MAX,
			sizeof(CT_Bucket) * bucketCount
		));*/

		CT_C_[CIdx(index, 0)] = std::max(1u, (unsigned int)(distrib(gen) % PRIME));
		CT_C_[CIdx(index, 1)] = distrib(gen) % PRIME;

		//printf("Data %lu and %lu for index %lu", CT_C_[CIdx(index, 0)], CT_C_[CIdx(index, 1)], index);

	}

	cudaErrorCheck(cudaMemcpyToSymbol(CT_C, CT_C_, sizeof(CT_C_)));
}

__global__
void compareTrie(DefaultGPUGraph* graphs, GPUGraph* _graphs, int labelCount) {

	for (unsigned int label = 1; label <= labelCount; label++) {
		auto graph = graphs[label];
		auto _graph = _graphs[label];
		for (unsigned int vertex = 0; vertex < graph.vertexCount; vertex++) {
			VertexRow row = graph.neighbours(vertex);
			VertexRow _row = _graph.neighbours(vertex);

			unsigned int delta = row.end - row.start;
			unsigned int _delta = _row.end - _row.start;


			if (delta != _delta) {
				//printf("Length mismatch vertex %lu %lu - %lu (%lu) vs %lu - %lu  (%lu) %lu\n", vertex, row.start, row.end, row.end - row.start, _row.start, _row.end, _row.end - _row.start, label);
				__trap();
			}

			for (unsigned int i3 = 0; i3 < delta; i3++) {
				if (graph.neighbourData[row.start + i3] != _graph._neighbourData[_row.start + i3]) {
					//printf("Row mismatch vertex %lu %lu - %lu (%lu) vs %lu - %lu  (%lu) %lu\n", vertex, row.start, row.end, row.end - row.start, _row.start, _row.end, _row.end - _row.start, label);
					printRow(row, label, vertex, graph.neighbourData);
					//printf("Differs to\n");
					printRow(_row, label, vertex, graph._neighbourData);
					__nanosleep(1'000'000);
					__trap();
					break;
				}
			}
		}
	}
}

__global__
void printCT(GPUGraph* graphs, unsigned elems, int labelCount) {
	for (int i = 1; i <= labelCount; i++) {
		printf("\nLabel %i: ", i);
		for (unsigned int i2 = 0; i2 < elems; i2++) {
			printf("%lu, ", graphs[i]._neighbourOffsets[i2]);
		}
	}
}

template<int BlockSize>
__global__
void populateHashes(GPUGraph* graphs, unsigned int* g_neighbourData, int labelCount, int vertexCount) {
	int tdx = threadIdx.x;
	int idx = tdx + blockIdx.x * blockDim.x;

	if (idx < vertexCount) {
		for (int label = 1; label <= labelCount; label++) {
			unsigned int vertex = idx;

			DefaultGPUGraph* _graph = (DefaultGPUGraph*)graphs + label;
			GPUGraph* graph = graphs + label;

			VertexRow _row = _graph->neighbours(vertex);

			if (_row.end != 0) {
				if (!graph->insert(vertex)) {
					graph->overflow = true;
					return;
				}
			}

		}
	}
}

template<int BlockSize>
__global__
void populateOffsets(GPUGraph* graphs, int bucketCount, int label) {
	int tdx = threadIdx.x;
	int idx = tdx + blockIdx.x * blockDim.x;

	if (idx < bucketCount) {
		DefaultGPUGraph* graph = (DefaultGPUGraph*)graphs + label;
		GPUGraph* _graph = graphs + label;

		CT_Bucket* bucket = _graph->vertices + idx;

		unsigned int* neighbourOffset = _graph->_neighbourOffsets + _graph->getOffsetLoc(idx);

		for (unsigned int i = 0; i < BUCKET_SIZE; i++) {
			unsigned int vertex;
			if ((vertex = bucket->data[i]) != 0xffffffff) {
				VertexRow row = graph->neighbours(vertex);
				neighbourOffset[i] = row.end - row.start;
			}
			else {
				neighbourOffset[i] = 0;
			}
		}
	}
}

template<int BlockSize>
__global__
void populateNeighbours(GPUGraph* graphs, int bucketCount, int label) {
	int tdx = threadIdx.x;
	int idx = tdx + blockIdx.x * blockDim.x;

	if (idx < bucketCount) {
		DefaultGPUGraph* graph = (DefaultGPUGraph*)graphs + label;
		GPUGraph* _graph = graphs + label;

		CT_Bucket bucket = _graph->vertices[idx];

		for (unsigned int i = 0; i < BUCKET_SIZE; i++) {
			unsigned int vertex;
			if ((vertex = bucket.data[i]) != 0xffffffff) {
				VertexRow row = graph->neighbours(vertex);
				VertexRow _row = _graph->neighbours(vertex);

				if (row.end - row.start != _row.end - _row.start) {
					//printf("Row issue %lu %lu - %lu (%lu) vs %lu - %lu  (%lu) %lu\n", vertex, _row.start, _row.end, _row.end - _row.start, row.start, row.end, row.end - row.start, label);
					__trap();
				}

				for (unsigned i2 = 0; i2 < _row.end - _row.start; i2++) {
					_graph->_neighbourData[_row.start + i2] = graph->neighbourData[row.start + i2];
				}
			}
		}
	}
}

__global__
void memsetKernel(void* dst, const unsigned int val, size_t size) {
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t step = blockDim.x * gridDim.x;

	unsigned int* _dst = (unsigned int*)dst;

	for (size_t i = idx; i < size; i += step) {
		_dst[i] = val;
	}
}

#define HashLayerSize (sizeof(CT_Bucket) * labelCount * vertexCount)
#define OffsetLayerSize (sizeof(unsigned int)* (BUCKET_SIZE * labelCount * vertexCount + 1))
#define NeighbourLayerSize (sizeof(unsigned int) * edgeCount)

template<int BlockSize>
__global__
void attemptPopulateHashes(TempCSRGraph input, GPUGraph* graphs, unsigned int* g_neighbourData, int labelCount, int vertexCount, int attempts, void* cubTempStorage, size_t cubTempSize) {

	bool incomplete = false;

	if (attempts == 0) {
		incomplete = true;
	}

	if (attempts > 64) {
		__trap();
	}

	for (int i = 1; i <= labelCount; i++) {
		if (graphs[i].overflow) {
			graphs[i].bucketCount *= 4;
			graphs[i].bucketCount /= 3;
			incomplete = true;
			//printf("Graph %i overflow, new buckets %lu\n", i, graphs[i].bucketCount);
		}
		if (i > 1) {
			graphs[i].vertices = graphs[i - 1].vertices + graphs[i - 1].bucketCount;
			graphs[i]._neighbourOffsets = graphs[i - 1]._neighbourOffsets + graphs[i - 1].bucketCount * BUCKET_SIZE;
		}
		graphs[i].overflow = false;

		//printf("Graph %i %lu %p %p\n", i, graphs[i].bucketCount, graphs[i].vertices, graphs[i]._neighbourOffsets);
	}

	int blockCount = _ceil<int, BlockSize>(vertexCount);

	if (incomplete) {
		//printf("%p with %llu and val %lu\n", graphs[1].vertices, HashLayerSize, 0xff);
		memsetKernel << <1024, 1024 >> > (graphs[1].vertices, 0xffffffff, HashLayerSize);
		populateHashes <BlockSize> << <blockCount, BlockSize >> > (graphs, g_neighbourData, labelCount, vertexCount);
		attemptPopulateHashes<BlockSize> << <1, 1, 0, cudaStreamTailLaunch >> > (input, graphs, g_neighbourData, labelCount, vertexCount, attempts + 1, cubTempStorage, cubTempSize);
	}
	else {
		for (int label = 1; label <= labelCount; label++) {
			GPUGraph* graph = graphs + label;
			int blockCount = _ceil<int, BlockSize>(graph->bucketCount);
			populateOffsets<BlockSize> << <blockCount, BlockSize >> > (graphs, graph->bucketCount, label);
		}

		unsigned int bucketTotal = 0;

		for (int label = 1; label <= labelCount; label++) {
			GPUGraph* graph = graphs + label;
			bucketTotal += graph->bucketCount;
			//printf("Scan for %i %i %p to %p (max %p)\n", label, graph->bucketCount * BUCKET_SIZE, graph->_neighbourOffsets, graph->_neighbourOffsets + graph->bucketCount * BUCKET_SIZE, graphs[1]. _neighbourOffsets + (OffsetLayerSize / sizeof(unsigned int)));
		}

		BasicAdd scanOp;
		//printf("buckCount %lu \n", bucketTotal);
		cub::DeviceScan::ExclusiveScan(cubTempStorage, cubTempSize, graphs[1]._neighbourOffsets, scanOp, 0, bucketTotal * BUCKET_SIZE);

		for (int label = 1; label <= labelCount; label++) {
			GPUGraph* graph = graphs + label;
			populateNeighbours<BlockSize> << <blockCount, BlockSize >> > (graphs, graph->bucketCount, label);
		}

		//compareTrie << <1, 1 >> > ((DefaultGPUGraph*)graphs, graphs, labelCount);

		for (int label = 1; label <= labelCount; label++) {
			GPUGraph* graph = graphs + label;
			cudaMemcpyAsync(&(graph->neighbourData), &(graph->_neighbourData), sizeof(graph->_neighbourData), cudaMemcpyDefault, 0);
		}
	}



}
#endif


template<int Amount, bool isWarp>
__device__ __forceinline__
int coopAdd(int* ptr, int gdx, int& broadcast) {

	if (gdx == 0) {
		broadcast = atomicAdd(ptr, Amount);
	}

	if constexpr (isWarp) {
		broadcast = __shfl_sync(0xffffffff, broadcast, 0);
	}
	else {
		__syncthreads();
	}
	

	return broadcast;
}

__device__ __forceinline__
unsigned int parseNum(char* & c) {
	unsigned int val = 0;
	for (; *c >= 48 && *c <= 57; c++) {
		unsigned char ch = (unsigned char) *c;
		val *= 10;
		val += (ch - 48);
	}

	c++;

	return val;
}

class VertexPairItr {
public:
	using iterator_category = std::forward_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using value_type = unsigned int;
	using pointer = unsigned int*;
	using reference = unsigned int&;

	__device__
	reference operator*() const { return pairs -> pointer; }

	__device__
	VertexPairItr(VertexPair* ptr) { pairs = ptr; }

	__device__
	reference operator[](std::size_t idx) {
		return pairs[idx].pointer;
	}

	__device__
	VertexPairItr operator+(difference_type n) {
		return pairs + n;
	}

private:
	VertexPair* pairs;
};

template<int Decomp, int BlockSize>
__global__
void bitfieldExpansion(DefaultGPUGraph* graphs, int threadCount) {
	int wid = threadIdx.x / 32;
	int wdx = threadIdx.x % 32;

	int idx = wid * Decomp + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	for (int xDisp = wdx; xDisp < Decomp; xDisp+=32) {
		int x = idx + xDisp;
		if (x < threadCount) {
			VertexPair* pair = (graphs[y+1].vertexPairs) + x;
			pair->pointer = __popc(pair->vertices);
		}
	}
}

template<int Decomp, int BlockSize>
__global__
void finishScan(DefaultGPUGraph* graphs, int threadCount, unsigned int label) {
	int wid = threadIdx.x / 32;
	int wdx = threadIdx.x % 32;

	int idx = wid * Decomp + blockIdx.x * blockDim.x;


	for (int xDisp = wdx; xDisp < Decomp; xDisp += 32) {
		int x = idx + xDisp;
		if (x < threadCount) {
			VertexPair* pair = (graphs[label].vertexPairs) + x;

			unsigned int missing = 0;
			for (unsigned int __label = 1; __label < label; __label++) {
				DefaultGPUGraph* graph = graphs  + __label;

				unsigned lastVertex = graph->vertexCount - 1;
				VertexPair lastPair = graph->getPair(lastVertex);

				missing += __popc(lastPair.vertices) + lastPair.pointer;
			}

			//pair->pointer += missing;
		}
	}
}

__global__
void csrToTrie(TempCSRGraph input, DefaultGPUGraph* graphs, unsigned int* g_neighbourData, int threadCount) {
	int tdx = threadIdx.x;
	int idx = tdx + blockIdx.x * blockDim.x;

	//Find boundary conditions along input edges

	LabelEdge src0;
	LabelEdge src1 = input.sortedEdges[idx];
	bool diff = false;

	if (idx < threadCount) {

		g_neighbourData[idx] = src1.right;

		if (idx > 0) {
			src0 = input.sortedEdges[idx - 1];
			bool diffLabel = src0.label != src1.label;
			bool diffsrc = src0.left != src1.left;

			diff = diffLabel || diffsrc;
		}
	}

	if (diff) {
		int label = src1.label;

		DefaultGPUGraph graph = graphs[label];

		unsigned neighbourOffset = graph.getOffset(src1.left);

		graph.neighbourOffsets[neighbourOffset] = idx;

	}

	if (idx == threadCount - 1) {
		int label = src1.label;

		DefaultGPUGraph graph = graphs[label];

		unsigned neighbourOffset = graph.getOffset(src1.left);

		graph.neighbourOffsets[neighbourOffset + 1] = threadCount;

	}
}

template<int ThreadDist, int BlockSize>
__global__ 
void parseCSR(char* input, int inputSize, TempCSRGraph output, int* parsed, int* completion, void* cubTempStorage, size_t cubTempSize,
	DefaultGPUGraph* triegraphs, VertexPair* g_vertexPairs, unsigned int* g_neighbourOffsets, unsigned int* g_neighbourData, int labelCount) {
	using BlockScan = cub::BlockScan<int, BlockSize>;

	const int BlockSplit = BlockSize * ThreadDist;

	int tdx = threadIdx.x;

	int splits;
	char* inputEnd = input + inputSize;

	__shared__ int broadcast;
	__shared__ unsigned int broadcast2;

	__shared__ typename BlockScan::TempStorage temp_storage;

	while ((splits = coopAdd<BlockSplit, false>(parsed, tdx, broadcast)) < inputSize) {

		
		LabelEdge edge;
		bool isEdge = false;
		int offset = tdx * ThreadDist;

		char* c = input + offset + splits;

		int steps;

		for (steps = 0; steps < ThreadDist; steps++, c++) {
			if (*c == '\n') {
				c++;
				break;
			}
		}

		if (c < inputEnd) {

			if (steps != ThreadDist) {
				switch (*c) {
				case 'e':
					isEdge = true;
				case 'v':
					break;

		#ifdef SAFEPARSE						
				default:
					__trap(); //Parsing Error!
		#endif
				}

				c++;
		#ifdef SAFEPARSE
				if (*c != ' ') {
					__trap(); //Parsing Error!
				}
		#endif

				c++;
				unsigned int val0 = parseNum(c);
				unsigned int val1 = parseNum(c);

				if (isEdge) {
					unsigned int val2 = parseNum(c);
					edge = { val0, val1, val2};

					unsigned int leftdiv = val0 / 32;
					unsigned int leftrem = val0 % 32;

					atomicOr(&(triegraphs[val2].vertexPairs[leftdiv].vertices), 1 << leftrem);
				}
				else {
					output.vertexLabels[val0] = val1;
				}
			}

		}

		int scan;

		int edgeAmount = isEdge;

		BlockScan(temp_storage).ExclusiveSum(edgeAmount, scan);

		__syncthreads();

		if (tdx == BlockSize - 1) {
			broadcast2 = atomicAdd(output.edgeCount, scan + isEdge);
		}

		__syncthreads();

		if (isEdge) {
			output.edges[broadcast2 + scan] = edge;
		}

	}
	
	if (tdx == 0) {
		if (atomicAdd(completion, 1) == gridDim.x - 1) {
			int edgeCount = atomicAdd(output.edgeCount, 0); //Force Propogation!
			int vertexCount = triegraphs[1].vertexCount;

			cub::DeviceRadixSort::SortKeys<LabelEdge, unsigned int, LabelEdge_decomposer>(cubTempStorage, cubTempSize, output.edges, output.sortedEdges, edgeCount, LabelEdge_decomposer{});

			const int ExpansionDecompSize = 128;

			int threadCount = _ceil<int, 32>(vertexCount);
			int blockCount = _ceil<int, BlockSize>(threadCount);


			dim3 dBlockCount(blockCount, labelCount, 1);
			dim3 dBlockShape(BlockSize, 1, 1);


			bitfieldExpansion<ExpansionDecompSize, BlockSize><<<dBlockCount, dBlockShape >>>(triegraphs, threadCount);
			
			BasicAdd scanOp;

			size_t requiredSize = vertexCount * labelCount * sizeof(VertexPair) / 32;

			if (requiredSize > cubTempSize) {
				__trap();
				for (unsigned int __label = 1; __label < labelCount; __label++) {
					VertexPairItr itr(triegraphs[1].vertexPairs);

					cub::DeviceScan::ExclusiveScan(cubTempStorage, cubTempSize, itr, itr, scanOp, 0, vertexCount/32);

				}

				for (unsigned int __label = 1; __label < labelCount; __label++) {

					finishScan<ExpansionDecompSize, BlockSize> << <blockCount, BlockSize >> > (triegraphs, threadCount, __label);
				}

				csrToTrie << < blockCount, BlockSize >> > (output, triegraphs, g_neighbourData, edgeCount);
			}
			else {
				VertexPairItr itr(triegraphs[1].vertexPairs);
				cub::DeviceScan::ExclusiveScan(cubTempStorage, cubTempSize, itr, itr, scanOp, 0, ((vertexCount/32) + 32) * labelCount );
				blockCount = edgeCount / BlockSize + 1;
				csrToTrie << < blockCount, BlockSize >> > (output, triegraphs, g_neighbourData, edgeCount);
			}

#ifdef CTGraph_Format
			attemptPopulateHashes<BlockSize> << <1, 1 >> > (output, (GPUGraph*)triegraphs, g_neighbourData, labelCount, vertexCount, 0, cubTempStorage, cubTempSize);
#endif
		
		}
	}

}

__global__
void printGraph1(TempCSRGraph output) {
	printf("Printing Unordered Edge Data:");
	for (unsigned int i = 0; i < *(output.edgeCount); i++) {
		LabelEdge edge = output.edges[i];
		printf("(%i, %i, %i), ", edge.left, edge.right, edge.label);
		if (i % 8 == 7) {
			printf("\n");
		}
	}
}

__global__
void printGraph2(TempCSRGraph output) {
	printf("Printing Ordered Edge Data:");

	for (unsigned int i = 0; i < *(output.edgeCount); i++) {
		LabelEdge edge = output.sortedEdges[i];
		printf("(%i, %i, %i), ", edge.left, edge.right, edge.label);
		if (i % 8 == 7) {
			printf("\n");
		}
	}
}

__global__
void printGraph3(GPUGraph* triegraphs) {

	printf("Offset Data:");

	GPUGraph graph = triegraphs[1];

	for (unsigned int vertex = 0; vertex < graph.vertexCount; vertex++) {
		VertexRow row = graph.neighbours(vertex);
		if (row.end != 0) {
			printf("(%lu: %lu - %lu), ", vertex, row.start, row.end);
		}
	}

	__nanosleep(1'000'000);
}

//Parse the values of the file
int parseHeader(char* start, char* end, unsigned int* values) {
	unsigned int currentValue = 0;

	for (char* chptr = start; chptr < end; chptr++) {

		char ch = *chptr;

		if (ch >= 48 && ch <= 57) {
			unsigned int i = ch - 48;
			values[currentValue] *= 10;
			values[currentValue] += i;
		}
		else if (ch == ' ') {
			currentValue++;
		}
		else if (ch == '\n') {
			return (int)(chptr - start + 1);
		}
		else {
			return -1;
		}
	}

	return -1;
}

__global__ void dummy() {

}

size_t preallocUsed = 0;

template<size_t MaxPrealloc = PREALLOC, typename T>
__host__
void p_malloc(T** alloc, size_t size, unsigned char* prealloc) {

	size_t allignedPop = size + (128 - size % 128);

	*alloc = (T*)(prealloc + preallocUsed);

	preallocUsed += allignedPop;

	if (preallocUsed > MaxPrealloc) {
		info_printf("Prealloc Overflow! %llu\n", size);
		csv_printf("Prealloc Overflow! %llu,", size);
		exit(1);
	}
}

__host__ GPUGraph* parseGraph(std::string graphLoc, unsigned char* prealloc, int& _vertexCount, int& _edgeCount) {

	auto fstart = std::chrono::steady_clock::now();
	FILE* f = fopen(graphLoc.c_str(), "r");

	if (f == NULL) {
		info_printf("\nMissing file %s", graphLoc.c_str());
		//std::cout << "\nMissing file " << graphLoc << std::endl;
		exit(12);
	}

	fseek(f, 0, SEEK_END);
	long csize = ftell(f);
	rewind(f);


	char* cbuf = new char[csize];

	fread(cbuf, sizeof(char), csize, f);

	auto fend = std::chrono::steady_clock::now();
	std::chrono::duration<double> felapsed_seconds = fend - fstart;
	info_printf("Access Time: %f\n", felapsed_seconds.count());

	if (strncmp(cbuf, "t # 0\n", 6)) {
		info_printf("Error: Wrong Data Graph Format, Missing \"t # 0\\n\" \n");
		exit(76);
	}

	if (strncmp(cbuf + csize - 6, "t # -1", 6)) {
		info_printf("Error: Wrong Data Graph Format, Missing \"t # -1\" \n");
		exit(77);
	}

	unsigned int header[4]{};

	//Parse Header
	int headerLength = parseHeader(cbuf + 6, cbuf + csize, header);
	if (headerLength < 0) {
		info_printf("Error: Malformed Data Graph Header");
		exit(78);
	}

	unsigned int vertexCount = header[0];
	unsigned int edgeCount = header[1];
	//unsigned int vertexLabelCount = header[2];
	unsigned int edgeLabelCount = header[3];

	//unsigned int padding = CurBlockSize;

	char* input;

	unsigned int* tempEdgeCount, * tempVertexLabels;
	LabelEdge *tempEdges, *tempSortedEdges;

	int* parsed, *completion;
	void* cubTempStorage;

	int trieCount = edgeLabelCount + 1;

	GPUGraph* triegraphs;
	VertexPair *g_vertexPairs;
	unsigned int *g_neighbourOffsets, *g_neighbourData;

	GPUGraph* h_triegraphs = new GPUGraph[trieCount];

	int inputSize = csize - headerLength - 6 - 6;
	size_t cubTempSize = edgeCount * sizeof(LabelEdge) * 2;

#ifdef CTGraph_Format

	int labelCount = trieCount;
	if (cubTempSize < 2 * OffsetLayerSize) {
		cubTempSize = 2 * OffsetLayerSize;
	}
#endif

	auto cstart = std::chrono::steady_clock::now();

	p_malloc(&input, inputSize, prealloc);

	p_malloc(&tempEdgeCount, sizeof(unsigned int), prealloc);
	p_malloc(&tempVertexLabels, vertexCount * sizeof(unsigned int), prealloc);
	p_malloc(&tempEdges, edgeCount * sizeof(LabelEdge), prealloc);
	p_malloc(&tempSortedEdges, edgeCount * sizeof(LabelEdge), prealloc);
	p_malloc(&completion, sizeof(int), prealloc);
	p_malloc(&parsed, sizeof(int), prealloc);
	p_malloc(&cubTempStorage, cubTempSize, prealloc);

	p_malloc(&triegraphs, sizeof(GPUGraph) * trieCount, prealloc);
	p_malloc(&g_vertexPairs, (sizeof(VertexPair) * trieCount * vertexCount) / 32 + vertexCount, prealloc);
	p_malloc(&g_neighbourOffsets, sizeof(unsigned int) * (trieCount * vertexCount + 1), prealloc);
	p_malloc(&g_neighbourData, sizeof(unsigned int) * edgeCount, prealloc);


#ifdef CTGraph_Format
	CT_Bucket* g_hashlayer;
	p_malloc(&g_hashlayer, HashLayerSize, prealloc);

	unsigned int* padded_g_neighbourOffsets; //Sadly their offsets is not contigious so needs to be reallocated.
	p_malloc(&padded_g_neighbourOffsets, OffsetLayerSize, prealloc);

	unsigned int* padded_g_neighbourData; //Sadly their offsets is not contigious so needs to be reallocated.
	p_malloc(&padded_g_neighbourData, NeighbourLayerSize, prealloc);

	buildConstantTable(labelCount);
#endif

	for (int i = 0; i < trieCount; i++) {

		h_triegraphs[i].edgeCount = (int)edgeCount;
		h_triegraphs[i].vertexCount = (int)vertexCount;
		h_triegraphs[i].vertexPairs = g_vertexPairs + ((i * vertexCount) / 32 + i);
		h_triegraphs[i].neighbourOffsets = g_neighbourOffsets;
		h_triegraphs[i].neighbourData = g_neighbourData;

#ifdef CTGraph_Format
		h_triegraphs[i].vertices = g_hashlayer;
		h_triegraphs[i].bucketCount = _ceil(vertexCount * 2, BUCKET_SIZE);
		h_triegraphs[i]._neighbourOffsets = padded_g_neighbourOffsets;
		h_triegraphs[i]._neighbourData = padded_g_neighbourData;
#endif
	}
/*
	for (int i = 0; i < trieCount; i++) {
#ifdef CTGraph_Format
		h_triegraphs[i] = { (int)edgeCount, (int)vertexCount, g_vertexPairs + ((i * vertexCount) / 32 + i), g_neighbourOffsets, g_neighbourData, (unsigned int)i + 1, 
			false, g_hashlayer, _ceil(vertexCount * 2, BUCKET_SIZE), padded_g_neighbourOffsets, padded_g_neighbourData };
#else
		h_triegraphs[i] = { (int)edgeCount, (int)vertexCount, g_vertexPairs + ((i * vertexCount) / 32 + i), g_neighbourOffsets, g_neighbourData, (unsigned int)i + 1 };
#endif

	}
*/
	_edgeCount = edgeCount;
	_vertexCount = vertexCount;

	cudaMemset(input, '\n', 1); //Pad the end!
	cudaMemset(input + inputSize, '\n', 1024); //Pad the end!
	cudaMemcpy(input, cbuf + 6 + headerLength, inputSize, cudaMemcpyDefault);
	cudaMemcpy(triegraphs + 1, h_triegraphs, sizeof(GPUGraph) * trieCount, cudaMemcpyDefault);

	cudaMemset(completion, 0, sizeof(int));
	cudaMemset(parsed, 0, sizeof(int));
	cudaMemset(tempEdgeCount, 0, sizeof(unsigned int));

	int BlockCount;

	int device;
	cudaDeviceProp deviceProp;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&deviceProp, device);
	cudaError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&BlockCount, parseCSR<4, CurBlockSize>, CurBlockSize, 0));

	info_printf("Block per SM %i SM count %i\n", BlockCount, deviceProp.multiProcessorCount);


	BlockCount *= deviceProp.multiProcessorCount;

	//(char* input, int inputSize, TempCSRGraph output, int* completion, void* cubTempStorage, size_t cubTempSize);

	TempCSRGraph output{ tempEdgeCount , tempVertexLabels, tempEdges, tempSortedEdges };

	cudaErrorSync();

	auto cend = std::chrono::steady_clock::now();
	std::chrono::duration<double> celapsed_seconds = cend - cstart;
	info_printf("Setup Time: %f\n", celapsed_seconds.count());

	info_printf("Parsing Workload: Size %i with BlockSize %u, Vertices %u edges %u input Size %u\n", BlockCount, CurBlockSize, vertexCount, edgeCount, inputSize);

	auto start = std::chrono::steady_clock::now();

	parseCSR<4, CurBlockSize> << <BlockCount, CurBlockSize>>> (input, inputSize, output, parsed, completion, cubTempStorage, cubTempSize, 
																(DefaultGPUGraph*) triegraphs, g_vertexPairs, g_neighbourOffsets, g_neighbourData, edgeLabelCount);
	cudaErrorSync();

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	info_printf("GPU Parse Time: %f\n", elapsed_seconds.count());
	csv_printf("%fs, ", elapsed_seconds.count());

	//printGraph << <1, 1 >> > (output);
	// 

	//printGraph2 << <1, 1 >> > (output);
	//printGraph1 << <1, 1 >> > (output);	
	//printGraph3 << <1, 1 >> > (triegraphs);

	//printTrie << <1, 1 >> > (triegraphs, edgeLabelCount);

	cudaErrorSync();

	return triegraphs;
}