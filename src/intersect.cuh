#pragma once
#include <cstdint>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>

template<typename T>
__device__ __forceinline__
T __ceil(T val, T div) {
	return (val + div - 1) / div;
}

template<typename T>
__device__ __host__
constexpr int __popcount(T val) {
	int result = 0;
	for (int i = 0; i < sizeof(T) * 8; i++) {
		result += ((val) >> (i)) % 2;
	}
	return result;
}

__device__ __forceinline__
unsigned int __lds(int address) {
	unsigned int val;
	asm("ld.shared.u32 %0, [%1];": "=r" (val) : "r" (address));
	return val;
}

__device__ __forceinline__
unsigned int __lds(int address, int offset) {
	return __lds(address + offset * sizeof(unsigned int));
}

__device__ __forceinline__
void __sts(int address, unsigned int val) {
	asm("st.shared.u32 [%0], %1;":: "r" (address), "r" (val));
}

__device__ __forceinline__
void __sts(int address, int offset, unsigned int val) {
	__sts(address + offset * sizeof(unsigned int), val);
}

template<unsigned int LeafCount>
__device__ __forceinline__
bool binarySearchFixed(unsigned int lval, unsigned int* list, unsigned int start, unsigned int end, unsigned int& index) {

	static_assert(__popcount(LeafCount) == 1, "Leaf Count must be a power of 2");

	unsigned int rval;

	index = start + LeafCount / 2;

	for (unsigned int delta = LeafCount / 4; delta > 0; delta /= 2) { //For loop produces worst codegen :(
		if (list[index] > lval) {
			index -= delta;
		}
		else {
			index += delta;
		}
	}

	index -= (list[index] > lval);

	if (index < end && index > start) {
		rval = list[index];
	}

	return lval == rval;

}

template<unsigned int Decomp = 0>
__device__ __forceinline__
bool binarySearch(unsigned int lval, unsigned int* list, unsigned int start, unsigned int end, unsigned int& index) {

	if constexpr (Decomp) {
		if (end - start == Decomp) {
			return binarySearchFixed<Decomp>(lval, list, start, end, index);
			//return binarySearchFixed2<Decomp>(lval, (char*) (list + start));
		}
	}

	unsigned int rval;

	do {
		index = (end + start) / 2;

		rval = list[index];

		if (lval < rval) {
			end = index;
		}
		else {
			start = index + 1;
		}
	} while (lval != rval && start != end);

	return lval == rval;

}


template<unsigned int CoopSize = 32, unsigned int Decomp = 0>
__device__ __forceinline__
void intersect(unsigned int* list0, int list0Size, unsigned int* list1, int list1Size, unsigned int* output, unsigned int& outputSize, unsigned int wdx)

{

	static_assert(__popcount(CoopSize) == 1, "CoopSize must be a power of 2");
	static_assert(CoopSize <= 32, "CoopSize must be less than or equal to a warp");
	static_assert((Decomp && CoopSize > 1) || !Decomp, "Decomposition requires a greater than 1 Coop Size");


	int l1Start = 0;
	unsigned int lastIndex = 0;

	int subWarpWork = __ceil(list0Size, (int)CoopSize) * CoopSize;

	for (int i = wdx; i < subWarpWork; i += CoopSize) {

		bool pred = false;
		unsigned int lval = list0[i];

		if (i < list0Size) {
			if constexpr (Decomp) {

				if (true) {

					l1Start = lastIndex;

					//int steps = 0;

					while (l1Start + Decomp < list1Size) {
						if (lval >= list1[l1Start + Decomp]) {
							l1Start += Decomp;
							//steps++;
						}
						else {
							break;
						}
					}

					int l1SubEnd = min(l1Start + Decomp, list1Size);

					pred = binarySearch<Decomp>(lval, list1, l1Start, l1SubEnd, lastIndex);

				}
			}
			else {

				pred = binarySearch(lval, list1, lastIndex, list1Size, lastIndex);
			}
		}

		if constexpr (CoopSize > 1) {

			if (pred) {
				unsigned int writeLoc = atomicAdd(&outputSize, 1);
				output[writeLoc] = lval;
			}

			/*
			unsigned int writeMask = __ballot_sync(coopMask, pred);
			if (pred) {
				unsigned int writeLoc = outputSize + __popc(writeMask & localMask);
				output[writeLoc] = lval;
			}

			outputSize += __popc(writeMask);*/
		}
		else {
			if (pred) {
				output[outputSize] = lval;
				outputSize++;
			}
		}

	}

}

__device__ __forceinline__
void scanIntersect(unsigned int* list0, unsigned int list0Size, unsigned int* list1, unsigned int list1Size, unsigned int* output, unsigned int& outputSize) {
	int i0 = 0;
	int i1 = 0;

	while (i0 < list0Size && i1 < list1Size) {
		unsigned int lval = list0[i0];
		unsigned int rval = list1[i1];

		if (lval >= rval) {
			i1++;
			if (lval == rval) {
				output[outputSize] = lval;
				outputSize ++;
			}
		}
		else {
			i0++;
		}
	}


}


__device__ long long int clockTiming() {
	__threadfence();
	long long int cl = clock64();
	__threadfence();
	return cl;
}

__global__ void testFunc(unsigned int* list0, unsigned int list0Size,
	unsigned int* list1, unsigned int list1Size,
	unsigned int* output, unsigned int* expected, unsigned int expectedSize) {

	unsigned int wdx = threadIdx.x % 32;
	__shared__ unsigned int outputSize[4];


	__syncwarp();

	long long int start = clockTiming();

	__syncwarp();

	const unsigned int subWarpSize = 4;
	const unsigned int samples = 10000;

	for (int i = 0; i < samples; i++) {
		if (wdx == 0) {
			outputSize[threadIdx.x / 32] = 0;
		}

		__syncwarp();

		if (wdx == 0) {
			if (list0Size < list1Size) {
				scanIntersect(list0, list0Size, list1, list1Size, output, outputSize[threadIdx.x / 32]);
				//intersect<subWarpSize, 128>(list0, list0Size, list1, list1Size, output, outputSize[threadIdx.x / 32], wdx);
			}
			else {
				scanIntersect(list1, list1Size, list0, list0Size, output, outputSize[threadIdx.x / 32]);
				//intersect<subWarpSize, 128>(list1, list1Size, list0, list0Size, output, outputSize[threadIdx.x / 32], wdx);
			}
		}
	}

	__syncwarp();

	long long int end = clockTiming();

	long long int avg = (end - start) / ((long long int)(samples));

	if (wdx == 0) {
		printf("\nTime Taken (Avg): %llu cycles, list0 Size: %i list1 Size: %i Output Size: %u\n", avg, list0Size, list1Size, outputSize[threadIdx.x / 32]);

		/*
		printf("\nList 0:");
		for (int i = 0; i < list0Size; i++) {
			printf("%i,", list0[i]);
		}

		printf("\nList 1:");
		for (int i = 0; i < list1Size; i++) {
			printf("%i,", list1[i]);
		}*/

		printf("\nExpected (%i):", expectedSize);
		for (int i = 0; i < expectedSize; i++) {
			printf("%i,", expected[i]);
		}


		printf("\nResult (%i):", outputSize[threadIdx.x / 32]);
		for (int i = 0; i < outputSize[threadIdx.x / 32]; i++) {
			printf("%i,", output[i]);
		}

		/*
		printf("\Expected (%i):", outputSize);
		for (int i = 0; i < outputSize; i++) {
			printf("%i,", expected[i]);
		}
		printf("\n");*/


		for (int i = 0; i < outputSize[threadIdx.x / 32] && i < expectedSize; i++) {
			if (output[i] != expected[i]) {
				printf("\nError at %i with value %i vs %i\n", i, output[i], expected[i]);
				//__trap());
			}
		}

		if (outputSize[threadIdx.x / 32] < expectedSize) {
			printf("\nError: Output is too small %i vs %i\n", outputSize[threadIdx.x / 32], expectedSize);
			//__trap());
		}

		if (outputSize[threadIdx.x / 32] > expectedSize) {
			printf("\nError: Output is too large %i vs %i\n", outputSize[threadIdx.x / 32], expectedSize);
			//__trap());
		}

	}
}

__host__ void profileIntersect() {
#define MAXSETSIZE 30
#define MAXVAL (MAXSETSIZE * 16)
#define SEED 111

	std::set<unsigned int> set0, set1;
	std::default_random_engine eng{ SEED };
	std::uniform_int_distribution<int> dist1(1, MAXSETSIZE);
	std::uniform_int_distribution<int> dist2(1, MAXVAL);

	int list0Size = dist1(eng);
	int list1Size = dist1(eng);

	for (int i = 0; i < list0Size; i++) {
		set0.insert(dist2(eng));
	}

	for (int i = 0; i < list1Size; i++) {
		set1.insert(dist2(eng));
	}

	std::vector <unsigned int> vec0, vec1, expected;

	vec0.reserve(set0.size());
	std::copy(set0.begin(), set0.end(), std::back_inserter(vec0));

	vec1.reserve(set1.size());
	std::copy(set1.begin(), set1.end(), std::back_inserter(vec1));

	for (int i : set0) {
		if (auto search = set1.find(i); search != set1.end()) {
			expected.push_back(i);
		}
	}

	list0Size = (int) vec0.size();
	list1Size = (int) vec1.size();
	int expectedSize = (int) expected.size();

	unsigned int* glist0, * glist1, * output, * gexpected;

	cudaMalloc(&glist0, sizeof(unsigned int) * list0Size * 4);
	cudaMemset(glist0, -1, sizeof(unsigned int) * list0Size * 4);
	cudaMemcpy(glist0, &vec0[0], sizeof(unsigned int) * list0Size, cudaMemcpyHostToDevice);

	cudaMalloc(&glist1, sizeof(unsigned int) * list1Size * 4);
	cudaMemset(glist1, -1, sizeof(unsigned int) * list1Size * 4);
	cudaMemcpy(glist1, &vec1[0], sizeof(unsigned int) * list1Size, cudaMemcpyHostToDevice);

	cudaMalloc(&output, std::min(list0Size, list1Size) * sizeof(unsigned int));


	cudaMalloc(&gexpected, expectedSize * sizeof(unsigned int));
	cudaMemcpy(gexpected, &expected[0], sizeof(unsigned int) * expectedSize, cudaMemcpyHostToDevice);

	testFunc << <1, 32 >> > (glist0, list0Size, glist1, list1Size, output, gexpected, expectedSize);

	if (cudaError_t error = cudaDeviceSynchronize()) {
		printf("Error: %s,", cudaGetErrorString(error));
		cudaDeviceReset();
		exit(1);
	}
}

void cudaErrorSync() {
	if (cudaError_t error = cudaDeviceSynchronize()) {
		info_printf("Error: %s\n", cudaGetErrorName(error));
		csv_printf("Error: %s\n", cudaGetErrorName(error));
		cudaDeviceReset();
		exit(error);
	}
}

void cudaErrorCheck(cudaError_t error) {
	if (error) {
		info_printf("Error: %s\n", cudaGetErrorName(error));
		csv_printf("Error: %s\n", cudaGetErrorName(error));
		cudaDeviceReset();
		exit(error);
	}
}