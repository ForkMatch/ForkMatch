#pragma once

#include <cstdint>
#include <bitset>
#include <algorithm>
#include <vector>
#include "environment.cuh"

struct CCSRElem {
	unsigned int elem;

	static const unsigned int vertmask = ((1 << CCSRVertSize) - 1);
	static const unsigned int relmask = ((1 << CCSRRelSize) - 1);
	static const unsigned int degmask = ((1 << CCSRDegSize) - 1);

	static const unsigned int relOff = CCSRVertSize;
	static const unsigned int degOff = CCSRVertSize + CCSRRelSize * (CCSRDegSize > 0);

	static const unsigned int s_vertmask = ((unsigned(1) << CCSRVertSize) - 1);
	static const unsigned int s_relmask = ((unsigned(1) << CCSRRelSize) - 1) << relOff;
	static const unsigned int s_degmask = ((unsigned(1) << CCSRDegSize) - 1) << degOff;

	static const unsigned int MaxDeg = degmask;

	__device__ __host__ __forceinline__ CCSRElem(unsigned int _vert, unsigned int _rel, unsigned int _deg) {

		unsigned int vert = (_vert & vertmask);
		unsigned int rel = (_rel & relmask);
		unsigned int deg = (_deg & degmask);

#ifdef CCSR_ENCODE_ASSERT
#ifndef __CUDA_ARCH__
		if (_deg != deg) {
			deg = MaxCCSREncode;
		}

		if (vert != _vert || rel != _rel) {
			info_printf("Graph data loss!\n", 0);
			csv_printf("Graph data loss!,", 0);
			exit(67);
		}
#endif 
#endif

		if constexpr (CCSRDegSize) {
			elem = vert | (rel << relOff) | (deg << degOff);
		}
		else {
			elem = vert | (rel << relOff);
		}

		
	}

	__device__ __host__ __forceinline__ CCSRElem(unsigned int _vert, unsigned int _rel) : CCSRElem(_vert, _rel, MaxCCSREncode) {}

	__device__ __host__ __forceinline__ CCSRElem(unsigned int _elem) {
		elem = _elem;
	}

	__device__ __host__ CCSRElem() {
		elem = 0;
	}


	__device__ __host__ __forceinline__ unsigned int vert() {
		return elem & s_vertmask;
	}

	__device__ __host__ __forceinline__ unsigned int rel() {
		return (elem & s_relmask) >> relOff;
	}

	__device__ __host__ __forceinline__ unsigned int deg() {
		return (elem & s_degmask) >> degOff;
	}

	__host__ __device__ __forceinline__ CCSRElem removeDeg() {
		if constexpr (CCSRDegSize == 0) {
			return *this;
		}
		return { elem | ~s_degmask };
	}

	__host__ __device__ __forceinline__ constexpr CCSRElem& operator=(const CCSRElem& _elem) {
		elem = _elem.elem;
		return *this;
	}

	//__host__ __device__ __forceinline__ constexpr bool operator==(CCSRElem const& rhs) const { 
	//	return elem == rhs.elem; 
	//}

	__host__ __device__ __forceinline__ operator unsigned int() const { return elem; }

};

static_assert(sizeof(CCSRElem) == sizeof(unsigned int), "CCSR Elems should be interchangable with Int 32");

__device__ __host__ bool nodeCompare(CCSRElem seg0, CCSRElem seg1) {
	return (seg0.vert()) < (seg1.vert());
}

struct StaggerInfo {
	unsigned int size;
	unsigned int offset;
	unsigned int normOffset;
};


//Legacy Data Structure kept around so I don't need to reimplement query optimiser
struct CCSR {
	unsigned int count; //Number of vertices
	unsigned int size; //Size of data (in elements)
	CCSRElem* data;

	//Conversion Processes
	unsigned int* vertices, *localVertices, *invVertices;
	StaggerInfo* staggerInfo;
	unsigned int maxDegree;

	void print() {

		unsigned int rowLen;

		const unsigned int maxPrint = 300;
		unsigned int curPrint = 0;

		printf("CCSR Masks: %s, %s, %s\n", std::bitset<32>(CCSRElem().s_vertmask).to_string().c_str(),
											std::bitset<32>(CCSRElem().s_relmask).to_string().c_str(),
											std::bitset<32>(CCSRElem().s_degmask).to_string().c_str());

		printf("Graph Data:\n");

		for (unsigned int i = 0; i < size; i += rowLen + 1) {
			rowLen = data[i];
			printf("Len (%i):|", rowLen);
			for (unsigned int i2 = 1; i2 <= rowLen; i2++) {
				CCSRElem elem = data[i + i2];
				printf("(%u, %u, (%u)), ", elem.vert(), elem.rel(), elem.elem);
			}
			printf("|\n");
			curPrint++;
			if (curPrint >= maxPrint) {
				return;
			}
		}

		printf("Graph Vertex Locs:\n");

		for (unsigned int i = 0; i < count; i++) {
			printf("%u, ", vertices[i]);
		}
	}

	template<bool useInv = false>
	unsigned int demangle(unsigned int vertice) {
		unsigned int mDegree = data[vertice].elem;
		StaggerInfo mSInfo = staggerInfo[mDegree - 1];
		unsigned int mNormIndex = mSInfo.normOffset + (vertice - mSInfo.offset) / (mDegree + 1);
		if constexpr (useInv) {
			return invVertices[mNormIndex];
		}
		else {
			return mNormIndex;
		}
	}

	//Using vertices gets vertex based on their input file ID, not where they are in the CCSR
	unsigned int relativeVertex(unsigned int id) {
		return 0;
	}
};



/*
* Get Index
* Find - Distance to beginning
*/
inline unsigned int getIndex(unsigned int* arr, unsigned int len, unsigned int value) {
	return (unsigned int) (std::find(arr, arr + len, value) - arr);
}

struct ReqRelPair {
	unsigned int req, rel, relRefl;
};

bool reqRelCompare(ReqRelPair l, ReqRelPair r) {
	return (l.req) < (r.req);
}


/*
* Convert Query Graph into requirements (to solve subgraph isomorphism over)
* Note: GFSM is unaware of the actual query at runtime (or even its nodes), and only
*		cares to enforce these requirements for each new mapping
*/

struct ReqHeader {
	unsigned int count, index, pkDist;

#ifdef SYMMETRY
	unsigned int lastSymmetry;
#endif
};

struct Requirements {
	ReqHeader* header;
	unsigned int* reqs;
	unsigned int size, len;
	int* mappingData;

#ifdef SYMMETRY
	unsigned int symmetryCount;
	unsigned int* permutations;
#endif
};

#define ReqCount 0
#define ReqIndex 1
#define PkDist 2

/*
* Req Structure:
* {(Index of next Mapping, Required Relation), ...}
*/

void genereateCCSRRequirements(CCSR graph, Requirements requirements,
	unsigned int maxValency, unsigned int depth,
	int* mappingData) {

	unsigned int numOfReqs = 0;

	unsigned int relativeDepth;
	bool found = false;

	//Find a vertex to map for.
	for (unsigned int i = 0; i < graph.count; i++) {
		if (mappingData[i] == 0 && (!found)) {
			mappingData[i] = 1;
			relativeDepth = i;
			found = true;
		}
		else if (mappingData[i] > 0) {
			mappingData[i]++;
		}

	}

	//Failed as couldn't traverse the graph! (i.e. disconnected query)
	if (!found) {
		exit(1);
	}

	CCSRElem* examinedRow = &graph.data[graph.localVertices[relativeDepth]];
	CCSRElem* currentRow = examinedRow + 1;

	unsigned int reqOffset = depth * maxValency * 3;
	unsigned int firstQ = 0;

	ReqRelPair* reqRels = new ReqRelPair[maxValency]();

	unsigned int currentVertex, currentIndex;


	for (unsigned int i = 0; i < *examinedRow; i++, currentRow++) {
		currentVertex = currentRow -> vert();
		currentIndex = getIndex(graph.localVertices, graph.count, currentVertex); //THIS IS BROKEN AND WRONG!!!!!!!!!!!!

		if (mappingData[currentIndex] < 0) {
			mappingData[currentIndex] = 0;
		}

		if (mappingData[currentIndex] > 0) {
			unsigned int relation = currentRow -> rel();
			reqRels[numOfReqs].rel = relation;
			CCSRElem* reflectedRow = &graph.data[currentVertex];
			for (unsigned int i2 = 0; i2 < *reflectedRow; i2++) {
				if (((reflectedRow + 1 + i2)->vert()) == graph.localVertices[relativeDepth]) {
					relation = (reflectedRow + 1 + i2) -> rel();
					break;
				}
			}

			reqRels[numOfReqs].req = (unsigned int)(mappingData[currentIndex] - 1);
			reqRels[numOfReqs].relRefl = relation;
			numOfReqs++;
		}

	}

	if (numOfReqs) {
		std::sort(reqRels, reqRels + numOfReqs, reqRelCompare);

		for (unsigned int i = 0; i < numOfReqs; i++) {
			requirements.reqs[reqOffset + i * 3] = reqRels[i].req;
			requirements.reqs[reqOffset + i * 3 + 1] = reqRels[i].rel;
			requirements.reqs[reqOffset + i * 3 + 2] = reqRels[i].relRefl;
		}

		firstQ = requirements.reqs[reqOffset];
		for (unsigned int i = 0; i < numOfReqs; i++) {
			firstQ = requirements.reqs[reqOffset + i * 3];
			if (firstQ) {
				break;
			}
		}
	}
	else {
		firstQ = 0;
	}

	/*
	if ((firstQ > depth && depth) || (depth && !firstQ)) {
		printf("\nQuery Process Failure: Depth %u, First Q %u", depth, firstQ);
		exit(1);
	}*/

	delete[] reqRels;
	//delete[] localVerticeLocs;

	requirements.header[depth].count = numOfReqs;
	requirements.header[depth].index = reqOffset;
	requirements.header[depth].pkDist = firstQ;
}

unsigned int getMaxValency(CCSR graph) {

	unsigned int maxValency = 0;
	CCSRElem* ceiling = graph.data + graph.size;

	for (CCSRElem* examinedRow = graph.data; examinedRow < ceiling; examinedRow += (*examinedRow) + 1) {
		if (*examinedRow > maxValency) {
			maxValency = *examinedRow;
		}
	}

	return maxValency;
}

template<unsigned int size>
struct QueryFunction {
	unsigned int mappings[size];

	template<unsigned int subSize>
	constexpr QueryFunction& operator=(QueryFunction<subSize> func) {

		unsigned int minSize = std::min(subSize, size);

		std::memset(mappings, 0, sizeof(mappings));
		for (unsigned int i = 0; i < minSize; i++) {
			mappings[i] = func.mappings[i];
		}
		return *this;
	}

	bool subEquals(QueryFunction<size> func, unsigned int subSize, unsigned int ignore) {

		unsigned int minSize = std::min(subSize, size);

		for (unsigned int i = 0; i < minSize; i++) {
			if (i != ignore) {
				if (mappings[i] != func.mappings[i]) {
					return false;
				}
			}
		}

		return true;
	}

	void print() {
		printf("\nFunc:");
		for (int i = 0; i < size; i++) {
			printf("%u,", mappings[i]);
		}
	}
};

#define MAXSYMS 20

template<unsigned int Depth, unsigned int MaxDepth = MAXSYMS>
void queryMatch(CCSR graph, QueryFunction<Depth> func, std::vector< QueryFunction <MaxDepth>>* matches) {
	if constexpr (Depth < MaxDepth) {
		if (Depth >= graph.count) {
			//matches[numMatches] = func;
			//func.print();
			QueryFunction <MaxDepth> _func; //Cba to make two overloads
			_func = func;
			matches->push_back(_func);
		}
		else {

			unsigned int preservedVert = graph.localVertices[Depth];
			CCSRElem *preservedRow = &(graph.data[preservedVert]);

			for (unsigned int i = 0; i < graph.count; i++) { 
				unsigned int newVert = graph.localVertices[i];

				bool pred = true;

				//Injective check
				for (unsigned int i2 = 0; i2 < Depth; i2++) {
					if (newVert == func.mappings[i2]) {
						pred = false;
						break;
					}
				}

				//Edge Check
				if (pred) {
					unsigned int rowLen = graph.data[newVert].elem;
					for (unsigned int i2 = 0; i2 < preservedRow->elem; i2++) {

						CCSRElem missingEdge = preservedRow[i2 + 1];

						unsigned int missingVert = missingEdge.vert();
						unsigned int missingLabel = missingEdge.rel();

						//Code Duplication, Condense this with smsm.cu HnWrite implementation when opportuntiy
						unsigned int node = graph.demangle(missingVert);

						if (node < Depth) {
							bool found = false;

							unsigned int mMissingVert = func.mappings[node];

							for (unsigned int i3 = 0; i3 < rowLen; i3++) {
								CCSRElem foundEdge = graph.data[newVert + 1 + i3];
								if (foundEdge.vert() == mMissingVert) {
									if (foundEdge.rel() == missingLabel) {
										found = true;
									}
								}

								if (found) {
									break;
								}
							}

							if (!found) {
								pred = false;
								break;
							}
						}

					}
				}
				

				if (pred) {
					QueryFunction<Depth + 1> newFunc;

					std::memcpy(&(newFunc.mappings), &(func.mappings), sizeof(func));
					newFunc.mappings[Depth] = newVert;

					queryMatch(graph, newFunc, matches);
				}
					

			}
		}
	}
}

void generateSymmetry() {

}

/*
* Preprocess
*
* - Sets up requirements for the query graph --
*   - Chooses a matching order
*   - Sets up the requirements buffer data
*   - Max degree in the query graph
*/

//Lazy so just leave this in global as all in same compilation unit
#ifdef SYMMETRY
std::vector< QueryFunction <MAXSYMS>> symmetries;
#endif

__host__ unsigned int preProcessQuery(CCSR query, Requirements* requirements) {
	
#ifdef SYMMETRY
	symmetries.reserve(10);

	for (unsigned int i = 0; i < query.count; i++) {
		QueryFunction<1> func{query.localVertices[i]};
		queryMatch(query, func, &symmetries);
	}

	requirements->permutations = new unsigned int[symmetries.size() * query.count];

	info_printf("\nSize of symmetry: %llu\n", symmetries.size());
	for (size_t i = 0; i < symmetries.size(); i++) {
		QueryFunction <MAXSYMS> func = symmetries[i];
		info_printf("Function %llu", i);
		for (unsigned int i2 = 0; i2 < query.count; i2++) {
			info_printf("(%lu, %lu)", query.demangle(query.localVertices[i2]), query.demangle(func.mappings[i2]));
			requirements->permutations[query.demangle(query.localVertices[i2])] = query.demangle(func.mappings[i2]); //I think this maybe incorrect
		}
		info_printf("\n");
	}
#endif


	unsigned int maxValency = getMaxValency(query);

	unsigned int reqSize = 3 * maxValency * query.count;

	requirements->reqs = new unsigned int[3 * maxValency * query.count]{};
	requirements->header = new ReqHeader[query.count];
	requirements->size = reqSize;
	requirements->len = query.count;
	requirements->mappingData = new int[query.count];

	memset(requirements->mappingData, -1, sizeof(int) * query.count);
	unsigned int degVal = query.data[0];
	unsigned int loc = 0;

	for (unsigned int i = 0; i < query.size; i += query.data[i] + 1) {
		if (degVal < query.data[i]) {
			degVal = query.data[i];
			loc = i;
		}
	}

	requirements->mappingData[loc] = 0;

	for (unsigned int depth = 0; depth < query.count; depth++) {
		genereateCCSRRequirements(query, *requirements, maxValency, depth, (requirements->mappingData)); //Fix this argument requirements->mappingData, legacy change
	}


	int* tempMappingData = new int[query.count]{};
	//unsigned int* orderingData = new unsigned int[query.count]{};

	for (unsigned int i = 0; i < query.count; i++) {
		for (int i2 = 0; i2 < (int) query.count; i2++) {
			if (requirements->mappingData[i2] == i + 1) {
				tempMappingData[i] = i2;
			}
		}
	}

	memcpy(requirements->mappingData, tempMappingData, sizeof(int) * query.count);

#ifdef SYMMETRY

	QueryFunction<MAXSYMS> identity;
	unsigned int idenId = 0;
	bool foundId = false;

	for (size_t i = 0; i < symmetries.size(); i++) {
		QueryFunction <MAXSYMS> func = symmetries[i];
		bool isIden = true;
		for (unsigned int i2 = 0; i2 < query.count; i2++) {
			if (query.localVertices[i2] != func.mappings[i2]) {
				isIden = false;
				break;
			}
		}
		if (isIden) {
			identity = func;
			foundId = true;
			idenId = (unsigned int) i;
		}
	}

	requirements->symmetryCount = (unsigned int) symmetries.size();

	if (query.count >= 10) {
		requirements->symmetryCount = 1;
	}

	csv_printf(" Sym: %lu,", requirements->symmetryCount);

	if (!foundId){
		exit(999);
	}

	info_printf("Identity: ");
	for (unsigned int i = 0; i < query.count; i++) {
		info_printf("(%lu, %lu)", query.localVertices[i], identity.mappings[i]);
	}

	info_printf("Mapping Data: ");
	for (unsigned int i = 0; i < query.count; i++) {
		info_printf("%i, ", requirements->mappingData[i]);
	}


	auto equivalentGroups = new std::set<unsigned int>[query.count] {};

	//Find Branches from identity
	for (unsigned int depth = 0; depth < query.count; depth++) {
		unsigned int jumps = 0xffffffff;
		for (size_t i = 0; i < symmetries.size(); i++) {
			if (i != idenId) {
				QueryFunction <MAXSYMS> func = symmetries[i];
				unsigned divergedMapping = func.mappings[depth];
				unsigned originalMapping = identity.mappings[depth];
				unsigned localJumps = 0xffffffff;
				if (divergedMapping != originalMapping) {
					if (depth > 0) {
						for (unsigned int backStep = depth - 1; backStep != 0xffffffff; backStep--) {
							if (identity.mappings[backStep] == divergedMapping) {
								if (func.mappings[backStep] == originalMapping) {
									if (identity.subEquals(func, depth, backStep)) {

										unsigned left = query.demangle(originalMapping);
										unsigned right = query.demangle(divergedMapping);

										left = getIndex((unsigned int*)requirements->mappingData, query.count, left);
										right = getIndex((unsigned int*)requirements->mappingData, query.count, right);

										equivalentGroups[left].insert(right);
										equivalentGroups[right].insert(left);

										//printf("%lu %lu %lu %lu\n", left, right, originalMapping, divergedMapping);

										localJumps = std::min(depth - backStep, localJumps);
									}
								}
							}
						}
					}
				}

				jumps = std::min(jumps, localJumps);
			}
		}

		//printf("Jumps for depth %u: %u\n", depth, jumps);
	}

	for (unsigned int i = 0; i < query.count; i++) {
		auto group = equivalentGroups[i];
		info_printf("Equivalent Groups: ");
		for (unsigned int vert : group) {
			info_printf("%u, ", vert);
		}
		info_printf("\n");
	}

	for (int i = query.count - 1; i >= 0; i--) {

		unsigned int* lastSymmetry = &(requirements->header[(query.count - 1) - i].lastSymmetry);
		*lastSymmetry = 0;

		unsigned int vert1 = getIndex((unsigned int*)requirements->mappingData, query.count, i);



		for (unsigned int i2 = i + 1; i2 < query.count; i2++) {
			unsigned int vert2 = getIndex((unsigned int*)requirements->mappingData, query.count, i2);

			if (equivalentGroups[i].contains(i2)) {
				*lastSymmetry = i2 - i;
			}

			if (*lastSymmetry) {
				break;
			}


		}
	}


#endif

	//Reformat for SMSM:


	for (unsigned int i = 0; i < query.count; i++) {
		unsigned int oldClimb = 0;
		ReqHeader header = (requirements)->header[i];
		for (unsigned int i2 = 0; i2 < header.count; i2++) {
			unsigned int _oldClimb = (requirements)->reqs[i2 * 3 + header.index];
			(requirements)->reqs[i2 * 3 + header.index] -= oldClimb;
			oldClimb = _oldClimb;
			//printf("%u total\n", oldClimb);
		}
		if (header.count) {
			(requirements)->reqs[header.index] -= 1;
		}
	}

	delete[] tempMappingData;
	//delete[] orderingData;

#ifdef QUERYDATAPRINT
	info_printf("\nMaxValency %i, Query Height %i: ", maxValency, query.count);

	for (unsigned int i = 0; i < maxValency * query.count; i++) {
		info_printf("\nReqClimb %i, ", (requirements)->reqs[i * 3]);
		info_printf("ReqRel %i, ", (requirements)->reqs[i * 3 + 1]);
		info_printf("ReqReflRel %i, ", (requirements)->reqs[i * 3 + 2]);
	}

	info_printf("\n\n");

	for (unsigned int i = 0; i < query.count; i++) {
		info_printf("\nNumReqs %i, ", (requirements)->header[i].count);
		info_printf("ReqIndex %i, ", (requirements)->header[i].index);
		info_printf("FirstQueryDepth %i, ", (requirements)->header[i].pkDist);
#ifdef SYMMETRY
		info_printf("LastSymmetry %i, ", (requirements)->header[i].lastSymmetry);
#endif
	}
#endif

	for (unsigned int i = 0; i < maxValency * query.count; i++) {
		if (requirements->reqs[i * 3] > query.count) {
			info_printf("\nPreprocessor Failure: Query Gen,");
			csv_printf("Preprocessor Failure: Query Gen,");
			exit(1);
		}
	}

	return maxValency;
}

template<unsigned int width, unsigned int height>
CCSR gridCCSR() {
	unsigned int data[height][width][5];
	unsigned int scan[width * height]{};
	int size = 0;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned int deg = 0;
			if (x > 0)
				data[y][x][1 + deg++] = x - 1 + y * width;
			if (x < width - 1)
				data[y][x][1 + deg++] = x + 1 + y * width;
			if (y > 0)
				data[y][x][1 + deg++] = x + (y-1) * width;
			if (y < height - 1)
				data[y][x][1 + deg++] = x + (y+1) * width;

			data[y][x][0] = deg;

			if (x == width && y == height)
				size = scan[x + y * width] + deg + 1;
			else
				scan[x + y * width + 1] = scan[x + y * width] + deg + 1;
		}
	}

	CCSRElem* elems = new CCSRElem[size];

	int c = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int deg = data[y][x][0];
			elems[c] = deg;
			c++;
			for (int d = 1; d < deg; d++, c++) {
				elems[c] = CCSRElem(data[y][x][d], 1, deg);
			}
		}
	}
	
}

CCSR triangleCCSR() {
	unsigned int count = 6;
	unsigned int size = 9;

	CCSRElem* data = new CCSRElem[size]{ 2, CCSRElem(3,1,2), CCSRElem(6,1,2), 2, CCSRElem(0,1,2), CCSRElem(6,1,2), 2, CCSRElem(0,1,2), CCSRElem(3,1,2) };
	unsigned int* vertices = new unsigned int[size]{ 0, 3, 6 };

	return { count, size, data, vertices};
}