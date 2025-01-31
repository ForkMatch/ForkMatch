#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <algorithm>
#include "ccsr.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"
#include <cub/cub.cuh>

#define ERR_MISSING_FILE 232
#define ERR_INVALID_DATA 233
#define ERR_MISSING_DATA 234
#define ERR_NONLINEAR_DATA 235

struct OrderPair {
	unsigned int mapping;
	unsigned int size;
};

inline bool operator<(const OrderPair& pair0, const OrderPair& pair1) {
	if (pair0.size == pair1.size) {
		return pair0.mapping < pair1.mapping;
	}
	return pair0.size > pair1.size;
}

/*
* TODO: Rework this, it could be better!
*
* Variable Length 2D Intermediate structure to hold CCSR data as its generated
* It is made of
*/

class Dynamic2D {

	/*
	* Structure:
		DynamicRow :: {Length, <Data>, location of more data}
	*	FrontRows: {DynamicRow0, DynamicRow1, DynamicRow2, ...}
	*	BackRows: Data for longer rows
	*/


private:
	unsigned int* frontRows, * backRows;
	unsigned int rows, width, reserved;

	unsigned int bAllocated;

	/*
	* Reused Init func (in constructors below)
	*/
	void construct(unsigned int _rows, unsigned int _width, unsigned int _reserve) {
		rows = _rows;
		width = _width;
		reserved = _reserve * (width + 2);
		bAllocated = width + 2;
		frontRows = (unsigned int*)calloc(rows * (width + 2), sizeof(unsigned int));
		backRows = (unsigned int*)calloc(reserved, sizeof(unsigned int));
	}

public:

	/*
	* Get an element
	*/
	unsigned int get(unsigned int* alloc, int i) const {
		if (width <= (unsigned int) i) {
			unsigned int offset = alloc[width + 1];
			if (offset) {
				return get(&backRows[offset], i - width);
			}
			return 0xffffffff; //Debug State
		}
		return alloc[i + 1];
	}

	/*
	* Get a row reference from the back rows
	*/
	unsigned int* getBottomRow(unsigned int* alloc) {
		if (alloc[0] >= width) {
			unsigned int offset = *(alloc + width + 1);
			if (offset) {
				return getBottomRow(&backRows[offset]);
			}

			*(alloc + width + 1) = bAllocated;
			unsigned int* result = &backRows[bAllocated];
			bAllocated += width + 2;

			return result;
		}
		return alloc;
	}

	/*
	* Add an element
	*/
	void put(unsigned int* alloc, unsigned int value) {
		unsigned int* row = getBottomRow(alloc);
		row[row[0] + 1] = value;
		row[0]++;
		if (row != alloc) {
			alloc[0]++;
		}
	}

	/*
	*	The variable size rows in the Dynamic2D
	*	Accessed via the [] operator on the Dynamic2D
	*/

	struct DynamicRow {
		unsigned int* allocation;
		unsigned int width, size;
		Dynamic2D* parent;

		unsigned int operator [](int i) const {
			if (width <= (unsigned int) i) {
				return parent->get(allocation, i);
			}
			return allocation[i + 1];
		}

		void add(unsigned int value) {
			size++;
			parent->put(allocation, value);
		}
	};

	/*
	*	Dynamic2D Constructors
	*/

	Dynamic2D(unsigned int _rows, unsigned int _width, unsigned int reserved) {
		construct(_rows, _width, reserved);
	}

	Dynamic2D(unsigned int _rows, unsigned int _width) {
		construct(_rows, _width, _rows);
	}

	~Dynamic2D() {
		free(frontRows);
		free(backRows);
	}

	/*
	*	How rows are accessed in the Dynamic2D
	*/

	DynamicRow operator [](int i) const {
		unsigned int* row = &frontRows[i * (width + 2)];
		return { row, width, row[0], (Dynamic2D*)this };
	}

	void copy(unsigned int* target, DynamicRow row) {
		if (width < row.size) {
			memcpy(target, row.allocation + 1, width * sizeof(unsigned int));
			copy(target + width, { &backRows[*(row.allocation + width + 1)], width, row.size - width, (Dynamic2D*)this });
		}
		else {
			memcpy(target, row.allocation + 1, row.size * sizeof(unsigned int));
		}
	}

};

/*
* The Parsing structure for parsing a CCSR
*/

//Parsing Modes and States
enum ParseMode { header_p, vertex_p, edge_p };
enum ParseState { complete_p, fail_p, edges_found_p, end_p };

//#define PARSE_PROTECT //Who knows what people will throw at it.

void clearValues(unsigned int* values, size_t size) {
	memset(values, 0, size * sizeof(unsigned int));
}

//Parse the values of the file
inline ParseState parseValues(char** str, char* end, unsigned int* values, const ParseMode pm) {

	unsigned int scanWidth;
	char ch = **str;

	if (ch == 't') {
		if (strncmp(*str, "t # -1", 6)) {
			return fail_p;
		}
		return end_p;
	}

	//"pm" is constant, so this should get optimised down!

	switch (pm) {
	case header_p:
		scanWidth = 4;
		(*str) -= 2; //This is bad, but lets me keep my for loop simple

		break;

	case vertex_p:
		if (ch == 'e') {
			return edges_found_p;
		}
		if (ch != 'v') {
			return fail_p;
		}

		scanWidth = 2;
		break;

	case edge_p:
		if (ch != 'e') {
			return fail_p;
		}

		scanWidth = 3;
		break;

	}

	unsigned int currentValue = 0;

	for ((*str) += 2; *str < end; (*str)++) {
		ch = **str;

		if (ch >= 48 && ch <= 57) {
			unsigned int i = ch - 48;
			values[currentValue] *= 10;
			values[currentValue] += i;
		}
		else {
			currentValue++;

			if (ch == '\n') {

#ifdef PARSE_PROTECT
				if (currentValue < scanWidth) {
					return fail_p;
				}
#endif
				(*str)++;
				break;
			}

#ifdef PARSE_PROTECT
			if (currentValue > scanWidth || ch != ' ') {
				return fail_p;
			}
#endif

		}
	}

	return complete_p;
}

//When things break, it's sometimes nicer to pretty print
void dumpBfr(unsigned int* bfr, unsigned int size) {
	printf("\nBuffer Dump:");
	for (unsigned int i = 0; i < size; i++) {
		if (!(i % 10)) {
			printf("\n");
		}

		CCSRElem elem(bfr[i]);

		printf("(%i, %i), ", elem.vert(), elem.rel());
	}
}

/*
* Parse a plain text graph file
*/

bool fileParse(std::string loc, CCSR* ccsr, bool directed) {

	auto start = std::chrono::steady_clock::now();
	auto fstart = std::chrono::steady_clock::now();
	FILE* f = fopen(loc.c_str(), "r");

	if (f == NULL) {


		std::cout << "\nMissing file " << loc << std::endl;
		exit(ERR_MISSING_FILE);
	}

	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	rewind(f);


	char* buf = (char*)calloc(size, sizeof(char));

	fread(buf, sizeof(char), size, f);
	auto fend = std::chrono::steady_clock::now();
	std::chrono::duration<double> felapsed_seconds = fend - fstart;
	info_printf("\nFile Access Time: %fs\n", felapsed_seconds.count());

	char* strEnd = buf + size * sizeof(char);

	char* strPtr = buf;

	if (strncmp(buf, "t # 0\n", 6)) {
		exit(ERR_INVALID_DATA); //BROKE
	}

	strPtr += 6;

	unsigned int values[4]{};
	ParseState pState;
	//Parse Header

	pState = parseValues(&strPtr, strEnd, values, header_p);
	if (pState == fail_p) {
		exit(ERR_INVALID_DATA);
	}

	unsigned int verticeCount = values[0];
	unsigned int edgeCount = values[1];
	unsigned int labelCount = values[2];
	//Throwaway values[3] as we don't use labelled vertices

	//Scan Vertex Data
	unsigned int vertexSeen = 0;

	for (pState = complete_p; pState == complete_p; ) {
		clearValues(values, 2);
		pState = parseValues(&strPtr, strEnd, values, vertex_p);
		if (pState == fail_p) {
			exit(ERR_INVALID_DATA);
		}
		if (pState == end_p) {
			break;
		}
		vertexSeen++;
	}

	if (vertexSeen < verticeCount) {
		exit(ERR_MISSING_DATA);
	}


	OrderPair* orderings = (OrderPair*)malloc(sizeof(OrderPair) * verticeCount);

	unsigned int preSize = ((edgeCount / verticeCount) + 1);

	auto start2 = std::chrono::steady_clock::now();
	Dynamic2D allocations(verticeCount, preSize, verticeCount);
	auto end2 = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
	info_printf("\nAlloc Time: %f s\n", elapsed_seconds2.count());

	auto startV = std::chrono::steady_clock::now();

	unsigned int edgeSeen = 0;
	for (pState = complete_p; pState == complete_p; edgeSeen++) {
		clearValues(values, 3);
		pState = parseValues(&strPtr, strEnd, values, edge_p);
		if (pState == fail_p) {
			exit(ERR_INVALID_DATA);
		}
		if (pState == end_p) {
			break;
		}

		allocations[values[0]].add(CCSRElem(values[1], values[2]));
	}

	if (edgeSeen < edgeCount) {
		exit(ERR_MISSING_DATA);
	}

	unsigned int allocationSize = 0;
	unsigned int maxLen = 0;
	for (unsigned int i = 0; i < verticeCount; i++) {
		auto row = allocations[i];
		if (row.size) {
			allocationSize += row.size + 1;
		}

		maxLen = std::max(maxLen, row.size);
		orderings[i] = { i, (unsigned int)row.size };
	}

	CCSRElem* ccsrData = new CCSRElem[allocationSize];
	unsigned int* verticeLocs = new unsigned int[verticeCount]{};
	unsigned int* localVerticeLocs = new unsigned int[verticeCount]{};
	unsigned int* invVerticeLocs = new unsigned int[verticeCount]{};

	CCSRElem* ccsrWrite = ccsrData;

	StaggerInfo* staggerInfo = new StaggerInfo[maxLen]{};
	unsigned int maxDegree = maxLen;
	unsigned int* degreeCounts = new unsigned int[maxLen]{};

	unsigned int vSize;

	auto endV = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_secondsV = endV - startV;
	info_printf("\nParse - First View Time: %fs\n", elapsed_secondsV.count());

	auto startS = std::chrono::steady_clock::now();

	std::sort(orderings, orderings + verticeCount);

	unsigned int actualSize = 0;

	for (unsigned int i = 0; i < verticeCount; i++) {
		unsigned int currentVertex = orderings[i].mapping;
		if (currentVertex >= verticeCount) {
			exit(1);
		}
		auto row = allocations[currentVertex];

		vSize = row.size;
		verticeLocs[currentVertex] = (unsigned int)(ccsrWrite - ccsrData);
		localVerticeLocs[i] = (unsigned int)(ccsrWrite - ccsrData);

		invVerticeLocs[i] = currentVertex;
		if (vSize != 0) {
			*ccsrWrite = vSize;
			allocations.copy((unsigned int*)(ccsrWrite + 1), row);
			staggerInfo[vSize - 1].size += vSize + 1;
			degreeCounts[vSize - 1]++;
			ccsrWrite += vSize + 1;
			actualSize++;
		}

	}

	auto endS = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_secondsS = endS - startS;
	info_printf("\nParse - Data Access Time: %fs\n", elapsed_secondsS.count());

	auto startM = std::chrono::steady_clock::now();

	for (unsigned int i = 0; i < allocationSize; i++) {
		CCSRElem value = ccsrData[i];
		if (value.rel()) { //TODO: Gonna assume that degree never greater than vertex size, probs should put a guard somewhere
			unsigned int vert = CCSRElem(verticeLocs[value.vert()]);
			unsigned int rel = value.rel();

			ccsrData[i] = CCSRElem(vert, rel);
		}
	}

	auto endM = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_secondsM = endM - startM;
	info_printf("\nParse - Remap Time: %fs\n", elapsed_secondsM.count());

	//Assuming top heavy CCSR
	staggerInfo[maxLen - 1].offset = 0;
	staggerInfo[maxLen - 1].normOffset = 0;
	for (int32_t i = maxLen - 2; i >= 0; i--) {
		staggerInfo[i].offset = staggerInfo[i + 1].offset + staggerInfo[i + 1].size;
		staggerInfo[i].normOffset = staggerInfo[i + 1].normOffset + degreeCounts[i + 1];
	}

	auto startSo = std::chrono::steady_clock::now();
	CCSRElem* temp = new CCSRElem[maxLen];

	unsigned int len = 0;
	for (unsigned int i = 0; i < allocationSize; i += len + 1) {
		if (len = ccsrData[i].elem) {
			CCSRElem* row = &ccsrData[i] + 1;

			if (!std::is_sorted(row, row + len, nodeCompare)) {
				std::sort(row, row + len, nodeCompare);
			}

			for (unsigned int i2 = 0; i2 < len; i2++) {
				CCSRElem elem = row[i2];
				row[i2] = { elem.vert(), elem.rel(), ccsrData[elem.vert()].elem};
			}
		}
	}

	auto endSo = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_secondsSo = endSo - startSo;
	info_printf("\nParse - Resort Time: %fs\n", elapsed_secondsSo.count());

	delete[] temp;


	*ccsr = {verticeCount, allocationSize, ccsrData, verticeLocs, localVerticeLocs, invVerticeLocs, staggerInfo, maxDegree}; //Fix this!

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	info_printf("\nOverall Parse Time: %fs\n", elapsed_seconds.count());
	csv_printf("%fs,", elapsed_seconds.count());

	return true;
}