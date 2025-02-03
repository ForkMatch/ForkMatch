#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cub/cub.cuh>

#include "trie.cuh"
#include "ccsr.cuh"
#include "tbs.cuh"
#include "hostparser.cuh"

#define ScratchSize CurBlockSize * 6

struct SMem_Scratch {
	unsigned char data[ScratchSize];
};

template<unsigned int Size>
struct IntArray {
    unsigned int rows[Size];
    __device__ __forceinline__
    unsigned int operator [](int i) const { return rows[i]; }
    __device__ __forceinline__
    unsigned int& operator [](int i) { return rows[i]; }
};

__device__ __forceinline__ 
unsigned int loopInc(unsigned int& sharedI, unsigned int& localI, const unsigned int inc, unsigned int wdx);

__device__ __forceinline__ 
FuncPair fetch(FuncPair* forest, FuncPair mapping, unsigned int steps);

template<unsigned int RowCount>
__device__ __forceinline__
bool hasNext(VertexRows<RowCount> rows, IntArray<RowCount> is);

template<unsigned int RowCount, bool isCoop>
__device__ __forceinline__
bool isoCompare(IntArray<RowCount>& is, unsigned int* data);

__device__ __forceinline__
bool attemptInsert(FuncPair* forest,
    unsigned int dst, unsigned int& outputSize, unsigned int outputSubMax, unsigned int outputTrueMax,
    unsigned int mapping, unsigned int parent, 
    unsigned int wdx, unsigned int warpMask, bool pred, bool& earlyBreak);

template<bool hasSymmetry>
__device__ __forceinline__
bool isInj(FuncPair func, unsigned int mapping, FuncPair* forest, unsigned int lastSymmetry);

template<unsigned int Decomp = 128>
__device__ __forceinline__
bool isoGenCoop(FuncPair func, VertexRows<2> rows, unsigned int* data, FuncPair* forest,
    unsigned int outputOffset, unsigned int& outputSize, unsigned int outputSubMax, unsigned int outputTrueMax,
    unsigned int parent, unsigned int wdx, unsigned int warpMask);

template<unsigned int BlockSize, unsigned int RowCount, bool hasSymmetry, bool isCoop = false>
__device__ __forceinline__
bool isoGen(FuncPair func, VertexRows<RowCount> rows, unsigned int* data, FuncPair* forest, 
    unsigned int dst, unsigned int& outputSize, unsigned int outputSubMax, unsigned int outputTrueMax,
    unsigned int parent, unsigned int wdx, unsigned int warpMask,
    unsigned int lastSymmetry);

template<unsigned int HeaderCount>
__device__
__forceinline__ void fetchRows(
    FuncPair* forest, GPUGraph* data, unsigned int* d_reqs, 
    FuncPair& func, unsigned int parent, VertexRow& firstRow, VertexRow& secondRow, VertexRow& thirdRow);

template<unsigned int BlockSize, bool WriteData, bool hasSymmetry>
__device__
__forceinline__ void isoWrite(FuncPair* forest, JobPosting posting, unsigned int* output, const Requirements& reqs);

template<unsigned int BlockSize, unsigned int HeaderCount, bool hasSymmetry>
__device__
__forceinline__ bool isoMatch(
    JobPosting& posting, 
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data, 
    Requirements reqs,
    unsigned int* output,
    unsigned int depth,
    SMem_Scratch* smem);

template<unsigned int BlockSize>
__device__
__forceinline__ void isoInit(
    JobPosting& posting,
    unsigned int* workSplits,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data, 
    Requirements reqs,
    SMem_Scratch* smem);

template<unsigned int BlockSize, bool hasSymmetry>
__device__ void isoLoop(
    JobPosting& posting, 
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs, unsigned int startDepth,
    unsigned int* output,
    SMem_Scratch* smem,
    bool dstOwnership);

template<unsigned int BlockSize, bool hasSymmetry>
__global__ void initPosting(
     unsigned int* workSplits,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data, 
    Requirements reqs,
    unsigned int* output);

template<unsigned int BlockSize, bool hasSymmetry>
__device__
void d_launchPosting(
    JobPosting _posting,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs,
    unsigned int* output,
    unsigned int depth,
    unsigned int attempts);

template<unsigned int BlockSize, bool hasSymmetry>
__global__
void launchPosting(
    JobPosting _posting,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs,
    unsigned int* output,
    unsigned int depth,
    unsigned int attempts);


template<unsigned int BlockSize, bool hasSymmetry>
__global__
void launchPosting(
    JobPosting _posting1,
    JobPosting _posting2,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs,
    unsigned int* output,
    unsigned int depth1,
    unsigned int depth2);

template<unsigned int BlockSize, bool hasSymmetry>
__global__
void asyncLaunchPosting(
    JobPosting _posting,
    JobBoard board,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs,
    unsigned int* output,
    unsigned int depth,
    unsigned int dst);

template<unsigned int BlockSize, bool hasSymmetry>
__global__
void asyncLaunchPosting2(
    JobPosting _posting1,
    JobPosting _posting2,
    JobBoard board,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs,
    unsigned int* output,
    unsigned int depth,
    unsigned int dst);


template<unsigned int BlockSize, bool hasSymmetry>
__global__
void launchPosting(
    JobPosting _posting,
    JobBoard board,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs,
    unsigned int* output,
    unsigned int depth);

__global__
void freeDst(
    MemoryPool memPool,
    unsigned int dst);

__global__ void printPoolStats(MemoryPool memPool);

__global__ void populatePool(MemoryPool memPool);


