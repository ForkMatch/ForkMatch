#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <cooperative_groups.h>
//#include <cuda/atomic>

#include <cub/cub.cuh>

#include "trie.cuh"
#include "ccsr.cuh"
#include "tbs.cuh"
#include "hostparser.cuh"

#define ScratchSize CurBlockSize * 6

struct SMem_Scratch {
	unsigned char data[ScratchSize];
};

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
