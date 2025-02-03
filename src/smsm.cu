
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <set>
#include <random>

#include "smsm.cuh"


/*
* Get Grid ID -- Used to profile DP launches
*/
static __device__ __forceinline__ unsigned long long __gridid()
{
    unsigned long long gridid;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(gridid));
    return gridid;
}

/*
* Increment the counter by atomicAdd, then propogate result to the warp.
*/
__device__ __forceinline__ 
unsigned int loopInc(unsigned int& sharedI, unsigned int& localI, const unsigned int inc, unsigned int wdx) {
    unsigned int local;

    if (wdx == 0) {
        local = atomicAdd(&sharedI, inc);
    }

    local = __shfl_sync(0xffffffff, local, 0);

    localI = local;

    return localI;

}
/*
* Get the mapping from TBS that is "N" steps away
*/
__device__ __forceinline__ 
FuncPair fetch(FuncPair* forest, FuncPair mapping, unsigned int steps) {
    for (unsigned int i = 0; i < steps; i++) {
        mapping = forest[mapping.prev];
    }

    return mapping;
}

/*
* Check if there is an element left in the IntArray
*/
template<unsigned int RowCount>
__device__ __forceinline__
bool hasNext(VertexRows<RowCount> rows, IntArray<RowCount> is){

    for (unsigned int i = 0; i < RowCount; i++) {
        if (is[i] >= rows[i].end) {
            return false;
        }
    }

    return true;
}

/*
* Check whether neighbours are identical, and increments otherwise
*/
template<unsigned int RowCount, bool isCoop>
__device__ __forceinline__
bool isoCompare(IntArray<RowCount>& is, unsigned int* data) {
    if (RowCount == 1) {
        if (isCoop) {
            is[0]+=32;
        }
        else {
            is[0]++;
        }
        return true;
    }

    if (RowCount == 2) {
        unsigned int lval = data[is[0]];
        unsigned int rval = data[is[1]];

        if (lval <= rval) {
            is[0]++;
            if (lval == rval) {
                is[1]++;
                return true;
            }
        }
        else {
            is[1]++;
        }
    }

    if (RowCount == 3) {
        unsigned int lval = data[is[0]];
        unsigned int mval = data[is[1]];
        unsigned int rval = data[is[2]];

        if (lval <= mval) {
            if (lval <= rval) {
                is[0]++;
                if(lval == mval){
                    is[1]++;
                    if (lval == rval) {
                        is[2]++;
                        return true;
                    }
                }
                else {
                    if (lval == rval) {
                        is[2]++;
                    }
                }

            }
            else {
                is[2]++;
            }
        }
        else {
            if (mval <= rval) {
                is[1]++;
                if (mval == rval) {
                    is[2]++;
                }
            }
            else {
                is[2]++;
            }
        }

    }
    return false;
}

/*
* Attempt to insert data into the TBS
*/
__device__ __forceinline__
bool attemptInsert(FuncPair* forest,
    unsigned int dst, unsigned int& outputSize, unsigned int outputSubMax, unsigned int outputTrueMax,
    unsigned int mapping, unsigned int parent, 
    unsigned int wdx, unsigned int warpMask, bool pred, bool& earlyBreak) {


    unsigned int dataMask = __ballot_sync(0xffffffff, pred);
    unsigned int localPop = __popc(dataMask & warpMask);
    unsigned int warpPop = __popc(dataMask);

    unsigned int broadcast;

    if (wdx == 0) {
        broadcast = atomicAdd(&outputSize, warpPop);
    }

    broadcast = __shfl_sync(0xffffffff, broadcast, 0);

    unsigned int writeLoc = dst + broadcast;

    if (writeLoc >= outputSubMax) {

        if (writeLoc >= outputTrueMax) {
            return true;
        }

        earlyBreak = true;
    }

    if (pred) {
        writeLoc += localPop;

        forest[writeLoc] = { mapping, parent };
    }

    return false;
}

/*
* Check whether new mapping is found in partial match
*/
template<bool hasSymmetry>
__device__ __forceinline__
bool isInj(FuncPair func, unsigned int mapping, FuncPair* forest, unsigned int lastSymmetry) {
    FuncPair prevF = func;

    //Change this as its bad practice with malformed inputs
    while (true) {

        if constexpr (hasSymmetry) {
            lastSymmetry--;
        }
        if constexpr (hasSymmetry) {
            if (lastSymmetry == 0) {
                if (prevF.mapping <= mapping) {
                    return false;
                }
            }
            else {
                if (prevF.mapping == mapping) {
                    return false;
                }
            }
        }
        else {
            if (prevF.mapping == mapping) {
                return false;
            }
        }



        if (prevF.prev != 0xffffffff) {
            prevF = forest[prevF.prev];
        }
        else {
            break;
        }
    }

    return true;
}


/*
* Cooperatively generate the partial matches
*/
template<unsigned int Decomp>
__device__ __forceinline__
bool isoGenCoop(FuncPair func, VertexRows<2> rows, unsigned int* data, FuncPair* forest,
    unsigned int outputOffset, unsigned int& outputSize, unsigned int outputSubMax, unsigned int outputTrueMax,
    unsigned int parent, unsigned int wdx, unsigned int warpMask)

{

    unsigned int l0Start = rows.rows[0].start;
    unsigned int l0End = rows.rows[0].end;

    unsigned int l1Start = rows.rows[1].start;
    unsigned int l1End = rows.rows[1].end;

    unsigned int lastIndex = l1Start;

    unsigned int l0SuperEnd = (_ceil(l0End - l0Start, (unsigned int) 32) * 32) + l0Start;

    bool earlyBreak = false;

    for (unsigned int l0index = wdx + l0Start; l0index < l0SuperEnd; l0index += 32) {

        bool pred = false;
        unsigned int mapping = data[l0index];

        if (l0index < l0End) {
            
            l1Start = lastIndex;

            while (l1Start + Decomp < l1End) {
                if (mapping >= data[l1Start + Decomp]) {
                    l1Start += Decomp;
                }
                else {
                    break;
                }
            }

            unsigned int l1SubEnd = min(l1Start + Decomp, l1End);

            pred = binarySearch<Decomp>(mapping, data, l1Start, l1End, lastIndex);

            if (pred) {
                pred = isInj<false>(func, mapping, forest, 0);
            }
        }

        if (attemptInsert(forest,
            outputOffset, outputSize, outputSubMax, outputTrueMax,
            mapping, parent,
            wdx, warpMask, pred, earlyBreak)) {
            break;
        }


    }

    return earlyBreak;

}

/*
* Generate the partial matches for some number of input rows
*/
template<unsigned int BlockSize, unsigned int RowCount, bool hasSymmetry, bool isCoop>
__device__ __forceinline__
bool isoGen(FuncPair func, VertexRows<RowCount> rows, unsigned int* data, FuncPair* forest, 
    unsigned int dst, unsigned int& outputSize, unsigned int outputSubMax, unsigned int outputTrueMax,
    unsigned int parent, unsigned int wdx, unsigned int warpMask,
    unsigned int lastSymmetry
    ) {
       
    IntArray<RowCount> is;

    for (unsigned int i = 0; i < RowCount; i++) {
        is[i] = rows[i].start;

        if constexpr (isCoop) {
            static_assert(RowCount == 1, "Coop only supported for non-intersections");
            is[i] += wdx;
        }
    }

    unsigned int pred;

    unsigned int mapping, mask;

    bool earlyBreak = false;

    while (mask = __ballot_sync(0xffffffff, pred = hasNext<RowCount>(rows, is))) {

        if (pred) {
            mapping = data[is[0]];

            pred = isoCompare<RowCount, isCoop>(is, data);

            if (pred) {
                pred = isInj<hasSymmetry>(func, mapping, forest, lastSymmetry);
            }

            
        }

        if (attemptInsert(forest,
            dst, outputSize, outputSubMax, outputTrueMax,
            mapping, parent,
            wdx, warpMask, pred, earlyBreak)) {
            break;
        }

    }
    return earlyBreak;
}

/*
* Get row data using Requirements
*/
template<unsigned int HeaderCount>
__device__
__forceinline__ void fetchRows(
    FuncPair* forest, GPUGraph* data, unsigned int* d_reqs, 
    FuncPair& func, unsigned int parent, VertexRow& firstRow, VertexRow& secondRow, VertexRow& thirdRow) {

    func = forest[parent];

    FuncPair f_firstRow = fetch(forest, func, d_reqs[0]);
    firstRow = data[d_reqs[1]].neighbours(f_firstRow.mapping);

    if constexpr (HeaderCount > 1) {
        FuncPair f_secondRow = fetch(forest, f_firstRow, d_reqs[3]);
        secondRow = data[d_reqs[4]].neighbours(f_secondRow.mapping);

        if constexpr (HeaderCount > 2) {
            FuncPair f_thirdRow = fetch(forest, f_secondRow, d_reqs[6]);
            thirdRow = data[d_reqs[7]].neighbours(f_thirdRow.mapping);
        }
    }
}

/*
* Shuffle sync for non u64 data types
*/
template<typename T>
__device__
__forceinline__ 
T s__shfl_sync(unsigned mask, T var, int srcLane) {

    static_assert(sizeof(T) == sizeof(long long), "Size Mismatch, Both Vars should have size 8");
    long long _v;

    memcpy(&_v, (long long*)&var, sizeof(long long));

    _v = __shfl_sync(mask, _v, srcLane);

    memcpy(&var, (T*)&_v, sizeof(T));

    return var;
}

/*
* Generate all partial match extensions for a given depth
*/
template<unsigned int BlockSize, unsigned int HeaderCount, bool hasSymmetry>
__device__
__forceinline__ bool isoMatch(
    JobPosting& posting, 
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data, 
    Requirements reqs,
    unsigned int* output,
    unsigned int depth,
    SMem_Scratch* smem) {

    using WarpReduceT = cub::WarpReduce<unsigned int>;
    using WTempStorageT = typename WarpReduceT::TempStorage;

    unsigned int bid = blockIdx.x;
    unsigned int tdx = threadIdx.x;
    unsigned int wdx = tdx % 32;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    WTempStorageT* w_temp_storage = (WTempStorageT*)(&(smem -> data));
    w_temp_storage += tdx / 32;

    const unsigned int WarpStorageSize = sizeof(WTempStorageT) * BlockSize / 32;

    static_assert(WarpStorageSize <= sizeof(*smem), "Scratch too small");

    unsigned int warpMask = (1 << wdx) - 1;

    ReqHeader header = reqs.header[depth];
    unsigned int* d_reqs = reqs.reqs + header.index;
#ifdef SYMMETRY
    unsigned int lastSymmetry = header.lastSymmetry;
#else
    unsigned int lastSymmetry = 0;
#endif

    const int WarpCount = BlockSize / 32;
    
    __shared__ unsigned int workCompleted;
    __shared__ unsigned int workGenerated;

    unsigned int warpSplit;

    if (tdx == 0) {
        workCompleted = 0;
        workGenerated = 0;
    }

    __syncthreads();

    Job job = posting.jobs[0];
    unsigned int workCount = job.count;
    unsigned int ceil_workCount = _ceil<unsigned int, 32>(workCount) * 32;

    unsigned int dst = job.dst;

    unsigned int outputMax = ((job.dst / CurBankSize) * (CurBankSize) + CurBankSize);

    unsigned int outputSubMax = outputMax - BlockSize * SubMaxMulSpill;
    unsigned int outputTrueMax = outputMax - 32;

    bool earlyBreak = false;
    bool globalEarlyBreak = false;


    while (loopInc(workCompleted, warpSplit, 32, wdx) < ceil_workCount) {

        FuncPair func;
        FuncPair f_firstRow, f_secondRow, f_thirdRow;
        VertexRow firstRow{}, secondRow{}, thirdRow{};

        unsigned int wi = warpSplit + wdx;
        unsigned int parent = job.src + wi;

        bool tooLarge = false;

        if (wi < workCount) {
            fetchRows<HeaderCount>(forest, data, d_reqs,
                func, parent, firstRow, secondRow, thirdRow);
#ifdef COOPGEN
            if constexpr (HeaderCount == 1) {
                unsigned int rowDelta = firstRow.end - firstRow.start;
                tooLarge = rowDelta >= 1; //256
            }
        }
        else {
            parent = 0;
#endif
        }

#ifdef COOPGEN
        if (HeaderCount == 1) {
            tooLarge = __any_sync(0xffffffff, tooLarge);
        }
#endif


        if constexpr (HeaderCount == 1) {

            if (tooLarge && depth == 2) {
                for (unsigned int srcLane = 0; srcLane < 32; srcLane++) {
                    unsigned int _parent = __shfl_sync(0xffffffff, parent, srcLane);

                    if (_parent) {
                        fetchRows<HeaderCount>(forest, data, d_reqs,
                            func, _parent, firstRow, secondRow, thirdRow);


                        if constexpr (HeaderCount == 1) {
                            earlyBreak = isoGen<BlockSize, 1, hasSymmetry, true>(func, { firstRow }, data[1].neighbourData,
                                forest, dst,
                                workGenerated, outputSubMax, outputTrueMax,
                                _parent, wdx, warpMask,
                                lastSymmetry
                            );
                        }
                    }

                }
            }
            else {
                earlyBreak = isoGen<BlockSize, 1, hasSymmetry>(func, { firstRow }, data[1].neighbourData,
                    forest, dst,
                    workGenerated, outputSubMax, outputTrueMax,
                    parent, wdx, warpMask,
                    lastSymmetry
                );
            }



        }

        if constexpr (HeaderCount == 2) {
            earlyBreak = isoGen<BlockSize, 2, hasSymmetry>(func, { firstRow, secondRow }, data[1].neighbourData,
                forest, dst,
                workGenerated, outputSubMax, outputTrueMax,
                parent, wdx, warpMask,
                lastSymmetry
            );
        }

        if constexpr (HeaderCount == 3) {
            earlyBreak = isoGen<BlockSize, 3, hasSymmetry>(func, { firstRow, secondRow, thirdRow }, data[1].neighbourData,
                forest, dst,
                workGenerated, outputSubMax, outputTrueMax,
                parent, wdx, warpMask,
                lastSymmetry
            );
        }

        if (earlyBreak) {
            break;
        }
    }

  
    globalEarlyBreak = __syncthreads_or(earlyBreak);

    if (globalEarlyBreak) {

        if (threadIdx.x == 0) {

            JobPosting _posting = posting;

            if (dst + workGenerated >= outputTrueMax) {

                unsigned int splitCount = 2;

#ifdef ZEALOUSFORK
                if (workCompleted < workCount) {
                    splitCount = _ceil(workCount, workCompleted);
                }
#endif

                unsigned int splitSize = workCount / splitCount;

                unsigned int _dst;

                unsigned offsetSrc = 0;

                unsigned dstBank = memPool.originalBank(job.dst);

                if (memPool.originalBank(job.src) != dstBank) {
                    _dst = dstBank;
                }
                else {
                    _dst = memPool.pop();
                }
#ifdef ZEALOUSFORK
                for (unsigned int i = 0; i < splitCount; i++) {

                    if (i == splitCount - 1) {
                        splitSize = workCount - offsetSrc;
                    }

                    if (i) {
                        _dst = 0;
                    }

                    Job currentJob = { .src = job.src + offsetSrc, .dst = _dst, .count = splitSize };

                    //That is one long print statement, oops xD
                    debug_printf("(%u out of %u) Restructuring Current (grid %llu offsetSrc %u splitCount %u workcount %u workcompleted %u) with src %u (sizeof %llu) dst %u count %u\n", i + 1, splitCount, __gridid(), offsetSrc, splitCount, workCount, workCompleted, currentJob.src, sizeof(currentJob.src), currentJob.dst, currentJob.count);
                    
                    _posting.jobs[0] = currentJob;

                    launchPosting <DPBlockSize, hasSymmetry> << <1, DPBlockSize, 0, cudaStreamFireAndForget >> > (
                        _posting,
                        memPool,
                        forest, data,
                        reqs,
                        output,
                        depth, 0);

                    offsetSrc += splitSize;
                }
#else

                Job job1 = { .src = job.src, .dst = _dst, .count = splitSize };
                Job job2 = { .src = job.src + splitSize, .dst = 0, .count = workCount - splitSize };
                JobPosting _posting2;
                _posting.jobs[0] = job1;
                _posting2.jobs[0] = job2;

                asyncLaunchPosting2 <DPBlockSize, hasSymmetry> << <1, 1, 0, cudaStreamFireAndForget >> > (
                    _posting,
                    _posting2,
                    memPool,
                    forest, data,
                    reqs,
                    output,
                    depth,
                    depth,
                    0);
#endif
            }
            else {
                Job nextJob = { .src = job.dst, .dst = 0, .count = workGenerated };
                _posting.jobs[0] = nextJob;

                if (workCompleted < workCount) {

                    Job currentJob = { .src = job.src + workCompleted, .dst = 0, .count = workCount - workCompleted };
                    JobPosting _posting2 = _posting;
                    _posting2.jobs[0] = currentJob;


                    debug_printf("Relauching Current with src %u dst %u count %u\nLauching New Post with src %u dst %u count %u\n", currentJob.src, currentJob.dst, currentJob.count, nextJob.src, nextJob.dst, nextJob.count);

                    asyncLaunchPosting2 <DPBlockSize, hasSymmetry> << <1, 1, 0, cudaStreamFireAndForget >> > (
                        _posting2,
                        _posting,
                        memPool,
                        forest, data,
                        reqs,
                        output,
                        depth,
                        depth + 1,
                        job.dst);

                }
                else {

                    debug_printf("Lauching New Post with src %u dst %u count %u (depth %u)\n", nextJob.src, nextJob.dst, nextJob.count, depth);

                    asyncLaunchPosting <DPBlockSize, hasSymmetry> << <1, 1, 0, cudaStreamFireAndForget >> > (
                        _posting,
                        memPool,
                        forest, data,
                        reqs,
                        output,
                        depth + 1,
                        job.dst);

                }
            }

        }

        asm("exit;");
        
    }

    if (threadIdx.x == 0) {

        posting.jobs[0] = { .src = job.dst, .dst = job.dst + workGenerated, .count = workGenerated };
    }

    __syncthreads();

    return false;


}

/*
* Convert the TBS into a Table
*/
template<unsigned int BlockSize, bool WriteData, bool hasSymmetry>
__device__
__forceinline__ void isoWrite(FuncPair* forest, JobPosting posting, unsigned int* output, const Requirements& reqs)
{

    unsigned int querySize = reqs.len;

    unsigned int ix = threadIdx.x;

    __shared__ unsigned int outputOff;

#ifdef SYMMETRY
    unsigned int cache[10];
#endif


    Job job = posting.jobs[0];

    if (ix == 0) {
#ifdef SYMMETRY
        outputOff = atomicAdd(output, job.count * reqs.symmetryCount);
#else
        outputOff = atomicAdd(output, job.count);
#endif
    }

    __syncthreads();

    if constexpr (WriteData) {
        unsigned int* _output = output + 1 + outputOff * querySize;

        for (unsigned int dx = ix; dx < job.count; dx += BlockSize) {

#ifdef SYMMETRY
            unsigned int* fInfo = &_output[dx * querySize * reqs.symmetryCount];
#else
            unsigned int* fInfo = &_output[dx * querySize];
#endif

            FuncPair function = { .prev = dx + job.src };
            for (unsigned int i = 0; i < querySize; i++) {

#ifdef DEBUGMODE
                if (function.prev >= SOLNSIZE) {
                    unsigned int __offset = dx + job.src;
                    unsigned int oBank = ((__offset / CurBankSize) * CurBankSize) + 1;
                    FuncPair _function = { .prev = __offset };
                    _function = forest[_function.prev];
                    info_printf("Error: Segfault! with mapping (%u, %u) with parent (%u, %u) from offset %u (%u) for src %u count %u\n", function.mapping, function.prev, _function.mapping, _function.prev, __offset, oBank, job.src, job.count);
                    return;
                }
#endif

                function = forest[function.prev];

#ifndef SYMMETRY
                fInfo[i] = function.mapping;
#else
                cache[i] = function.mapping;
#endif
            }

#ifdef SYMMETRY
            for (unsigned int i = 0; i < reqs.symmetryCount; i++) {
                unsigned int* perms = reqs.permutations + i * querySize;
                for (unsigned int i2 = 0; i2 < querySize; i2++) {
                    fInfo[i * querySize + i2] = cache[perms[i2]];
                }
            }
#endif
        }

    }
    

}

/*
* Initialise the first depth of the TBS
*/
template<unsigned int BlockSize>
__device__
__forceinline__ void isoInit(
    JobPosting& posting,
    unsigned int* workSplits,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data, 
    Requirements reqs,
    SMem_Scratch* smem) {

    using BlockScanT = cub::BlockScan<unsigned int, BlockSize>;
    using TempStorageT = typename BlockScanT::TempStorage;

    TempStorageT* temp_storage = (TempStorageT*)(smem);
    static_assert(sizeof(TempStorageT) <= sizeof(*smem), "Scratch too small");

    ReqHeader header = reqs.header[1];
    unsigned int* d_reqs = reqs.reqs + header.index;
    unsigned int reqGraph = d_reqs[1];

    unsigned int workSplit = data[reqGraph].vertexCount / gridDim.x; // This is the same for all graphs btw! As it is global vertex count

    unsigned int workStart = workSplit * blockIdx.x;
    unsigned int workEnd = workStart + workSplit;

    if (blockIdx.x == gridDim.x - 1) {
        workEnd = data[reqGraph].vertexCount;
        
    }

    int tdx = threadIdx.x;

    if (tdx == 0) {
        posting.jobs[0] = { .dst = memPool.pop() };
    }

    __syncthreads();

    unsigned int pseudoEnd = _ceil<unsigned int, BlockSize>(workEnd - workStart) * BlockSize + workStart; //Need this so non-divergent block

    for (unsigned int i = tdx + workStart; i < pseudoEnd; i += BlockSize) {
        __syncthreads();

        Job& job = posting.jobs[0];

        unsigned int pred = false;

        FuncPair func;

        if (i < workEnd) {
            pred = true;

            if (data[reqGraph].neighbours(i).end == 0) {
                pred = false;
            }

            func = { i, 0xffffffff };
        }

        unsigned int scan;

        BlockScanT(*temp_storage).ExclusiveSum(pred, scan);

        if (pred) {
            forest[job.count + scan + job.dst] = func;
        }

        __syncthreads();
        
        if (tdx == BlockSize - 1) {
            job.count += scan + pred;
        }
    }

    __syncthreads();

    if (tdx == 0) {
        Job& job = posting.jobs[0];
        job.src = job.dst;
        job.dst = job.dst + job.count;
    }

    __syncthreads();
    
}

/*
* Iterately generate each layer of the TBS for all requirements
*/
template<unsigned int BlockSize, bool hasSymmetry>
__device__ void isoLoop(
    JobPosting& posting, 
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs, unsigned int startDepth,
    unsigned int* output,
    SMem_Scratch* smem,
    bool dstOwnership) {


    for (unsigned int depth = startDepth; depth < reqs.len; depth++) {

        ReqHeader header = reqs.header[depth];

        switch (header.count) {
        case 1:
            isoMatch<BlockSize, 1, hasSymmetry>(posting, memPool, forest, data, reqs, output, depth, smem);
            break;
        case 2:
            isoMatch<BlockSize, 2, hasSymmetry>(posting, memPool, forest, data, reqs, output, depth, smem);
            break;
        case 3:
            isoMatch<BlockSize, 3, hasSymmetry>(posting, memPool, forest, data, reqs, output, depth, smem);
            break;
        default:
            __trap();
        }

        
    }


    isoWrite<BlockSize, WriteSolnFlag, hasSymmetry>(forest, posting, output, reqs);

    __syncthreads();

    if (threadIdx.x == 0) {
        if (dstOwnership) {
            unsigned oBank = memPool.originalBank(posting.jobs[0].dst);

#ifdef DEBUGMODE
            for (unsigned int i = 0; i < CurBankSize - 1; i++) {
                forest[oBank + i] = { 0xffffffff, 0xffffffff };
            }
#endif
            memPool.push(oBank);
        }
    }

}

/*
* This command is only launched host side to start the problem
*/
template<unsigned int BlockSize, bool hasSymmetry>
__global__ void initPosting(
     unsigned int* workSplits,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data, 
    Requirements reqs,
    unsigned int* output) {

    __shared__ SMem_Scratch scratch;
    __shared__ JobPosting posting;

    isoInit<BlockSize>(posting, workSplits, memPool, forest, data, reqs, &scratch);

    unsigned int startDepth = 1;

    isoLoop<BlockSize, hasSymmetry>(posting, memPool, forest, data, reqs, startDepth, output, &scratch, true);
}

/*
* Initialise the first depth of the TBS
*/
template<unsigned int BlockSize, bool hasSymmetry>
__device__
void d_launchPosting(
    JobPosting _posting,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs,
    unsigned int* output,
    unsigned int depth,
    unsigned int attempts) {

    __shared__ SMem_Scratch scratch;
    __shared__ JobPosting posting;
    bool dstOwnership;

#if MemAttempts > 1
    bool retryTime = false;
#endif

    if (threadIdx.x == 0) {


        if (_posting.jobs[0].dst == 0) {
#if MemAttempts > 1
            _posting.jobs[0].dst = memPool.pop<true>();
            dstOwnership = true;
            if (_posting.jobs[0].dst == 0) {
                if (attempts > MemAttempts) {
                    __trap();
                }
                else {
                    launchPosting <DPBlockSize, hasSymmetry> << < 1, DPBlockSize, 0, cudaStreamDefault >> > (_posting, memPool, forest, data, reqs, output, depth, attempts + 1); //Force serialise in NULL stream to try resolve grid lock
                    retryTime = true;
                }
            }
            else {
                if (attempts > 0) {
                    //If we resolve issue, re-parallelise
                    launchPosting <DPBlockSize, hasSymmetry> << < 1, DPBlockSize, 0, cudaStreamFireAndForget >> > (_posting, memPool, forest, data, reqs, output, depth, 0);
                    retryTime = true;
                }
            }
#else

#endif
        }

        posting = _posting;
    }


#if MemAttempts > 1
    retryTime = __syncthreads_or(retryTime);
    if (retryTime) {
        return;
    }
#else
    __syncthreads();
#endif


    isoLoop<BlockSize, hasSymmetry>(posting, memPool, forest, data, reqs, depth, output, &scratch, dstOwnership);
}

/*
* Launched Kernel to continue the TBS search with a grid with a single block
*/
template<unsigned int BlockSize, bool hasSymmetry>
__global__
void launchPosting(
    JobPosting _posting,
    
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs,
    unsigned int* output,
    unsigned int depth,
    unsigned int attempts) {

    d_launchPosting<BlockSize, hasSymmetry>(_posting, memPool, forest, data, reqs, output, depth, attempts);
}

/*
* Launched Kernel to continue the TBS search with a grid with two blocks
*/
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
    unsigned int depth2) {

    if (blockIdx.x == 0) {
        d_launchPosting<BlockSize, hasSymmetry>(_posting1, memPool, forest, data, reqs, output, depth1, 0);
    }
    else {
        d_launchPosting<BlockSize, hasSymmetry>(_posting2, memPool, forest, data, reqs, output, depth2, 0);
    }

}

/*
* Free a memory bank (Tail Reference)
*/
__global__
void freeDst(
    MemoryPool memPool,
    unsigned int dst) {
    unsigned oBank = memPool.originalBank(dst);

    memPool.push(oBank);
}

/*
* Launched Kernel to two new blocks to search the TBS
*/
template<unsigned int BlockSize, bool hasSymmetry>
__global__
void asyncLaunchPosting2(
    JobPosting _posting1,
    JobPosting _posting2,
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs,
    unsigned int* output,
    unsigned int depth1,
    unsigned int depth2,
    unsigned int dst) {

#ifdef CONSOLIDATELAUNCHES
    launchPosting <DPBlockSize, hasSymmetry> << < 2, DPBlockSize, 0, cudaStreamFireAndForget >> > (
        _posting1,
        _posting2,
        memPool,
        forest, data,
        reqs,
        output,
        depth1,
        depth2);
#else
    launchPosting <DPBlockSize, hasSymmetry> << < 1, DPBlockSize, 0, cudaStreamFireAndForget >> > (
        _posting1,
        memPool,
        forest, data,
        reqs,
        output,
        depth1, 0);

    launchPosting <DPBlockSize, hasSymmetry> << < 1, DPBlockSize, 0, cudaStreamFireAndForget >> > (
        _posting2,
        memPool,
        forest, data,
        reqs,
        output,
        depth2, 0);
#endif

    if (dst) {
        freeDst << < 1, 1, 0, cudaStreamTailLaunch >> > (memPool, dst);
    }
    
}

/*
* Launched Kernel to have a block search the TBS
*/
template<unsigned int BlockSize, bool hasSymmetry>
__global__
void asyncLaunchPosting(
    JobPosting _posting,
    
    MemoryPool memPool,
    FuncPair* forest, GPUGraph* data,
    Requirements reqs,
    unsigned int* output,
    unsigned int depth,
    unsigned int dst) {

    launchPosting <DPBlockSize, hasSymmetry> <<< 1, DPBlockSize, 0, cudaStreamFireAndForget >>> (
        _posting,
                memPool,
        forest, data,
        reqs,
        output,
        depth, 0);

    freeDst<<< 1, 1, 0, cudaStreamTailLaunch >>> (memPool, dst);
}

/*
* Prints how many times the memory bank has been used
*/
#ifdef POOLSTATS
__global__ void printPoolStats(MemoryPool memPool) {
    unsigned long long count = *(memPool.popCount);
    info_printf("Pop Count: %llu\n", count);
    csv_printf("Pop Count: %llu, ", count);
}
#endif

/*
* Populates memory pool with allocations
*/
__global__ void populatePool(MemoryPool memPool) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < SubMemPoolCapacity) {
        memPool[idx] = CurBankSize * idx + 1;
        if (idx == 0) {
            *(memPool.head) = 0;
            *(memPool.tail) = SubMemPoolCapacity;
        }
    }

}

/*
* Creates preallocation for all future work
*/
template<size_t MaxPrealloc = PREALLOC>
unsigned char* prealloc() {
    unsigned char* allocation;

    cudaMalloc(&allocation, PREALLOC);
    dummy << <1, 1 >> > ();
    cudaErrorSync();

    return allocation;
}

__host__ void match(std::string queryStr, std::string dataStr, unsigned char* prealloc) {

    info_printf("Query Graph - %s Data Graph - %s\n", queryStr.c_str(), dataStr.c_str());
    csv_printf("%s, %s, ", queryStr.c_str(), dataStr.c_str());
    auto start = std::chrono::steady_clock::now();

    int vertexCount, edgeCount;
    GPUGraph* data = parseGraph(dataStr, prealloc, vertexCount, edgeCount);

    CCSR query;
    fileParse(queryStr, &query, false);

    Requirements requirements;
    preProcessQuery(query, &requirements);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    info_printf("\nParse Time: %fs\n", elapsed_seconds.count());
    csv_printf("%fs, ", elapsed_seconds.count());

    ReqHeader* req_header;
    unsigned int* req_data;
    int32_t* req_mappingData;

    info_printf("\nReq Mapping Data: ");
    for (unsigned int i = 0; i < query.count; i++) {
        info_printf("%i, ", requirements.mappingData[i]);
    }
    info_printf("\n");


    p_malloc(&req_header, sizeof(ReqHeader) * query.count, prealloc);
    p_malloc(&req_data, sizeof(unsigned int) * requirements.size, prealloc);
    p_malloc(&req_mappingData, sizeof(int32_t) * query.count, prealloc);
    cudaMemcpy(req_header, requirements.header, sizeof(ReqHeader) * query.count, cudaMemcpyHostToDevice);
    cudaMemcpy(req_data, requirements.reqs, sizeof(unsigned int) * requirements.size, cudaMemcpyHostToDevice);
    cudaMemcpy(req_mappingData, requirements.mappingData, sizeof(int32_t) * query.count, cudaMemcpyHostToDevice);

    Requirements reqsG = { req_header, req_data, requirements.size, requirements.len, req_mappingData };

#ifdef SYMMETRY
    reqsG.symmetryCount = requirements.symmetryCount;

    unsigned int* req_permutations;
    cudaMalloc(&req_permutations, sizeof(unsigned int) * symmetries.size() * query.count);
    reqsG.permutations = req_permutations;
#endif

    unsigned int* memPoolData;
    unsigned int* boardData;

    p_malloc(&memPoolData, sizeof(unsigned int) * (MemPoolCapacity + 2), prealloc);
    cudaMemset(memPoolData, 0, sizeof(unsigned int) * (MemPoolCapacity + 2));

    p_malloc(&boardData, sizeof(unsigned int) * 2, prealloc);
    cudaMemset(boardData, 0, sizeof(unsigned int) * 2);

    JobPosting* postings;
    p_malloc(&postings, sizeof(JobPosting) * JobPoolCapacity, prealloc);

    MemoryPool memPool{ memPoolData, memPoolData + 1,  memPoolData + 2 };

#ifdef POOLSTATS
    unsigned long long int* popCount;
    p_malloc(&popCount, sizeof(unsigned long long int), prealloc);
    cudaMemset(popCount, 0, sizeof(unsigned long long int));
    memPool.popCount = popCount;
#endif

    JobBoard board{ boardData , boardData + 1, postings };

    FuncPair* funcpairs;
    p_malloc(&funcpairs, sizeof(FuncPair) * SOLNSIZE, prealloc);
    cudaMemset(funcpairs, 0, sizeof(FuncPair) * SOLNSIZE);

    unsigned int* outputG;
    p_malloc(&outputG, sizeof(unsigned int) * OUTPUTSIZE, prealloc);
    cudaMemset(outputG, 0, sizeof(unsigned int) * OUTPUTSIZE); //Debug reasons

    unsigned int* workSplitsG;
    p_malloc(&workSplitsG, sizeof(unsigned int) * 2, prealloc);

    unsigned int workSplits[] = { 0, 1090920 };
    cudaMemcpy(workSplitsG, workSplits, sizeof(workSplits), cudaMemcpyHostToDevice);

    dummy << <1, 1 >> > ();
    cudaErrorSync();

    auto kstart = std::chrono::steady_clock::now();

    unsigned int solnSize;

    int blockCount;


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockCount, initPosting<CurBlockSize, false>, CurBlockSize, 0);

    info_printf("Max blocks per SM, %i\n", blockCount);

    info_printf("Output Size %llu TBS Size %llu\n", OUTPUTSIZE, SOLNSIZE);

    blockCount *= deviceProp.multiProcessorCount;

    if (edgeCount > 2*vertexCount) {
        blockCount /= 2;
        int multiplier = (edgeCount / vertexCount) * (edgeCount / vertexCount);
        if (multiplier > 40) {
            multiplier = 40;
        }
        
        blockCount *= multiplier;
        
    }

    info_printf("Block Count, %i\n", blockCount);
    csv_printf("%i, ", blockCount);

    info_printf("Mem Pool Bank Size: %i, Bank Count %i, Max Bank Count %i\n", CurBankSize, SubMemPoolCapacity, MemPoolCapacity);
    csv_printf("%i, %i, %i, ", CurBankSize, SubMemPoolCapacity, MemPoolCapacity);

    unsigned int populateCount = (SubMemPoolCapacity / CurBlockSize) + 1;

    populatePool<<<populateCount, CurBlockSize >>> (memPool); //This will BREAK!!! needs variable launch args

#ifdef SYMMETRY
        if (requirements.symmetryCount > 1) {
            initPosting<CurBlockSize, true> << <blockCount, CurBlockSize >> > (
                workSplitsG,
                memPool,
                funcpairs, data,
                reqsG,
                outputG);
        }
        else {
            initPosting<CurBlockSize, false> << <blockCount, CurBlockSize >> > (
                workSplitsG,
                memPool,
                funcpairs, data,
                reqsG,
                outputG);
        }
#else
        initPosting<CurBlockSize, false> << <blockCount, CurBlockSize >> > (
            workSplitsG,
            memPool,
            funcpairs, data,
            reqsG,
            outputG);
#endif

    cudaMemcpyAsync(&solnSize, outputG, sizeof(unsigned int), cudaMemcpyDeviceToHost, 0);

    cudaErrorSync();

    auto kend = std::chrono::steady_clock::now();
    std::chrono::duration<double> kelapsed_seconds = kend - kstart;
    info_printf("\nKernel Time: %fs\n", kelapsed_seconds.count());
    csv_printf("%fs, ", kelapsed_seconds.count());

    info_printf("Soln Size: %u\n", solnSize);
    csv_printf("%u, ", solnSize);

    unsigned int* output;
    unsigned int solnLenSize = solnSize * query.count;
    output = new unsigned int[solnLenSize];
    cudaMemcpy(output, outputG + 1, sizeof(unsigned int) * solnLenSize, cudaMemcpyDeviceToHost);

#ifdef POOLSTATS
    printPoolStats<<<1,1>>>(memPool);
    cudaErrorSync();
#endif
}

int main(int argc, char* argv[])
{
#ifdef CTGRAPH_USAGE
    info_printf("CT Graph Problem\n");
    csv_printf("CT Gr,");
#else
    info_printf("Trie Graph Problem\n");
    csv_printf("Trie Gr,");
#endif

    cudaDeviceReset();

    if (cudaError_t error = cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, LaunchLimit)) {
        info_printf("Error: %s", cudaGetErrorName(error));
        cudaDeviceReset();
        exit(error);
    }

    //Warmup the device!
    for (int i = 0; i < 1; i++) {
        unsigned int* test;
        cudaMalloc(&test, 6'000'000);
        dummy << <1, 1 >> > ();
        cudaFree(test);
    }
    cudaDeviceSynchronize();

    unsigned char* allocation = prealloc();

    auto start = std::chrono::steady_clock::now();

    if (argc == 3) {
        std::string queryStr(argv[1]);
        std::string dataStr(argv[2]);
        match(queryStr, dataStr, allocation);
    }
    else {
        exit(99);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    info_printf("\nEnd to End Time: %fs\n", elapsed_seconds.count());
    csv_printf("%fs\n", elapsed_seconds.count());

    return 0;
}
