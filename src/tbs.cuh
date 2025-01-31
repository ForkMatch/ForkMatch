#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>

#include <cstdint>
#include "environment.cuh"

struct FuncPair {
	unsigned int mapping, prev;
};

struct Job {
	unsigned int src, dst, count;
};

#define MaxJobs 1
#define MaxAllocations 8
#define MaxPostings 2048

struct JobPosting {
	Job jobs[MaxJobs];
};

//static const unsigned int CurBlockSize = 512;
//static const unsigned int CurBankCount = 1;

static const unsigned int SubMemPoolCapacity = (SOLNSIZE / CurBankSize);
static const unsigned int MemPoolCapacity = SubMemPoolCapacity * PaddingMul;
static const unsigned int JobPoolCapacity = 1024;

template<typename T>
struct SharedPool {
	unsigned int* head, * tail;
	T* elements;
#ifdef POOLSTATS
	unsigned long long int* popCount;
#endif

	__device__ __host__
		SharedPool(unsigned int* _head, unsigned int* _tail, T* _elements) {
		head = _head;
		tail = _tail;
		elements = _elements;
	}

	__device__ __host__
		SharedPool() {}

	__device__
	const char* this_name() {
		return "Shared Pool";
	}

	template<bool Retry = false>
	__device__
		void poolError(const char* str) {
		info_printf("Error in %s: %s\n", this_name(), str);
		ecsv_printf("!!");
		__nanosleep(1'000'000);

		if constexpr (!Retry) {
			__trap();
		}


	}

	__device__
		T& operator[](unsigned int idx) {
		return elements[idx];
	}
};

struct MemoryPool : SharedPool <unsigned int>
{

	__device__ __host__
	MemoryPool(unsigned int* _head, unsigned int* _tail, unsigned int* _elements) : SharedPool <unsigned int>(_head, _tail, _elements) {}

	__device__ void push(unsigned int bank) { //Do not hammer this or you may UB badly!! 

		const unsigned int cap = capacity();

		//printf("!");

		/*
		unsigned int random = ((clock() * 7621) + 1) % 32768; //Slow down pushes!*/

		unsigned int addr = atomicInc(tail, cap);
		if (addr == *head) {
			poolError("Push failed -- Buffer Overflow");
		}

		unsigned int oldBank = atomicExch(&elements[addr], bank);
		if (oldBank) {
			poolError("Push failed -- Bank Raced");
		}
	}

	template<bool Retry = false>
	__device__ unsigned int pop() {

		const unsigned int cap = capacity();

		unsigned int addr = atomicInc(head, cap);
		unsigned int bank = atomicExch(&elements[addr], 0);

#ifdef POOLSTATS
		atomicAdd(popCount, 1);
#endif

		if (!bank) {
			poolError<Retry>("Incomplete bank");
			atomicDec(head, cap);
		}

		if (*head == (*tail) - 1) {
			poolError<Retry>("Ran out of banks");
			atomicDec(head, cap);
		}

		return bank;
	}

	__device__ __forceinline__
	unsigned int originalBank(unsigned int bank) {
		return ((bank / CurBankSize) * CurBankSize) + 1;
	}

	__device__
		constexpr unsigned int capacity() {
		return MemPoolCapacity;
	}

	__device__
	const char* this_name() {
		return "Memory Pool";
	}
};

struct JobBoard : SharedPool <JobPosting>
{

	__device__ __host__
	JobBoard(unsigned int* _head, unsigned int* _tail, JobPosting* _elements) : SharedPool <JobPosting>(_head, _tail, _elements) {}

	__device__ __host__
	JobBoard() {}

	__device__ void push(JobPosting bank) { //Do not hammer this or you may UB badly!! There is no busy waits to prevent reading incomplete elements

		const unsigned int cap = capacity();

		unsigned int addr = atomicAdd(tail, 1);
		if (addr > cap) {
			poolError("Push failed -- Buffer Overflow");
		}

		elements[addr] = bank;
	}

	__device__ JobPosting pop() {
		return {};
	}

	__device__
		constexpr unsigned int capacity() {
		return JobPoolCapacity;
	}

	__device__
		const char* this_name() {
		return "Job Board";
	}
};