#pragma once

//Allocation Config
#define PREALLOC 45'600'000'000
#define SOLN_PREALLOC 24'000'000'000
#define OUTPUT_PREALLOC 17'300'000'000
#define SOLNSIZE (SOLN_PREALLOC / sizeof(FuncPair))
#define OUTPUTSIZE (OUTPUT_PREALLOC / sizeof(unsigned int))

//Launch Config
#define CurBlockSize 256
#define DPBlockSize 1024
#define MemAttempts 1 << 16
#define LaunchLimit 100'000
#define SubMaxMulSpill 512

static const unsigned int CurBankSize = 1 << 20;
static const unsigned int PaddingMul = 16;

//Build Preprocessors

//#define DEBUGMODE
//#define TRACKDP
#define POOLSTATS
#define QUERYDATAPRINT
#define SYMMETRY
//#define DEBUGPRINT
#define INFOPRINT
//#define CSVPRINT
//#define ECSVPRINT
//#define CTGRAPH_USAGE
//#define ZEALOUSFORK
//#define COOPGEN
#define CONSOLIDATELAUNCHES
#define SAFEPARSE

#ifdef DEBUGPRINT
#define debug_printf(f_, ...) printf((f_), ## __VA_ARGS__)
#else
#define debug_printf(f_, ...) do {} while(0)
#endif

#ifdef INFOPRINT
#define info_printf(f_, ...) printf((f_), ## __VA_ARGS__)
#else
#define info_printf(f_, ...) do {} while(0)
#endif

#ifdef CSVPRINT
#define csv_printf(f_, ...) printf((f_), ## __VA_ARGS__)
#else
#define csv_printf(f_, ...) do {} while(0)
#endif

#ifdef ECSVPRINT
#define ecsv_printf(f_, ...) printf((f_), ## __VA_ARGS__)
#else
#define ecsv_printf(f_, ...) do {} while(0)
#endif


//Disables Table Construction if false
static const bool WriteSolnFlag = true;

//Legacy Preprocessors for CCSR Data Type
#define CCSRDegSize 0

#ifdef CCSR_USE_DEG_ENCODES
#define CCSRDegSize 5
#endif

#define MaxCCSREncode ((1 << CCSRDegSize) - 1)
#define CCSRRelSize 8
#define CCSRVertSize (32 - CCSRRelSize - CCSRDegSize)
