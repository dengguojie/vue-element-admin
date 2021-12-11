/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file transpose.h
 * \brief
 */
#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

#include <vector>
#include <string>
#include <map>
#include <queue>
#include <memory>
#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "register/op_tiling.h"
#include "graph/utils/op_desc_utils.h"
#include "external/graph/operator.h"

namespace optiling {

#define TRANSPOSE_MAX_AXIS_NUM 8
#define MAX_CORE_NUM 64 
#define BORROW_SRC_AXIS_NUM 2
#define BORROW_DST_AXIS_NUM 2
#define BORROW_MAX_AXIS_NUM (BORROW_SRC_AXIS_NUM + BORROW_DST_AXIS_NUM)
#define BORROW_SRC_AXIS_NUM_LT 3
#define BORROW_DST_AXIS_NUM_LT BORROW_SRC_AXIS_NUM_LT
#define BORROW_MAX_AXIS_NUM_LT (BORROW_SRC_AXIS_NUM_LT + BORROW_DST_AXIS_NUM_LT)
#define UB_REORDER_NUM 3
#define BORROW_OTHER_AXIS_NUM 6
#define UB_REORDER_COMBINATION 4  // src: intact|tail dst:intact|tail 4 = 2*2
#define UB_REORDER_LOOP 3
#define BYTES_PER_BLOCK 32
#define BYTES_PER_KB 1024
#define UB_REORDER_FACTOR 33
#define ELE_NUM_PER_BLOCK_B8 32
#define ELE_NUM_PER_BLOCK_B16 16
#define ELE_NUM_PER_BLOCK_B32 8
#define ELE_NUM_PER_BLOCK_B64 4
#define BLOCK_NUM_256K 8192
#define BLOCK_NUM_248K 7936
#define BLOCK_NUM_192K 6144
#define EPB8 8
#define EPB16 16
#define EPB32 32
#define LAST_AXIS_N_TRANS_MAX_SIZE_B16 (32 * 1024)  // unit: 2B
#define LAST_TWO_TRANS_MAX_SIZE_B16 (255 * 16 * 16)  // unit: 2B
#define MDMS 0
#define MDTS 1
#define TDMS 2
#define TDTS 3
#define COL_UNIT 240 
#define ROW_UNIT 128 
#define MAX_INFO_NUM 16 // keep same with GE threads, too huge will result in init timeout 
#define MAX_TILING_NUM 16 * 1024 
#define TILING_HEAD_SIZE 4
#define MAX_RETRY_SCENARIO 5

/*
 * 4 * 32 block = 4KB, this value should be consistent with the variable in transpose.py
 * 1KB : reserved
 * 2KB : store overlap data for dirtyData
 * 3KB : reserved
 * 4KB : reserved
 */
#define UB_RESERVED_KB 4                                         // keep same with transpose.py
#define UB_RESERVED_BLOCK_SIZE (UB_RESERVED_KB * 1024 / 32)      // 128 blocks, 4KB
#define LAST_AXIS_HUGE_THRESHOLD 100 * 1024                      // unit B
#define HUGE_BLOCKS_UNIT (LAST_AXIS_HUGE_THRESHOLD / 32)         // unit blocks, 3200 blocks
#define LAST_AXIS_BLOCK_ALIGN_N_BORROW_B8_THRESHOLD 3000         // unit B
#define LAST_AXIS_BLOCK_ALIGN_Y_BORROW_B8_THRESHOLD (128 * 1024) // unit B
#define LAST_AXIS_BLOCK_ALIGN_LARGE_THRESHOLD 4096               // unit B
#define LAST_AXIS_BLOCK_ALIGN_N_BORROW_THRESHOLD 512             // unit B
#define LAST_AXIS_NOT_BLOCK_ALIGN_LARGE_THRESHOLD 256            // unit B
#define UB_CAP_BLOCKS 7800                                       // for 310 256 - 8 = 248KB = 7936 Blocks
#define B8_HUGE_SIZE 1024

#define STRIDE_BOUNDARY 65535
#define NBURST_BOUNDARY 4095
#define ACCU_BLOCK_SIZE 128  // keep same with transpose.py , not gt 240 for both 310 and 910
#define ACCU_BLOCK_SIZE_IDENTICAL 1024

#define MAX_COL_FP16_VNCHWCONV_FULL 256  // verify 128 is better than 248 //496
#define MAX_COL_FP16_VNCHWCONV_PARTIAL 256
#define MAX_ROW_FP16_VNCHWCONV_FULL 128
#define SMALL_SHAPE_SIZE_THRESHOLD 1024
#define F2T_THRESHOLD_B16 64       // unit : byte
#define F2T_THRESHOLD_B32 32       // unit : byte
#define TILING_FIXED_MAX_LEN 1024  // unit: bytes, keep same with transpose.py

enum TransposeScenario {
  SCENARIO_0 = 0,    // identical shape
  SCENARIO_1 = 1,    // large last axis and not transpose
  SCENARIO_2 = 2,    // small last axis and not transpose
  SCENARIO_3 = 3,    // huge  last axis and not transpose
  SCENARIO_4 = 4,    // borrow axis scenario with last axis not transposed
  SCENARIO_5 = 5,    // borrow axis scenario with last axis transposed
  SCENARIO_6 = 6,    // small shape
  SCENARIO_7 = 7,    // last axis transpose
  SCENARIO_8 = 8,    // a100 verifaction
  SCENARIO_9 = 9,    // last axis block aligned and not transpose
  SCENARIO_10 = 10,  // last two axis: block aligned & transpose & not huge
  SCENARIO_11 = 11,  // last two axis: block aligned & transpose & huge
  SCENARIO_INIT = 99,// SCENARIO_INIT
};

enum TilingModelPri {
  TM_PRI_0 = 0,
  TM_PRI_1 = 1,
  TM_PRI_2 = 2,
  TM_PRI_3 = 3,
  TM_PRI_4 = 4,
  TM_PRI_5 = 5,
  TM_PRI_6 = 6,
  TM_PRI_7 = 7,
  TM_PRI_8 = 8,
  TM_PRI_9 = 9,
  TM_PRI_10 = 10,
  TM_PRI_11 = 11,
  TM_PRI_MAX = 99,
};

enum SubScenarioLastAxisTrans {
  LAST_AXIS_TR_COMMON = 0,
  LAST_AXIS_TR_F2T = 1,  // fat 2 thin
  LAST_AXIS_TR_T2F = 2,  // thin 2 fat
};

enum MteMode {
  MTE_MODE_NULL = 0,  //default
  MTE_MODE_DST = 1,   // dst contiguous
  MTE_MODE_SRC = 2    // src contiguous
};

struct PermInfo{
    int64_t perm[UB_REORDER_NUM + 1][BORROW_MAX_AXIS_NUM_LT]; // 1 : for init status
};

struct SplitParam {
  int64_t nFactor;
  int64_t colFactor;
  int64_t rowFactor;

  void Reset() {
    nFactor = 1;
    colFactor = 1;
    rowFactor = 1;
  }

  SplitParam() {
    Reset();
  }

  void Set(int64_t n, int64_t c, int64_t r) {
    nFactor = n;
    colFactor = c;
    rowFactor = r;
  }
};

struct NCR {
  int64_t n[TRANSPOSE_MAX_AXIS_NUM];
  int64_t col[TRANSPOSE_MAX_AXIS_NUM];
  int64_t row[TRANSPOSE_MAX_AXIS_NUM];
  int64_t nVol;
  int64_t cVol;
  int64_t rVol;
  int64_t nSize;
  int64_t colSize;
  int64_t rowSize;

  const NCR& operator = (const NCR& rhs) {
    nVol = rhs.nVol;
    cVol = rhs.cVol;
    rVol = rhs.rVol;
    nSize = rhs.nSize;
    colSize = rhs.colSize;
    rowSize = rhs.rowSize;
    for (int64_t i = 0; i < nSize; i++) {
      n[i] = rhs.n[i];
    }
    for (int64_t i = 0; i < colSize; i++) {
      col[i] = rhs.col[i];
    }
    for (int64_t i = 0; i < rowSize; i++) {
      row[i] = rhs.row[i];
    }
    return *this;
  }
  void Reset() {
    nVol = 0;
    cVol = 0;
    rVol = 0;
    nSize = 0;
    colSize = 0;
    rowSize = 0;
  }

  NCR() {
    Reset();
  }
};

class TilingModel {
public:
  TilingModel(){
    Reset();
  }
  TilingModel(int64_t p, int64_t c, int64_t u, SubScenarioLastAxisTrans scenario, std::string name)
      : coreNum(c), ubBlocks(u), priority(p), maxCol(0), maxRow(0), subScenario(scenario),
        isf2t(false), ist2f(false) {
  }
  virtual ~TilingModel(){};
  virtual bool Decision(const NCR& ncr, int64_t dim){
    return false;
  }
  void copy(const TilingModel& tm) {
    sp = tm.sp;
    ncr = tm.ncr;
    coreNum = tm.coreNum;
    ubBlocks = tm.ubBlocks;
    priority = tm.priority;
    maxCol = tm.maxCol;
    maxRow = tm.maxRow;
    subScenario = tm.subScenario;
    isf2t = tm.isf2t;
    ist2f = tm.ist2f;
  }
  void Reset() {
    priority = 0;
    isf2t = false;
    ist2f = false;
    ncr.Reset();
  }

  SplitParam sp;
  NCR ncr;
  int64_t coreNum;
  int64_t ubBlocks;
  int64_t priority;
  int64_t maxCol;  // unit: bytes
  int64_t maxRow;
  SubScenarioLastAxisTrans subScenario;
  bool isf2t;
  bool ist2f;
};

struct ShapeInfo {
  int64_t id;
  std::vector<int64_t> inShape;
  std::vector<int64_t> outShape;
  std::vector<int64_t> perm;
  std::vector<int64_t> reducedInShape;
  std::vector<int64_t> reducedOutShape;
  std::vector<int64_t> reducedPerm;

  int64_t inShapeSize;
  int64_t outShapeSize;
  int64_t permSize;
  
  int64_t origDim;
  int64_t dim;
  int64_t totalVolumeActual;
  int64_t identical;
  int64_t lastAxisLen;
  int64_t lastAxisBurstLen;
  int64_t elePerBlock;
  int64_t eleLenInBytes;
  int64_t alignElement;  // unit: element number padding.  eg. dtype=float, axis[0]= 11, alignElement=8-(11-8)%8=5
  bool isLastAxisTranspose;
  bool isLastAxisHuge;
  bool isLastTwoAlignedAndTrans;
  TransposeScenario scenario;
  MteMode mteMode;

  ShapeInfo() {
    inShape.resize(TRANSPOSE_MAX_AXIS_NUM);
    outShape.resize(TRANSPOSE_MAX_AXIS_NUM);
    perm.resize(TRANSPOSE_MAX_AXIS_NUM);
    reducedInShape.resize(TRANSPOSE_MAX_AXIS_NUM);
    reducedOutShape.resize(TRANSPOSE_MAX_AXIS_NUM);
    reducedPerm.resize(TRANSPOSE_MAX_AXIS_NUM);
    Reset();
  }

  void Reset() {
    id = 0;
    inShapeSize = 0;
    outShapeSize = 0;
    permSize = 0;
    origDim = 0;
    dim = 0;
    totalVolumeActual = 0;
    identical = 0;
    lastAxisLen = 0;
    lastAxisBurstLen = 0;
    elePerBlock = 8;
    eleLenInBytes = 0;
    alignElement = 0;
    isLastAxisTranspose = false;
    isLastAxisHuge = false;
    isLastTwoAlignedAndTrans = false;
    scenario = SCENARIO_0;
    mteMode = MTE_MODE_NULL;
  }
};

struct CompilerInfo {
  int64_t coreNum;
  int64_t usedCoreNum;
  int64_t ubSize;          // unit: block
  int64_t ubSizeCouldUse;  // unit: block
  int64_t fp16Times;
  int64_t blockSize;
  ge::DataType dType;
  std::string mode;
  std::string opType;
  CompilerInfo() {
    coreNum = 0;
    usedCoreNum = 0;
    ubSize = 0;
    ubSizeCouldUse = 0;
    fp16Times = 1;
    blockSize = 2;
  }
};

struct LastAxisNTLoopInfo {
  int64_t headMajorLoop;
  int64_t headMajorNum;
  int64_t headTailNum;

  int64_t bodyLoopNum;
  int64_t bodyMajorLoop;
  int64_t bodyMajorNum;
  int64_t bodyTailNum;

  int64_t tailMajorLoop;
  int64_t tailMajorNum;
  int64_t tailTailNum;

  void Reset() {
    headMajorLoop = 0;
    headMajorNum = 0;
    headTailNum = 0;
    bodyLoopNum = 0;
    bodyMajorLoop = 0;
    bodyMajorNum = 0;
    bodyTailNum = 0;
    tailMajorLoop = 0;
    tailMajorNum = 0;
    tailTailNum = 0;
  }
  LastAxisNTLoopInfo() {
    Reset();
  }
};

struct LastAxisNTHugeInfo {
  int64_t majorLoopNum;
  int64_t majorBlocks;
  int64_t tailBlocks;
  int64_t backEle;

  void Reset() {
    majorLoopNum = 0;
    majorBlocks = 0;
    tailBlocks = 0;
    backEle = 0;
  }

  LastAxisNTHugeInfo() {
    Reset();
  }
};

struct IdenticalInfo {
  int64_t base;
  int64_t eleNum;
  int64_t majorLoop;
  int64_t majorNum;
  int64_t tailNum;
  int64_t notAlignEle;

  void Reset() {
    base = 0;
    eleNum = 0;
    majorLoop = 0;
    majorNum = 0;
    tailNum = 0;
    notAlignEle = 0;
  }
  IdenticalInfo() {
    Reset();
  }
};

struct InfoPerCoreLastAxisNT {
  int64_t base;
  int64_t num;
  int64_t aggregateLoopUnit;
  int64_t aggregateLoopNum;
  int64_t aggregateLoopTail;
  int64_t initTuple[TRANSPOSE_MAX_AXIS_NUM];
  LastAxisNTLoopInfo loopInfo;

  void Reset() {
    base = 0;
    num = 0;
    aggregateLoopUnit = 0;
    aggregateLoopNum = 0;
    aggregateLoopTail = 0;
    for (int64_t i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
      initTuple[i] = 0;
    }
    loopInfo.Reset();
  }
  InfoPerCoreLastAxisNT() {
    Reset();
  }
};

struct InfoN {
  int64_t loopOnN;
  int64_t nOffsetLogic;
  int64_t initNTupleSize;
  int64_t initNTuple[TRANSPOSE_MAX_AXIS_NUM];

  void Reset(int64_t coreNum = MAX_CORE_NUM) {
    initNTupleSize = (coreNum == MAX_CORE_NUM) ? TRANSPOSE_MAX_AXIS_NUM : initNTupleSize;
    for (int64_t i = 0; i < initNTupleSize; i++) {
      initNTuple[i] = 0;
    }
    loopOnN = 0;
    nOffsetLogic = 0;
    initNTupleSize = 0;
  }
  InfoN() {
    Reset();
  }
};

struct InfoCol {
  int64_t colPerMC;
  int64_t colBlockPerMC;
  int64_t loopOnMC;
  int64_t colTC;
  int64_t colBlockTC;
  int64_t colOffset;
  int64_t backStepLeft;
  int64_t initDstTupleSize;
  int64_t tailDstTupleSize;
  int64_t initDstTuple[TRANSPOSE_MAX_AXIS_NUM];
  int64_t tailDstTuple[TRANSPOSE_MAX_AXIS_NUM];

  void Reset(int64_t coreNum = MAX_CORE_NUM) {
    initDstTupleSize = (coreNum == MAX_CORE_NUM) ? TRANSPOSE_MAX_AXIS_NUM : initDstTupleSize;
    for (int64_t i = 0; i < initDstTupleSize; i++) {
      initDstTuple[i] = 0;
    }
    tailDstTupleSize = (coreNum == MAX_CORE_NUM) ? TRANSPOSE_MAX_AXIS_NUM : tailDstTupleSize;
    for (int64_t i = 0; i < tailDstTupleSize; i++) {
      tailDstTuple[i] = 0;
    }
    colPerMC = 0;
    colBlockPerMC = 0;
    loopOnMC = 0;
    colTC = 0;
    colBlockTC = 0;
    colOffset = 0;
    backStepLeft = 0;
    initDstTupleSize = 0;
    tailDstTupleSize = 0;
  }
  InfoCol() {
    Reset();
  }
};

struct InfoRow {
  int64_t rowPerMR;
  int64_t rowBlockPerMR;
  int64_t loopOnMR;
  int64_t rowTR;
  int64_t rowBlockTR;
  int64_t rowOffset;
  int64_t backStepUp;
  int64_t initSrcTupleSize;
  int64_t tailSrcTupleSize;
  int64_t initSrcTuple[TRANSPOSE_MAX_AXIS_NUM];
  int64_t tailSrcTuple[TRANSPOSE_MAX_AXIS_NUM];

  void Reset(int64_t coreNum = MAX_CORE_NUM) {
    initSrcTupleSize = (coreNum == MAX_CORE_NUM) ? TRANSPOSE_MAX_AXIS_NUM : initSrcTupleSize;
    for (int64_t i = 0; i < initSrcTupleSize; i++) {
      initSrcTuple[i] = 0;
    }
    tailSrcTupleSize = (coreNum == MAX_CORE_NUM) ? TRANSPOSE_MAX_AXIS_NUM : tailSrcTupleSize;
    for (int64_t i = 0; i < tailSrcTupleSize; i++) {
      tailSrcTuple[i] = 0;
    }
    rowPerMR = 0;
    rowBlockPerMR = 0;
    loopOnMR = 0;
    rowTR = 0;
    rowBlockTR = 0;
    rowOffset = 0;
    backStepUp = 0;
    initSrcTupleSize = 0;
    tailSrcTupleSize = 0;
  }

  InfoRow() {
    Reset();
  }
};

struct InfoCol2D {
  int64_t loopOnMC;
  int64_t colPerMC;
  int64_t colBlockPerMC;
  int64_t colTC;
  int64_t colBlockTC;
  int64_t colOffset;

  void Reset() {
    loopOnMC = 0;
    colPerMC = 0;
    colBlockPerMC = 0;
    colTC = 0;
    colBlockTC = 0;
    colOffset = 0;
  }

  InfoCol2D() {
    Reset();
  }
};

struct InfoRow2D {
  int64_t loopOnMR;
  int64_t rowPerMR;
  int64_t rowBlockPerMR;
  int64_t rowTR;
  int64_t rowBlockTR;
  int64_t rowOffset;

  void Reset() {
    loopOnMR = 0;
    rowPerMR = 0;
    rowBlockPerMR = 0;
    rowTR = 0;
    rowBlockTR = 0;
    rowOffset = 0;
  }

  InfoRow2D() {
    Reset();
  }
};

struct InfoPerCore {
  InfoN infoN;
  InfoCol infoCol;
  InfoRow infoRow;

  void Reset(int64_t coreNum = MAX_CORE_NUM) {
    infoN.Reset(coreNum);
    infoCol.Reset(coreNum);
    infoRow.Reset(coreNum);
  }

  InfoPerCore() {
    Reset();
  }
};

struct InfoPerCore2D {
  InfoN infoN;
  InfoCol2D infoCol2D;
  InfoRow2D infoRow2D;

  void Reset() {
    infoN.Reset();
    infoCol2D.Reset();
    infoRow2D.Reset();
  }
};

struct BorrowAxisPerCore {
  int64_t initTupleLogic;
  int64_t initTuple[BORROW_MAX_AXIS_NUM_LT];
  BorrowAxisPerCore() {
    initTupleLogic = 0;
    for (int i = 0; i < BORROW_MAX_AXIS_NUM_LT; i++) {
      initTuple[i] = 0;
    }
  }
};

struct OtherAxisPerCore {
  int64_t idx;
  int64_t base;
  int64_t loop;
  int64_t initTuple[TRANSPOSE_MAX_AXIS_NUM];
  OtherAxisPerCore() {
    idx = 0;
    base = 0;
    loop = 0;
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
      initTuple[i] = 0;
    }
  }
};

struct IndexInfo {
  int64_t idx_in;
  int64_t idx_out;
  int64_t loop;
  int64_t step;
  int64_t tail;
  int64_t pivot;
  int64_t dup;

  void Reset() {
    idx_in = 0;
    idx_out = 0;
    loop = 0;
    step = 1;
    tail = 0;
    pivot = 0;
    dup = 0;
  }
  IndexInfo() {
    Reset();
  }
};

// L:loop; R:repeat; S:stride; B:burstLen
struct LRSB {
  int64_t n;
  int64_t vol;
  int64_t loop;
  int64_t repeat;
  int64_t srcStride;
  int64_t dstStride;
  int64_t burstLen;
  int64_t srcOffset;
  int64_t dstOffset;
  LRSB() {
    Reset();
  }
  void Reset() {
    n = 0;
    vol = 0;
    loop = 0;
    repeat = 0;
    srcStride = 0;
    dstStride = 0;
    burstLen = 0;
    srcOffset = 0;
    dstOffset = 0;
  }
  void Set(int64_t l, int64_t r, int64_t ss, int64_t ds, int64_t b, int64_t st, int64_t dt) {
    loop = l;
    repeat = r;
    srcStride = ss;
    dstStride = ds;
    burstLen = b;
    srcOffset = st;
    dstOffset = dt;
  }
  void Set(int64_t nn, int64_t v, int64_t l, int64_t r, int64_t ss, int64_t ds, int64_t b, int64_t st, int64_t dt) {
    n = nn;
    vol = v;
    loop = l;
    repeat = r;
    srcStride = ss;
    dstStride = ds;
    burstLen = b;
    srcOffset = st;
    dstOffset = dt;
  }
};

struct BorrowInfo {
  int64_t srcNum;
  int64_t dstNum;
  int64_t srcNumNoDup;
  int64_t dstNumNoDup;
  int64_t otherNum;
  int64_t srcVol;
  int64_t dstVol;
  int64_t dupAxis;
  int64_t dstAxisPerm;
  int64_t srcAxisPerm;
  int64_t axisPerm;
  int64_t pivotSrcAxisDup;
  int64_t pivotDstAxisDup;
  int64_t majorDstLoop_in;
  int64_t tailDstLoop_in;
  int64_t majorSrcLoop_out;
  int64_t tailSrcLoop_out;
  int64_t majorBurstLen_in;
  int64_t tailBurstLen_in;
  int64_t majorBurstLen_out;
  int64_t tailBurstLen_out;
  int64_t majorInEle;
  int64_t tailInEle;
  int64_t majorInTailEle;
  int64_t tailInTailEle;
  int64_t majorOutEle;
  int64_t tailOutEle;
  int64_t majorOutTailEle;
  int64_t tailOutTailEle;
  int64_t srcStep;
  int64_t dstStep;
  int64_t xdxsVol[UB_REORDER_COMBINATION];
  int64_t lastTwoLoop;
  int64_t lastTwoRepeat;
  int64_t lastTwosListRepeat;
  int64_t lastTwodListRepeat;
  int64_t lastTwosStride;
  int64_t lastTwodStride;
  int64_t lastTwoOffset;
  int64_t loopPerCore[MAX_CORE_NUM];
  int64_t srcJumpFactorLogic_in;
  int64_t srcJumpFactorMod_in;
  int64_t dstJumpFactorLogic_in;
  int64_t dstJumpFactorMod_in;
  int64_t dstStrideCopyIn[BORROW_DST_AXIS_NUM_LT];
  int64_t dstFactorCopyIn[BORROW_DST_AXIS_NUM_LT];
  int64_t srcFactorCopyOut[BORROW_SRC_AXIS_NUM_LT];
  int64_t srcStrideCopyOut[BORROW_SRC_AXIS_NUM_LT];
  int64_t otherJumpFactor_in[BORROW_OTHER_AXIS_NUM];
  int64_t otherJumpStride_in[BORROW_OTHER_AXIS_NUM];
  int64_t otherJumpFactorMod_in[BORROW_OTHER_AXIS_NUM];
  int64_t otherInitTuple_in[BORROW_OTHER_AXIS_NUM];
  int64_t srcJumpFactorMod_out;
  int64_t dstJumpFactorMod_out;
  int64_t dstJumpstride_out;
  int64_t otherJumpFactor_out[BORROW_OTHER_AXIS_NUM];
  int64_t otherJumpStride_out[BORROW_OTHER_AXIS_NUM];
  int64_t otherJumpFactorMod_out[BORROW_OTHER_AXIS_NUM];
  int64_t otherInitTuple_out[BORROW_OTHER_AXIS_NUM];
  int64_t ubPermNum;
  int64_t ubPermRaw[BORROW_MAX_AXIS_NUM_LT];
  int64_t ubPerm[BORROW_MAX_AXIS_NUM_LT];
  LRSB lrsb[UB_REORDER_COMBINATION][UB_REORDER_LOOP];
  PermInfo permInfo;

  IndexInfo srcIndexIn[BORROW_SRC_AXIS_NUM_LT];
  IndexInfo srcIndexOut[BORROW_SRC_AXIS_NUM_LT];
  IndexInfo dstIndexIn[BORROW_DST_AXIS_NUM_LT];
  IndexInfo dstIndexOut[BORROW_DST_AXIS_NUM_LT];
  IndexInfo srcIndexInNoDup[BORROW_SRC_AXIS_NUM_LT];
  IndexInfo dstIndexInNoDup[BORROW_DST_AXIS_NUM_LT];
  IndexInfo srcIndexOutNoDup[BORROW_SRC_AXIS_NUM_LT];
  IndexInfo dstIndexOutNoDup[BORROW_DST_AXIS_NUM_LT];
  IndexInfo otherIndex[BORROW_OTHER_AXIS_NUM];

  BorrowAxisPerCore srcAxis_in[MAX_CORE_NUM];
  BorrowAxisPerCore dstAxis_in[MAX_CORE_NUM];
  OtherAxisPerCore otherAxis_in[MAX_CORE_NUM];

  void Reset(int64_t coreNum = MAX_CORE_NUM) {
    int loop = (coreNum == MAX_CORE_NUM) ? BORROW_SRC_AXIS_NUM_LT : srcNum;
    for (int i = 0; i < loop; i++) {
      srcIndexIn[i].Reset();
      srcIndexOut[i].Reset();
      srcIndexInNoDup[i].Reset();
      srcIndexOutNoDup[i].Reset();
      srcFactorCopyOut[i] = 0;
      srcStrideCopyOut[i] = 0;
    }

    loop = (coreNum == MAX_CORE_NUM) ? BORROW_DST_AXIS_NUM_LT : dstNum;
    for (int i = 0; i < loop; i++) {
      dstIndexIn[i].Reset();
      dstIndexOut[i].Reset();
      dstIndexInNoDup[i].Reset();
      dstIndexOutNoDup[i].Reset();
      dstFactorCopyIn[i] = 0;
      dstStrideCopyIn[i] = 0;
    }

    loop = (coreNum == MAX_CORE_NUM) ? BORROW_OTHER_AXIS_NUM : otherNum;
    for (int i = 0; i < loop; i++) {
      otherIndex[i].Reset();
      otherJumpFactor_in[i] = 0;
      otherJumpStride_in[i] = 0;
      otherJumpFactorMod_in[i] = 1;
      otherInitTuple_in[i] = 0;
      otherJumpFactor_out[i] = 0;
      otherJumpStride_out[i] = 0;
      otherJumpFactorMod_out[i] = 1;
      otherInitTuple_out[i] = 0;
    }

    memset_s(&ubPerm, sizeof(ubPerm), 0, sizeof(ubPerm));
    memset_s(&ubPermRaw, sizeof(ubPermRaw), 0, sizeof(ubPermRaw));
    memset_s(&lrsb, sizeof(lrsb), 0, sizeof(lrsb));
    memset_s(&permInfo, sizeof(permInfo), 0, sizeof(permInfo));
    memset_s(&loopPerCore, sizeof(loopPerCore), 0, sizeof(loopPerCore));
    memset_s(&srcAxis_in, sizeof(srcAxis_in), 0, sizeof(srcAxis_in));
    memset_s(&dstAxis_in, sizeof(dstAxis_in), 0, sizeof(dstAxis_in));
    memset_s(&otherAxis_in, sizeof(otherAxis_in), 0, sizeof(otherAxis_in));

    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
      xdxsVol[i] = 1;
    }

    srcNum = 0;
    dstNum = 0;
    srcNumNoDup = 0;
    dstNumNoDup = 0;
    otherNum = 0;
    srcVol = 1;
    dstVol = 1;
    dupAxis = 0;
    dstAxisPerm = 0x0;
    srcAxisPerm = 0x0;
    axisPerm = 0x0;
    pivotSrcAxisDup = 0;
    pivotDstAxisDup = 0;
    majorDstLoop_in = 1;
    tailDstLoop_in = 1;
    majorSrcLoop_out = 1;
    tailSrcLoop_out = 1;
    majorBurstLen_in = 1;
    tailBurstLen_in = 1;
    majorBurstLen_out = 1;
    tailBurstLen_out = 1;
    majorInEle = 0;
    tailInEle = 0;
    majorInTailEle = 0;
    tailInTailEle = 0;
    majorOutEle = 0;
    tailOutEle = 0;
    majorOutTailEle = 0;
    tailOutTailEle = 0;
    srcStep = 1;
    dstStep = 1;
    lastTwoLoop = 0;
    lastTwoRepeat = 0;
    lastTwosListRepeat = 0;
    lastTwodListRepeat = 0;
    lastTwosStride = 0;
    lastTwodStride = 0;
    lastTwoOffset = 0;

    srcJumpFactorLogic_in = 1;
    srcJumpFactorMod_in = 0;
    dstJumpFactorLogic_in = 1;
    dstJumpFactorMod_in = 0;
    srcJumpFactorMod_out = 0;
    dstJumpFactorMod_out = 0;
    dstJumpstride_out = 0;
    ubPermNum = 0;
  }

  BorrowInfo() {
    Reset();
  }
};

struct TwoDInfo {
  int64_t nAxisNum;
  int64_t colPerMC;
  int64_t colBlockPerMC;
  int64_t colBlockTC;
  int64_t rowPerMR;
  int64_t rowBlockPerMR;
  int64_t rowBlockTR;
  int64_t srcStrideIn;
  int64_t srcStrideInTail;
  int64_t dstStrideOut;
  int64_t dstStrideOutTail;
  int64_t nUnit;

  int64_t nFactor[TRANSPOSE_MAX_AXIS_NUM];
  int64_t nSrcStride[TRANSPOSE_MAX_AXIS_NUM];
  int64_t nDstStride[TRANSPOSE_MAX_AXIS_NUM];
  InfoPerCore2D infoPerCore2D[MAX_CORE_NUM];
  
  int64_t nFactorSize;
  int64_t nSrcStrideSize;
  int64_t nDstStrideSize;
  int64_t infoPerCore2DSize;

  void Reset(int64_t coreNum = MAX_CORE_NUM) {
    for (int64_t i = 0; i < coreNum; i++) {
      infoPerCore2D[i].Reset();
    }
    nAxisNum = 0;
    colPerMC = 0;
    colBlockPerMC = 0;
    colBlockTC = 0;
    rowPerMR = 0;
    rowBlockPerMR = 0;
    rowBlockTR = 0;
    srcStrideIn = 0;
    srcStrideInTail = 0;
    dstStrideOut = 0;
    dstStrideOutTail = 0;
    nUnit = 1;
    nFactorSize = 0;
    nSrcStrideSize = 0;
    nDstStrideSize = 0;
    infoPerCore2DSize = 0;
  }

  TwoDInfo() {
    Reset();
  }
};

struct RuntimeInfo {
  /*
   * last axis not transposed
   */
  int64_t id;
  int64_t coreNum;
  int64_t scenarioSize;
  TransposeScenario scenarios[MAX_RETRY_SCENARIO];
  int64_t ubReorderFactor;
  int64_t ubThreshold;  // unit: block number
  int64_t fp16Offset1;
  int64_t fp16Offset2;
  int64_t fp16Offset3;  // store overlap data for dirty data
  int64_t lastAxisElementNum;
  int64_t lastAxisElementNumAligned;
  int64_t srcStrideLogic;
  int64_t srcStride;
  int64_t dstStride;
  int64_t backNum;
  int64_t skipEle;
  int64_t colPermSize;
  //int64_t rowPermSize;
  int64_t ncrsSize;
  int64_t nJumpAxisNum;
  int64_t srcJumpAxisNum;
  int64_t dstJumpAxisNum;
  int64_t infoNSize;
  int64_t infoColSize;
  int64_t infoRowSize;
  int64_t nRangeSize;
  int64_t colRangeSize;
  int64_t rowRangeSize;

  /*
   * scenario_0: identical
   */
  IdenticalInfo infoPerCoreIdentical[MAX_CORE_NUM];

  /*
   * scenario_1: last axis large not transposed
   */
  InfoPerCoreLastAxisNT infoPerCoreLastAxisNT[MAX_CORE_NUM];

  /*
   * scenario_3: last axis huge not transposed
   */
  LastAxisNTHugeInfo hugeInfo;

  /*
   * scenario_4: borrow axis scenario
   */
  BorrowInfo borrowInfo;

  /*
   * scenario_11: borrow axis scenario
   */
  TwoDInfo twoDInfo;

  /*
   *
   * last axis transposed
   *
   */
  int64_t colPerm[TRANSPOSE_MAX_AXIS_NUM];

  NCR ncrs[TRANSPOSE_MAX_AXIS_NUM];
  NCR ncr;
  SplitParam sp;
  TilingModel tilingModel;

  InfoN infoN[MAX_CORE_NUM];
  InfoCol infoCol[MAX_CORE_NUM];
  InfoRow infoRow[MAX_CORE_NUM];
  InfoPerCore infoPerCore[MAX_CORE_NUM];
  int64_t nRange[MAX_CORE_NUM][2];
  int64_t colRange[MAX_CORE_NUM][2];
  int64_t rowRange[MAX_CORE_NUM][2];

  int64_t nJumpFactor[TRANSPOSE_MAX_AXIS_NUM];
  int64_t nJumpStrideIn[TRANSPOSE_MAX_AXIS_NUM];
  int64_t nJumpStrideOut[TRANSPOSE_MAX_AXIS_NUM];
  int64_t nJumpFactorMod[TRANSPOSE_MAX_AXIS_NUM];

  int64_t srcJumpFactor[TRANSPOSE_MAX_AXIS_NUM];
  int64_t srcJumpStride[TRANSPOSE_MAX_AXIS_NUM];
  int64_t srcJumpFactorMod[TRANSPOSE_MAX_AXIS_NUM];

  int64_t dstJumpFactor[TRANSPOSE_MAX_AXIS_NUM];
  int64_t dstJumpStride[TRANSPOSE_MAX_AXIS_NUM];
  int64_t dstJumpFactorMod[TRANSPOSE_MAX_AXIS_NUM];

  void ResetCommon() {
    id = 0;
    coreNum = MAX_CORE_NUM;
    scenarioSize = 0; 
    ubReorderFactor = 1;
    ubThreshold = 1;
    fp16Offset1 = 0;
    fp16Offset2 = 0;
    fp16Offset3 = 0;
    lastAxisElementNum = 0;
    lastAxisElementNumAligned = 0;
    srcStrideLogic = 0;
    srcStride = 0;
    dstStride = 0;
    backNum = 0;
    skipEle = 0;
    nJumpAxisNum = 0;
    srcJumpAxisNum = 0;
    dstJumpAxisNum = 0;
    ncrsSize = 0;
    colPermSize = 0;
    infoNSize = 0;
    infoColSize = 0;
    infoRowSize = 0;
    nRangeSize = 0;
    colRangeSize = 0;
    rowRangeSize = 0;
  }

  void ResetScenario0() {
    for (int64_t i = 0; i < coreNum; i++) {
      infoPerCoreIdentical[i].Reset();
    }
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
      nJumpFactor[i] = 0;
      nJumpStrideIn[i] = 0;
      nJumpStrideOut[i] = 0;
      nJumpFactorMod[i] = 1;
      srcJumpFactor[i] = 0;
      srcJumpStride[i] = 0;
      srcJumpFactorMod[i] = 1;
      dstJumpFactor[i] = 0;
      dstJumpStride[i] = 0;
      dstJumpFactorMod[i] = 1;
    }
  }

  void ResetScenario1() {
    for (int64_t i = 0; i < coreNum; i++) {
      infoPerCoreLastAxisNT[i].Reset();
    }
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
      nJumpFactor[i] = 0;
      nJumpStrideIn[i] = 0;
      nJumpStrideOut[i] = 0;
      nJumpFactorMod[i] = 1;
      srcJumpFactor[i] = 0;
      srcJumpStride[i] = 0;
      srcJumpFactorMod[i] = 1;
      dstJumpFactor[i] = 0;
      dstJumpStride[i] = 0;
      dstJumpFactorMod[i] = 1;
    }
  }

  void ResetScenario2() {
    for (int64_t i = 0; i < coreNum; i++) {
      infoPerCoreLastAxisNT[i].Reset();
    }
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
      nJumpFactor[i] = 0;
      nJumpStrideIn[i] = 0;
      nJumpStrideOut[i] = 0;
      nJumpFactorMod[i] = 1;
      srcJumpFactor[i] = 0;
      srcJumpStride[i] = 0;
      srcJumpFactorMod[i] = 1;
      dstJumpFactor[i] = 0;
      dstJumpStride[i] = 0;
      dstJumpFactorMod[i] = 1;
    }
  }

  void ResetScenario3() {
    hugeInfo.Reset();
    for (int64_t i = 0; i < coreNum; i++) {
      infoPerCoreLastAxisNT[i].Reset();
    }
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
      nJumpFactor[i] = 0;
      nJumpStrideIn[i] = 0;
      nJumpStrideOut[i] = 0;
      nJumpFactorMod[i] = 1;
      srcJumpFactor[i] = 0;
      srcJumpStride[i] = 0;
      srcJumpFactorMod[i] = 1;
      dstJumpFactor[i] = 0;
      dstJumpStride[i] = 0;
      dstJumpFactorMod[i] = 1;
    }
  }

  void ResetScenario4() {
    borrowInfo.Reset(coreNum);
  }

  void ResetScenario5() {
    borrowInfo.Reset(coreNum);
  }

  void ResetScenario6() {
    ResetScenario2();
  }

  void ResetScenario7() {
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
      nJumpFactor[i] = 0;
      nJumpStrideIn[i] = 0;
      nJumpStrideOut[i] = 0;
      nJumpFactorMod[i] = 1;
      srcJumpFactor[i] = 0;
      srcJumpStride[i] = 0;
      srcJumpFactorMod[i] = 1;
      dstJumpFactor[i] = 0;
      dstJumpStride[i] = 0;
      dstJumpFactorMod[i] = 1;
      ncrs[i].Reset();
    }
    ncr.Reset();
    sp.Reset();
    tilingModel.Reset();
    for (int64_t i = 0; i < coreNum; i++) {
      infoN[i].Reset();
    }
    for (int64_t i = 0; i < coreNum; i++) {
      infoCol[i].Reset();
    }
    for (int64_t i = 0; i < coreNum; i++) {
      infoRow[i].Reset();
    }
    for (int64_t i = 0; i < coreNum; i++) {
      infoPerCore[i].Reset(coreNum);
    }
  }

  void ResetScenario9() {
    for (int64_t i = 0; i < coreNum; i++) {
      infoPerCoreLastAxisNT[i].Reset();
    }
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
      nJumpFactor[i] = 0;
      nJumpStrideIn[i] = 0;
      nJumpStrideOut[i] = 0;
      nJumpFactorMod[i] = 1;
      srcJumpFactor[i] = 0;
      srcJumpStride[i] = 0;
      srcJumpFactorMod[i] = 1;
      dstJumpFactor[i] = 0;
      dstJumpStride[i] = 0;
      dstJumpFactorMod[i] = 1;
    }
  }

  void ResetScenario10() {
    borrowInfo.Reset(coreNum);
  }

  void ResetScenario11() {
    twoDInfo.Reset(coreNum);
  }

  void Reset() {

    if (scenarioSize == 0) {
        ResetScenario0();
        ResetScenario1();
        ResetScenario2();
        ResetScenario3();
        ResetScenario4();
        ResetScenario5();
        ResetScenario6();
        ResetScenario7();
        ResetScenario9();
        ResetScenario10();
        ResetScenario11();
    } else {
      for (int64_t i = 0; i < scenarioSize; i++) {
        switch(scenarios[i]) {
          case SCENARIO_0:
            ResetScenario0();
            break;
          case SCENARIO_1:
            ResetScenario1();
            break;
          case SCENARIO_2:
            ResetScenario2();
            break;
          case SCENARIO_3:
            ResetScenario3();
            break;
          case SCENARIO_4:
            ResetScenario4();
            break;
          case SCENARIO_5:
            ResetScenario5();
            break;
          case SCENARIO_6:
            ResetScenario6();
            break;
          case SCENARIO_7:
            ResetScenario7();
            break;
          case SCENARIO_9:
            ResetScenario9();
            break;
          case SCENARIO_10:
            ResetScenario10();
            break;
          case SCENARIO_11:
            ResetScenario11();
            break;
          default:
            break;
        }
      }
    }

    ResetCommon();
  }

  RuntimeInfo(): coreNum(MAX_CORE_NUM), scenarioSize(0) {
    // set coreNum & scenarioSize before Reset for in Reset() will use them 
    Reset();
  }
};


void ReduceAxis(const std::string& opType, CompilerInfo& compilerInfo, ShapeInfo& shapeInfo);

void RemoveAxis(ShapeInfo& shapeInfo);

void MergeAxis(ShapeInfo& shapeInfo);

bool TransposeCalcTilingData(const std::string& opType, const CompilerInfo& compilerInfo, ShapeInfo& shapeInfo,
                             RuntimeInfo& runtimeInfo);

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool TransposeTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                     OpRunInfo& runInfo);

}  // namespace optiling

#endif  //__TRANSPOSE_H__
