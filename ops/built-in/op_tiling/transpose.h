/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

namespace optiling {

#define TRANSPOSE_MAX_AXIS_NUM 8
#define BYTES_PER_BLOCK 32
#define UB_REORDER_FACTOR 33
#define ELE_NUM_PER_BLOCK_FP16 16
#define ELE_NUM_PER_BLOCK_FP32 8
#define ELE_NUM_PER_BLOCK_INT64 4
/*
 * 4 * 32 block = 4KB, this value should be consistent with the variable in transpose.py
 * 1KB : reserved
 * 2KB : store overlap data for dirtyData
 * 3KB : reserved
 * 4KB : reserved
 */
#define UB_RESERVED_BLOCK_SIZE 4 * 32
#define UB_RESERVED_KB 4
#define LAST_AXIS_HUGE_THRESHOLD 100 * 1024 //unit B
#define HUGE_BLOCKS_UNIT (LAST_AXIS_HUGE_THRESHOLD / 32) //unit blocks, 3200 blocks
#define LAST_AXIS_BLOCK_ALIGN_LARGE_THRESHOLD 4096 //unit B
#define LAST_AXIS_NOT_BLOCK_ALIGN_LARGE_THRESHOLD 128 //unit B
#define WORKSPACE_MAX_SIZE (int64_t)(16 * 1024 * 1024 * 1024) // 16GB
#define UB_CAP_BLOCKS 7800 // for 310 256 - 8 = 248KB = 7936 Blocks

#define STRIDE_BOUNDARY 65535
#define NBURST_BOUNDARY 4095
#define ACCU_BLOCK_SIZE 128 // keep same with transpose.py , not gt 240 for both 310 and 910
#define ACCU_BLOCK_SIZE_IDENTICAL 1024

#define MAX_COL_FP16_VNCHWCONV_FULL    256 // verify 128 is better than 248 //496
#define MAX_COL_FP16_VNCHWCONV_PARTIAL 256
#define MAX_ROW_FP16_VNCHWCONV_FULL    128
#define UB_SIZE_1_16_FP16              3968 // 3968 * 33 < 256 * 1024
#define SMALL_SHAPE_SIZE_THRESHOLD     1024
#define F2T_THRESHOLD_B16  64 // unit : byte
#define F2T_THRESHOLD_B32  32 // unit : byte
#define TILING_FIXED_MAX_LEN 1024 // unit: bytes, keep same with transpose.py

enum TransposeScenario {
    SCENARIO_0 = 0,     //identical shape
    SCENARIO_1 = 1,     //large last axis and not transpose
    SCENARIO_2 = 2,     //small last axis and not transpose
    SCENARIO_3 = 3,     //huge  last axis and not transpose
    SCENARIO_6 = 6,     //small shape
    SCENARIO_7 = 7,     //last axis transpose
    SCENARIO_8 = 8,     //920A verifaction
};

enum SubScenarioLastAxisTrans {
    LAST_AXIS_TR_COMMON = 0,
    LAST_AXIS_TR_F2T = 1, // fat 2 thin
    LAST_AXIS_TR_T2F = 2, // thin 2 fat
};

struct SplitParam {
    int64_t nFactor;
    int64_t colFactor;
    int64_t rowFactor;

    SplitParam() {
        nFactor  = 1;
        colFactor = 1;
        rowFactor = 1;
    }

    void Set(int64_t n, int64_t c, int64_t r) {
        nFactor = n;
        colFactor = c;
        rowFactor = r;
    }
};

struct NCR {
    vector<int64_t> n;
    vector<int64_t> col;
    vector<int64_t> row;
    int64_t nVol;
    int64_t cVol;
    int64_t rVol;
};

class TilingModel {
public:
    TilingModel(int64_t p, int64_t c, int64_t u,
                SubScenarioLastAxisTrans scenario, std::string name) : coreNum(c),
                                                                       ubBlocks(u),
                                                                       priority(p),
                                                                       maxCol(0),
                                                                       maxRow(0),
                                                                       subScenario(scenario),
                                                                       modelName(name) {}
    virtual ~TilingModel(){};
    virtual void Decision(const NCR & ncr, int64_t dim) = 0;
    virtual bool Isf2t() {
        return false;
    }
    virtual bool Ist2f() {
        return false;
    }
    SplitParam sp;
    NCR ncr;
    int64_t coreNum;
    int64_t ubBlocks;
    int64_t priority;
    int64_t maxCol; //unit: bytes
    int64_t maxRow;
    SubScenarioLastAxisTrans subScenario;
    std::string modelName;
};

struct TMCompare {
    bool operator()(const std::shared_ptr<TilingModel> & lhs, const std::shared_ptr<TilingModel> & rhs) const {
        return lhs->priority > rhs->priority;
    }
};

struct ShapeInfo {
    std::vector<int64_t> inShape;
    std::vector<int64_t> outShape;
    std::vector<int64_t> perm;
    std::vector<int64_t> reducedInShape;
    std::vector<int64_t> reducedOutShape;
    std::vector<int64_t> reducedPerm;
    std::vector<int64_t> reducedPermGrad;

    int64_t dim;
    int64_t totalVolumeLogic;
    int64_t totalVolumeActual;
    int64_t identical;
    int64_t lastAxisLen;
    int64_t lastAxisBurstLen;
    int64_t elePerBlock;
    int64_t eleLenInBytes;
    int64_t alignElement; //unit: element number padding.  eg. dtype=float, axis[0]= 11, alignElement=8-(11-8)%8=5
    bool isLastAxisTranspose;
    bool isLastAxisHuge;
    TransposeScenario scenario;

    ShapeInfo() : reducedPermGrad(TRANSPOSE_MAX_AXIS_NUM, 0) {
        dim = 0;
        totalVolumeLogic = 0;
        totalVolumeActual = 0;
        identical = 0;
        lastAxisLen = 0;
        lastAxisBurstLen = 0;
        elePerBlock = 8;
        eleLenInBytes = 0;
        alignElement = 0;
        isLastAxisTranspose = false;
        isLastAxisHuge = false;
        scenario = SCENARIO_0;
    }
};

struct CompilerInfo {
    int64_t coreNum;
    int64_t usedCoreNum;
    int64_t ubSize; //unit: block
    int64_t ubSizeCouldUse;//unit: block
    int64_t fp16Times;
    int64_t blockSize;
    std::string dType;
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

    LastAxisNTLoopInfo() {
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
};

struct LastAxisNTHugeInfo {
    int64_t majorLoopNum;
    int64_t majorBlocks;
    int64_t tailBlocks;
    int64_t backEle;

    LastAxisNTHugeInfo() {
        majorLoopNum = 0;
        majorBlocks = 0;
        tailBlocks = 0;
        backEle = 0;
    }
};

struct WorkspaceInfo {
    int64_t loop;
    int64_t repeat1;
    int64_t repeat2;
    int64_t repeat3;

    WorkspaceInfo() {
        loop = 0;
        repeat1 = 0;
        repeat2 = 0;
        repeat3 = 0;
    }
};

struct IdenticalInfo {
    int64_t base;
    int64_t eleNum;
    int64_t majorLoop;
    int64_t majorNum;
    int64_t tailNum;
    int64_t notAlignEle;

    IdenticalInfo() {
        base = 0;
        eleNum = 0;
        majorLoop = 0;
        majorNum = 0;
        tailNum = 0;
        notAlignEle = 0;
    }
};

struct InfoPerCoreLastAxisNT {
    int64_t base;
    int64_t num;
    int64_t initTuple[TRANSPOSE_MAX_AXIS_NUM];
    LastAxisNTLoopInfo loopInfo;
    WorkspaceInfo workspaceInfo;
    InfoPerCoreLastAxisNT() {
        base = 0;
        num = 0;
        for (int64_t i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
            initTuple[i] = 0;
        }
    }
};

struct InfoN {
    int64_t loopOnN;
    int64_t nOffsetLogic;
    int64_t nOffsetActual;
    std::vector<int64_t> initNTuple;

    InfoN() {
        loopOnN = 0;
        nOffsetLogic = 0;
        nOffsetActual = 0;
        initNTuple.resize(TRANSPOSE_MAX_AXIS_NUM, 0);
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
    std::vector<int64_t> initDstTuple;
    std::vector<int64_t> tailDstTuple;

    InfoCol() {
        colPerMC = 0;
        colBlockPerMC  = 0;
        loopOnMC = 0;
        colTC = 0;
        colBlockTC = 0;
        colOffset = 0;
        backStepLeft = 0;
        initDstTuple.resize(TRANSPOSE_MAX_AXIS_NUM, 0);
        tailDstTuple.resize(TRANSPOSE_MAX_AXIS_NUM, 0);
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
    std::vector<int64_t> initSrcTuple;
    std::vector<int64_t> tailSrcTuple;

    InfoRow() {
        rowPerMR = 0;
        rowBlockPerMR = 0;
        loopOnMR = 0;
        rowTR = 0;
        rowBlockTR = 0;
        rowOffset = 0;
        backStepUp = 0;
        initSrcTuple.resize(TRANSPOSE_MAX_AXIS_NUM, 0);
        tailSrcTuple.resize(TRANSPOSE_MAX_AXIS_NUM, 0);
    }
};

struct InfoPerCore {
    InfoN infoN;
    InfoCol infoCol;
    InfoRow infoRow;
};

struct BorrowAxis {
    int64_t index;
    int64_t loop;
    int64_t step;
    int64_t tail;
    int64_t stride;
    BorrowAxis() {
        index = -1;
        loop = 0;
        step = 0;
        tail = 0;
        stride = 1;
    }
};

struct BorrowInfo {
    BorrowAxis src_1;
    BorrowAxis src_2;
    BorrowAxis dst_1;
    BorrowAxis dst_2;
};

struct RuntimeInfo {
    /*
     * last axis not transposed
     */
    int64_t byWorkspace;
    int64_t ubReorderFactor;
    int64_t ubThreshold; //unit: block number
    int64_t fp16Offset1;
    int64_t fp16Offset2;
    int64_t fp16Offset3;//store overlap data for dirty data
    int64_t cycleNumWorkspace;
    int64_t loopNumWorkspace;
    int64_t nBurstWorkspace;
    int64_t nBurstTailWorkspace;
    int64_t lastAxisElementNum;
    int64_t lastAxisElementNumAligned;
    int64_t srcStrideWorkspace;
    int64_t dstStrideWorkspace;
    int64_t workspaceSizeInBytes;
    int64_t srcStrideLogic;
    int64_t backNum;
    int64_t skipEle;
    std::vector<std::pair<int64_t,int64_t>> initRanges;
    std::vector<std::pair<int64_t,int64_t>> extendRanges;
    std::vector<int64_t> dstBaseAddr;
    std::vector<int64_t> srcBaseAddr;
    std::vector<int64_t> srcBaseAddrWorkspace;
    std::vector<int64_t> dirtyDataStartAddrPerCore;

    /*
     * scenario_0: identical
     */
    std::vector<IdenticalInfo> infoPerCoreIdentical;

    /*
     * scenario_1: last axis large not transposed
     */
    std::vector<InfoPerCoreLastAxisNT> infoPerCoreLastAxisNT;

    /*
     * scenario_3: last axis huge not transposed
     */
    LastAxisNTHugeInfo hugeInfo;

    /*
     *
     * last axis transposed
     *
     */
    std::vector<int64_t> colPerm;
    std::vector<int64_t> rowPerm;
    vector<NCR> ncrs;
    NCR ncr;
    SplitParam sp;
    //priority_queue top() return const reference, unique_ptr no copy ctro
    std::priority_queue<std::shared_ptr<TilingModel>, std::vector<std::shared_ptr<TilingModel>>, TMCompare> pqtm;

    int64_t nJumpAxisNum;
    int64_t srcJumpAxisNum;
    int64_t dstJumpAxisNum;
    int64_t rPartVol;

    std::vector<InfoN> infoN;
    std::vector<InfoCol> infoCol;
    std::vector<InfoRow> infoRow;

    std::vector<InfoPerCore> infoPerCore;

    std::vector<std::pair<int64_t, int64_t>> nRange;
    std::vector<std::pair<int64_t, int64_t>> colRange;
    std::vector<std::pair<int64_t, int64_t>> rowRange;

    int64_t nJumpFactor[TRANSPOSE_MAX_AXIS_NUM];
    int64_t nJumpStride[TRANSPOSE_MAX_AXIS_NUM];
    int64_t nJumpFactorMod[TRANSPOSE_MAX_AXIS_NUM];

    int64_t srcJumpFactor[TRANSPOSE_MAX_AXIS_NUM];
    int64_t srcJumpStride[TRANSPOSE_MAX_AXIS_NUM];
    int64_t srcJumpFactorMod[TRANSPOSE_MAX_AXIS_NUM];

    int64_t dstJumpFactor[TRANSPOSE_MAX_AXIS_NUM];
    int64_t dstJumpStride[TRANSPOSE_MAX_AXIS_NUM];
    int64_t dstJumpFactorMod[TRANSPOSE_MAX_AXIS_NUM];

    RuntimeInfo() {
        byWorkspace = 0;
        ubReorderFactor = 1;
        ubThreshold = 1;
        fp16Offset1 = 0;
        fp16Offset2 = 0;
        fp16Offset3 = 0;
        cycleNumWorkspace = 0;
        loopNumWorkspace = 0;
        nBurstWorkspace = 0;
        nBurstTailWorkspace = 0;
        lastAxisElementNum = 0;
        lastAxisElementNumAligned = 0;
        srcStrideWorkspace = 0;
        dstStrideWorkspace = 0;
        workspaceSizeInBytes = 0;
        srcStrideLogic = 0;
        backNum = 0;
        skipEle = 0;
        nJumpAxisNum = 0;
        srcJumpAxisNum = 0;
        dstJumpAxisNum = 0;
        rPartVol = 0;

        for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
            nJumpFactor[i] = 0;
            nJumpStride[i] = 0;
            nJumpFactorMod[i] = 1;
            srcJumpFactor[i] = 0;
            srcJumpStride[i] = 0;
            srcJumpFactorMod[i] = 1;
            dstJumpFactor[i] = 0;
            dstJumpStride[i] = 0;
            dstJumpFactorMod[i] = 1;
        }
    }
};

void ReduceAxis(const std::string & opType,
                CompilerInfo & compilerInfo,
                ShapeInfo & shapeInfo);

void RemoveAxis(ShapeInfo & shapeInfo);

void MergeAxis(ShapeInfo & shapeInfo);

bool TransposeCalcTilingData(const std::string &opType,
                             const CompilerInfo & compilerInfo,
                             ShapeInfo & shapeInfo,
                             RuntimeInfo & runtimeInfo);

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool TransposeTiling(const std::string &opType,
                     const TeOpParas &opParas,
                     const nlohmann::json &op_info,
                     OpRunInfo &runInfo);

}// namespace optiling

#endif  //__TRANSPOSE_H__
