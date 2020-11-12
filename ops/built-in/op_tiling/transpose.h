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
#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"
#include "register/op_tiling.h"

namespace optiling {

#define TRANSPOSE_MAX_AXIS_NUM 8
#define BYTES_PER_BLOCK 32
#define UB_REORDER_FACTOR 33
#define ELE_NUM_PER_BLOCK_FP16 16
/*
 * 4 * 32 block = 4KB, this value should be consistent with the variable in transpose.py
 * 1KB : reserved
 * 2KB : store overlap data for dirtyData
 * 3KB : reserved
 * 4KB : reserved
 */
#define UB_RESERVED_BLOCK_SIZE 4 * 32
#define LAST_AXIS_HUGE_THRESHOLD 1000
#define STRIDE_BOUNDARY 65535
#define NBURST_BOUNDARY 4095
#define MAX_COL_FP16_VNCHWCONV_FULL    496
#define MAX_COL_FP16_VNCHWCONV_PARTIAL 256 
#define MAX_ROW_FP16_VNCHWCONV_FULL    128

enum TransposeScenario {
    e_last_axis_transposed = 0,
    e_last_axis_not_transposed = 1
};

struct ShapeInfo {
    std::vector<int64_t> inShape;
    std::vector<int64_t> outShape;
    std::vector<int64_t> perm;
    std::vector<int64_t> reducedInShape;
    std::vector<int64_t> reducedOutShape;
    std::vector<int64_t> reducedOutShapeAxis8;
    std::vector<int64_t> reducedPerm;
    std::vector<int64_t> reducedPermGrad;
    TransposeScenario scenario;
    int64_t dim;
    int64_t totalVolumeLogic;
    int64_t totalVolumeActual;
    int64_t volumePerCore;
    int64_t identical;
    bool isLastAxisTranspose;
    bool isLastAxisHuge;
    ShapeInfo() : reducedPermGrad(TRANSPOSE_MAX_AXIS_NUM, 0) {
        scenario = e_last_axis_transposed;
        dim = 0;
        totalVolumeLogic = 0;
        totalVolumeActual = 0;
        volumePerCore = 0;
        identical = 0;
        isLastAxisTranspose = false;
        isLastAxisHuge = false;
    }
};


struct CompilerInfo {
    int64_t coreNum;
    int64_t ubSize; //unit: block
    int64_t ubSizeCouldUse;//unit: block
    std::string dType;
    std::string opType;
    CompilerInfo() {
        coreNum = 0;
        ubSize = 0;
        ubSizeCouldUse = 0;
    }
};

struct RuntimeInfo {
    /*
     *
     * last axis not transposed
     *
     */
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
    std::vector<std::pair<int64_t,int64_t>> initRanges;
    std::vector<std::pair<int64_t,int64_t>> extendRanges;
    std::vector<int64_t> dstBaseAddr;
    std::vector<int64_t> srcBaseAddr;
    std::vector<int64_t> srcBaseAddrWorkspace;
    std::vector<int64_t> dirtyDataStartAddrPerCore;

    /*
     *
     * last axis transposed
     *
     */
    std::vector<int64_t> colElePerMC;
    std::vector<int64_t> colBlockPerMC;
    std::vector<int64_t> loopOnMC;
    std::vector<int64_t> colEleTC;
    std::vector<int64_t> colBlockTC;
    std::vector<int64_t> colOffset;

    std::vector<int64_t> rowPerMR;
    std::vector<int64_t> rowBlockPerMR;
    std::vector<int64_t> loopOnMR;
    std::vector<int64_t> rowTR;
    std::vector<int64_t> rowBlockTR;
    std::vector<int64_t> rowOffset;

    std::vector<int64_t> backStepUp;
    std::vector<int64_t> backStepLeft;

    vector<vector<int64_t>> initJumpCounter;
    vector<vector<int64_t>> tailJumpCounter;

    int64_t srcJumpFactor[TRANSPOSE_MAX_AXIS_NUM];
    int64_t srcJumpFactorMod[TRANSPOSE_MAX_AXIS_NUM];
    int64_t srcJumpStride[TRANSPOSE_MAX_AXIS_NUM];
    int64_t dstJumpStride;
    int64_t srcJumpAxisNum;

    std::vector<int64_t> rowPerCore;

    RuntimeInfo() {
        /*
         *
         *
         */
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
        /*
         *
         *
         */
        dstJumpStride = 1;
        srcJumpAxisNum = 0;
        for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
            srcJumpFactor[i] = 0;
            srcJumpFactorMod[i] = 1;
            srcJumpStride[i] = 0;
        }
    }
};


struct LevelInfo {
    int64_t nBurst;
    int64_t nBurstTail;
    int64_t burstLen; //unit: block number
    int64_t burstLenTail; //unit: block number, only used for shape indetical scenario
    int64_t alignElement; //unit: element number padding.  eg. dtype=float, axis[0]= 11, alignElement=8-(11-8)%8=5
    int64_t byWorkspace;
    int64_t elementNumPerBurst;
    int64_t srcStride;
    int64_t identicalLoopNum;//used only for shape identical scenario
    int64_t levelLoopNum[TRANSPOSE_MAX_AXIS_NUM];
    int64_t srcGapPerRound[TRANSPOSE_MAX_AXIS_NUM];
    int64_t hasTail[TRANSPOSE_MAX_AXIS_NUM];
    bool allLevelOneInUb;
    vector<int64_t> dstLevelAccuVolume;
    vector<int64_t> srcLevelAccuVolume;
    LevelInfo() : dstLevelAccuVolume(TRANSPOSE_MAX_AXIS_NUM, 1),
                  srcLevelAccuVolume(TRANSPOSE_MAX_AXIS_NUM, 1) {
        nBurst = 1;
        nBurstTail = 0;
        burstLen = 0;
        burstLenTail = 0;
        alignElement = 0;
        byWorkspace = 0;
        elementNumPerBurst = 0;
        srcStride = 0;
        identicalLoopNum = 0;
        allLevelOneInUb = true;
        for(int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
            levelLoopNum[i] = 1;
            srcGapPerRound[i] = 0;
            hasTail[i] = 0;
        }
    }
};


void ReduceAxis(const std::string & opType,
                const CompilerInfo & compilerInfo,
                ShapeInfo & shapeInfo);

void RemoveAxis(ShapeInfo & shapeInfo);

void MergeAxis(ShapeInfo & shapeInfo);

bool TransposeCalcTilingData(const CompilerInfo & compilerInfo,
                             const ShapeInfo & shapeInfo,
                             RuntimeInfo & runtimeInfo,
                             LevelInfo & levelInfo);

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
