/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * dynamic TransposeD op tiling
 */

#include "transpose.h"

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <math.h>

#include "op_tiling.h"
#include "op_log.h"
//#include "error_util.h"
#include "securec.h"
#include <nlohmann/json.hpp>

using namespace std;


namespace optiling {

struct BLOCK_STRUCT {
    char buff[BYTES_PER_BLOCK];
};

#define CHECK_TRANSPOSE_TILING_RESULT(ret)\
    if (ret == false) {\
        OP_LOGE(compilerInfo.opType.c_str(), "Failed to calc runtimeinfo because of ub capacity.");\
        return false;\
    }

#define XXX(s) for(int i = 0; i < 100; i++)printf(s);printf("\n");

static void hexdump(const void *ptr, int buflen) {
  printf("hexdump: %d\n", buflen);
  unsigned char *buf = (unsigned char*)ptr;
  int i, j;
  for (i=0; i<buflen; i+=16) {
    printf("%06x: ", i);
    for (j=0; j<16; j++)
      if (i+j < buflen)
        printf("%02x ", buf[i+j]);
      else
        printf("   ");
    printf("\n");
  }
}


static int GCD(int a, int b) {
    if (b == 0) return a;
    return GCD(b, a % b);
}


static string vec_to_string(const vector<int64_t> & v) {
    string s;
    bool first = true;
    for (auto i : v) {
        if (first) {
            s += std::to_string(i);
            first = false;
        } else {
            s += "," + std::to_string(i);
        }
    }
    return s;
}


string vec_to_string(const int64_t * v, int64_t size) {
    string s;
    bool first = true;

    if (v == nullptr) {
        return s;
    }

    for (int i = 0; i < size; i ++) {
        if (first) {
            s += std::to_string(v[i]);
            first = false;
        } else {
            s += "," + std::to_string(v[i]);
        }
    }
    return s;
}


static int64_t SizeofDType(const string & dType) {
    if (dType == "int8" || dType == "uint8") {
        return 1;
    } else if (dType == "int16" || dType == "uint16" || dType == "float16" || dType == "bool") {
        return 2;
    } else if (dType == "int32" || dType == "uint32" || dType == "float" || dType == "float32") {
        return 4;
    } else if (dType == "int64" || dType == "uint64" || dType == "float64" || dType == "double") {
        return 8;
    }
    return 1;
}


static int64_t ElementNumPerBlock(const string & dType) {
    if (dType == "int8" || dType == "uint8") {
        return 32;
    } else if (dType == "int16" || dType == "uint16" || dType == "float16" || dType == "bool") {
        return 16;
    } else if (dType == "int32" || dType == "uint32" || dType == "float" || dType == "float32") {
        return 8;
    } else if (dType == "int64" || dType == "uint64" || dType == "float64" || dType == "double") {
        return 4;
    }
    return 32;
}


static int64_t GmMemOffsetFromBlock(uint64_t blockOffset,
                                    uint64_t burstLen,
                                    uint64_t alignElement,
                                    const string & dType) {
    return blockOffset * (burstLen * ElementNumPerBlock(dType) - alignElement);
}



static bool Is32BAligned(const CompilerInfo & compilerInfo, const vector<int64_t> & reducedOutShape) {
    int64_t dim = reducedOutShape.size();
    return reducedOutShape[dim - 1] % ElementNumPerBlock(compilerInfo.dType) == 0;
}


static bool IsStrideTooHuge(const LevelInfo & levelInfo) {
    //Since srcStride is logicl one, so we need to mul burstLen to get real stride
    return levelInfo.srcStride * levelInfo.burstLen > STRIDE_BOUNDARY;
}


static bool IsLastAxisJoinTranspose(ShapeInfo & shapeInfo) {
    int dim = shapeInfo.reducedPerm.size();
    if (dim <= 1) {
        return false;
    }

    if (shapeInfo.reducedPerm[dim - 1] != dim - 1) {
        shapeInfo.scenario = e_last_axis_transposed;
        shapeInfo.isLastAxisTranspose = true;
        return true;
    } else {
        shapeInfo.scenario = e_last_axis_not_transposed;
        shapeInfo.isLastAxisTranspose = false;
        return false;
    }
}

/*
 * If last axis is huge and not 32B aligned, do not use nchwconv, ans just copy one by one.
 */
static bool IsCopyOneByOne(const CompilerInfo & compilerInfo, const ShapeInfo & shapeInfo) {
    if (Is32BAligned(compilerInfo, shapeInfo.reducedOutShape)) {
        return false;
    }
    if ((shapeInfo.isLastAxisTranspose == false) && (shapeInfo.isLastAxisHuge == true)) {
        return true;
    }
    return false;
}


static void Reshape(ShapeInfo & shapeInfo) {
    int dim = shapeInfo.reducedPerm.size();
    shapeInfo.reducedPerm.push_back(dim);
    shapeInfo.reducedInShape.push_back(1);
    shapeInfo.reducedOutShape.push_back(1);
}


static bool GetShapePerm(const string & opType, const TeOpParas & paras, ShapeInfo & shapeInfo) {
    OP_LOGD(opType.c_str(), "Entering GetShapePerm.");

    if (paras.const_inputs.find("perm") == paras.const_inputs.end()) {
        //ge::OpsOneInputShapeErrReport("Transpose", "perm", "No perm in const_inputs");
        OP_LOGE(opType.c_str(), "No perm in const_inputs.");
        return false;
    }

    const int32_t* pPerm = reinterpret_cast<const int32_t*>(std::get<0>(paras.const_inputs.at("perm")));
    if (pPerm == nullptr) {
        //ge::OpsOneInputShapeErrReport("Transpose", "perm", "Failed to get perm pointer");
        OP_LOGE(opType.c_str(), "Failed to get perm pointer.");
        return false;
    }

    int32_t size = std::get<1>(paras.const_inputs.at("perm"));
    for (int32_t i = 0; i < size/sizeof(int32_t); i++) {
        shapeInfo.perm.push_back(pPerm[i]);
    }

    if (paras.inputs.size() == 0 || paras.inputs[0].tensor.size() == 0 ||
        paras.outputs.size() == 0 || paras.outputs[0].tensor.size() == 0) {
        //ge::OpsOneInputShapeErrReport("Transpose", "inputs/outputs", "TeOpParas size error.");
        OP_LOGE(opType.c_str(), "inputs.size=%u, inputs[0].tensor.size=%u, outputs.size=%u, outputs[0].tensor.size=%u",
                paras.inputs.size(), paras.inputs[0].tensor.size(),
                paras.outputs.size(), paras.outputs[0].tensor.size());
    }
    shapeInfo.inShape = paras.inputs[0].tensor[0].shape;
    shapeInfo.outShape = paras.outputs[0].tensor[0].shape;

    return true;
}


static bool GetInputX(const string & opType, const TeOpParas & paras, const char * & pInputX, int64_t & inputLen) {
    OP_LOGD(opType.c_str(), "Entering GetInputX.");

    if (paras.const_inputs.find("x") == paras.const_inputs.end()) {
        //ge::OpsOneInputShapeErrReport("Transpose", "x", "No x in const_inputs");
        OP_LOGE(opType.c_str(), "No x in const_inputs.");
        return false;
    }

    pInputX = reinterpret_cast<const char*>(std::get<0>(paras.const_inputs.at("x")));
    if (pInputX == nullptr) {
        //ge::OpsOneInputShapeErrReport("Transpose", "x", "Failed to get x pointer");
        OP_LOGE(opType.c_str(), "Failed to get x pointer.");
        return false;
    }

    inputLen = std::get<1>(paras.const_inputs.at("x"));
    OP_LOGD(opType.c_str(), "In GetInputX, inputLen = %ld", inputLen);
    return true;
}


static bool CheckTensorShape(const string & opType,
                             const ShapeInfo & shapeInfo) {
    OP_LOGD(opType.c_str(), "Entering CheckTensorShape.");

    int64_t inDims = shapeInfo.inShape.size();
    int64_t outDims = shapeInfo.outShape.size();
    int64_t permDims = shapeInfo.perm.size();

    for (int i = 0; i < shapeInfo.perm.size();i++) {
        OP_LOGE(opType.c_str(), "inPerm[%d] = %d\n", i, shapeInfo.perm[i]);
    }

    if (inDims <= 1 || inDims != outDims || inDims != permDims) {
        OP_LOGE(opType.c_str(), "The dim of inputs is invalid, inDims = %d, outDims = %d, permDims = %d",
                opType.c_str(), inDims, outDims, permDims);
        return false;
    }

    for (int64_t i=0; i < inDims; i++){
        if (shapeInfo.inShape[shapeInfo.perm[i]] != shapeInfo.outShape[i]) {
            OP_LOGE(opType.c_str(), "The dim of inputs or outputs conflict with perm.", opType.c_str());
            return false;
        }
    }

    return true;
}


/*
 *   4D shape(6,4,10,stride)
 *   return 6 x 4 x 10
 */
static int64_t CalcTotalVolumeLogic(const std::vector<int64_t> & reducedInShape) {
    int64_t vol = 1;
    for (auto i : reducedInShape) {
       vol = vol * i;
    }
    //Last dim is stride, not in volume
    return vol / reducedInShape[reducedInShape.size()-1];
}


/*
 *   4D shape(6,4,10,stride)
 *   return 6 x 4 x 10 x stride
 */
static int64_t CalcTotalVolumeActual(const std::vector<int64_t> & reducedInShape) {
    int64_t vol = 1;
    for (auto i : reducedInShape) {
       vol = vol * i;
    }
    return vol;
}


/*
 *   case1:
 *       reducedPerm     = (1, 0, 2)
 *       reducedPermGard = (0, 1, 2)
 *
 *   case2:
 *       reducedPerm     = (2, 1, 3, 0, 4)
 *       reducedPermGard = (3, 1, 0, 2, 4)
 */
static void CalcReducePermGrad(const vector<int64_t> & reducedPerm , vector<int64_t> & reducedPermGrad) {
    vector<pair<int32_t, int32_t>> sortedList;
    for (int32_t i = 0; i < reducedPerm.size(); i++) {
        sortedList.push_back({i, reducedPerm[i]});
    }

    sort(sortedList.begin(), sortedList.end(), [](pair<int32_t, int32_t> lhs, pair<int32_t, int32_t> rhs) -> bool {
        return lhs.second < rhs.second;
    });

    for (int32_t i = 0; i < sortedList.size(); i++) {
         reducedPermGrad[i] = (sortedList[i].first);
    }
}


static bool IsIdentical(const ShapeInfo & shapeInfo) {
    for(int64_t i = 0; i < shapeInfo.reducedPerm.size(); i++) {
        if(shapeInfo.reducedPerm[i] != i) {
            return false;
        }
    }
    return true;
}


static void  CalcOutShape(ShapeInfo & shapeInfo) {
    vector<int64_t> & inShape = shapeInfo.reducedInShape;
    vector<int64_t> & perm = shapeInfo.reducedPerm;
    vector<int64_t> & outShape = shapeInfo.reducedOutShape;
    outShape.clear();
    int64_t dim = perm.size();
    outShape.resize(dim);
    for(int64_t i = 0; i < dim; i++) {
        outShape[i] = inShape[perm[i]];
    }
}

/*
 * If axis value is 1, then remove it.
 *
 *     inShape              perm                    reducedInShape       reducedPerm
 *     ---------------------------------------------------------------------------------
 *     Shape(4,1,6,1)       perm(0,1,2,3)	        Shape(4,6)           perm(0,1)
 */
void RemoveAxis(ShapeInfo & shapeInfo) {
    int64_t dim = shapeInfo.inShape.size();
    if (dim == 1) {
        return;
    }
    vector<int64_t> & shape = shapeInfo.reducedInShape;
    shape.clear();
    vector<int64_t> delPerm;
    vector<int64_t> newPerm;
    for (int64_t i = 0; i < dim; i++) {
        if (shapeInfo.inShape[i] != 1) {
            shape.push_back(shapeInfo.inShape[i]);
        }else {
            for (int64_t j = 0 ; j < shapeInfo.perm.size(); j++) {
                if (shapeInfo.perm[j] == i) {
                    delPerm.push_back(shapeInfo.perm[j]);
                }
            }
        }
    }
    std::sort(delPerm.begin(), delPerm.end(), greater<int64_t>());

    for (int64_t i = 0; i < dim; i++) {
        bool delFlag = false;
        for(int64_t j = 0; j < delPerm.size(); j++) {
            if (shapeInfo.perm[i] == delPerm[j]) {
                delFlag = true;
            }
        }
        if (delFlag == false) {
            newPerm.push_back(shapeInfo.perm[i]);
        }
    }


    for (int64_t i = 0; i < delPerm.size(); i++) {
        for (int64_t j = 0; j < newPerm.size(); j++) {
            if (newPerm[j] > delPerm[i]) {
                newPerm[j] = newPerm[j] - 1;
            }
        }
    }

    shapeInfo.reducedPerm = newPerm;
    CalcOutShape(shapeInfo);
}



void MergeAxis(ShapeInfo & shapeInfo) {
    int64_t dim = shapeInfo.reducedInShape.size();
    if (dim == 1) {
        return;
    }
    vector<int64_t> newPerm;
    vector<int64_t> newShape;
    vector<int64_t> newDimPosition(dim, -1);
    vector<int64_t> mergedShape(dim, 0);
    vector<int64_t> perm = shapeInfo.reducedPerm;
    vector<int64_t> shape = shapeInfo.reducedInShape;

    int64_t curHead = shapeInfo.reducedPerm[0];
    newDimPosition[curHead] = 0;
    mergedShape[0] = shape[curHead];
    int dimIndex = 0;
    for (int permIndex = 1; permIndex < dim; ++permIndex) {
        //If two indices in permutation are consecutive numbers, combine their dimensions.
        if (curHead + 1 == perm[permIndex]) {
            curHead = perm[permIndex];
            mergedShape[dimIndex] *= shape[curHead];
        } else {
            // Else start a new dimension.
            curHead = perm[permIndex];
            dimIndex++;
            newDimPosition[curHead] = dimIndex;
            mergedShape[dimIndex] = shape[curHead];
        }
    }
    //Compact the new permutations and dimension sizes.
    newPerm.resize(dimIndex + 1);
    newShape.resize(dimIndex + 1);
    dimIndex = 0;
    for (int i = 0; i < newDimPosition.size(); ++i) {
        if (newDimPosition[i] >= 0) {
            int newPermIndex = newDimPosition[i];
            newPerm[dimIndex] = newPermIndex;
            newShape[dimIndex] = mergedShape[newPermIndex];
            dimIndex++;
        }
    }
    shapeInfo.reducedInShape = newShape;
    shapeInfo.reducedPerm.resize(newPerm.size());
    CalcReducePermGrad(newPerm, shapeInfo.reducedPerm);
    CalcOutShape(shapeInfo);
}


/*
 *     inShape              perm                    reducedInShape     reducedOutShape    reducedPerm
 *     --------------------------------------------------------------------------------------------------
 *     Shape(4,5,6,7)       perm(1,0,2,3)	        Shape(4,5,42)      Shape(5,4,42)      perm(1,0,2)
 *     Shape(2,3,4,5)       perm(0,2,3,1)	        Shape(2,3,20)      Shape(2,20,3)      perm(0,2,1)
 *     Shape(2,3,4,5,6)     perm(0,4,1,2,3)	        Shape(2,60,6)      Shape(2,6,60)      perm(0,2,1)
 *     Shape(2,3,4,5,6)     perm(2,3,4,0,1)	        Shape(6,120)       Shape(120,6)       perm(1,0)
 *
 *     If last axis join transpose, the implementation now is add a axis with value 1.
 */
void ReduceAxis(const string & opType,
                const CompilerInfo & compilerInfo,
                ShapeInfo & shapeInfo) {
    OP_LOGD(opType.c_str(), "Entering ReduceAxis.");

    RemoveAxis(shapeInfo);

    MergeAxis(shapeInfo);

    /*
     * If in shape and out shape is idential, than just do it by copy
     */
    if(IsIdentical(shapeInfo)) {
        shapeInfo.identical = 1;
    }

    /*
     * If last axis joined the transpose , such as shape = (20,30,40), perm = (1,2,0),
     * we can reshape shape to (20,30,40,1), and perm = (1,2,0,3).
     */
    if (IsLastAxisJoinTranspose(shapeInfo)) {
        //Reshape(shapeInfo);
    }

    shapeInfo.totalVolumeLogic = CalcTotalVolumeLogic(shapeInfo.reducedInShape);
    shapeInfo.totalVolumeActual = CalcTotalVolumeActual(shapeInfo.reducedInShape);
    shapeInfo.volumePerCore = (int64_t)ceil(shapeInfo.totalVolumeLogic * 1.0 / compilerInfo.coreNum);
    shapeInfo.dim = shapeInfo.reducedInShape.size();
    CalcReducePermGrad(shapeInfo.reducedPerm, shapeInfo.reducedPermGrad);

    if (shapeInfo.reducedInShape[shapeInfo.dim - 1] > LAST_AXIS_HUGE_THRESHOLD) {
        shapeInfo.isLastAxisHuge = true;
    }

    shapeInfo.reducedOutShapeAxis8 = shapeInfo.reducedOutShape;
    for(int i = 0; i < TRANSPOSE_MAX_AXIS_NUM - shapeInfo.dim; i++) {
        shapeInfo.reducedOutShapeAxis8.insert(shapeInfo.reducedOutShapeAxis8.begin(),1);
    }

    return;
}


static int64_t GetIdByLevel(int64_t level) {
    return TRANSPOSE_MAX_AXIS_NUM - level - 1;
}


static int64_t GetLevelById(int64_t id) {
    return TRANSPOSE_MAX_AXIS_NUM - id - 1;
}


static int64_t GetOrigIdById(int64_t dim, int64_t id) {
    return id - dim + 2;
}


static int64_t GetOrigIdByLevel(int64_t dim, int64_t level) {
    return GetIdByLevel(level) - dim;
}


/*
 *   reducedOutShape(6,4,10,stride),suppose stride is 16/32/..., not huge ,so 10 * stride can be move to ub
 *   dstLevelAccuVolume = [1,1,1,1,240,40,10,1]
 *   if coreNum = 2
 *      extendRanges = [(0,119),(120,239)]
 *      levelLoopNum = [1, 1, 1, 1, 1, 3, 1, 1]
 *                                     |
 *                                     3 = 120 / (4 * 10)
 *   if coreNum = 4
 *      extendRanges = [(0,79),(40,119),(120,199),(160,239)]
 *      levelLoopNum = [1, 1, 1, 1, 1, 2, 1, 1]
 *                                     |
 *                                     2 = 80 / (4 * 10)
 */
static void CalcLevelLoopNum(int64_t level,
                             const ShapeInfo & shapeInfo,
                             RuntimeInfo & runtimeInfo,
                             LevelInfo & levelInfo){
    int64_t volPerCore = runtimeInfo.extendRanges[0].second - runtimeInfo.extendRanges[0].first + 1;
    int64_t dim = shapeInfo.dim;
    int64_t levelOneAxisNumPerCore = shapeInfo.reducedOutShape[dim - 2];
    if (level == 0) {
        /*
         * If level is zero, it means split core on level one,
         * so we need calc loop by splited axis value, not the whole one
         */
        levelOneAxisNumPerCore = volPerCore;
    }

    if (levelInfo.nBurst == shapeInfo.reducedOutShape[dim - 2]) {
        levelInfo.levelLoopNum[TRANSPOSE_MAX_AXIS_NUM - 2] = 1;
    }else {
        levelInfo.levelLoopNum[TRANSPOSE_MAX_AXIS_NUM - 2] = levelOneAxisNumPerCore / levelInfo.nBurst;
    }

    int64_t index = GetIdByLevel(level);

    //left side index, all left ones should be one (which is default) exclude [index -1]  need be updated
    if (level != 0) {
        levelInfo.levelLoopNum[index - 1] = volPerCore / levelInfo.dstLevelAccuVolume[index];
    }

    //right side index,
    for (int64_t i = index; i < TRANSPOSE_MAX_AXIS_NUM - 2; i++) {
        levelInfo.levelLoopNum[i] = shapeInfo.reducedOutShapeAxis8[i];
    }
}


static void CalcUbReorderFactor(const CompilerInfo & compilerInfo,
                                const ShapeInfo & shapeInfo,
                                RuntimeInfo & runtimeInfo) {
    if (Is32BAligned(compilerInfo, shapeInfo.reducedOutShape)) {
        runtimeInfo.ubReorderFactor  = 1;
        return;
    }

    if (shapeInfo.isLastAxisHuge == true) {
        runtimeInfo.ubReorderFactor  = 1;
        return;
    }

    runtimeInfo.ubReorderFactor = UB_REORDER_FACTOR;
    return;
}



/*
 * 1. If alignElement = 0, there is no need to reorder.
 * 2. If alignElement != 0 , reorder is necessory, suppose burstLen = 3(blocks)
 *    Because we will use vnchwconv twice, so we need 3 * 16 + 3 * 16 (blocks)
 *    So for each block ,we nedd 32 additional blocks, that is 33 blocks.
 */
static bool CalcUbThreshold(const CompilerInfo & compilerInfo,
                            const ShapeInfo & shapeInfo,
                            LevelInfo & levelInfo,
                            RuntimeInfo & runtimeInfo) {

    int64_t blockNum = levelInfo.nBurst * levelInfo.burstLen;

    if (IsCopyOneByOne(compilerInfo, shapeInfo)) {
        runtimeInfo.ubThreshold = levelInfo.burstLen;
        return true;
    }

    /*
     * since allLevelOneInUb is false, it means ub capacity is not enough for level one.
     * So just set threashold to be nburst * burstLen
     */
    if (levelInfo.allLevelOneInUb == false) {
        runtimeInfo.ubThreshold = blockNum;
        return true;
    }

    int64_t maxUbBlockNum = compilerInfo.ubSizeCouldUse / runtimeInfo.ubReorderFactor;
    if (maxUbBlockNum < blockNum) {
        OP_LOGE(compilerInfo.opType.c_str(), "ubSizeCouldUse=%ld, ubReorderFactor=%ld\n",
                compilerInfo.ubSizeCouldUse,
                runtimeInfo.ubReorderFactor);
        return false;
    }

    for (int64_t i = TRANSPOSE_MAX_AXIS_NUM - 2; i >= 0; i--) {
        if (levelInfo.levelLoopNum[i] * blockNum <= maxUbBlockNum) {
           blockNum = levelInfo.levelLoopNum[i] * blockNum;
           levelInfo.hasTail[i] = 0;
           runtimeInfo.ubThreshold = blockNum;
        } else {
            int64_t k = maxUbBlockNum / blockNum;
            runtimeInfo.ubThreshold = k * blockNum;
            if (levelInfo.levelLoopNum[i] % k == 0) {
                levelInfo.hasTail[i] = 0;
            } else {
                levelInfo.hasTail[i] = 1;
            }
            break;
        }
    }
    return true;
}


static void CalcUbOffset(const CompilerInfo & compilerInfo, const ShapeInfo & shapeInfo, RuntimeInfo & runtimeInfo) {
    runtimeInfo.fp16Offset3 = (compilerInfo.ubSizeCouldUse + 1) * ELE_NUM_PER_BLOCK_FP16;

    if (Is32BAligned(compilerInfo, shapeInfo.reducedOutShape) == true) {
         runtimeInfo.fp16Offset1 = 0;
         runtimeInfo.fp16Offset2 = 0;
         return;
    }

   if (IsCopyOneByOne(compilerInfo, shapeInfo)) {
         runtimeInfo.fp16Offset1 = 0;
         runtimeInfo.fp16Offset2 = 0;
         return;
   }

    runtimeInfo.fp16Offset1 = runtimeInfo.ubThreshold * ELE_NUM_PER_BLOCK_FP16;
    /*
     * Each block will become 16 line, and each line with 16 fp16 elements
     * For example, ubThreshold = 2  , we need 512 fp16 elements capacity for vnchwconv
     */
    runtimeInfo.fp16Offset2 = runtimeInfo.fp16Offset1 +\
                            ELE_NUM_PER_BLOCK_FP16 * runtimeInfo.ubThreshold * ELE_NUM_PER_BLOCK_FP16;
    /*
    runtimeInfo.fp16Offset3 = runtimeInfo.fp16Offset2 +\
                            ELE_NUM_PER_BLOCK_FP16 * runtimeInfo.ubThreshold * ELE_NUM_PER_BLOCK_FP16;
    */
}


/*
 *  For float32:
 *   alignElement   gcd   cycleNum
 *   0              NA    NA
 *   1              1     8
 *   2              2     4
 *   3              1     8
 *   4              4     2
 *   5              1     8
 *   6              2     4
 *   7              1     8
 */
static void CalcWorkspaceParams(const CompilerInfo & compilerInfo,
                          const ShapeInfo & shapeInfo,
                          const LevelInfo & levelInfo,
                          RuntimeInfo & runtimeInfo) {
    int64_t totalVolLogic = CalcTotalVolumeLogic(shapeInfo.reducedInShape);
    int64_t logicVolPerCore = ceil(totalVolLogic * 1.0 / compilerInfo.coreNum);
    int64_t elementsPerBlock = ElementNumPerBlock(compilerInfo.dType);
    int64_t gcd = GCD(levelInfo.alignElement, elementsPerBlock);
    int64_t cycleNum = elementsPerBlock / gcd;
    if (logicVolPerCore % cycleNum != 0) {
        logicVolPerCore += cycleNum - logicVolPerCore % cycleNum;
    }
    int64_t addrOffset = 0;
    for(int64_t i = 0; i < compilerInfo.coreNum; i++) {
        addrOffset = totalVolLogic - (i + 1) * logicVolPerCore;
        if (addrOffset < 0) {
            addrOffset = 0;
        }
        //reverse insert
        runtimeInfo.srcBaseAddrWorkspace.insert(runtimeInfo.srcBaseAddrWorkspace.begin(), addrOffset);
    }

    int64_t logicVolPerCycle = logicVolPerCore / cycleNum;
    runtimeInfo.nBurstWorkspace = compilerInfo.ubSizeCouldUse / levelInfo.burstLen;
    runtimeInfo.nBurstWorkspace = min(NBURST_BOUNDARY, (int)runtimeInfo.nBurstWorkspace);
    runtimeInfo.loopNumWorkspace = logicVolPerCycle / runtimeInfo.nBurstWorkspace;
    runtimeInfo.nBurstTailWorkspace = logicVolPerCycle - runtimeInfo.loopNumWorkspace * runtimeInfo.nBurstWorkspace;
    runtimeInfo.cycleNumWorkspace = cycleNum;
    runtimeInfo.lastAxisElementNum = shapeInfo.reducedInShape[shapeInfo.dim - 1];
    runtimeInfo.lastAxisElementNumAligned = runtimeInfo.lastAxisElementNum + levelInfo.alignElement;
    runtimeInfo.srcStrideWorkspace = (runtimeInfo.lastAxisElementNum * cycleNum -\
                                      runtimeInfo.lastAxisElementNumAligned) / elementsPerBlock;
    runtimeInfo.dstStrideWorkspace = levelInfo.burstLen * (cycleNum - 1);
    runtimeInfo.workspaceSizeInBytes = shapeInfo.totalVolumeLogic * levelInfo.burstLen * BYTES_PER_BLOCK;
}


/*
 * Level one default gap is the stride - 1
 * Since level one may be cut because of UB capacity
 * For example:
 *     dtype = float
 *     ub_size = 64
 *     reduceInShape  = (5,4,6,7,8)
 *     reduceOutShape = (6,4,7,5,8)
 *     At first, srcGapPerRound(Logic) = (0,0,0,7,42,1,168,1)
 *     LoopNum =(1,1,1,1,2,7,2,1)
 *                           |
 *                           ---This 2 means level one is cut because of ub capacity
 *     factor = (reduceOutShape[4] - nBurstTail) / LoopNum[6] = (5 - 1)/ 2 = 2
 */
static void UpdateLevelOneGap(const ShapeInfo & shapeInfo,  LevelInfo & levelInfo) {

    //If level loop num is not one, it means this axis is cut
    if (levelInfo.levelLoopNum[TRANSPOSE_MAX_AXIS_NUM - 2] != 1) {
        int64_t factor = (shapeInfo.reducedOutShape[shapeInfo.dim - 2] -\
                levelInfo.nBurstTail) / levelInfo.levelLoopNum[TRANSPOSE_MAX_AXIS_NUM - 2];
        levelInfo.srcGapPerRound[TRANSPOSE_MAX_AXIS_NUM - 2] *= factor;
    }
    return;
}


/*
 *   reducedOutShape(6,4,10,stride)
 *   dstLevelAccuVolume=[1,1,1,1,240,40,10,1]
 */
static void CalcLevelAccuVolume(const ShapeInfo & shapeInfo, LevelInfo & levelInfo) {
    int64_t id = shapeInfo.dim - 2;
    for (int64_t i = TRANSPOSE_MAX_AXIS_NUM - 2; i >= 0; i--) {
        if (id >= 0) {
            levelInfo.dstLevelAccuVolume[i] = levelInfo.dstLevelAccuVolume[i + 1] * shapeInfo.reducedOutShape[id];
            levelInfo.srcLevelAccuVolume[i] = levelInfo.srcLevelAccuVolume[i + 1] * shapeInfo.reducedInShape[id];
            id--;
        }
    }
}


static int64_t AlignTo32BByWorkspace(const CompilerInfo & compilerInfo,
                                 const ShapeInfo & shapeInfo,
                                 const LevelInfo & levelInfo) {
    if (Is32BAligned(compilerInfo, shapeInfo.reducedOutShape) == true) {
        return 0;
    }

    if (IsStrideTooHuge(levelInfo) == true) {
        //Since stride is too huge, aligned to 32B is useless.
        return 0;
    }

    //If last axis is huge , no need to align to 32B because may result in poor perf
    if (shapeInfo.isLastAxisHuge == false) {
        return 1;
    }

    return 0;
}


/*
 *   coreNum = 32
 *   level = 1
 *   reducedOutShape(6,5,7,4,stride)
 *   volumePerCore = 27
 *   dstLevelAccuVolume = (0,0,0,840,140,28,4,1)
 *   initRange = [(0,26), (27,53), (54,80), ...(837,839)]
 *   Level-2                        Level-2
 *   ----------------------------   -----------------------------
 *   0  1  2  3 ...... 25  26  27 | 28  29  30  ...... 53  54  55
 *   0  1  2  3 ...... 25  26
 *   ------------------------
 *       initRnage[0]
 *                             27   28  29  30 ...... 53
 *                             -------------------------
 *                                  initRnage[1]
 *   we found initRange[1] cross Level-2
 */
static bool IsCrossUpperLevel(int64_t level,
                              const vector<pair<int64_t,int64_t>> & initRanges,
                              const vector<int64_t> & dstLevelAccuVolume) {
    int64_t id = GetIdByLevel(level + 1);
    for (auto range : initRanges) {
        int64_t begin = range.first/dstLevelAccuVolume[id];
        int64_t end  = range.second/dstLevelAccuVolume[id];
        if( begin != end) {
            return true;
        }
    }
    return false;
}


/*
 *   shape(6,4,10,stride),
 *   dim = 4
 *   volumePerCore = 60
 *               level: 7  6  5  4  3    2   1   0
 *               index: 0  1  2  3  4    5   6   7
 *   dstLevelAccuVolume =[ 1, 1, 1, 1, 240, 40, 10, 1]
 *   40 < 60 < 240
 *   Level =  2
 */
static int64_t LocateLevel(const ShapeInfo & shapeInfo,
                           const LevelInfo & levelInfo,
                           const RuntimeInfo & runtimeInfo) {
    int64_t level = 0;
    for (int64_t i = 0; i < shapeInfo.dim - 1; i++) {
        if (shapeInfo.volumePerCore < levelInfo.dstLevelAccuVolume[TRANSPOSE_MAX_AXIS_NUM - shapeInfo.dim + i]) {
            continue;
        }else {
            level = shapeInfo.dim - i - 1;
            break;
        }
    }
    if (IsCrossUpperLevel(level, runtimeInfo.initRanges, levelInfo.dstLevelAccuVolume)) {
        level = level + 1;
    }
    return level;
}


/*
 *   reducedOutShape(6,4,10,stride)
 *   coreNum = 4
 *   volumePerCore = 60
 *   initRanges = [(0,59), (60,119), (120,179), (180,239)]
 */
static void CalcInitRange(const CompilerInfo & compilerInfo,
                          const ShapeInfo & shapeInfo,
                          RuntimeInfo & runtimeInfo) {
    int64_t coreNum = compilerInfo.coreNum;
    int64_t beginPos = 0;
    int64_t endPos = 0;
    for (int64_t i = 0; i < coreNum; i++) {
        beginPos = shapeInfo.volumePerCore * i;
        if (i == coreNum - 1) {
            endPos = shapeInfo.totalVolumeLogic - 1;
        }else {
            endPos = shapeInfo.volumePerCore * (i + 1) - 1;
        }

        if (beginPos >= shapeInfo.totalVolumeLogic) {
            beginPos = shapeInfo.totalVolumeLogic - shapeInfo.volumePerCore;
        }
        if (endPos >= shapeInfo.totalVolumeLogic) {
            endPos= shapeInfo.totalVolumeLogic - 1;
        }
        runtimeInfo.initRanges.push_back({beginPos, endPos});
    }
}


static int64_t GetStartIndexByPos(int64_t pos, int64_t currentLevelVolume) {
    return (int64_t)floor(pos * 1.0 / currentLevelVolume);
}


static int64_t GetEndIndexByPos(int64_t pos, int64_t currentLevelVolume) {
    return (int64_t)floor(pos * 1.0 / currentLevelVolume);
}


static int64_t GetPosByIndex(int64_t index, int64_t currentLevelVolume) {
    return index * currentLevelVolume;
}


/*
 *   case1:
 *       in:
 *           coreNum = 4
 *           level = 2
 *           reducedOutShape(6,4,10,stride)
 *           dstLevelAccuVolume = (0,0,0,0,240,40,10,1)
 *           initRange = [(0,59), (60,119), (120,179), (180,239)]
 *       out:
 *           crossLevelNum = 2
 *
 *   case2:
 *       in:
 *           coreNum = 32
 *           level = 1
 *           reducedOutShape(6,5,7,4,stride)
 *           dstLevelAccuVolume = (0,0,0,840,140,28,4,1)
 *           initRange = [(0,26), (27,53), (54,80), ...(837,839)]
 *       out:
 *           crossLevelNum = 2
 *           level = 2
 *
 *       because (27,53) cross level 2, so we need to update level from 1 to 2, and set crossLevelNum = 2
 */
static void CountRangeCrossLevelNum(int64_t & level,
                                    const RuntimeInfo & runtimeInfo,
                                    const LevelInfo & levelInfo,
                                    int64_t & crossLevelNum) {
    int64_t currentLevelVolume = levelInfo.dstLevelAccuVolume[TRANSPOSE_MAX_AXIS_NUM - level - 1];
    printf("level=%ld, currentLevelVolume = %ld\n", level, currentLevelVolume);
    for (auto eachRange : runtimeInfo.initRanges) {
        int64_t startIndex = GetStartIndexByPos(eachRange.first, currentLevelVolume);
        int64_t endIndex = GetEndIndexByPos(eachRange.second, currentLevelVolume);
        printf("(%ld,%ld), startIndex=%ld, endIndex=%ld, %ld\n",
                eachRange.first, eachRange.second, startIndex, endIndex, endIndex - startIndex + 1);
        if (endIndex - startIndex + 1 > crossLevelNum) {
            crossLevelNum = endIndex - startIndex + 1;
        }
    }
}


static bool IsLastCore(int64_t pos, int64_t totalVolumeLogic) {
    return pos + 1 == totalVolumeLogic;
}


/*
 *   case1:
 *       reducedOutShape(6,4,10,stride)
 *       coreNum = 4
 *       volumePerCore = 60
 *       initRange = [(0,59), (60,119), (120,179), (180,239)]
 *       extendRanges = [(0,79), (40,119), (120,119), (160,239)]
 *   case2:
 */
static void CalcExtendRange(int64_t level,
                            int64_t crossLevelNum,
                            const ShapeInfo & shapeInfo,
                            LevelInfo & levelInfo,
                            RuntimeInfo & runtimeInfo) {
    int64_t currentLevelVolume = levelInfo.dstLevelAccuVolume[TRANSPOSE_MAX_AXIS_NUM - level - 1];
    for (auto eachRange : runtimeInfo.initRanges) {
       int64_t startIndex = GetStartIndexByPos(eachRange.first, currentLevelVolume);
       int64_t endIndex = GetEndIndexByPos(eachRange.second, currentLevelVolume);
       int64_t startPos = GetPosByIndex(startIndex, currentLevelVolume);
       int64_t endPos = GetPosByIndex(endIndex+1, currentLevelVolume)-1;
       while (endIndex - startIndex + 1 < crossLevelNum) {
           if (IsLastCore(endPos, shapeInfo.totalVolumeLogic)) {
               startIndex -= 1;
           }else {
               endIndex += 1;
           }
       }
       startPos = GetPosByIndex(startIndex, currentLevelVolume);
       endPos = GetPosByIndex(endIndex + 1, currentLevelVolume) - 1;
       runtimeInfo.extendRanges.push_back({startPos, endPos});
    }
}


static bool CalcNBurstByUbSizeForIdentical(const CompilerInfo & compilerInfo,
                                           const ShapeInfo & shapeInfo,
                                           LevelInfo & levelInfo) {
    int64_t singleEleLen = SizeofDType(compilerInfo.dType);
    int64_t loopNum = (shapeInfo.totalVolumeActual *  singleEleLen) / (compilerInfo.ubSizeCouldUse * BYTES_PER_BLOCK);
    int64_t totalVolumeInBytes = shapeInfo.totalVolumeActual * singleEleLen;
    int64_t ubSizeInBytes = compilerInfo.ubSizeCouldUse * BYTES_PER_BLOCK;
    levelInfo.burstLen = compilerInfo.ubSizeCouldUse;
    levelInfo.burstLenTail = ceil(1.0 * (totalVolumeInBytes - ubSizeInBytes * loopNum)/ BYTES_PER_BLOCK);
    levelInfo.identicalLoopNum = loopNum;
    return true;
}

/*
 *  Consider 32 aligned scenario:
 *       ubSizeCouldUse = 20 (blocks),  levelOneSize = 56, burstLen = 3 (blocks)
 *       because 6 * 3 <= 20, 7 * 3 > 20
 *       so, nBurst = 6 , nBurstTail = 56 - (56 / 3) * 3 = 56 - 54 = 2
 */
static bool CalcNBurstByUbSize(const CompilerInfo & compilerInfo,
                               const ShapeInfo & shapeInfo,
                               const RuntimeInfo & runtimeInfo,
                               const vector<int64_t> & reducedOutShape,
                               int64_t levelOneSize,
                               LevelInfo & levelInfo) {

    if (shapeInfo.identical) {
        return CalcNBurstByUbSizeForIdentical(compilerInfo, shapeInfo, levelInfo);
    }

    //If stride is more than 65535, nBurst should be 1 right now
    if (IsStrideTooHuge(levelInfo) == true) {
        levelInfo.nBurst = 1;
        return true;
    }

    //If last axis is not 32B aligned , if do not aligned to 32B, we need to copy one by one
    if (Is32BAligned(compilerInfo, reducedOutShape) == false) {
        //If not 32B aligned by workspace, nBurst should be one
        if (levelInfo.byWorkspace == 0) {
            levelInfo.nBurst = 1;
            return true;
        }
    }

    if (levelOneSize * levelInfo.burstLen * runtimeInfo.ubReorderFactor <= compilerInfo.ubSizeCouldUse) {
        //All level one can be moved to ub
        levelInfo.nBurst = levelOneSize;
        levelInfo.nBurstTail = 0;
        return true;
    }

    levelInfo.allLevelOneInUb = false;

    levelInfo.nBurst = compilerInfo.ubSizeCouldUse / runtimeInfo.ubReorderFactor / levelInfo.burstLen;
    if (levelInfo.nBurst == 0) return false;

    levelInfo.nBurstTail = levelOneSize - (levelOneSize / levelInfo.nBurst) * levelInfo.nBurst;
    if (levelInfo.nBurstTail != 0) {
        levelInfo.hasTail[TRANSPOSE_MAX_AXIS_NUM - 2] = 1; //level one has tail
    }
    return true;
}


/*
 *  reducedInShape:  4  5  6  7  8
 *  reducedOutShape: 6  5  7  4  8
 *
 *  level                      7   6   5   4    3     2     1    0
 *  index                      0   1   2   3    4     5     6    7
 *  origGap:                   0   0   0   210  42    7     1    1
 *  levelInfo.srcGapPerRound:  0   0   0   7    42    1     210  1
 *  levelInfo.srcStride        210-1 = 209
 */
static bool CalcLevelInfo(const CompilerInfo & compilerInfo,
                          const ShapeInfo & shapeInfo,
                          const RuntimeInfo & runtimeInfo,
                          LevelInfo & levelInfo) {

    int64_t dim = shapeInfo.dim;

    //1. srcGapPerRound
    vector<int64_t> origGap(TRANSPOSE_MAX_AXIS_NUM, 0);
    origGap[TRANSPOSE_MAX_AXIS_NUM - 1] = 1;
    int64_t index = dim - 2;
    bool first = false;
    for (int64_t i = TRANSPOSE_MAX_AXIS_NUM - 2; i >= 0; i--) {
        if (index >= 0) {
            if (first == false) {
                origGap[i] = 1;
                first = true;
            } else {
                origGap[i] = origGap[i + 1] * shapeInfo.reducedInShape[index--];
            }
        }
    }
    int64_t diff = TRANSPOSE_MAX_AXIS_NUM - dim;
    for (int i = 0; i < dim ; i++) {
        levelInfo.srcGapPerRound[i + diff] = origGap[shapeInfo.reducedPerm[i]+ diff];
    }

    //2. srcStride
    levelInfo.srcStride = levelInfo.srcGapPerRound[TRANSPOSE_MAX_AXIS_NUM - 2] - 1;

    //3. burstLen
    levelInfo.burstLen = ceil(shapeInfo.reducedOutShape[dim - 1] * 1.0 / ElementNumPerBlock(compilerInfo.dType));

    //4. byWorkspace
    levelInfo.byWorkspace = AlignTo32BByWorkspace(compilerInfo, shapeInfo, levelInfo);

    //5. nBurst
    bool ret = CalcNBurstByUbSize(compilerInfo,
                                  shapeInfo,
                                  runtimeInfo,
                                  shapeInfo.reducedOutShape,
                                  shapeInfo.reducedOutShape[dim - 2],
                                  levelInfo);

    if (ret == false) return false;

    //6. alignElement
    int64_t eleNumPerBlock = ElementNumPerBlock(compilerInfo.dType);
    if ((shapeInfo.reducedOutShape[dim - 1] % eleNumPerBlock) != 0) {
        //Not 32B aligned
        levelInfo.alignElement = eleNumPerBlock - (shapeInfo.reducedOutShape[dim - 1] % eleNumPerBlock);
    }

    //7. elementNumPerBurst
    levelInfo.elementNumPerBurst = levelInfo.burstLen * eleNumPerBlock - levelInfo.alignElement;

    return true;
}


/*
 *   case1:
 *         dim = 3
 *         reducedPerm = (1,0,2)
 *         dstData = (0,0,0,0,0,3,2,1)
 *         srcData = (0,0,0,0,0,2,3,1)
 *
 *   case2:
 *         dim = 5
 *         reducedPerm = (2,1,3,0,4)
 *         dstData = (0,0,0,6,5,4,7,8)
 *         srcData = (0,0,0,4,5,6,7,8)
 */
static void SwitchLevelFromDst2Src(int64_t dim,
                                   const vector<int64_t> & dstData,
                                   vector<int64_t> & srcData,
                                   const vector<int64_t> & reducedPerm) {
    vector<int64_t> reducedPermGrad(TRANSPOSE_MAX_AXIS_NUM, 0);
    CalcReducePermGrad(reducedPerm, reducedPermGrad);
    int64_t diff = TRANSPOSE_MAX_AXIS_NUM - dim;
    for (int64_t i = 0; i < dim - 1; i++) {
        srcData[i + diff] = dstData[diff + reducedPermGrad[i]];
    }
}


/*
 *   reducedInShape = (10,64,8)
 *   perm = (1,0,2)
 *   core:    0   1   2   3   4   5         29   30   31
 *   ----------------------------------------------------
 *   dstAddr: 0,  20, 40, 60, 80, 100  ...  580, 600, 620
 *   srcAddr: 0,  2,  4,  6,  8,  10   ...  58,  60,  62
 */
static void CalcBaseAddr(const CompilerInfo & compilerInfo,
                         const ShapeInfo & shapeInfo,
                         const LevelInfo & levelInfo,
                         RuntimeInfo & runtimeInfo) {
    //step 1. calc dst base addr
    for (int i = 0 ; i < compilerInfo.coreNum; i++) {
        runtimeInfo.dstBaseAddr.push_back(runtimeInfo.extendRanges[i].first);
    }

    int64_t dim = shapeInfo.reducedInShape.size();
    int64_t diff = TRANSPOSE_MAX_AXIS_NUM - dim;
    for (auto eachDstBaseAddr : runtimeInfo.dstBaseAddr) {
        vector<int64_t> dstAxisIndex(TRANSPOSE_MAX_AXIS_NUM, 0);
        vector<int64_t> srcAxisIndex(TRANSPOSE_MAX_AXIS_NUM, 0);
        int64_t left = eachDstBaseAddr;

        //step 2. calc dst axis index by dst addr
        for (int64_t i = 0; i < dim - 1; i++) {
            dstAxisIndex[i + diff] = (int64_t)floor(left * 1.0  / levelInfo.dstLevelAccuVolume[i + diff + 1]);
            left -= dstAxisIndex[i + diff] * levelInfo.dstLevelAccuVolume[i + diff + 1];
        }

        //step 3. calc src axis index by dst axis index and perm
        SwitchLevelFromDst2Src(dim, dstAxisIndex, srcAxisIndex, shapeInfo.reducedPerm);

        //step 4. calc src addr by src axis index
        int64_t addr = 0;
        for (int64_t i = 0; i < dim - 1; i++) {
            int64_t index = i + diff;
            addr += srcAxisIndex[index] * levelInfo.srcLevelAccuVolume[index+1];
        }
        runtimeInfo.srcBaseAddr.push_back(addr);
    }
}


static bool CalcRuntimeInfo(const CompilerInfo & compilerInfo,
                            const ShapeInfo & shapeInfo,
                            LevelInfo & levelInfo, //TODO should  const
                            RuntimeInfo & runtimeInfo) {


    if (CalcUbThreshold(compilerInfo, shapeInfo, levelInfo, runtimeInfo) == false) {
        return false;
    }

    CalcUbOffset(compilerInfo, shapeInfo, runtimeInfo);

    if (levelInfo.byWorkspace != 0) {
        CalcWorkspaceParams(compilerInfo, shapeInfo, levelInfo, runtimeInfo);
    }

    return true;
}


static void PrintTilingInfo(int64_t level,
                            int64_t crossLevelNum,
                            const CompilerInfo & compilerInfo,
                            const ShapeInfo & shapeInfo,
                            const RuntimeInfo & runtimeInfo,
                            const LevelInfo & levelInfo) {
    string logStr = "==================================================================\n";
    logStr += "ShapeInfo:\n";
    logStr += "    in: " + vec_to_string(shapeInfo.inShape) + "\n";
    logStr += "    out: " + vec_to_string(shapeInfo.outShape) + "\n";
    logStr += "    perm: " + vec_to_string(shapeInfo.perm) + "\n";
    logStr += "    reducedIn: " + vec_to_string(shapeInfo.reducedInShape) + "\n";
    logStr += "    reducedOut: " + vec_to_string(shapeInfo.reducedOutShape) + "\n";
    logStr += "    reducedPerm: " + vec_to_string(shapeInfo.reducedPerm) + "\n";
    logStr += "    reducedPermGrad: " + vec_to_string(shapeInfo.reducedPermGrad) + "\n";
    logStr += "    dim: " + to_string(shapeInfo.dim) + "\n";
    logStr += "    identical: " + to_string(shapeInfo.identical) + "\n";
    logStr += "    totalVolumeLogic: " + to_string(shapeInfo.totalVolumeLogic) + "\n";
    logStr += "    totalVolumeActual: " + to_string(shapeInfo.totalVolumeActual) + "\n";
    logStr += "    volumePerCore(init): " + to_string(shapeInfo.volumePerCore) + "\n";
    logStr += "    volumePerCore(extend): " + to_string(runtimeInfo.extendRanges[0].second -\
                                                      runtimeInfo.extendRanges[0].first + 1) + "\n\n";

    logStr += "CompilerInfo:\n";
    logStr += "    coreNum=" + to_string(compilerInfo.coreNum) + ", ";
    logStr += "ubSize=" + to_string(compilerInfo.ubSize) + ", ";
    logStr += "ubSizeCouldUse=" + to_string(compilerInfo.ubSizeCouldUse) + ", ";
    logStr += "dType=" + compilerInfo.dType +"\n\n";

    logStr += "LevelInfo:\n";
    logStr += "    nBurst=" + to_string(levelInfo.nBurst) + ", ";
    logStr += "nBurstTail=" + to_string(levelInfo.nBurstTail) + ", ";
    logStr += "burstLen=" + to_string(levelInfo.burstLen) + ", ";
    logStr += "burstLenTail=" + to_string(levelInfo.burstLenTail) + ", ";
    logStr += "alignElement=" + to_string(levelInfo.alignElement) + ", ";
    logStr += "byWorkspace=" + to_string(levelInfo.byWorkspace) + ", ";
    logStr += "elementNumPerBurst=" + to_string(levelInfo.elementNumPerBurst) + ", ";
    logStr += "identicalLoopNum=" + to_string(levelInfo.identicalLoopNum) + ", ";
    logStr += "srcStride(Logic)=" + to_string(levelInfo.srcStride) + ", ";
    logStr += "srcStride(Actual)=" + to_string(levelInfo.srcStride * levelInfo.burstLen) + ", " + "\n";
    logStr += "    accuVolume:\n";
    logStr += "        dest: " + vec_to_string(levelInfo.dstLevelAccuVolume) + "\n";
    logStr += "        src: " + vec_to_string(levelInfo.srcLevelAccuVolume) + "\n";
    logStr += "    LoopNum: " + vec_to_string(levelInfo.levelLoopNum, TRANSPOSE_MAX_AXIS_NUM) + "\n";
    logStr += "    hasTail: " + vec_to_string(levelInfo.hasTail, TRANSPOSE_MAX_AXIS_NUM) + "\n";
    logStr += "    srcGapPerRound(Logic): " + vec_to_string(levelInfo.srcGapPerRound, TRANSPOSE_MAX_AXIS_NUM) + "\n";
    logStr += "    srcGapPerRound(Actual Elements): ";
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
        logStr += to_string(GmMemOffsetFromBlock(levelInfo.srcGapPerRound[i],
                                                 levelInfo.burstLen,
                                                 levelInfo.alignElement,
                                                 compilerInfo.dType)) + ",";
    }
    logStr += "\n    srcGapPerRound(Actual 32BAlgined Elements): ";
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
        logStr += to_string(GmMemOffsetFromBlock(levelInfo.srcGapPerRound[i],
                                                 levelInfo.burstLen,
                                                 0,
                                                 compilerInfo.dType)) + ",";
    }
    logStr += "\n\nRuntimeInfo:\n";
    logStr += "    ubThreshold=" + to_string(runtimeInfo.ubThreshold) + "\n";
    logStr += "    ubReorderFactor=" + to_string(runtimeInfo.ubReorderFactor) + "\n";
    logStr += "    fp16Offset1=" + to_string(runtimeInfo.fp16Offset1) + "\n";
    logStr += "    fp16Offset2=" + to_string(runtimeInfo.fp16Offset2) + "\n";
    logStr += "    fp16Offset3=" + to_string(runtimeInfo.fp16Offset3) + "\n";
    logStr += "    cycleNumWorkspace=" + to_string(runtimeInfo.cycleNumWorkspace) + "\n";
    logStr += "    loopNumWorkspace=" + to_string(runtimeInfo.loopNumWorkspace) + "\n";
    logStr += "    nBurstWorkspace=" + to_string(runtimeInfo.nBurstWorkspace) + "\n";
    logStr += "    nBurstTailWorkspace=" + to_string(runtimeInfo.nBurstTailWorkspace) + "\n";
    logStr += "    lastAxisElementNum=" + to_string(runtimeInfo.lastAxisElementNum) + "\n";
    logStr += "    lastAxisElementNumAligned=" + to_string(runtimeInfo.lastAxisElementNumAligned) + "\n";
    logStr += "    workspaceSizeInBytes=" + to_string(runtimeInfo.workspaceSizeInBytes) + "\n";
    logStr += "    srcStrideWorkspace=" + to_string(runtimeInfo.srcStrideWorkspace) + "\n";
    logStr += "    dstStrideWorkspace=" + to_string(runtimeInfo.dstStrideWorkspace) + "\n";
    logStr += "    dstBaseAddr(Logic):" + vec_to_string(runtimeInfo.dstBaseAddr) + "\n";
    logStr += "    srcBaseAddr(Logic):" + vec_to_string(runtimeInfo.srcBaseAddr) + "\n";
    logStr += "    srcBaseAddrWorkspace(Logic):" + vec_to_string(runtimeInfo.srcBaseAddrWorkspace) + "\n";
    logStr += "    BaseAddrWorkspace(Logic):" + vec_to_string(runtimeInfo.srcBaseAddrWorkspace) + "\n";
    logStr += "    dstBaseAddr(Actual):";
    for (auto i : runtimeInfo.dstBaseAddr) {
        logStr += to_string(GmMemOffsetFromBlock(i,
                                                 levelInfo.burstLen,
                                                 levelInfo.alignElement,
                                                 compilerInfo.dType)) + ",";
    }
    logStr += "\n    srcBaseAddr(Actual):";
    for (auto i : runtimeInfo.srcBaseAddr) {
        logStr += to_string(GmMemOffsetFromBlock(i,
                                                 levelInfo.burstLen,
                                                 levelInfo.alignElement,
                                                 compilerInfo.dType)) + ",";
    }
    logStr += "\n    srcBaseAddr(Actual 32BAligned Elements):";
    for (auto i : runtimeInfo.srcBaseAddr) {
        logStr += to_string(GmMemOffsetFromBlock(i,
                                                 levelInfo.burstLen,
                                                 0,
                                                 compilerInfo.dType)) + ",";
    }
    logStr += "\n    srcBaseAddrWorkspace(Actual):";
    for (auto i : runtimeInfo.srcBaseAddrWorkspace) {
        logStr += to_string(GmMemOffsetFromBlock(i,
                                                 levelInfo.burstLen,
                                                 levelInfo.alignElement,
                                                 compilerInfo.dType)) + ",";
    }
    logStr += "\n    dstBaseAddrWorkspace(Actual):";
    for (auto i : runtimeInfo.srcBaseAddrWorkspace) {
        logStr += to_string(GmMemOffsetFromBlock(i,
                                                 levelInfo.burstLen,
                                                 0,
                                                 compilerInfo.dType)) + ",";
    }
    logStr += "\n    dirtyDataStartAddrPerCore(Actual):" + vec_to_string(runtimeInfo.dirtyDataStartAddrPerCore) + "\n";
    logStr += "\n    initRanges:\n";
    for (auto item : runtimeInfo.initRanges) {
        logStr += "        [" + to_string(item.first)  + "," +\
                                to_string(item.second) + "]" +\
                                to_string(item.second - item.first + 1) + "\n";
    }
    logStr += "\n    extendRanges:\n";
    for (auto item : runtimeInfo.extendRanges) {
        logStr += "        [" + to_string(item.first)  + "," +\
                                to_string(item.second) + "]" +\
                                to_string(item.second - item.first + 1) + "\n";
    }
    logStr += "\n";
    cout<<logStr<<endl;
}


static void PrintTilingInfo(const CompilerInfo & compilerInfo,
                            const ShapeInfo & shapeInfo,
                            const RuntimeInfo & runtimeInfo) {
    string logStr = "==================================================================\n";
    logStr += "ShapeInfo:\n\t";
    logStr += "in: " + vec_to_string(shapeInfo.inShape) + "\n\t";
    logStr += "out: " + vec_to_string(shapeInfo.outShape) + "\n\t";
    logStr += "perm: " + vec_to_string(shapeInfo.perm) + "\n\t";
    logStr += "reducedIn: " + vec_to_string(shapeInfo.reducedInShape) + "\n\t";
    logStr += "reducedOut: " + vec_to_string(shapeInfo.reducedOutShape) + "\n\t";
    logStr += "reducedPerm: " + vec_to_string(shapeInfo.reducedPerm) + "\n\t";
    logStr += "reducedPermGrad: " + vec_to_string(shapeInfo.reducedPermGrad) + "\n\t";
    logStr += "dim: " + to_string(shapeInfo.dim) + "\n\n";

    logStr += "CompilerInfo:\n\t";
    logStr += "coreNum=" + to_string(compilerInfo.coreNum) + ", ";
    logStr += "ubSize=" + to_string(compilerInfo.ubSize) + ", ";
    logStr += "ubSizeCouldUse=" + to_string(compilerInfo.ubSizeCouldUse) + ", ";
    logStr += "dType=" + compilerInfo.dType +"\n\n";

    logStr += "RuntimeInfo:\n\t";
    logStr += "dstJumpStride=" + to_string(runtimeInfo.dstJumpStride) + ", ";
    logStr += "srcJumpAxisNum=" + to_string(runtimeInfo.srcJumpAxisNum) + "\n\tJumpFactor: ";
    for (int j = 0; j < TRANSPOSE_MAX_AXIS_NUM - 1; j++) {
        logStr += to_string(runtimeInfo.srcJumpFactor[j]) + " ";
    }
    logStr +=  "\n\tsrcJumpStride: ";
    for (int j = 0; j < TRANSPOSE_MAX_AXIS_NUM - 1; j++) {
        logStr += to_string(runtimeInfo.srcJumpStride[j]) + " ";
    }
    logStr +=  "\n\t";
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        logStr += "core" + to_string(i) + ":\n\t\t";
        logStr += to_string(runtimeInfo.colElePerMC[i]) + "  ";
        logStr += to_string(runtimeInfo.loopOnMC[i]) + "  ";
        logStr += to_string(runtimeInfo.colEleTC[i]) + "  ";
        logStr += to_string(runtimeInfo.colOffset[i]) + "  ";
        logStr += to_string(runtimeInfo.backStepLeft[i]) + "\n\t\t";
        logStr += to_string(runtimeInfo.rowPerMR[i]) + "  ";
        logStr += to_string(runtimeInfo.loopOnMR[i]) + "  ";
        logStr += to_string(runtimeInfo.rowTR[i]) + "  ";
        logStr += to_string(runtimeInfo.rowOffset[i]) + "  ";
        logStr += to_string(runtimeInfo.backStepUp[i]) + "\n\t\t";

        for (int j =0; j < TRANSPOSE_MAX_AXIS_NUM - 1; j++) {
            logStr += to_string(runtimeInfo.initJumpCounter[i][j]) + " ";
        }
        logStr +=  "\n\t\t";
        for (int j =0; j < TRANSPOSE_MAX_AXIS_NUM - 1; j++) {
            logStr += to_string(runtimeInfo.tailJumpCounter[i][j]) + " ";
        }
        logStr +=  "\n\t";
    }
    cout<<logStr;
}


bool TilingDataLastAxisNotTransposed(const CompilerInfo & compilerInfo,
                             const ShapeInfo & shapeInfo,
                             RuntimeInfo & runtimeInfo,
                             LevelInfo & levelInfo) {
    int64_t crossLevelNum = 0;
    int64_t level = 0;
    bool ret = false;
    CalcUbReorderFactor(compilerInfo, shapeInfo, runtimeInfo);
    CalcLevelAccuVolume(shapeInfo, levelInfo);
    ret = CalcLevelInfo(compilerInfo, shapeInfo, runtimeInfo, levelInfo);
    CHECK_TRANSPOSE_TILING_RESULT(ret)
    CalcInitRange(compilerInfo, shapeInfo, runtimeInfo);
    level = LocateLevel(shapeInfo, levelInfo, runtimeInfo);
    CountRangeCrossLevelNum(level, runtimeInfo, levelInfo, crossLevelNum);
    CalcExtendRange(level, crossLevelNum, shapeInfo, levelInfo, runtimeInfo);
    CalcLevelLoopNum(level, shapeInfo, runtimeInfo, levelInfo);
    UpdateLevelOneGap(shapeInfo, levelInfo);
    ret = CalcRuntimeInfo(compilerInfo, shapeInfo, levelInfo, runtimeInfo);
    CHECK_TRANSPOSE_TILING_RESULT(ret)
    CalcBaseAddr(compilerInfo, shapeInfo, levelInfo, runtimeInfo);
    PrintTilingInfo(level, crossLevelNum, compilerInfo, shapeInfo, runtimeInfo, levelInfo);
    return true;
}

static int64_t CalcStride(const ShapeInfo & shapeInfo, int index) {
    int64_t vol = 1;
    for (int i = index + 1; i < shapeInfo.dim; i++) {
        vol *= shapeInfo.reducedInShape[i];
    }
    return vol;
}


static int64_t Align32BFloor(int64_t i, int64_t elementNumPerBlock) {
    return i - i % elementNumPerBlock;
}

static int64_t Align32BCeil(int64_t i, int64_t elementNumPerBlock) {
    if (i % elementNumPerBlock == 0) {
        return i;
    }
    return i + (elementNumPerBlock - i % elementNumPerBlock);
}


static void SplitRow(const CompilerInfo & compilerInfo, const ShapeInfo & shapeInfo, RuntimeInfo & runtimeInfo) {
    int64_t vol = 1;
    int64_t dim = shapeInfo.dim;
    int64_t coreNum =  compilerInfo.coreNum;
    int64_t elementNumPerBlock = ElementNumPerBlock(compilerInfo.dType);
    runtimeInfo.rowPerCore.resize(coreNum, 0);
    runtimeInfo.rowPerMR.resize(coreNum, 0);
    runtimeInfo.rowBlockPerMR.resize(coreNum, 0);
    runtimeInfo.loopOnMR.resize(coreNum, 0);
    runtimeInfo.rowTR.resize(coreNum, 0);
    runtimeInfo.rowBlockTR.resize(coreNum, 0);
    runtimeInfo.backStepUp.resize(coreNum, 0);
    runtimeInfo.rowOffset.resize(coreNum, 0);

    for (int64_t i = 0; i < dim - 1; i++) {
        vol *= shapeInfo.reducedInShape[i];
    }
    int64_t volPerCore = Align32BCeil(vol / coreNum, elementNumPerBlock);
    //0 - N-2 core
    for (int64_t i = 0; i < coreNum - 1; i++) {
        runtimeInfo.rowPerCore[i] = volPerCore;
    }
    //N-1 core
    runtimeInfo.rowPerCore[coreNum - 1] = vol - (coreNum - 1) * volPerCore;

    while (runtimeInfo.rowPerCore[coreNum - 1] < elementNumPerBlock) {
        for (int64_t i = 0; i < coreNum - 1; i++) {
            runtimeInfo.rowPerCore[i] -= elementNumPerBlock;
            runtimeInfo.rowPerCore[coreNum - 1] += elementNumPerBlock;
            if(runtimeInfo.rowPerCore[coreNum - 1] >= elementNumPerBlock) {
                break;
            }
        }
    }

    /*
     * vnchwconv max repeat is 255, so max block number is 255 * 16 = 4080
     */
    int64_t maxLine = 128;
    int64_t rowOffset = 0;
    for (int64_t i = 0; i < coreNum; i++) {
        if (runtimeInfo.colBlockPerMC[i] == 1) {
           maxLine = 2048;  // 2048 * 1 = 2048
        }
        else if (runtimeInfo.colBlockPerMC[i] <= 3) {
           maxLine = 1024;  // 1024 * 3 = 3072
        }
        else if (runtimeInfo.colBlockPerMC[i] <= 7) {
           maxLine = 512; // 512 * 7 = 3584
        }
        else if (runtimeInfo.colBlockPerMC[i] <= 15) {
           maxLine = 256; // 256 * 15 = 3840
        }
        int64_t rowPerCore = runtimeInfo.rowPerCore[i];
        int64_t k = rowPerCore % elementNumPerBlock;
        if (k != 0) {
            rowPerCore = rowPerCore - k;
        }
        runtimeInfo.rowPerMR[i] = min(maxLine, rowPerCore);
        runtimeInfo.rowBlockPerMR[i] = runtimeInfo.rowPerMR[i] / elementNumPerBlock;
        runtimeInfo.loopOnMR[i] = rowPerCore / runtimeInfo.rowPerMR[i];
        cout<<"i="<<i<<endl;
        cout<<"rowPerCore="<<rowPerCore<<endl;
        cout<<"rowTR="<<runtimeInfo.rowTR[i]<<endl<<endl;
        runtimeInfo.rowTR[i] = runtimeInfo.rowPerCore[i] - runtimeInfo.loopOnMR[i] * runtimeInfo.rowPerMR[i];
        runtimeInfo.rowBlockTR[i] = ceil(runtimeInfo.rowTR[i] / elementNumPerBlock);
        runtimeInfo.backStepUp[i] = (k != 0) ? (elementNumPerBlock - k) : 0;
        runtimeInfo.rowOffset[i] = rowOffset;
        rowOffset += runtimeInfo.rowPerCore[i];
    }
}


static void SplitCol(const CompilerInfo & compilerInfo, const ShapeInfo & shapeInfo, RuntimeInfo & runtimeInfo) {
    int64_t dim = shapeInfo.dim;
    int64_t baseLen = MAX_COL_FP16_VNCHWCONV_FULL / SizeofDType(compilerInfo.dType) * 2;
    int64_t lastAxisLen = shapeInfo.reducedInShape[dim - 1];
    int64_t elementNumPerBlock = ElementNumPerBlock(compilerInfo.dType);
    int64_t colElePerMC = 0;
    int64_t colBlockPerMC = 0;
    int64_t loopOnMC = 0;
    int64_t colEleTC = 0;
    int64_t colBlockTC = 0;
    int64_t backStepLeft = 0;
    int64_t colOffset = 0;

    /* max 248 scenario */
    if (shapeInfo.reducedInShape[dim - 1] > baseLen) {
        colElePerMC = baseLen;
    }else {
        /* align 32 B as long as possible */
        colElePerMC = Align32BFloor(lastAxisLen, elementNumPerBlock);
    }
    colBlockPerMC = colElePerMC / elementNumPerBlock;
    loopOnMC = lastAxisLen / colElePerMC;
    colEleTC = lastAxisLen - loopOnMC * colElePerMC;
    colBlockTC = ceil(colEleTC / elementNumPerBlock);
    int64_t k = colEleTC % elementNumPerBlock;
    if (k != 0) {
        backStepLeft = elementNumPerBlock - k;
    }

    if (colEleTC != 0) {
        if (colEleTC % elementNumPerBlock != 0) {
            backStepLeft = elementNumPerBlock - colEleTC % elementNumPerBlock;
        }
    }

    for (int64_t i = 0; i < compilerInfo.coreNum; i++) {
        runtimeInfo.colElePerMC.push_back(colElePerMC);
        runtimeInfo.colBlockPerMC.push_back(colBlockPerMC);
        runtimeInfo.loopOnMC.push_back(loopOnMC);
        runtimeInfo.colEleTC.push_back(colEleTC);
        runtimeInfo.colBlockTC.push_back(colBlockTC);
        runtimeInfo.backStepLeft.push_back(backStepLeft);
        runtimeInfo.colOffset.push_back(colOffset);
        //colOffset += colElePerMC; //TODO: only split core on vertical, if mixed ,need reset.
    }
    /* max 128 scenario */
}


static void CalcJumpFactorMod(const CompilerInfo & compilerInfo, RuntimeInfo & runtimeInfo) {
    runtimeInfo.initJumpCounter.resize(compilerInfo.coreNum);
    runtimeInfo.tailJumpCounter.resize(compilerInfo.coreNum);

    //TODO: axis before last axis in out shape
    for (int64_t i = 1; i < runtimeInfo.srcJumpAxisNum; i++) {
        runtimeInfo.srcJumpFactorMod[i] *= runtimeInfo.srcJumpFactorMod[i - 1] * runtimeInfo.srcJumpFactor[i - 1];
    }

    for (int64_t i = 0; i < 8; i++) {
    }

    for (int64_t i = 0; i < compilerInfo.coreNum; i++) {
        runtimeInfo.initJumpCounter[i].resize(TRANSPOSE_MAX_AXIS_NUM, 0);
        runtimeInfo.tailJumpCounter[i].resize(TRANSPOSE_MAX_AXIS_NUM, 0);
        for (int64_t j = 0; j < runtimeInfo.srcJumpAxisNum; j++) {
            runtimeInfo.initJumpCounter[i][j] = (runtimeInfo.rowOffset[i] / runtimeInfo.srcJumpFactorMod[j]) %\
                                                 runtimeInfo.srcJumpFactor[j];
            //cout<<runtimeInfo.initJumpCounter[i][j]<<"="<<runtimeInfo.rowOffset[i]<<"/"<< runtimeInfo.srcJumpFactorMod[j]<<"%"<<runtimeInfo.srcJumpFactor[j]<<", ";
            if (runtimeInfo.rowTR[i] != 0) {
                int64_t row = runtimeInfo.rowOffset[i] + runtimeInfo.rowPerMR[i] * runtimeInfo.loopOnMR[i] -\
                              runtimeInfo.backStepUp[i];
                runtimeInfo.tailJumpCounter[i][j] = (row / runtimeInfo.srcJumpFactorMod[j]) %\
                                                    runtimeInfo.srcJumpFactor[j];
            cout<<runtimeInfo.tailJumpCounter[i][j]<<"="<<row<<"/"<< runtimeInfo.srcJumpFactorMod[j]<<"%"<<runtimeInfo.srcJumpFactor[j]<<endl;
            }
        }
        cout<<endl;
    }
}

static bool TilingDataLastAxisTransposed(const CompilerInfo & compilerInfo,
                             const ShapeInfo & shapeInfo,
                             RuntimeInfo & runtimeInfo) {
    int64_t srcAxisIndex = 0;
    int64_t lastAxisPerm = shapeInfo.dim - 1;
    for (int i = shapeInfo.reducedPerm.size() - 1; i >= 0; i--) {
       runtimeInfo.srcJumpFactor[srcAxisIndex] = shapeInfo.reducedInShape[shapeInfo.reducedPerm[i]];
       runtimeInfo.srcJumpStride[srcAxisIndex] = CalcStride(shapeInfo, shapeInfo.reducedPerm[i]);
       srcAxisIndex++;
    }

    for(int64_t i = 0; i < shapeInfo.dim; i++) {
        if (lastAxisPerm == shapeInfo.reducedPerm[i]) {
            runtimeInfo.srcJumpAxisNum = shapeInfo.dim - i - 1;
            break;
        }
    }

    for(int64_t i = 0; i < shapeInfo.dim - 1; i++) {
        runtimeInfo.dstJumpStride *= shapeInfo.reducedInShape[i];
    }

    /* support scenario : col [8, unlimited], row [8, unlimited]
     * better perfromance scenario : col >= 128, row >= 128 * coreNum
     */
    SplitCol(compilerInfo, shapeInfo, runtimeInfo);
    SplitRow(compilerInfo, shapeInfo, runtimeInfo);
    CalcJumpFactorMod(compilerInfo, runtimeInfo);
    PrintTilingInfo(compilerInfo, shapeInfo, runtimeInfo);
    return true;
}


bool TransposeCalcTilingData(const CompilerInfo & compilerInfo,
                             const ShapeInfo & shapeInfo,
                             RuntimeInfo & runtimeInfo,
                             LevelInfo & levelInfo) {
    if (shapeInfo.scenario == e_last_axis_transposed) {
        return TilingDataLastAxisTransposed(compilerInfo, shapeInfo, runtimeInfo);
    }else {
        return TilingDataLastAxisNotTransposed(compilerInfo, shapeInfo, runtimeInfo, levelInfo);
    }

}


static void CalcDirtyDataStartAddr(const CompilerInfo & compilerInfo,
                                   const LevelInfo & levelInfo,
                                   RuntimeInfo & runtimeInfo) {
    cout<<"Entering CalcDirtyDataStartAddr"<<endl;
    for (const auto & item : runtimeInfo.extendRanges) {
        runtimeInfo.dirtyDataStartAddrPerCore.push_back((item.second + 1) * levelInfo.elementNumPerBurst);
    }
}


static void PrepareOverlapData(const CompilerInfo & compilerInfo,
                               const ShapeInfo & shapeInfo,
                               const LevelInfo & levelInfo,
                               RuntimeInfo & runtimeInfo,
                               const char * pInputX,
                               int64_t inputLen,
                               char * overlapData,
                               int32_t overlapDataLen) {
    cout<<"Entering PrepareOverlapData"<<endl;
    int64_t dim = shapeInfo.reducedPerm.size();
    int64_t perm[TRANSPOSE_MAX_AXIS_NUM + 1];
    int64_t inShapeAccuVol[TRANSPOSE_MAX_AXIS_NUM + 1];
    int64_t outShapeAccuVol[TRANSPOSE_MAX_AXIS_NUM + 1];

    for(int i = 0; i < TRANSPOSE_MAX_AXIS_NUM + 1; i++ ) {
        inShapeAccuVol[i] = 1;
        outShapeAccuVol[i] = 1;
    }
    int64_t singleEleLen = SizeofDType(compilerInfo.dType);
    int64_t offset = 0;
    int64_t elementNumPerBlock = BYTES_PER_BLOCK / singleEleLen;

    CalcDirtyDataStartAddr(compilerInfo, levelInfo, runtimeInfo);

    inShapeAccuVol[dim - 1] = shapeInfo.reducedInShape[dim - 1];
    outShapeAccuVol[dim - 1] = shapeInfo.reducedOutShape[dim - 1];

    for (int64_t i = dim - 2; i >= 0; i--) {
        inShapeAccuVol[i] = inShapeAccuVol[i + 1] * shapeInfo.reducedInShape[i];
        outShapeAccuVol[i] = outShapeAccuVol[i + 1] * shapeInfo.reducedOutShape[i];
    }

    for (int64_t i = 0; i < compilerInfo.coreNum - 1; i++) {
        for (int64_t j = 0; j < elementNumPerBlock; j++){
            /*
             * float32: elementNumPerBlock = 8
             * float16: elementNumPerBlock = 16
             */
            int64_t outAddr = runtimeInfo.dirtyDataStartAddrPerCore[i];
            int64_t inAddr = 0;
            outAddr += j;
            printf("outAddr=%ld\n",outAddr);
            for (int64_t k = 0; k < dim ; k++) {
                perm[shapeInfo.reducedPerm[k]] = outAddr / outShapeAccuVol[k + 1];
                outAddr -= perm[shapeInfo.reducedPerm[k]] * outShapeAccuVol[k + 1];
            }
            for (int64_t k = 0; k < dim ; k++) {
                printf("perm[%ld]=%ld\n",k,perm[k]);
            }
            for (int64_t k = 0; k < dim ; k++) {
                inAddr += perm[k] * inShapeAccuVol[k + 1];
            }
            printf("inAddr=%ld, offset=%ld\n", inAddr, offset);
            XXX("xxxxxxxxxxxxxxxxx");
            (void)memcpy_s(overlapData + offset,
                           overlapDataLen - offset,
                           pInputX + inAddr * singleEleLen,
                           singleEleLen);
            offset += singleEleLen;
        }
    }
}


bool GetCompileParams(const string & opType, const nlohmann::json &opCompileInfoJson, CompilerInfo & info) {
    OP_LOGD(opType.c_str(), "Entering GetCompileParams.");

    using namespace nlohmann;

    auto allVars = opCompileInfoJson["vars"];

    info.opType = opType;

    if (allVars.count("core_num") == 0) {
        OP_LOGE(opType.c_str(), "GetCompileParams, get core_num error");
        return false;
    }
    if (allVars.count("ub_size") == 0) {
        OP_LOGE(opType.c_str(), "GetCompileParams, get ub_size error");
        return false;
    }
    if (allVars.count("dtype") == 0) {
        OP_LOGE(opType.c_str(), "GetCompileParams, get dtype error");
        return false;
    }

    info.coreNum = allVars["core_num"].get<std::int64_t>();
    info.ubSize = allVars["ub_size"].get<std::int64_t>();
    info.ubSizeCouldUse = info.ubSize - UB_RESERVED_BLOCK_SIZE;
    info.dType = allVars["dtype"].get<std::string>();

    OP_LOGD(opType.c_str(), "GetCompileParams, coreNum[%d], ubSize[%d] blocks, dType[%s].",
           info.coreNum, info.ubSize, info.dType.c_str());

    return true;
}

bool IsNeedPrepare(const ShapeInfo & shapeInfo, const LevelInfo & levelInfo) {
    if (shapeInfo.identical == 1) {
        return false;
    }
    if (levelInfo.alignElement == 0) {
        return false;
    }
    return true;
}

/*
     0               1               2               3
     0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                      overlap_data_1                           |
    |                      ...                                      |   32 block(32 core)
    |                      overlap_data_n(n=core_num)               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |    core_num   |  ub_size      | ub_threshold  |  nburst       |   1 block
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |  nburst_tail  |  burst_len    | align_element |  src_stride   |   1 block
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    | fp16_offset_1 | fp16_offset_2 | fp16_offset_3 | cycle_num_wsp |   1 block
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    | loop_num_wsp  | nburst_wsp    |nburst_tail_wsp| by_workspace  |   1 block
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    | last_axis_ele |last_axis_ele_a|str_stride_wsp | dst_stride_wsp|   1 block
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                     level_loop_num (8)                        |   2 block
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                     level_gap (8)                             |   2 block
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                     has_tail (8)                              |   2 block
    |                                                               |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                     dst_base_addr_1                           |
    |                     ...                                       |   8 block(32 core)
    |                     dst_base_addr_n                           |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                     src_base_addr_1                           |
    |                     ...                                       |   8 block(32 core)
    |                     src_base_addr_n                           |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                     src_base_addr_wsp_1                       |
    |                     ...                                       |   8 block(32 core) exist when by_workspace = 1
    |                     src_base_addr_wsp_n                       |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                     dirty_data_start_addr_1                   |
    |                     ...                                       |   8 block(32 core)
    |                     dirty_data_start_addr_n                   |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
                                                                       --------
 */

static void SerializeTilingDataLastAxisTranspoed(OpRunInfo &runInfo,
                                                 const CompilerInfo & compilerInfo,
                                                 const ShapeInfo & shapeInfo,
                                                 const RuntimeInfo & runtimeInfo) {
    ByteBufferPut(runInfo.tiling_data, (int64_t)e_last_axis_transposed);          //0
    ByteBufferPut(runInfo.tiling_data, (int64_t)0);                               //0
    ByteBufferPut(runInfo.tiling_data, (int64_t)0);                               //0
    ByteBufferPut(runInfo.tiling_data, (int64_t)0);                               //0

    ByteBufferPut(runInfo.tiling_data, compilerInfo.coreNum);                     //1
    ByteBufferPut(runInfo.tiling_data, compilerInfo.ubSize);                      //2
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.dstJumpStride);                //3
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.srcJumpAxisNum);               //3
    for (int j = 0; j < TRANSPOSE_MAX_AXIS_NUM - 1; j++) {
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.srcJumpFactor[j]);
    }
    for (int j = 0; j < TRANSPOSE_MAX_AXIS_NUM - 1; j++) {
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.srcJumpStride[j]);
    }
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.colElePerMC[i]);          //0
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.loopOnMC[i]);             //1
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.colEleTC[i]);             //2
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.colOffset[i]);            //3
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.backStepLeft[i]);         //4
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.rowPerMR[i]);             //5
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.loopOnMR[i]);             //6
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.rowTR[i]);                //7
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.rowOffset[i]);            //8
        ByteBufferPut(runInfo.tiling_data, runtimeInfo.backStepUp[i]);           //9
        for (int j =0; j < TRANSPOSE_MAX_AXIS_NUM - 1; j++) {
            ByteBufferPut(runInfo.tiling_data, runtimeInfo.initJumpCounter[i][j]);  //10-16
        }
    }
}


static void SerializeTilingDataLastAxisNotTransposed(OpRunInfo &runInfo,
                                                     unique_ptr<BLOCK_STRUCT[]> & overlapData,
                                                     const CompilerInfo & compilerInfo,
                                                     const ShapeInfo & shapeInfo,
                                                     const RuntimeInfo & runtimeInfo,
                                                     const LevelInfo & levelInfo) {
    ByteBufferPut(runInfo.tiling_data, (int64_t)e_last_axis_transposed);          //0
    ByteBufferPut(runInfo.tiling_data, (int64_t)0);                               //1
    ByteBufferPut(runInfo.tiling_data, (int64_t)0);                               //2
    ByteBufferPut(runInfo.tiling_data, (int64_t)0);                               //3

    for (int i = 0; i < compilerInfo.coreNum; i++) {
        ByteBufferPut(runInfo.tiling_data, overlapData[i]);
    }
    ByteBufferPut(runInfo.tiling_data, compilerInfo.coreNum);                     //0
    ByteBufferPut(runInfo.tiling_data, compilerInfo.ubSize);                      //1
    ByteBufferPut(runInfo.tiling_data, shapeInfo.identical);                      //2
    ByteBufferPut(runInfo.tiling_data, levelInfo.nBurst);                         //3
    ByteBufferPut(runInfo.tiling_data, levelInfo.nBurstTail);                     //4
    ByteBufferPut(runInfo.tiling_data, levelInfo.burstLen);                       //5
    ByteBufferPut(runInfo.tiling_data, levelInfo.burstLenTail);                   //6
    ByteBufferPut(runInfo.tiling_data, levelInfo.srcStride * levelInfo.burstLen); //7
    ByteBufferPut(runInfo.tiling_data, levelInfo.alignElement);                   //8
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.ubReorderFactor);              //9
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.ubThreshold);                  //10
    ByteBufferPut(runInfo.tiling_data, levelInfo.identicalLoopNum);               //11
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.fp16Offset1);                  //12
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.fp16Offset2);                  //13
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.fp16Offset3);                  //14
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.cycleNumWorkspace);            //15
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.loopNumWorkspace);             //16
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.nBurstWorkspace);              //l7
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.nBurstTailWorkspace);          //18
    ByteBufferPut(runInfo.tiling_data, levelInfo.byWorkspace);                    //19
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.lastAxisElementNum);           //20
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.lastAxisElementNumAligned);    //21
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.srcStrideWorkspace);           //22
    ByteBufferPut(runInfo.tiling_data, runtimeInfo.dstStrideWorkspace);           //23
    int64_t alignElement = levelInfo.byWorkspace == 1 ? 0 : levelInfo.alignElement;

    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
        ByteBufferPut(runInfo.tiling_data, levelInfo.levelLoopNum[i]);
    }
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
        ByteBufferPut(runInfo.tiling_data, GmMemOffsetFromBlock(levelInfo.srcGapPerRound[i],
                                                                levelInfo.burstLen,
                                                                alignElement,
                                                                compilerInfo.dType));
    }
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
        ByteBufferPut(runInfo.tiling_data, levelInfo.hasTail[i]);
    }
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        ByteBufferPut(runInfo.tiling_data, GmMemOffsetFromBlock(runtimeInfo.dstBaseAddr[i],
                                                                levelInfo.burstLen,
                                                                levelInfo.alignElement,
                                                                compilerInfo.dType));
    }
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        ByteBufferPut(runInfo.tiling_data, GmMemOffsetFromBlock(runtimeInfo.srcBaseAddr[i],
                                                                levelInfo.burstLen,
                                                                alignElement,
                                                                compilerInfo.dType));
    }
    if(levelInfo.byWorkspace) {
        for (int i = 0; i < compilerInfo.coreNum; i++) {
            ByteBufferPut(runInfo.tiling_data, GmMemOffsetFromBlock(runtimeInfo.srcBaseAddrWorkspace[i],
                                                                   levelInfo.burstLen,
                                                                   levelInfo.alignElement,
                                                                   compilerInfo.dType));
        }
        for (int i = 0; i < compilerInfo.coreNum; i++) {
            ByteBufferPut(runInfo.tiling_data, GmMemOffsetFromBlock(runtimeInfo.srcBaseAddrWorkspace[i],
                                                                    levelInfo.burstLen,
                                                                    0,
                                                                    compilerInfo.dType));
        }
    }else {
        for (int i = 0; i < compilerInfo.coreNum; i++) {
            ByteBufferPut(runInfo.tiling_data, (int64_t)0);
        }
        for (int i = 0; i < compilerInfo.coreNum; i++) {
            ByteBufferPut(runInfo.tiling_data, (int64_t)0);
        }
    }
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        if (i < runtimeInfo.dirtyDataStartAddrPerCore.size()) {
            ByteBufferPut(runInfo.tiling_data, runtimeInfo.dirtyDataStartAddrPerCore[i]);
        } else {
            ByteBufferPut(runInfo.tiling_data, (int64_t)0);
        }
    }

    runInfo.block_dim = compilerInfo.coreNum;
    std::vector<int64_t> workspace;
    workspace.push_back(runtimeInfo.workspaceSizeInBytes);
    runInfo.workspaces = workspace;

    /**********************test code******************************/
    char buf[4096] = {0};
    memset(buf, 0, 4096);
    runInfo.tiling_data.seekg(0, std::ios::end);
    int64_t tiling_data_len = runInfo.tiling_data.tellg();
    runInfo.tiling_data.seekg(0, std::ios::beg);
    runInfo.tiling_data.read(buf, tiling_data_len);
    hexdump(buf, tiling_data_len);
    runInfo.tiling_data.seekg(0, std::ios::beg);
    /*************************************************************/
}


void SerializeTilingData(OpRunInfo &runInfo,
                         unique_ptr<BLOCK_STRUCT[]> & overlapData,
                         const CompilerInfo & compilerInfo,
                         const ShapeInfo & shapeInfo,
                         const RuntimeInfo & runtimeInfo,
                         const LevelInfo &levelInfo)
{
    if (shapeInfo.scenario == e_last_axis_transposed) {
        SerializeTilingDataLastAxisTranspoed(runInfo, compilerInfo, shapeInfo, runtimeInfo);
    }else {
        SerializeTilingDataLastAxisNotTransposed(runInfo, overlapData, compilerInfo, shapeInfo, runtimeInfo, levelInfo);
    }
}



bool TransposeTiling(const std::string &opType,
                     const TeOpParas &opParas,
                     const nlohmann::json &opInfo,
                     OpRunInfo &runInfo)
{
    OP_LOGI(opType.c_str(), "Tiling is running.");

    CompilerInfo compilerInfo;
    LevelInfo levelInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;
    const char * pInputX = NULL;
    int64_t inputLen = 0;
    int64_t reserved = 0;

    if (GetCompileParams(opType, opInfo, compilerInfo) == false) {
        return false;
    }
    if (GetShapePerm(opType, opParas, shapeInfo) == false) {
        return false;
    }
    if (CheckTensorShape(opType, shapeInfo) == false) {
        return false;
    }
    if (GetInputX(opType, opParas, pInputX, inputLen) == false) {
        return false;
    }

    ReduceAxis(opType, compilerInfo, shapeInfo);

    if (TransposeCalcTilingData(compilerInfo, shapeInfo, runtimeInfo, levelInfo) == false) {
        return false;
    }

    int64_t overlapDataLen = compilerInfo.coreNum * BYTES_PER_BLOCK;


    unique_ptr<BLOCK_STRUCT[]> overlapData(new BLOCK_STRUCT[compilerInfo.coreNum]);
    if (IsNeedPrepare(shapeInfo, levelInfo)) {
        PrepareOverlapData(compilerInfo,
                           shapeInfo,
                           levelInfo,
                           runtimeInfo,
                           pInputX,
                           inputLen,
                           (char*)overlapData.get(),
                           overlapDataLen);
    }

    hexdump((void*)overlapData.get(), 1024);//TODO, consider core num

    SerializeTilingData(runInfo, overlapData, compilerInfo, shapeInfo, runtimeInfo, levelInfo); 

    return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(Transpose, TransposeTiling);

}

