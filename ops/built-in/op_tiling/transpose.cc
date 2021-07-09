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

#include "register/op_tiling.h"
#include "op_log.h"
#include "op_tiling.h"
#include "securec.h"
#include <nlohmann/json.hpp>
#include "error_log.h"

using namespace std;

namespace optiling {

#define INVALID_SPLIT 999
#define LOOP_FOR_UB_PADDING 10

#define TRANSPOSE_CHECK_RET(res) \
    if (res == false) { \
        return false; \
    }

static void PrintScreen(const string & logStr) {
    const char * pLevel = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    const char * pStdout = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");

    if (pLevel == NULL || pStdout == NULL) {
        return;
    }
    if (pLevel[0] == '0' && pStdout[0] == '1') {
        cout << logStr << endl;
    }
}

static int64_t AlignX(int64_t a, int64_t x) {
    if (a % x == 0) {
        return a;
    }
    return a + x - a % x;
}

// 1/16 usage of UB with vnchwconv as b16
static int64_t CalcVnchwconvPartialUbSize(int64_t coreNum, int64_t ubBlocks) {
    return (ubBlocks * 32 - UB_RESERVED_KB * 1024) / 32 / 2;
}

// full usage of UB with vnchwconv
static int64_t CalcVnchwconvFullColSize(int64_t coreNum, int64_t ubBlocks) {
    if (coreNum > 2 && ubBlocks == 8192) {
        return 256; //910, 224*2 is better
    }
    else if (coreNum == 2 && ubBlocks == 8192) {
        return 256; //310, 224*2 is better
    }
    else if (coreNum == 1 && ubBlocks == 6144) {
        return 256; // cs and es
    } else {
        return 256;
    }
}

static int GCD(int a, int b) {
    if (b == 0) return a;
    return GCD(b, a % b);
}

static int64_t Align16(int64_t val, int64_t factor, int64_t upLimit = 0) {
    int64_t res = val / factor;
    int64_t k = res % 16;
    if (k != 0) {
        res = res + 16 - k;
    }
    if (upLimit != 0) {
        while (res * factor > upLimit)  {
            res -= 16;
        }
    }
    return res;
}

static void SplitEvenly(int64_t coreNum, int64_t vol, int64_t & x, int64_t & y,
                        int64_t & m, int64_t & n, int64_t unit = 1) {
    if (vol <= unit) {
        m = vol;
        n = 0;
        x = 1;
        y = 0;
    } else if (vol < coreNum * unit) {
        m = unit;
        if (vol % unit != 0) {
            n = unit + vol % unit;
        } else {
            n = 0;
        }
        x = vol / unit;
        if (n != 0) {
            y = 1;
        } else {
            y = 0;
        }
    } else {
        m = ceil(vol * 1.0 / coreNum);
        n = m - 1;
        x = vol - coreNum * (m - 1);
        y = coreNum - x;
    }
}

static string PadString(string &in, int width = 0) {
    string s = in;
    if (width == 0) {
        return s;
    }
    if ((int)s.size() < width) {
        for (int i = 0; i < width - (int)in.size(); i++) {
            s += " ";
        }
    }
    return s;
}

template<typename T> static string to_string(T in, int width = 0) {
    string s = std::to_string(in);
    return PadString(s, width);
}

static string hex_perm_to_string(int64_t hexPerm, int width = 0) {
    string s;
    if (hexPerm == 0x10) {
        s = "0x10";
    }
    if (hexPerm == 0x01) {
        s = "0x01";
    }
    if (hexPerm == 0x00) {
        s = "0x0";
    }
    if (hexPerm == 0x210) {
        s = "0x210";
    }
    return PadString(s, width);
}

static string vec_to_string(const vector<int64_t> & v, int width = 0) {
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
    return PadString(s, width);
}

template<typename T> static string arr_to_string(const T * v, int64_t size, int width = 0) {
    string s;
    bool first = true;

    if (v == nullptr) {
        return s;
    }

    for (int i = 0; i < size; i++) {
        if (first) {
            s += std::to_string(v[i]);
            first = false;
        } else {
            s += "," + std::to_string(v[i]);
        }
    }
    return PadString(s, width);
}

static bool IsSubset(vector<int64_t> a, vector<int64_t> b){
    int i = 0;
    int j = 0;
    int m = a.size();
    int n = b.size();
    if (m < n) {
        return false;
    }

    sort(a.begin(), a.end());
    sort(b.begin(), b.end());
    while ((i < n) && (j < m)) {
        if (a[j] < b[i]) {
            j++;
        }
        else if(a[j] == b[i]) {
            j++;
            i++;
        }
        else if(a[j] > b[i]) {
            return false;
        }
    }
    if (i < n) {
        return false;
    }
    return true;
}

static void VectorSub(vector<int64_t> a, vector<int64_t> b, vector<int64_t> &c) {
    sort(a.begin(), a.end());
    sort(b.begin(), b.end());
    set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(c));
}

static void VectorAdd(vector<int64_t> a, vector<int64_t> b, vector<int64_t> &c) {
    for (size_t i = 0; i < a.size(); i++) {
        c.push_back(a[i]);
    }
    for (size_t i = 0; i < b.size(); i++) {
        c.push_back(b[i]);
    }
}

template<typename T> static void ReverseArray(T * array,  int64_t size) {
    for (int64_t i = 0; i < size / 2; i++) {
        T temp = array[i];
        array[i] = array[size - 1 - i];
        array[size - 1 - i] = temp;
    }
}

static int64_t SizeofDType(const string & dType) {
    if (dType == "int8" || dType == "uint8" || dType == "bool") {
        return 1;
    } else if (dType == "int16" || dType == "uint16" || dType == "float16") {
        return 2;
    } else if (dType == "int32" || dType == "uint32" || dType == "float" || dType == "float32") {
        return 4;
    } else if (dType == "int64" || dType == "uint64" || dType == "float64" || dType == "double") {
        return 8;
    }
    return 1;
}

static int64_t ElementNumPerBlock(const string & dType) {
    if (dType == "int8" || dType == "uint8" || dType == "bool") {
        return 32;
    } else if (dType == "int16" || dType == "uint16" || dType == "float16") {
        return 16;
    } else if (dType == "int32" || dType == "uint32" || dType == "float" || dType == "float32") {
        return 8;
    } else if (dType == "int64" || dType == "uint64" || dType == "float64" || dType == "double") {
        return 4;
    }
    return 32;
}

static bool Is32BAligned(const CompilerInfo & compilerInfo, const vector<int64_t> & reducedOutShape) {
    int64_t dim = reducedOutShape.size();
    return reducedOutShape[dim - 1] % ElementNumPerBlock(compilerInfo.dType) == 0;
}

static void BlockAlign(vector<int64_t> & vec) {
    int i = vec.size();
    int k = i % ELE_NUM_PER_BLOCK_INT64;
    int align = 0;
    if (k != 0)  {
        align = ELE_NUM_PER_BLOCK_INT64 - k;
    }
    for (int i = 0; i < align; i++) {
        vec.push_back(0);
    }
}

static int64_t GetPermIndex(const vector<int64_t> & perm, int p) {
    for (size_t i = 0; i < perm.size(); i++) {
        if (perm[i] == p) {
            return i;
        }
    }
    return 0;
}

static bool IsStrideTooHuge(const ShapeInfo &shapeInfo, const RuntimeInfo &runtimeInfo) {
    return runtimeInfo.srcStrideLogic * shapeInfo.lastAxisBurstLen > STRIDE_BOUNDARY;
}

static bool IsLastAxisJoinTranspose(ShapeInfo & shapeInfo) {
    int dim = shapeInfo.reducedPerm.size();
    if (dim <= 1) {
        return false;
    }

    if (shapeInfo.reducedPerm[dim - 1] != dim - 1) {
        return true;
    } else {
        return false;
    }
}

static void Reshape(ShapeInfo & shapeInfo) {
    int dim = shapeInfo.reducedPerm.size();
    shapeInfo.dim = shapeInfo.dim + 1;
    shapeInfo.reducedPerm.push_back(dim);
    shapeInfo.reducedInShape.push_back(1);
    shapeInfo.reducedOutShape.push_back(1);
    shapeInfo.lastAxisLen = 1;
    shapeInfo.lastAxisBurstLen = 1;
    shapeInfo.alignElement = shapeInfo.elePerBlock - 1;
    shapeInfo.isLastAxisTranspose = 0;
    shapeInfo.isLastAxisHuge = false;
}

static bool GetShapePerm(const string & opType, const TeOpParas & paras, ShapeInfo & shapeInfo) {
    OP_LOGD(opType.c_str(), "Entering GetShapePerm.");

    if (paras.const_inputs.find("perm") == paras.const_inputs.end()) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "No perm in const_inputs.");
        return false;
    }

    auto tensor = std::get<2>(paras.const_inputs.at("perm"));
    auto dType = tensor.GetTensorDesc().GetDataType();
    
    if (dType == ge::DataType::DT_INT64 || dType == ge::DataType::DT_UINT64) {
        const int64_t* pPerm = reinterpret_cast<const int64_t*>(std::get<0>(paras.const_inputs.at("perm")));
        if (pPerm == nullptr) {
            VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get perm pointer.");
            return false;
        }
        int32_t size = std::get<1>(paras.const_inputs.at("perm"));
        for (size_t i = 0; i < size / sizeof(int64_t); i++) {
            shapeInfo.perm.push_back(pPerm[i]);
        }
    } else {
        const int32_t* pPerm = reinterpret_cast<const int32_t*>(std::get<0>(paras.const_inputs.at("perm")));
        if (pPerm == nullptr) {
            VECTOR_INNER_ERR_REPORT_TILIING(opType, "Failed to get perm pointer.");
            return false;
        }
        int32_t size = std::get<1>(paras.const_inputs.at("perm"));
        for (size_t i = 0; i < size / sizeof(int32_t); i++) {
            shapeInfo.perm.push_back(pPerm[i]);
        }
    }

    if (paras.inputs.size() == 0 || paras.outputs.size() == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                        "inputs.size=%ld, outputs.size=%ld,",
                                        paras.inputs.size(), paras.outputs.size());
        return false;
    } 
    if (paras.inputs[0].tensor.size() == 0 || paras.outputs[0].tensor.size() == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                        "inputs tensor size=%ld, outputs tensor size=%ld,",
                                        paras.inputs[0].tensor.size(), paras.outputs[0].tensor.size());
        return false;
    }
    shapeInfo.inShape = paras.inputs[0].tensor[0].shape;
    shapeInfo.outShape = paras.outputs[0].tensor[0].shape;

    return true;
}

static bool AddShapePerm(const string& opType, const TeOpParas& paras, const CompilerInfo& info, ShapeInfo& shapeInfo) {
    OP_LOGD(opType.c_str(), "Entering AddShapePerm.");

    if (paras.inputs.size() == 0 || paras.outputs.size() == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                        "inputs.size=%ld, outputs.size=%ld,",
                                        paras.inputs.size(), paras.outputs.size());
        return false;
    }
    if (paras.inputs[0].tensor.size() == 0 || paras.outputs[0].tensor.size() == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                        "inputs tensor size=%ld, outputs tensor size=%ld,",
                                        paras.inputs[0].tensor.size(), paras.outputs[0].tensor.size());
        return false;
    }
    shapeInfo.inShape = paras.inputs[0].tensor[0].shape;
    shapeInfo.outShape = paras.outputs[0].tensor[0].shape;

    // for depthtospace
    if (opType == "DepthToSpace") {
        // check input and block
        if (shapeInfo.inShape.size() != 4) {
            VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                            "The length of input shape must be 4, but got %lu.",
                                            shapeInfo.inShape.size());
            return false;
        }
        if (shapeInfo.inShape[3] % (info.blockSize * info.blockSize) != 0) {
            VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                           "Depth size must be divisible by block size, but got depth[%ld], block[%ld].",
                                           shapeInfo.inShape[3], info.blockSize);
            return false;
        }
        // calc input and output shape and perm
        std::vector<int64_t> tmpVector;
        tmpVector.push_back(shapeInfo.inShape[0] * shapeInfo.inShape[1]);
        tmpVector.push_back(shapeInfo.inShape[2]);
        tmpVector.push_back(info.blockSize);
        tmpVector.push_back(shapeInfo.inShape[3] / info.blockSize);
        shapeInfo.inShape.clear();
        shapeInfo.inShape = tmpVector;
        shapeInfo.outShape.clear();
        shapeInfo.outShape.push_back(tmpVector[0]);
        shapeInfo.outShape.push_back(tmpVector[2]);
        shapeInfo.outShape.push_back(tmpVector[1]);
        shapeInfo.outShape.push_back(tmpVector[3]);
        shapeInfo.perm.clear();
        shapeInfo.perm.push_back(0);
        shapeInfo.perm.push_back(2);
        shapeInfo.perm.push_back(1);
        shapeInfo.perm.push_back(3);
    }

    // for spacetodepth
    if (opType == "SpaceToDepth") {
        // check input and block
        if (shapeInfo.inShape.size() != 4) {
            VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                            "The length of input shape must be 4, but got %lu,",
                                            shapeInfo.inShape.size());
            return false;
        }
        if (shapeInfo.inShape[1] % info.blockSize != 0) {
            VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                        "Height size must be divisible by block size, but got height[%ld], block[%ld].",
                                        shapeInfo.inShape[1], info.blockSize);
            return false;
        }
        if (shapeInfo.inShape[2] % info.blockSize != 0) {
            VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                       "Width size must be divisible by block size, but got width[%ld], block[%ld].",
                                       shapeInfo.inShape[2], info.blockSize);
            return false;
        }
        // calc input and output shape and perm
        std::vector<int64_t> tmpVector;
        tmpVector.push_back(shapeInfo.inShape[0] * shapeInfo.inShape[1] / info.blockSize);
        tmpVector.push_back(info.blockSize);
        tmpVector.push_back(shapeInfo.inShape[2] / info.blockSize);
        tmpVector.push_back(shapeInfo.inShape[3] * info.blockSize);
        shapeInfo.inShape.clear();
        shapeInfo.inShape = tmpVector;
        shapeInfo.outShape.clear();
        shapeInfo.outShape.push_back(tmpVector[0]);
        shapeInfo.outShape.push_back(tmpVector[2]);
        shapeInfo.outShape.push_back(tmpVector[1]);
        shapeInfo.outShape.push_back(tmpVector[3]);
        shapeInfo.perm.clear();
        shapeInfo.perm.push_back(0);
        shapeInfo.perm.push_back(2);
        shapeInfo.perm.push_back(1);
        shapeInfo.perm.push_back(3);
    }

    return true;
}

static bool SetElePerBlock(const CompilerInfo & compilerInfo, ShapeInfo & shapeInfo) {
    shapeInfo.elePerBlock = ElementNumPerBlock(compilerInfo.dType);
    shapeInfo.eleLenInBytes = SizeofDType(compilerInfo.dType);
    return true;
}

static bool CheckTensorShape(const string & opType,
                             const ShapeInfo & shapeInfo) {
    OP_LOGD(opType.c_str(), "Entering CheckTensorShape.");

    int64_t inDims = shapeInfo.inShape.size();
    int64_t outDims = shapeInfo.outShape.size();
    int64_t permDims = shapeInfo.perm.size();

    if (inDims < 1 || inDims != outDims || inDims != permDims) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                        "The dim of inputs is invalid, inDims = %ld, outDims = %ld, permDims = %ld",
                                        inDims, outDims, permDims);
        return false;
    }

    for (int64_t i = 0; i < inDims; i++) {
        if (shapeInfo.inShape[shapeInfo.perm[i]] != shapeInfo.outShape[i]) {
            VECTOR_INNER_ERR_REPORT_TILIING(opType, "The dim of inputs or outputs conflict with perm.");
            return false;
        }
    }

    for (int64_t i = 0; i < inDims; i++) {
        if (shapeInfo.inShape[i] <= 0 || shapeInfo.outShape[i] <= 0) {
            VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                            "Invalid shape, %ld, %ld, %ld",
                                            i, shapeInfo.inShape[i], shapeInfo.outShape[i]);
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
    for (size_t i = 0; i < reducedPerm.size(); i++) {
        sortedList.push_back({i, reducedPerm[i]});
    }

    sort(sortedList.begin(), sortedList.end(), [](pair<int32_t, int32_t> lhs, pair<int32_t, int32_t> rhs)->bool {
        return lhs.second < rhs.second;
    });

    for (size_t i = 0; i < sortedList.size(); i++) {
        reducedPermGrad[i] = (sortedList[i].first);
    }
}

static bool IsIdentical(const ShapeInfo & shapeInfo) {
    for(size_t i = 0; i < shapeInfo.reducedPerm.size(); i++) {
        if((size_t)shapeInfo.reducedPerm[i] != i) {
            return false;
        }
    }
    return true;
}

static bool IsSmallShape(const ShapeInfo & shapeInfo) {
    return shapeInfo.totalVolumeActual * shapeInfo.eleLenInBytes < SMALL_SHAPE_SIZE_THRESHOLD;
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

static bool IsAllOne(const ShapeInfo& shapeInfo) {
    for (auto it : shapeInfo.inShape) {
        if (it != 1) {
            return false;
        }
    }
    return true;
}

/*
 * If axis value is 1, then remove it.
 *
 *     inShape              perm                    reducedInShape       reducedPerm
 *     ---------------------------------------------------------------------------------
 *     Shape(4,1,6,1)       perm(0,1,2,3)           Shape(4,6)           perm(0,1)
 */
void RemoveAxis(ShapeInfo & shapeInfo) {
    int64_t dim = shapeInfo.inShape.size();
    if (dim == 1) {
        shapeInfo.reducedInShape = shapeInfo.inShape;
        shapeInfo.reducedPerm = shapeInfo.perm;
        shapeInfo.reducedOutShape = shapeInfo.outShape;
        return;
    }

    if (IsAllOne(shapeInfo)) {
        shapeInfo.reducedInShape.push_back(1);
        shapeInfo.reducedPerm.push_back(0);
        shapeInfo.reducedOutShape.push_back(1);
        shapeInfo.dim = 1;
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
            for (size_t j = 0 ; j < shapeInfo.perm.size(); j++) {
                if (shapeInfo.perm[j] == i) {
                    delPerm.push_back(shapeInfo.perm[j]);
                }
            }
        }
    }
    std::sort(delPerm.begin(), delPerm.end(), greater<int64_t>());

    for (int64_t i = 0; i < dim; i++) {
        bool delFlag = false;
        for(size_t j = 0; j < delPerm.size(); j++) {
            if (shapeInfo.perm[i] == delPerm[j]) {
                delFlag = true;
            }
        }
        if (delFlag == false) {
            newPerm.push_back(shapeInfo.perm[i]);
        }
    }

    for (size_t i = 0; i < delPerm.size(); i++) {
        for (size_t j = 0; j < newPerm.size(); j++) {
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
    for (size_t i = 0; i < newDimPosition.size(); ++i) {
        if (newDimPosition[i] >= 0) {
            int newPermIndex = newDimPosition[i];
            newPerm[dimIndex] = newPermIndex;
            newShape[dimIndex] = mergedShape[newPermIndex];
            dimIndex++;
        }
    }
    shapeInfo.reducedInShape = newShape;
    shapeInfo.reducedPerm.resize(newPerm.size());
    shapeInfo.lastAxisLen = shapeInfo.reducedInShape[shapeInfo.reducedInShape.size() - 1];
    shapeInfo.lastAxisBurstLen = (int64_t)ceil(shapeInfo.lastAxisLen * 1.0 /  shapeInfo.elePerBlock);
    CalcReducePermGrad(newPerm, shapeInfo.reducedPerm);
    CalcOutShape(shapeInfo);
}

//Since small shape with too much core will result in data less than one block, so use less core
void UpdateCoreNum(CompilerInfo &compilerInfo, ShapeInfo &shapeInfo) {
    if (shapeInfo.totalVolumeActual >= shapeInfo.elePerBlock * compilerInfo.coreNum) {
        compilerInfo.usedCoreNum = compilerInfo.coreNum;
        return;
    }
    if (shapeInfo.totalVolumeActual < shapeInfo.elePerBlock) {
        compilerInfo.usedCoreNum = 1;
    } else {
        compilerInfo.usedCoreNum = shapeInfo.totalVolumeActual / shapeInfo.elePerBlock;
    }
}

static void SetScenario(const string & opType,
                        CompilerInfo & compilerInfo,
                        ShapeInfo & shapeInfo) {
    OP_LOGD(opType.c_str(), "Entering SetScenario.");

    if(IsIdentical(shapeInfo)) {
        shapeInfo.identical = 1;
        shapeInfo.scenario = SCENARIO_0;
    } else if(compilerInfo.coreNum == 96) {
        shapeInfo.scenario = SCENARIO_8;
    } else if (IsSmallShape(shapeInfo)) {
        shapeInfo.scenario = SCENARIO_6;
        Reshape(shapeInfo);
        UpdateCoreNum(compilerInfo, shapeInfo);
    } else if (shapeInfo.elePerBlock == ELE_NUM_PER_BLOCK_B8) {
        shapeInfo.scenario = SCENARIO_2;
        shapeInfo.isLastAxisHuge = true;
        if (IsLastAxisJoinTranspose(shapeInfo)) {
            Reshape(shapeInfo);
        }
    } else if (IsLastAxisJoinTranspose(shapeInfo)) {
        shapeInfo.scenario = SCENARIO_7;
        shapeInfo.isLastAxisTranspose = true;
    } else {
        if (shapeInfo.lastAxisLen * shapeInfo.eleLenInBytes >= LAST_AXIS_HUGE_THRESHOLD) {
            shapeInfo.scenario = SCENARIO_3;
        }
        else if (shapeInfo.lastAxisLen % shapeInfo.elePerBlock == 0) {
            if (shapeInfo.lastAxisLen * shapeInfo.eleLenInBytes >= LAST_AXIS_BLOCK_ALIGN_LARGE_THRESHOLD) {
                shapeInfo.scenario = SCENARIO_1;
            }
            else if (shapeInfo.lastAxisLen * shapeInfo.eleLenInBytes > LAST_AXIS_BLOCK_ALIGN_N_BORROW_THRESHOLD) {
                shapeInfo.scenario = SCENARIO_2;
            } else {
                shapeInfo.scenario = SCENARIO_4;
            }
        } else {
            if (shapeInfo.lastAxisLen * shapeInfo.eleLenInBytes > LAST_AXIS_NOT_BLOCK_ALIGN_LARGE_THRESHOLD) {
                shapeInfo.scenario = SCENARIO_1;
            } else {
                shapeInfo.scenario = SCENARIO_4;
            }

        }
        shapeInfo.isLastAxisTranspose = false;
    }

    if ((shapeInfo.lastAxisLen % shapeInfo.elePerBlock) != 0) {
        shapeInfo.alignElement = shapeInfo.elePerBlock - (shapeInfo.lastAxisLen % shapeInfo.elePerBlock);
    }

    if (shapeInfo.lastAxisLen * shapeInfo.eleLenInBytes > LAST_AXIS_NOT_BLOCK_ALIGN_LARGE_THRESHOLD) {
        shapeInfo.isLastAxisHuge = true;
    }

    return;
}

/*
 *     inShape              perm                    reducedInShape     reducedOutShape    reducedPerm
 *     --------------------------------------------------------------------------------------------------
 *     Shape(4,5,6,7)       perm(1,0,2,3)           Shape(4,5,42)      Shape(5,4,42)      perm(1,0,2)
 *     Shape(2,3,4,5)       perm(0,2,3,1)           Shape(2,3,20)      Shape(2,20,3)      perm(0,2,1)
 *     Shape(2,3,4,5,6)     perm(0,4,1,2,3)         Shape(2,60,6)      Shape(2,6,60)      perm(0,2,1)
 *     Shape(2,3,4,5,6)     perm(2,3,4,0,1)         Shape(6,120)       Shape(120,6)       perm(1,0)
 *
 *     If last axis join transpose, the implementation now is add a axis with value 1.
 */
void ReduceAxis(const string & opType,
                CompilerInfo & compilerInfo,
                ShapeInfo & shapeInfo) {
    OP_LOGD(opType.c_str(), "Entering ReduceAxis.");

    SetElePerBlock(compilerInfo, shapeInfo);
    RemoveAxis(shapeInfo);
    MergeAxis(shapeInfo);

    shapeInfo.totalVolumeLogic = CalcTotalVolumeLogic(shapeInfo.reducedInShape);
    shapeInfo.totalVolumeActual = CalcTotalVolumeActual(shapeInfo.reducedInShape);
    shapeInfo.dim = shapeInfo.reducedInShape.size();

    SetScenario(opType, compilerInfo, shapeInfo);
    return;
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

static void PrintShapeInfo(const ShapeInfo &shapeInfo, string & logStr) {
    logStr += "\nscenario  in                  out                 perm            reducedIn           reducedOut     ";
    logStr += "     reducedPerm     dim  lastAxisLen  lastAxisBurstLen  alignElement\n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "-----------------------------------------------------------------\n";
    logStr += to_string(shapeInfo.scenario, 10);
    logStr += vec_to_string(shapeInfo.inShape, 20);
    logStr += vec_to_string(shapeInfo.outShape, 20);
    logStr += vec_to_string(shapeInfo.perm, 16);
    logStr += vec_to_string(shapeInfo.reducedInShape, 20);
    logStr += vec_to_string(shapeInfo.reducedOutShape, 20);
    logStr += vec_to_string(shapeInfo.reducedPerm, 16);
    logStr += to_string(shapeInfo.dim, 5);
    logStr += to_string(shapeInfo.lastAxisLen, 13);
    logStr += to_string(shapeInfo.lastAxisBurstLen, 18);
    logStr += to_string(shapeInfo.alignElement, 14);
    logStr += "\n\n";
}

static void PrintCompilerInfo(const CompilerInfo &compilerInfo, string &logStr) {
    logStr += "coreNum    usedCoreNum    ubSize    ubSizeCouldUse\n";
    logStr += "--------------------------------------------------\n";
    logStr += to_string(compilerInfo.coreNum, 11);
    logStr += to_string(compilerInfo.usedCoreNum, 15);
    logStr += to_string(compilerInfo.ubSize, 10);
    logStr += to_string(compilerInfo.ubSizeCouldUse,14);
    logStr += "\n\n";
}

static string PrintTilingInfoScenario0(const CompilerInfo & compilerInfo,
                                       const ShapeInfo & shapeInfo,
                                       const RuntimeInfo & runtimeInfo) {
    string logStr;
    PrintShapeInfo(shapeInfo, logStr);
    PrintCompilerInfo(compilerInfo, logStr);
    logStr += "base    eleNum    majorLoop    majorNum    tailNum    notAlignEle\n";
    logStr += "------------------------------------------------------------------\n";
    for (int64_t i = 0; i < compilerInfo.coreNum; i++) {
        logStr += to_string(runtimeInfo.infoPerCoreIdentical[i].base, 8);
        logStr += to_string(runtimeInfo.infoPerCoreIdentical[i].eleNum, 10);
        logStr += to_string(runtimeInfo.infoPerCoreIdentical[i].majorLoop, 13);
        logStr += to_string(runtimeInfo.infoPerCoreIdentical[i].majorNum, 12);
        logStr += to_string(runtimeInfo.infoPerCoreIdentical[i].tailNum, 11);
        logStr += to_string(runtimeInfo.infoPerCoreIdentical[i].notAlignEle, 14);
        logStr += "\n";
    }
    logStr += "\n\n";
    PrintScreen(logStr);
    return logStr;
}

static string PrintTilingInfoScenario1(const CompilerInfo & compilerInfo,
                                       const ShapeInfo & shapeInfo,
                                       const RuntimeInfo & runtimeInfo) {
    string logStr;
    PrintShapeInfo(shapeInfo, logStr);
    PrintCompilerInfo(compilerInfo, logStr);

    logStr += "srcStrideLogic  srcJumpStride                 dstJumpStride                 ";
    logStr += "dstJumpFactor                dstJumpFactorMod\n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "-----------------------------------------------------\n";
    logStr += to_string(runtimeInfo.srcStrideLogic, 16);
    logStr += arr_to_string(runtimeInfo.srcJumpStride, shapeInfo.dim - 1, 30);
    logStr += arr_to_string(runtimeInfo.dstJumpStride, shapeInfo.dim - 1, 30);
    logStr += arr_to_string(runtimeInfo.dstJumpFactor, shapeInfo.dim - 1, 30);
    logStr += arr_to_string(runtimeInfo.dstJumpFactorMod, shapeInfo.dim - 1, 30);
    logStr += "\n\n";

    logStr += "base    num    initTuple\n";
    logStr += "--------------------------\n";
    for (int64_t i = 0; i < compilerInfo.coreNum; i++) {
        logStr += to_string(runtimeInfo.infoPerCoreLastAxisNT[i].base, 8);
        logStr += to_string(runtimeInfo.infoPerCoreLastAxisNT[i].num, 7);
        logStr += arr_to_string(runtimeInfo.infoPerCoreLastAxisNT[i].initTuple, shapeInfo.dim - 1);
        logStr += "\n";
    }
    logStr += "\n\n";
    PrintScreen(logStr);
    return logStr;
}

static string PrintTilingInfoScenario2(const CompilerInfo & compilerInfo,
                                       const ShapeInfo & shapeInfo,
                                       const RuntimeInfo & runtimeInfo) {
    string logStr;
    PrintShapeInfo(shapeInfo, logStr);
    PrintCompilerInfo(compilerInfo, logStr);

    logStr += "backNum  skipEle  srcStrideLogic  srcJumpStride                 dstJumpStride                 ";
    logStr += "dstJumpFactor                 dstJumpFactorMod\n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "-------------------------------------------------\n";
    logStr += to_string(runtimeInfo.backNum, 9);
    logStr += to_string(runtimeInfo.skipEle, 9);
    logStr += to_string(runtimeInfo.srcStrideLogic, 16);
    logStr += arr_to_string(runtimeInfo.srcJumpStride, shapeInfo.dim - 1, 30);
    logStr += arr_to_string(runtimeInfo.dstJumpStride, shapeInfo.dim - 1, 30);
    logStr += arr_to_string(runtimeInfo.dstJumpFactor, shapeInfo.dim - 1, 30);
    logStr += arr_to_string(runtimeInfo.dstJumpFactorMod, shapeInfo.dim - 1);
    logStr += "\n\n";

    logStr += "base        num        initTuple                     headMajorLoop  headMajorNum  headTailNum  ";
    logStr += "bodyLoopNum  bodymajorLoop  bodyMajorNum  bodyTailNum  tailMajorLoop  tailMajorNum  tailTailNum\n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "-------------------------------------------------------------------------------------------------\n";
    for (int64_t i = 0; i < compilerInfo.coreNum; i++) {
        const InfoPerCoreLastAxisNT & infoPerCore = runtimeInfo.infoPerCoreLastAxisNT[i];
        const LastAxisNTLoopInfo & loopInfo = infoPerCore.loopInfo;
        logStr += to_string(runtimeInfo.infoPerCoreLastAxisNT[i].base, 12);
        logStr += to_string(runtimeInfo.infoPerCoreLastAxisNT[i].num, 11);
        logStr += arr_to_string(runtimeInfo.infoPerCoreLastAxisNT[i].initTuple, shapeInfo.dim - 1, 30);
        logStr += to_string(loopInfo.headMajorLoop, 15);
        logStr += to_string(loopInfo.headMajorNum, 14);
        logStr += to_string(loopInfo.headTailNum, 13);
        logStr += to_string(loopInfo.bodyLoopNum, 13);
        logStr += to_string(loopInfo.bodyMajorLoop, 15);
        logStr += to_string(loopInfo.bodyMajorNum, 14);
        logStr += to_string(loopInfo.bodyTailNum, 13);
        logStr += to_string(loopInfo.tailMajorLoop, 15);
        logStr += to_string(loopInfo.tailMajorNum, 14);
        logStr += to_string(loopInfo.tailTailNum, 13);
        logStr += "\n";
    }
    PrintScreen(logStr);
    return logStr;
}

static string PrintTilingInfoScenario3(const CompilerInfo & compilerInfo,
                                       const ShapeInfo & shapeInfo,
                                       const RuntimeInfo & runtimeInfo) {
    string logStr;
    PrintShapeInfo(shapeInfo, logStr);
    PrintCompilerInfo(compilerInfo, logStr);

    logStr += "srcStrideLogic  srcJumpStride                 dstJumpStride                 ";
    logStr += "dstJumpFactor                 majorLoopNum  majorBlocks  tailBlocks  ";
    logStr += "backEle\n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "-------------------------------------------------------\n";
    logStr += to_string(runtimeInfo.srcStrideLogic, 16);
    logStr += arr_to_string(runtimeInfo.srcJumpStride, shapeInfo.dim - 1, 30);
    logStr += arr_to_string(runtimeInfo.dstJumpStride, shapeInfo.dim - 1, 30);
    logStr += arr_to_string(runtimeInfo.dstJumpFactor, shapeInfo.dim - 1, 30);
    logStr += to_string(runtimeInfo.hugeInfo.majorLoopNum, 14);
    logStr += to_string(runtimeInfo.hugeInfo.majorBlocks, 13);
    logStr += to_string(runtimeInfo.hugeInfo.tailBlocks, 12);
    logStr += to_string(runtimeInfo.hugeInfo.backEle, 7);
    logStr += "\n\n";

    logStr += "base    num    initTuple\n";
    logStr += "--------------------------\n";
    for (int64_t i = 0; i < compilerInfo.coreNum; i++) {
        logStr += to_string(runtimeInfo.infoPerCoreLastAxisNT[i].base, 8);
        logStr += to_string(runtimeInfo.infoPerCoreLastAxisNT[i].num, 7);
        logStr += arr_to_string(runtimeInfo.infoPerCoreLastAxisNT[i].initTuple, shapeInfo.dim - 1);
        logStr += "\n";
    }
    logStr += "\n\n";
    PrintScreen(logStr);
    return logStr;
}

static string PrintTilingInfoScenario4(const CompilerInfo & compilerInfo,
                                       const ShapeInfo & shapeInfo,
                                       const RuntimeInfo & runtimeInfo) {
    string logStr;
    const BorrowInfo& borrowInfo = runtimeInfo.borrowInfo;
    PrintShapeInfo(shapeInfo, logStr);
    PrintCompilerInfo(compilerInfo, logStr);

    logStr += "srcNum dstNum dupAxis srcVol dstVol srcIndexIn.i dstIndexIn.i srcIndexIn.o dstIndexIn.o ";
    logStr += "otherIndex.i srcIndexInNoDup.i dstIndexInNoDup.i srcIndexInNoDup.o dstIndexInNoDup.o\n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "----------------------------------------------------------------------------------------------------\n";
    logStr += to_string(borrowInfo.srcNum, 7);
    logStr += to_string(borrowInfo.dstNum, 7);
    logStr += to_string(borrowInfo.dupAxis, 8);
    logStr += to_string(borrowInfo.srcVol, 7);
    logStr += to_string(borrowInfo.dstVol, 7);
    int64_t arr[TRANSPOSE_MAX_AXIS_NUM];
    for (int64_t i = 0; i < borrowInfo.srcNum; i++) {
        arr[i] = borrowInfo.srcIndexIn[i].idx_in;
    }
    logStr += arr_to_string(arr, borrowInfo.srcNum, 13);
    
    for (int64_t i = 0; i < borrowInfo.dstNum; i++) {
        arr[i] = borrowInfo.dstIndexIn[i].idx_in;
    }
    logStr += arr_to_string(arr, borrowInfo.dstNum, 13);

    for (int64_t i = 0; i < borrowInfo.srcNum; i++) {
        arr[i] = borrowInfo.srcIndexIn[i].idx_out;
    }
    logStr += arr_to_string(arr, borrowInfo.srcNum, 13);

    for (int64_t i = 0; i < borrowInfo.dstNum; i++) {
        arr[i] = borrowInfo.dstIndexIn[i].idx_out;
    }
    logStr += arr_to_string(arr, borrowInfo.dstNum, 13);

    for (int64_t i = 0; i < borrowInfo.otherNum; i++) {
        arr[i] = borrowInfo.otherIndex[i].idx_in;
    }
    logStr += arr_to_string(arr, borrowInfo.otherNum, 13);

    for (int64_t i = 0; i < borrowInfo.srcNumNoDup; i++) {
        arr[i] = borrowInfo.srcIndexInNoDup[i].idx_in;
    }
    logStr += arr_to_string(arr, borrowInfo.srcNumNoDup, 18);

    for (int64_t i = 0; i < borrowInfo.dstNumNoDup; i++) {
        arr[i] = borrowInfo.dstIndexInNoDup[i].idx_in;
    }
    logStr += arr_to_string(arr, borrowInfo.dstNumNoDup, 18);

    for (int64_t i = 0; i < borrowInfo.srcNumNoDup; i++) {
        arr[i] = borrowInfo.srcIndexInNoDup[i].idx_out;
    }
    logStr += arr_to_string(arr, borrowInfo.srcNumNoDup, 18);

    for (int64_t i = 0; i < borrowInfo.dstNumNoDup; i++) {
        arr[i] = borrowInfo.dstIndexInNoDup[i].idx_out;
    }
    logStr += arr_to_string(arr, borrowInfo.dstNumNoDup, 16);

    logStr += "\n\n";
    logStr += "srcTail dstTail ubPermRaw    ubPerm      srcAxisPerm dstAxisPerm axisPerm\n";
    logStr += "------------------------------------------------------------------------------\n";
    logStr += to_string(borrowInfo.srcIndexIn[0].tail, 8);
    logStr += to_string(borrowInfo.dstIndexOut[borrowInfo.dstNum - 1].tail, 8);
    logStr += arr_to_string(borrowInfo.ubPermRaw, borrowInfo.ubPermNum, 13);
    logStr += arr_to_string(borrowInfo.ubPerm, borrowInfo.ubPermNum, 12);
    logStr += hex_perm_to_string(borrowInfo.srcAxisPerm, 12);
    logStr += hex_perm_to_string(borrowInfo.dstAxisPerm, 12);
    logStr += hex_perm_to_string(borrowInfo.axisPerm, 12);
    logStr += "\n\n";

    logStr += "majorDstLoop_in tailDstLoop_in majorSrcLoop_out tailSrcLoop_out majorBurstLen_in ";
    logStr += "tailBurstLen_in majorBurstLen_out tailBurstLen_out\n";
    logStr += "--------------------------------------------------------------------------------";
    logStr += "---------------------------------------------------------------------\n";
    logStr += to_string(borrowInfo.majorDstLoop_in, 16);
    logStr += to_string(borrowInfo.tailDstLoop_in, 15);
    logStr += to_string(borrowInfo.majorSrcLoop_out, 17);
    logStr += to_string(borrowInfo.tailSrcLoop_out, 16);
    logStr += to_string(borrowInfo.majorBurstLen_in, 17);
    logStr += to_string(borrowInfo.tailBurstLen_in, 16);
    logStr += to_string(borrowInfo.majorBurstLen_out, 18);
    logStr += to_string(borrowInfo.tailBurstLen_out, 17);
    logStr += "\n\n";
    logStr += "majorInEle tailInEle majorInTailEle tailInTailEle ";
    logStr += "majorOutEle tailOutEle majorOutTailEle tailOutTailEle\n";
    logStr += "--------------------------------------------------------------------------------";
    logStr += "---------------------------------------------------------------------\n";
    logStr += to_string(borrowInfo.majorInEle, 11);
    logStr += to_string(borrowInfo.tailInEle, 10);
    logStr += to_string(borrowInfo.majorInTailEle, 15);
    logStr += to_string(borrowInfo.tailInTailEle, 14);
    logStr += to_string(borrowInfo.majorOutEle, 12);
    logStr += to_string(borrowInfo.tailOutEle, 11);
    logStr += to_string(borrowInfo.majorOutTailEle, 16);
    logStr += to_string(borrowInfo.tailOutTailEle, 15);
    logStr += "\n\n";
    logStr += "loop1 repeat1 srcStride1 dstStride1 burstLen1 srcOffset1 dstOffset1 ";
    logStr += "loop2 repeat2 srcStride2 dstStride2 burstLen2 srcOffset2 dstOffset2 ";
    logStr += "loop3 repeat3 srcStride3 dstStride3 burstLen3 srcOffset3 dstOffset3 \n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "----------------------------------------------------------------------------------------------------\n";
    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        const LRSB* lrsb = borrowInfo.lrsb[i];
        for (int j = 0; j < UB_REORDER_LOOP; j++) {
            logStr += to_string(lrsb[j].loop, 6);
            logStr += to_string(lrsb[j].repeat, 8);
            logStr += to_string(lrsb[j].srcStride, 11);
            logStr += to_string(lrsb[j].dstStride, 11);
            logStr += to_string(lrsb[j].burstLen, 10);
            logStr += to_string(lrsb[j].srcOffset, 11);
            logStr += to_string(lrsb[j].dstOffset, 11);
        }
        logStr += "\n";
    }
    logStr += "\n";

    logStr += "srcJumpFactorLogic_in dstJumpFactorLogic_in srcStep dstStep dstFactorCopyIn dstStrideCopyIn      ";
    logStr += "srcFactorCopyOut srcStrideCopyOut      srcJumpFactorMod_in dstJumpFactorMod_in\n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "-----------------------------------------------------------------------------------\n";
    logStr += to_string(borrowInfo.srcJumpFactorLogic_in, 22);
    logStr += to_string(borrowInfo.dstJumpFactorLogic_in, 22);
    logStr += to_string(borrowInfo.srcIndexIn[0].step, 8);
    logStr += to_string(borrowInfo.dstIndexOut[borrowInfo.dstNum - 1].step, 8);
    logStr += arr_to_string(borrowInfo.dstFactorCopyIn, borrowInfo.dstNumNoDup, 16);
    logStr += arr_to_string(borrowInfo.dstStrideCopyIn, borrowInfo.dstNumNoDup, 21);
    logStr += arr_to_string(borrowInfo.srcFactorCopyOut, borrowInfo.srcNumNoDup, 17);
    logStr += arr_to_string(borrowInfo.srcStrideCopyOut, borrowInfo.srcNumNoDup, 22);
    logStr += to_string(borrowInfo.srcJumpFactorMod_in, 20);
    logStr += to_string(borrowInfo.dstJumpFactorMod_in, 20);
    logStr += "\n\n";

    logStr += "flag      idxIn idxOut loop    step    tail    intact \n";
    logStr += "-------------------------------------------------------------------------------------------\n";
    for(int i = 0; i < borrowInfo.srcNum; i++) {
        logStr += "src       ";
        logStr += to_string(borrowInfo.srcIndexIn[i].idx_in, 6);
        logStr += to_string(borrowInfo.srcIndexIn[i].idx_out,7);
        logStr += to_string(borrowInfo.srcIndexIn[i].loop, 8);
        logStr += to_string(borrowInfo.srcIndexIn[i].step, 8);
        logStr += to_string(borrowInfo.srcIndexIn[i].tail, 8);
        logStr += to_string(borrowInfo.srcIndexIn[i].intact, 7);
        logStr += "\n";
    }
    for(int i = 0; i < borrowInfo.dstNum; i++) {
        logStr += "dst       ";
        logStr += to_string(borrowInfo.dstIndexIn[i].idx_in, 6);
        logStr += to_string(borrowInfo.dstIndexIn[i].idx_out,7);
        logStr += to_string(borrowInfo.dstIndexIn[i].loop, 8);
        logStr += to_string(borrowInfo.dstIndexIn[i].step, 8);
        logStr += to_string(borrowInfo.dstIndexIn[i].tail, 8);
        logStr += to_string(borrowInfo.dstIndexIn[i].intact, 7);
        logStr += "\n";
    }
    for(int i = 0; i < borrowInfo.srcNumNoDup; i++) {
        logStr += "srcNoDup  ";
        logStr += to_string(borrowInfo.srcIndexInNoDup[i].idx_in, 6);
        logStr += to_string(borrowInfo.srcIndexInNoDup[i].idx_out,7);
        logStr += to_string(borrowInfo.srcIndexInNoDup[i].loop, 8);
        logStr += to_string(borrowInfo.srcIndexInNoDup[i].step, 8);
        logStr += to_string(borrowInfo.srcIndexInNoDup[i].tail, 8);
        logStr += to_string(borrowInfo.srcIndexInNoDup[i].intact, 7);
        logStr += "\n";
    }
    for(int i = 0; i < borrowInfo.dstNumNoDup; i++) {
        logStr += "dstNoDup  ";
        logStr += to_string(borrowInfo.dstIndexInNoDup[i].idx_in, 6);
        logStr += to_string(borrowInfo.dstIndexInNoDup[i].idx_out,7);
        logStr += to_string(borrowInfo.dstIndexInNoDup[i].loop, 8);
        logStr += to_string(borrowInfo.dstIndexInNoDup[i].step, 8);
        logStr += to_string(borrowInfo.dstIndexInNoDup[i].tail, 8);
        logStr += to_string(borrowInfo.dstIndexInNoDup[i].intact, 7);
        logStr += "\n";
    }
    logStr += "\n";

    logStr += "flag      idxIn idxOut loop    step    tail    intact \n";
    logStr += "-------------------------------------------------------------------------------------------\n";
    for(int i = 0; i < borrowInfo.srcNum; i++) {
        logStr += "src       ";
        logStr += to_string(borrowInfo.srcIndexOut[i].idx_in, 6);
        logStr += to_string(borrowInfo.srcIndexOut[i].idx_out,7);
        logStr += to_string(borrowInfo.srcIndexOut[i].loop, 8);
        logStr += to_string(borrowInfo.srcIndexOut[i].step, 8);
        logStr += to_string(borrowInfo.srcIndexOut[i].tail, 8);
        logStr += to_string(borrowInfo.srcIndexOut[i].intact, 7);
        logStr += "\n";
    }
    for(int i = 0; i < borrowInfo.dstNum; i++) {
        logStr += "dst       ";
        logStr += to_string(borrowInfo.dstIndexOut[i].idx_in, 6);
        logStr += to_string(borrowInfo.dstIndexOut[i].idx_out,7);
        logStr += to_string(borrowInfo.dstIndexOut[i].loop, 8);
        logStr += to_string(borrowInfo.dstIndexOut[i].step, 8);
        logStr += to_string(borrowInfo.dstIndexOut[i].tail, 8);
        logStr += to_string(borrowInfo.dstIndexOut[i].intact, 7);
        logStr += "\n";
    }
    for(int i = 0; i < borrowInfo.srcNumNoDup; i++) {
        logStr += "srcNoDup  ";
        logStr += to_string(borrowInfo.srcIndexOutNoDup[i].idx_in, 6);
        logStr += to_string(borrowInfo.srcIndexOutNoDup[i].idx_out,7);
        logStr += to_string(borrowInfo.srcIndexOutNoDup[i].loop, 8);
        logStr += to_string(borrowInfo.srcIndexOutNoDup[i].step, 8);
        logStr += to_string(borrowInfo.srcIndexOutNoDup[i].tail, 8);
        logStr += to_string(borrowInfo.srcIndexOutNoDup[i].intact, 7);
        logStr += "\n";
    }
    for(int i = 0; i < borrowInfo.dstNumNoDup; i++) {
        logStr += "dstNoDup  ";
        logStr += to_string(borrowInfo.dstIndexOutNoDup[i].idx_in, 6);
        logStr += to_string(borrowInfo.dstIndexOutNoDup[i].idx_out,7);
        logStr += to_string(borrowInfo.dstIndexOutNoDup[i].loop, 8);
        logStr += to_string(borrowInfo.dstIndexOutNoDup[i].step, 8);
        logStr += to_string(borrowInfo.dstIndexOutNoDup[i].tail, 8);
        logStr += to_string(borrowInfo.dstIndexOutNoDup[i].intact, 7);
        logStr += "\n";
    }
    logStr += "\n";

    logStr += "otherJumpFactor_in otherJumpStride_in otherJumpStride_out otherJumpFactorMod_in\n";
    logStr += "----------------------------------------------------------------------------------------------------\n";
    logStr += arr_to_string(borrowInfo.otherJumpFactor_in, borrowInfo.otherNum, 19);
    logStr += arr_to_string(borrowInfo.otherJumpStride_in, borrowInfo.otherNum, 19);
    logStr += arr_to_string(borrowInfo.otherJumpStride_out, borrowInfo.otherNum, 20);
    logStr += arr_to_string(borrowInfo.otherJumpFactorMod_in, borrowInfo.otherNum, 22);
    logStr += "\n\n";

    logStr += "core_id loopPerCore srcTupleLogic dstTupleLogic otherTuple    srcTuple       dstTuple       \n";
    logStr += "-------------------------------------------------------------------------------------------\n";
    for (int64_t i = 0; i < compilerInfo.coreNum; i++) {
        logStr += to_string(i, 8);
        logStr += to_string(borrowInfo.loopPerCore[i], 12);
        logStr += to_string(borrowInfo.srcAxis_in[i].initTupleLogic, 14);
        logStr += to_string(borrowInfo.dstAxis_in[i].initTupleLogic, 14);
        logStr += arr_to_string(borrowInfo.otherAxis_in[i].initTuple, borrowInfo.otherNum, 14);
        logStr += arr_to_string(borrowInfo.srcAxis_in[i].initTuple, borrowInfo.srcNumNoDup, 15);
        logStr += arr_to_string(borrowInfo.dstAxis_in[i].initTuple, borrowInfo.dstNumNoDup, 15);
        logStr += "\n";
    }
    logStr += "\n\n";
    PrintScreen(logStr);
    return logStr;
}

static string PrintTilingInfoScenario6(const CompilerInfo & compilerInfo,
                                       const ShapeInfo & shapeInfo,
                                       const RuntimeInfo & runtimeInfo) {
    return PrintTilingInfoScenario2(compilerInfo, shapeInfo, runtimeInfo);
}

static string PrintTilingInfoScenario7(const CompilerInfo & compilerInfo,
                            const ShapeInfo & shapeInfo,
                            const RuntimeInfo & runtimeInfo) {
    string logStr;
    PrintShapeInfo(shapeInfo, logStr);
    PrintCompilerInfo(compilerInfo, logStr);

    logStr += "n                   col                 row                 nFactor  colFactor  rowFactor  ";
    logStr += "priority  modelName\n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "--------------------------\n";
    priority_queue<shared_ptr<TilingModel>, vector<shared_ptr<TilingModel>>, TMCompare> pqtm = runtimeInfo.pqtm;
    while (!pqtm.empty()) {
        shared_ptr<TilingModel> tm = pqtm.top();
        pqtm.pop();
        logStr += vec_to_string(tm->ncr.n, 20);
        logStr += vec_to_string(tm->ncr.col, 20);
        logStr += vec_to_string(tm->ncr.row, 20);
        logStr += to_string(tm->sp.nFactor, 9);
        logStr += to_string(tm->sp.colFactor, 11);
        logStr += to_string(tm->sp.rowFactor, 11);
        logStr += to_string(tm->priority, 10);
        logStr += tm->modelName;
        logStr += "\n";
    }
    logStr += "\n\n";

    logStr += "nJumpAxisNum  srcJumpAxisNum  dstJumpAxisNum  nJumpFactor         nJumpStride         ";
    logStr += "srcJumpFactor       srcJumpStride       dstJumpFactor       dstJumpStride       rPartVol\n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "----------------------------------------------------------------------------------------------\n";
    logStr += to_string(runtimeInfo.nJumpAxisNum, 14);
    logStr += to_string(runtimeInfo.srcJumpAxisNum, 16);
    logStr += to_string(runtimeInfo.dstJumpAxisNum, 16);
    logStr += arr_to_string(runtimeInfo.nJumpFactor, runtimeInfo.nJumpAxisNum, 20);
    logStr += arr_to_string(runtimeInfo.nJumpStride, runtimeInfo.nJumpAxisNum, 20);
    logStr += arr_to_string(runtimeInfo.srcJumpFactor,runtimeInfo.srcJumpAxisNum, 20);
    logStr += arr_to_string(runtimeInfo.srcJumpStride, runtimeInfo.srcJumpAxisNum, 20);
    logStr += arr_to_string(runtimeInfo.dstJumpFactor,runtimeInfo.dstJumpAxisNum, 20);
    logStr += arr_to_string(runtimeInfo.dstJumpStride, runtimeInfo.dstJumpAxisNum, 20);
    logStr += to_string(runtimeInfo.rPartVol, 16);
    logStr += "\n\n";

    logStr += "loopN  nOffsetActual  initNTuple          colPerMC  loopMC  colTC  colOffset  bsl  ";
    logStr += "initDstTuple        tailDstTuple        rowPerMR  loopMR  rowTR  rowOffset  bsu  ";
    logStr += "initSrcTuple        tailSrcTuple\n";
    logStr += "------------------------------------------------------------------------------------------------------";
    logStr += "----------------------------------------------------------------------------------------------------\n";
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        logStr += to_string(runtimeInfo.infoPerCore[i].infoN.loopOnN, 7);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoN.nOffsetActual, 15);
        logStr += vec_to_string(runtimeInfo.infoPerCore[i].infoN.initNTuple, 20);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoCol.colPerMC, 10);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoCol.loopOnMC, 8);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoCol.colTC, 7);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoCol.colOffset, 11);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoCol.backStepLeft, 5);
        logStr += vec_to_string(runtimeInfo.infoPerCore[i].infoCol.initDstTuple, 20);
        logStr += vec_to_string(runtimeInfo.infoPerCore[i].infoCol.tailDstTuple, 20);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoRow.rowPerMR, 10);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoRow.loopOnMR, 8);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoRow.rowTR, 7);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoRow.rowOffset, 11);
        logStr += to_string(runtimeInfo.infoPerCore[i].infoRow.backStepUp, 5);
        logStr += vec_to_string(runtimeInfo.infoPerCore[i].infoRow.initSrcTuple, 20);
        logStr += vec_to_string(runtimeInfo.infoPerCore[i].infoRow.tailSrcTuple, 20);
        logStr += "\n";
    }
    logStr += "\n\n";
    PrintScreen(logStr);
    return logStr;
}

static void CalcTuple(const CompilerInfo &compilerInfo, const ShapeInfo &shapeInfo, RuntimeInfo &runtimeInfo) {
    int64_t dim = shapeInfo.dim;
    int64_t vol = 1;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v1Num = 0;
    int64_t v2Num = 0;
    InfoPerCoreLastAxisNT infoPerCore;
    for (int64_t i = dim - 3; i >= 0; i--) {
        runtimeInfo.dstJumpFactorMod[i] = shapeInfo.reducedOutShape[i + 1] * runtimeInfo.dstJumpFactorMod[i + 1];
    }
    for (int64_t i = 0; i < dim - 1; i++) {
        vol = vol * shapeInfo.reducedOutShape[i];
    }

    if(shapeInfo.scenario == SCENARIO_6) {
        SplitEvenly(compilerInfo.usedCoreNum, vol, v1Num, v2Num, v1, v2);
    } else {
        SplitEvenly(compilerInfo.coreNum, vol, v1Num, v2Num, v1, v2);
    }

    vol = 0;
    for (int64_t i = 0; i < v1Num; i++) {
        infoPerCore.base = vol;
        infoPerCore.num = v1;
        for (int64_t j = 0; j < dim - 1; j++) {
            infoPerCore.initTuple[j] = (vol / runtimeInfo.dstJumpFactorMod[j]) % shapeInfo.reducedOutShape[j];
        }
        ReverseArray(infoPerCore.initTuple, dim - 1); //since in ops, left tuple change faster than right
        runtimeInfo.infoPerCoreLastAxisNT.emplace_back(infoPerCore);
        vol += v1;
    }
    for (int64_t i = 0; i < v2Num; i++) {
        infoPerCore.base = vol;
        infoPerCore.num = v2;
        for (int64_t j = 0; j < dim - 1; j++) {
            infoPerCore.initTuple[j] = (vol / runtimeInfo.dstJumpFactorMod[j]) % shapeInfo.reducedOutShape[j];
        }
        ReverseArray(infoPerCore.initTuple, dim - 1); //since in ops, left tuple change faster than right
        runtimeInfo.infoPerCoreLastAxisNT.emplace_back(infoPerCore);
        vol += v2;
    }
    for(int64_t i = v1Num + v2Num; i < compilerInfo.coreNum; i++) {
        //compilerInfo.usedCoreNum != 0, and gt compilerInfo.coreNum
        InfoPerCoreLastAxisNT infoPerCore;
        runtimeInfo.infoPerCoreLastAxisNT.emplace_back(infoPerCore);
    }
    ReverseArray(runtimeInfo.dstJumpFactorMod, dim - 1);
}

static void CalcHugeInfo(const ShapeInfo &shapeInfo, RuntimeInfo &runtimeInfo) {
    runtimeInfo.hugeInfo.majorBlocks =  HUGE_BLOCKS_UNIT;
    runtimeInfo.hugeInfo.majorLoopNum = shapeInfo.lastAxisBurstLen / HUGE_BLOCKS_UNIT;
    runtimeInfo.hugeInfo.tailBlocks = shapeInfo.lastAxisBurstLen - runtimeInfo.hugeInfo.majorLoopNum * HUGE_BLOCKS_UNIT;
    runtimeInfo.hugeInfo.backEle = shapeInfo.alignElement;
}

static void CalcLoopInfo(int64_t &majorLoop, int64_t &majorNum, int64_t &tailNum,
                         int64_t toSplitNum, int64_t burstLen, int64_t maxUbBlockNum) {
    majorNum = maxUbBlockNum / burstLen; // maxUbBlockNum always gt burstLen
    majorLoop = toSplitNum / majorNum;
    tailNum = toSplitNum - majorLoop * majorNum;
}

static void CalcLoopInfo(const CompilerInfo &compilerInfo, const ShapeInfo &shapeInfo, RuntimeInfo &runtimeInfo) {
    CalcUbReorderFactor(compilerInfo, shapeInfo, runtimeInfo);
    for(int64_t i = 0; i < compilerInfo.coreNum; i++) {
        InfoPerCoreLastAxisNT & infoPerCore = runtimeInfo.infoPerCoreLastAxisNT[i];
        LastAxisNTLoopInfo & loopInfo = infoPerCore.loopInfo;
        int64_t dim = shapeInfo.dim;
        int64_t tuple0 = infoPerCore.initTuple[0];
        int64_t num = infoPerCore.num - runtimeInfo.backNum;
        int64_t levelOneSize = shapeInfo.reducedOutShape[dim - 2];
        int64_t ubBlockNum = ACCU_BLOCK_SIZE; //128 = 4KB; 200 = 6.4KB
        int64_t left = num;

        if (infoPerCore.num == 0) {
            return;
        }

        //0: stride detect, if stride is huge, copy in one by one
        if (IsStrideTooHuge(shapeInfo, runtimeInfo)) {
            ubBlockNum = shapeInfo.lastAxisBurstLen;
        }

        //1: workspace detect, if not block aligned and workspace not work , copy in one by one
        if (shapeInfo.lastAxisLen % shapeInfo.elePerBlock != 0) {
            if (runtimeInfo.byWorkspace == 0) {
                ubBlockNum = shapeInfo.lastAxisBurstLen;
            }
        }

        //2: head loop info
        if (tuple0 != 0) {
            if (levelOneSize - tuple0 < num) {
                CalcLoopInfo(loopInfo.headMajorLoop, loopInfo.headMajorNum, loopInfo.headTailNum,
                             levelOneSize - tuple0, shapeInfo.lastAxisBurstLen, ubBlockNum);
                left = num - (levelOneSize - tuple0);
            } else {
                CalcLoopInfo(loopInfo.headMajorLoop, loopInfo.headMajorNum, loopInfo.headTailNum,
                             num, shapeInfo.lastAxisBurstLen, ubBlockNum);
                left = 0;
            }
        }

        //3: body loop info
        if (left > 0 && left >= levelOneSize) {
            CalcLoopInfo(loopInfo.bodyMajorLoop, loopInfo.bodyMajorNum, loopInfo.bodyTailNum,
                         levelOneSize, shapeInfo.lastAxisBurstLen, ubBlockNum);
            loopInfo.bodyLoopNum = left / levelOneSize;
            left -= loopInfo.bodyLoopNum * levelOneSize;
        }

        //4: tail loop info
        if (left > 0) {
            CalcLoopInfo(loopInfo.tailMajorLoop, loopInfo.tailMajorNum, loopInfo.tailTailNum,
                         left, shapeInfo.lastAxisBurstLen, ubBlockNum);
        }
    }
}

static int64_t CalcStride(const vector<int64_t> & shape, int64_t dim, int index) {
    int64_t vol = 1;
    for (int i = index + 1; i < dim; i++) {
        vol *= shape[i];
    }
    return vol;
}

/*
 *reoder the stride by dst shape.
 *
 *            0  1  2      3  4                                    0   3  2      1  4
 *         -----------------------                              ---------------------------
 * inShape = (6, 4, 12800, 8, 200), perm =(0,3,2,1,4), outShape = (6,  8, 12800, 4, 200)
 *
 *                           0              1          2      3
 *                        ---------------------------------------
 * at first,      stirde = (4*12800*8*200, 12800*8*200, 8*200, 200)
 *
 *                          1               2          3      0
 *                         ---------------------------------------
 * after reorder, stirde = (12800*8*200, 8*200,        200,   4*12800*8*200)
 *
 */
static void ReorderSrcStride(const vector<int64_t> & perm, int64_t * stride, int64_t size) {
    int64_t index = 0;
    int64_t temp[TRANSPOSE_MAX_AXIS_NUM];
    for(int64_t i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
        temp[i] = stride[i];
    }
    for (int64_t i = size - 1; i >= 0; i--) {
        stride[index++] = temp[perm[i]];
    }
}

bool TilingDataScenario0(const CompilerInfo & compilerInfo,
                         const ShapeInfo & shapeInfo,
                         RuntimeInfo & runtimeInfo) {
    int64_t perCoreSize1 = 0;
    int64_t perCoreSize2 = 0;
    int64_t p1Num = 0;
    int64_t p2Num = 0;
    int64_t base = 0;
    int64_t blocks = 0;

    SplitEvenly(compilerInfo.coreNum,
                shapeInfo.totalVolumeActual,
                p1Num,
                p2Num,
                perCoreSize1,
                perCoreSize2,
                shapeInfo.elePerBlock);

    vector<IdenticalInfo> & identicalInfo = runtimeInfo.infoPerCoreIdentical;
    identicalInfo.resize(compilerInfo.coreNum);

    for (int64_t i = 0; i < p1Num; i++) {
        IdenticalInfo & info = identicalInfo[i];
        blocks = perCoreSize1 / shapeInfo.elePerBlock;
        info.base = base;
        info.eleNum = perCoreSize1;
        info.majorLoop =  blocks / ACCU_BLOCK_SIZE_IDENTICAL;
        info.majorNum = ACCU_BLOCK_SIZE_IDENTICAL;
        info.tailNum = blocks - info.majorLoop * ACCU_BLOCK_SIZE_IDENTICAL;
        info.notAlignEle = perCoreSize1 - blocks * shapeInfo.elePerBlock;
        base += perCoreSize1;
    }

    for (int64_t i = 0; i < p2Num; i++) {
        IdenticalInfo & info = identicalInfo[i + p1Num];
        blocks = perCoreSize2 / shapeInfo.elePerBlock;
        info.base = base;
        info.eleNum = perCoreSize2;
        info.majorLoop =  blocks / ACCU_BLOCK_SIZE_IDENTICAL;
        info.majorNum = ACCU_BLOCK_SIZE_IDENTICAL;
        info.tailNum = blocks - info.majorLoop * ACCU_BLOCK_SIZE_IDENTICAL;
        info.notAlignEle = perCoreSize2 - blocks * shapeInfo.elePerBlock;
        base += perCoreSize2;
    }

    return true;
}

bool TilingDataScenario1(const CompilerInfo & compilerInfo,
                         const ShapeInfo & shapeInfo,
                         RuntimeInfo & runtimeInfo) {
    int64_t dim = shapeInfo.dim;
    int64_t index = 0;

    //1. src stride
    for (int64_t i = 0; i < dim; i++) {
        runtimeInfo.srcJumpStride[i] = CalcStride(shapeInfo.reducedInShape, dim, i);
    }
    ReorderSrcStride(shapeInfo.reducedPerm, runtimeInfo.srcJumpStride, dim - 1);
    runtimeInfo.srcStrideLogic = runtimeInfo.srcJumpStride[0] / shapeInfo.reducedInShape[dim - 1] - 1;

    //2. dst stride
    for (int64_t i = 0; i < dim; i++) {
        runtimeInfo.dstJumpStride[i] = CalcStride(shapeInfo.reducedOutShape, dim, i);
    }
    ReverseArray(runtimeInfo.dstJumpStride, dim - 1);

    //3. dst factor
    for (int64_t i = dim - 2; i >= 0; i--) {
        runtimeInfo.dstJumpFactor[index++] = shapeInfo.reducedOutShape[i];
    }

    //4. init tuple
    CalcTuple(compilerInfo, shapeInfo, runtimeInfo);

    return true;
}

static void CalcBackNum(const ShapeInfo &shapeInfo, RuntimeInfo &runtimeInfo) {
    if (shapeInfo.lastAxisLen < shapeInfo.elePerBlock) {
        runtimeInfo.backNum = ceil((shapeInfo.elePerBlock - shapeInfo.lastAxisLen) * 1.0 / shapeInfo.lastAxisLen) + 1;
        runtimeInfo.skipEle = runtimeInfo.backNum * shapeInfo.lastAxisLen - shapeInfo.elePerBlock;
    }else if (shapeInfo.alignElement != 0 && shapeInfo.lastAxisLen > shapeInfo.elePerBlock) {
        runtimeInfo.backNum = 1;
    }
}

bool TilingDataScenario2(const CompilerInfo & compilerInfo,
                         const ShapeInfo & shapeInfo,
                         RuntimeInfo & runtimeInfo) {

    TilingDataScenario1(compilerInfo, shapeInfo, runtimeInfo);
    CalcBackNum(shapeInfo, runtimeInfo);
    CalcLoopInfo(compilerInfo, shapeInfo, runtimeInfo);
    return true;
}

bool TilingDataScenario3(const CompilerInfo & compilerInfo,
                         const ShapeInfo & shapeInfo,
                         RuntimeInfo & runtimeInfo) {
    int64_t dim = shapeInfo.dim;
    int64_t index = 0;

    //1. src stride
    for (int64_t i = 0; i < dim; i++) {
        runtimeInfo.srcJumpStride[i] = CalcStride(shapeInfo.reducedInShape, dim, i);
    }
    ReorderSrcStride(shapeInfo.reducedPerm, runtimeInfo.srcJumpStride, dim - 1);
    runtimeInfo.srcStrideLogic = runtimeInfo.srcJumpStride[0] / shapeInfo.reducedInShape[dim - 1] - 1;

    //2. dst stride
    for (int64_t i = 0; i < dim; i++) {
        runtimeInfo.dstJumpStride[i] = CalcStride(shapeInfo.reducedOutShape, dim, i);
    }
    ReverseArray(runtimeInfo.dstJumpStride, dim - 1);

    //3. dst factor
    for (int64_t i = dim - 2; i >= 0; i--) {
        runtimeInfo.dstJumpFactor[index++] = shapeInfo.reducedOutShape[i];
    }

    //4. init tuple
    CalcTuple(compilerInfo, shapeInfo, runtimeInfo);

    //5. huge info
    CalcHugeInfo(shapeInfo, runtimeInfo);

    return true;
}

static void CalcStrideS4(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    BorrowInfo& borrowInfo = runtimeInfo.borrowInfo;
    int64_t srcIdx = borrowInfo.srcIndexIn[0].idx_in;

    for (int i = borrowInfo.dstNumNoDup - 1, j = 0; i >= 0; i--) {
        int64_t idx = borrowInfo.dstIndexInNoDup[i].idx_in;
        borrowInfo.dstStrideCopyIn[j++] = CalcStride(shapeInfo.reducedInShape, shapeInfo.dim, idx);
    }

    for (int i = 0; i < borrowInfo.srcNumNoDup; i++) {
        int64_t idx = borrowInfo.srcIndexOutNoDup[i].idx_out;
        borrowInfo.srcStrideCopyOut[i] = CalcStride(shapeInfo.reducedOutShape, shapeInfo.dim, idx);
    }

    for (int i = 0; i < borrowInfo.otherNum; i++) {
        borrowInfo.otherJumpStride_in[i] = CalcStride(shapeInfo.reducedInShape,
                                                      shapeInfo.dim,
                                                      borrowInfo.otherIndex[i].idx_in);
    }
    for (int i = 0; i < borrowInfo.otherNum; i++) {
        borrowInfo.otherJumpStride_out[i] = CalcStride(shapeInfo.reducedOutShape,
                                                       shapeInfo.dim,
                                                       borrowInfo.otherIndex[i].idx_out);
    }
}

static void CalcSrcDstPerm(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    if (bi.srcNum == 2) {
        if (bi.srcIndexIn[0].idx_out > bi.srcIndexIn[1].idx_out) {
            bi.srcAxisPerm = 0x10;
        } else {
            bi.srcAxisPerm = 0x01;
        }
    }
    if (bi.dstNum == 2) {
        if (bi.dstIndexIn[0].idx_out > bi.dstIndexIn[1].idx_out) {
            bi.dstAxisPerm = 0x10;
        } else {
            bi.dstAxisPerm = 0x01;
        }
    }
}

static void SplitCore(const CompilerInfo& compilerInfo, const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    int64_t sum = 1;
    int64_t coreNum = compilerInfo.coreNum;
    BorrowInfo & bi = runtimeInfo.borrowInfo;

    for (int64_t i = 0; i < bi.srcNum; i++) {
        sum *= bi.srcIndexIn[i].loop;
    }
    for (int64_t i = 0; i < bi.dstNum; i++) {
        if (bi.dstIndexIn[i].dup == 0) {
            sum *= bi.dstIndexIn[i].loop;
        } else {
            bi.dupAxis++;
        }
    }
    for (int64_t i = 0; i < bi.otherNum; i++) {
        sum *= bi.otherIndex[i].loop;
    }

    int64_t base = 0;
    int64_t stride1 = 0;
    int64_t stride2 = 0;
    int64_t s1Num = 0;
    int64_t s2Num = 0;

    SplitEvenly(coreNum, sum, s1Num, s2Num, stride1, stride2);

    int srcFactor = bi.srcIndexIn[0].loop;
    int dstFactor = 1;
    if (bi.dstNumNoDup > 0) {
        dstFactor = bi.dstIndexOutNoDup[bi.dstNumNoDup - 1].loop;
    }

    bi.srcJumpFactorLogic_in = srcFactor;
    bi.dstJumpFactorLogic_in = dstFactor;

    for (int64_t i = bi.dstNumNoDup - 1, j = 0; i >= 0; i--) {
        int64_t idx = bi.dstIndexInNoDup[i].idx_in;
        bi.dstFactorCopyIn[j++] = shapeInfo.reducedInShape[idx];
    }

    for (int64_t i = 0; i < bi.srcNumNoDup; i++) {
        int64_t idx = bi.srcIndexOutNoDup[i].idx_out;
        bi.srcFactorCopyOut[i] = shapeInfo.reducedOutShape[idx];
    }

    for (int i = 0; i < bi.otherNum; i++) {
        bi.otherJumpFactor_in[i] = shapeInfo.reducedInShape[bi.otherIndex[i].idx_in];
    }

    bi.srcJumpFactorMod_in  = 1;
    bi.dstJumpFactorMod_in  = srcFactor;
    bi.otherJumpFactorMod_in[0] = srcFactor * dstFactor;

    for (int64_t i = 1; i < bi.otherNum; i++) {
        bi.otherJumpFactorMod_in[i] *= bi.otherJumpFactorMod_in[i - 1] * bi.otherJumpFactor_in[i - 1];
    }

    bi.loopPerCore.resize(compilerInfo.coreNum, 0);
    bi.srcAxis_in.resize(compilerInfo.coreNum);
    bi.dstAxis_in.resize(compilerInfo.coreNum);
    bi.otherAxis_in.resize(compilerInfo.coreNum);

    for (int64_t i = 0; i < s1Num; i++) {
        bi.loopPerCore[i] = stride1;
        bi.srcAxis_in[i].initTupleLogic = base % srcFactor;
        bi.dstAxis_in[i].initTupleLogic = base / srcFactor % dstFactor;
        bi.srcAxis_in[i].initTuple[0] =  (base % srcFactor) * bi.srcIndexIn[0].step;
        bi.dstAxis_in[i].initTuple[0] =  (base / srcFactor % dstFactor) * bi.dstIndexOut[bi.dstNum - 1].step;

        for (int j = 0; j < bi.otherNum; j++) {
            bi.otherAxis_in[i].initTuple[j] = base / bi.otherJumpFactorMod_in[j] % bi.otherJumpFactor_in[j];
        }
        base += stride1;
    }
    for (int64_t i = s1Num; i < s1Num + s2Num; i++) {
        bi.loopPerCore[i] = stride2;
        bi.srcAxis_in[i].initTupleLogic = base % srcFactor;
        bi.dstAxis_in[i].initTupleLogic = base / srcFactor % dstFactor;
        bi.srcAxis_in[i].initTuple[0] =  (base % srcFactor) * bi.srcIndexIn[0].step;
        bi.dstAxis_in[i].initTuple[0] =  (base / srcFactor % dstFactor) * bi.dstIndexOut[bi.dstNum - 1].step;

        for (int j = 0; j < bi.otherNum; j++) {
            bi.otherAxis_in[i].initTuple[j] = base / bi.otherJumpFactorMod_in[j] % bi.otherJumpFactor_in[j];
        }
        base += stride2;
    }
}

static void UpdateIndexInfo(IndexInfo& info, int64_t step, int64_t axisVol) {
    if (step > axisVol) {
        step = axisVol;
    }
    info.step = step;
    info.loop = axisVol / step;
    info.tail =  axisVol % step;
    info.dup = 1;
}

static void MergeDupAxis(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    BorrowInfo& borrowInfo = runtimeInfo.borrowInfo;
    for (int64_t i = 0; i < borrowInfo.srcNum; i++) {
        int64_t p = borrowInfo.srcIndexIn[i].idx_in;
        for (int64_t j = 0; j < borrowInfo.dstNum; j++) {
            if (p == borrowInfo.dstIndexIn[j].idx_in) {
                int64_t step = max(borrowInfo.srcIndexIn[i].step, borrowInfo.dstIndexIn[j].step); 
                int64_t axisVol = shapeInfo.reducedInShape[p]; 
                UpdateIndexInfo(borrowInfo.srcIndexIn[i], step, axisVol);
                UpdateIndexInfo(borrowInfo.dstIndexIn[j], step, axisVol);
                break;
            }
        }
    }
    for (int64_t i = 0; i < borrowInfo.srcNum; i++) {
        if (borrowInfo.srcIndexIn[i].dup == 0) {
            borrowInfo.srcIndexInNoDup[borrowInfo.srcNumNoDup++] = borrowInfo.srcIndexIn[i];
        }
    }
    for (int64_t i = 0; i < borrowInfo.dstNum; i++) {
        if (borrowInfo.dstIndexIn[i].dup == 0) {
            borrowInfo.dstIndexInNoDup[borrowInfo.dstNumNoDup++] = borrowInfo.dstIndexIn[i];
        }
    }
}

static void CalcLeftVol(const CompilerInfo& compilerInfo, const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    int64_t ubSize = 0;
    if (shapeInfo.alignElement == 0) {
        ubSize = 32 * 1024; // as b16
    } else {
        ubSize = CalcVnchwconvPartialUbSize(compilerInfo.coreNum, compilerInfo.ubSize);
    }

    int64_t leftVol = ubSize / (shapeInfo.lastAxisLen * compilerInfo.fp16Times);
    // since block align padding may result ub size not enough
    // loop 10 is ok for lastAxisLen from 1 to 256
    for (int i = 0; i < LOOP_FOR_UB_PADDING; i++) {
        int64_t vol = sqrt(leftVol) - i;
        if (vol * AlignX(vol * shapeInfo.lastAxisLen, shapeInfo.elePerBlock) * compilerInfo.fp16Times <= ubSize) {
            runtimeInfo.borrowInfo.srcVol = vol;
            runtimeInfo.borrowInfo.dstVol = vol;
            break;
        }
        else {
            continue;
        }
    }
}

static void MakeSrcIndexAsInShape(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    IndexInfo indexInfo[BORROW_MAX_AXIS_NUM];
    BorrowInfo& borrowInfo = runtimeInfo.borrowInfo;
    int k = 0;
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        for (int64_t j = 0; j < borrowInfo.srcNum; j++) {
            if (i == borrowInfo.srcIndexIn[j].idx_in) {
                indexInfo[k++] = borrowInfo.srcIndexIn[j];
            }
        }
    }
    for (int64_t i = 0; i < borrowInfo.srcNum; i++) {
        borrowInfo.srcIndexIn[i] = indexInfo[i];
    }

    k = 0;
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        for (int64_t j = 0; j < borrowInfo.srcNumNoDup; j++) {
            if (i == borrowInfo.srcIndexInNoDup[j].idx_in) {
                indexInfo[k++] = borrowInfo.srcIndexInNoDup[j];
            }
        }
    }
    for (int64_t i = 0; i < borrowInfo.srcNumNoDup; i++) {
        borrowInfo.srcIndexInNoDup[i] = indexInfo[i];
    }
}

static void MakeSrcIndexAsOutShape(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    BorrowInfo& borrowInfo = runtimeInfo.borrowInfo;
    int k = 0;
    for (int64_t i = shapeInfo.dim - 1; i >= 0; i--) {
        for (int64_t j = 0; j < borrowInfo.srcNum; j++) {
            if (shapeInfo.reducedPerm[i] == borrowInfo.srcIndexIn[j].idx_in) {
                borrowInfo.srcIndexOut[k++] = borrowInfo.srcIndexIn[j];
                break;
            }
        }
    }
    k = 0;
    for (int64_t i = shapeInfo.dim - 1; i >= 0; i--) {
        for (int64_t j = 0; j < borrowInfo.srcNumNoDup; j++) {
            if (shapeInfo.reducedPerm[i] == borrowInfo.srcIndexInNoDup[j].idx_in) {
                borrowInfo.srcIndexOutNoDup[k++] = borrowInfo.srcIndexInNoDup[j];
                break;
            }
        }
    }
}

static void MakeDstIndexAsInShape(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    IndexInfo indexInfo[BORROW_MAX_AXIS_NUM];
    BorrowInfo& borrowInfo = runtimeInfo.borrowInfo;
    int k = 0;
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        for (int64_t j = 0; j < borrowInfo.dstNum; j++) {
            if (i == borrowInfo.dstIndexIn[j].idx_in) {
                indexInfo[k++] = borrowInfo.dstIndexIn[j];
            }
        }
    }
    for (int64_t i = 0; i < borrowInfo.dstNum; i++) {
        borrowInfo.dstIndexIn[i] = indexInfo[i];
    }
    k = 0;
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        for (int64_t j = 0; j < borrowInfo.dstNumNoDup; j++) {
            if (i == borrowInfo.dstIndexInNoDup[j].idx_in) {
                indexInfo[k++] = borrowInfo.dstIndexInNoDup[j];
            }
        }
    }
    for (int64_t i = 0; i < borrowInfo.dstNumNoDup; i++) {
        borrowInfo.dstIndexInNoDup[i] = indexInfo[i];
    }
}

static void MakeDstIndexAsOutShape(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    BorrowInfo& borrowInfo = runtimeInfo.borrowInfo;
    int k = 0;
    for (int64_t i = shapeInfo.dim - 1; i >= 0; i--) {
        for (int64_t j = 0; j < borrowInfo.dstNum; j++) {
            if (shapeInfo.reducedPerm[i] == borrowInfo.dstIndexIn[j].idx_in) {
                borrowInfo.dstIndexOut[k++] = borrowInfo.dstIndexIn[j];
                break;
            }
        }
    }
    k = 0;
    for (int64_t i = shapeInfo.dim - 1; i >= 0; i--) {
        for (int64_t j = 0; j < borrowInfo.dstNumNoDup; j++) {
            if (shapeInfo.reducedPerm[i] == borrowInfo.dstIndexInNoDup[j].idx_in) {
                borrowInfo.dstIndexOutNoDup[k++] = borrowInfo.dstIndexIn[j];
                break;
            }
        }
    }
}

/*
 * ubPermRaw: 5,4,1,0; ubPerm: 3,2,1,0
 */
static void MakeDiscreteBeContiguous(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    BorrowInfo& borrowInfo = runtimeInfo.borrowInfo;
    int64_t perm[BORROW_MAX_AXIS_NUM];
    for (int i = 0; i < BORROW_MAX_AXIS_NUM; i++) {
        perm[i] = borrowInfo.ubPermRaw[i];
    }

    for (int64_t i = 0; i < borrowInfo.ubPermNum; i++) {
        int idx = 0;
        int minVal = TRANSPOSE_MAX_AXIS_NUM;
        for (int64_t j = 0; j < borrowInfo.ubPermNum; j++) {
            if (perm[j] < minVal) {
                minVal = perm[j];
                idx = j;
            }
        }
        perm[idx] = TRANSPOSE_MAX_AXIS_NUM;
        runtimeInfo.borrowInfo.ubPerm[idx] = TRANSPOSE_MAX_AXIS_NUM + i;
    }

    for (int64_t i = 0; i < borrowInfo.ubPermNum; i++) {
        runtimeInfo.borrowInfo.ubPerm[i] = runtimeInfo.borrowInfo.ubPerm[i] - TRANSPOSE_MAX_AXIS_NUM;
    }
}

static void CalcPermInUb(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    BorrowInfo& borrowInfo = runtimeInfo.borrowInfo;
    for (int64_t i = 0; i < shapeInfo.reducedPerm.size(); i++) {
        bool exist = false;
        int64_t p = shapeInfo.reducedPerm[i];
        for (int64_t j = 0; j < borrowInfo.srcNum; j++) {
            if(p == borrowInfo.srcIndexIn[j].idx_in) {
                for (int64_t k = 0; k < borrowInfo.ubPermNum; k++) {
                    if (borrowInfo.ubPermRaw[k] == p) {
                        exist = true;
                        break;
                    }
                }
                if (!exist) {
                    borrowInfo.ubPermRaw[borrowInfo.ubPermNum++] = p;
                    break;
                }
            }
        }

        exist = false;
        for (int64_t j = 0; j < borrowInfo.dstNum; j++) {
            if(p == borrowInfo.dstIndexIn[j].idx_in) {
                for (int64_t k = 0; k < borrowInfo.ubPermNum; k++) {
                    if (borrowInfo.ubPermRaw[k] == p) {
                        exist = true;
                        break;
                    }
                }
                if (!exist) {
                    borrowInfo.ubPermRaw[borrowInfo.ubPermNum++] = p;
                    break;
                }
            }
        }
    }
    MakeDiscreteBeContiguous(shapeInfo, runtimeInfo);
}

static int64_t GetDupAxisInSrc(const RuntimeInfo& runtimeInfo, int64_t index) {
    const BorrowInfo& bi = runtimeInfo.borrowInfo;
    for (int64_t i = 0; i < bi.srcNum; i++) {
        if (index == bi.srcIndexIn[i].idx_in) {
            return i;
        }
    }
    return -1;
}

static void CalcSrcBorrowAxisIndex(const ShapeInfo& si, RuntimeInfo& runtimeInfo) {
    int64_t dim = si.dim;
    int64_t borrowed = 1;
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    for (int i = 0; i < BORROW_SRC_AXIS_NUM; i++) {
        if (dim >= i + 2) {
            borrowed *= si.reducedInShape[dim - i - 2];
            int64_t srcNum = bi.srcNum;
            bi.srcNum++;
            bi.srcIndexIn[srcNum].idx_in = dim - i - 2;
            bi.srcIndexIn[srcNum].idx_out = GetPermIndex(si.reducedPerm, bi.srcIndexIn[srcNum].idx_in);
            if (borrowed >= bi.srcVol) {
                borrowed /= si.reducedInShape[dim - i - 2];
                bi.srcIndexIn[srcNum].step =  bi.srcVol / borrowed;
                bi.srcIndexIn[srcNum].loop =  si.reducedInShape[dim - i - 2] / bi.srcIndexIn[srcNum].step;
                bi.srcIndexIn[srcNum].tail =  si.reducedInShape[dim - i - 2] % \
                                                (bi.srcIndexIn[srcNum].step * bi.srcIndexIn[srcNum].loop);
                break;
            } else {
                bi.srcIndexIn[srcNum].loop = 1;
                bi.srcIndexIn[srcNum].step = si.reducedInShape[dim - i - 2];
                bi.srcIndexIn[srcNum].intact = 1;
            }
        }
    }
}

static void CalcBorrowLoop(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;

    for (int64_t i = 0; i < bi.srcNumNoDup; i++) {
        bi.majorSrcLoop_out *= bi.srcIndexInNoDup[i].step;
    }

    bi.tailSrcLoop_out = bi.srcIndexInNoDup[0].tail;
    for (int64_t i = 1; i < bi.srcNumNoDup; i++) {
        bi.tailSrcLoop_out *= bi.srcIndexInNoDup[i].step;
    }

    for (int64_t i = 0; i < bi.dstNumNoDup; i++) {
        bi.majorDstLoop_in *= bi.dstIndexInNoDup[i].step;
    }

    if (bi.dstNumNoDup == bi.dstNum) {
        for (int64_t i = 0; i < bi.dstNumNoDup; i++) {
            if (i == bi.dstNumNoDup - 1) {
                bi.tailDstLoop_in *= bi.dstIndexOut[i].tail;
            } else {
                bi.tailDstLoop_in *= bi.dstIndexOut[i].step;
            }
        }
    } else {
        bi.tailDstLoop_in = bi.dstIndexOutNoDup[0].tail;
    }
}

static void CalcBorrowBurstLen(const ShapeInfo& si, RuntimeInfo& runtimeInfo) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;

    for (int64_t i = 0; i < bi.srcNum; i++) {
        if (i == 0) {
            bi.majorBurstLen_in *= bi.srcIndexIn[i].step;
            bi.tailBurstLen_in *= bi.srcIndexIn[i].tail;
        } else {
            bi.majorBurstLen_in *= bi.srcIndexIn[i].loop * bi.srcIndexIn[i].step;
            bi.tailBurstLen_in *= bi.srcIndexIn[i].loop * bi.srcIndexIn[i].step;
        }
    }
    bi.majorInEle = bi.majorBurstLen_in * si.lastAxisLen;
    bi.tailInEle = bi.tailBurstLen_in * si.lastAxisLen;
    bi.majorInTailEle = bi.majorBurstLen_in * si.lastAxisLen % si.elePerBlock;
    bi.tailInTailEle = bi.tailBurstLen_in * si.lastAxisLen % si.elePerBlock;
    bi.majorBurstLen_in = (bi.majorBurstLen_in * si.lastAxisLen + si.elePerBlock - 1) / si.elePerBlock;
    bi.tailBurstLen_in = (bi.tailBurstLen_in * si.lastAxisLen + si.elePerBlock -1) / si.elePerBlock;

    for (int64_t i = 0; i < bi.dstNum; i++) {
        if (i == bi.dstNum - 1) {
            bi.majorBurstLen_out *= bi.dstIndexOut[i].step;
            bi.tailBurstLen_out *= bi.dstIndexOut[i].tail;
        } else {
            bi.majorBurstLen_out *= bi.dstIndexOut[i].step;
            bi.tailBurstLen_out *= bi.dstIndexOut[i].step;
        }
    }

    bi.majorOutEle = bi.majorBurstLen_out * si.lastAxisLen;
    bi.tailOutEle = bi.tailBurstLen_out * si.lastAxisLen;
    bi.majorOutTailEle = bi.majorBurstLen_out * si.lastAxisLen % si.elePerBlock;
    bi.tailOutTailEle = bi.tailBurstLen_out * si.lastAxisLen % si.elePerBlock;
    bi.majorBurstLen_out = (bi.majorBurstLen_out * si.lastAxisLen + si.elePerBlock - 1) / si.elePerBlock;
    bi.tailBurstLen_out = (bi.tailBurstLen_out * si.lastAxisLen + si.elePerBlock - 1) / si.elePerBlock;
}

void CalcDstBorrowAxisIndex(const ShapeInfo& si, RuntimeInfo& runtimeInfo) {
    int64_t dim = si.dim;
    int64_t borrowed = 1;
    int64_t tailEle = si.lastAxisLen;
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    for (int i = 0; i < BORROW_DST_AXIS_NUM; i++) {
        if (dim >= i + 2) {
            int64_t id = dim -i - 2;
            int64_t dstNum = bi.dstNum;
            int64_t index = si.reducedPerm[id];
            int64_t dupId = GetDupAxisInSrc(runtimeInfo, index);
            bi.dstIndexIn[dstNum].idx_out = id;
            bi.dstIndexIn[dstNum].idx_in = index;

            if (dupId != -1) {
                bi.dstVol = bi.dstVol * bi.srcIndexIn[dupId].step;
            }

            if (si.reducedOutShape[id] * borrowed <= bi.dstVol) {
                borrowed *= si.reducedOutShape[id];
                bi.dstNum++;
                bi.dstIndexIn[dstNum].loop = 1;
                bi.dstIndexIn[dstNum].step = si.reducedOutShape[dim - i - 2];
                bi.dstIndexIn[dstNum].intact = 1;
                tailEle *= bi.dstIndexIn[dstNum].step;
            } else {
                if (borrowed * 2 > bi.dstVol) {
                    break;
                }

                bi.dstNum++;
                bi.dstIndexIn[dstNum].step =  bi.dstVol / borrowed;
                bi.dstIndexIn[dstNum].loop =  si.reducedOutShape[id] / bi.dstIndexIn[dstNum].step;
                bi.dstIndexIn[dstNum].tail =  si.reducedOutShape[id] % \
                                              (bi.dstIndexIn[dstNum].step * bi.dstIndexIn[dstNum].loop);

                for (int j = 0; j < ELE_NUM_PER_BLOCK_FP16; j++) {
                    if ((bi.dstIndexIn[dstNum].tail != 0) && (tailEle * bi.dstIndexIn[dstNum].tail < si.elePerBlock)) {
                        bi.dstIndexIn[dstNum].step -= 1;
                        bi.dstIndexIn[dstNum].loop = si.reducedOutShape[id] / bi.dstIndexIn[dstNum].step;
                        bi.dstIndexIn[dstNum].tail = si.reducedOutShape[id] % \
                                                     (bi.dstIndexIn[dstNum].step * bi.dstIndexIn[dstNum].loop);
                    } else {
                        break;
                    }
                }
                break;
            }
        }
    }
}

static void ReorderIndexInfo(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    MakeSrcIndexAsInShape(shapeInfo, runtimeInfo);
    MakeSrcIndexAsOutShape(shapeInfo, runtimeInfo);
    MakeDstIndexAsInShape(shapeInfo, runtimeInfo);
    MakeDstIndexAsOutShape(shapeInfo, runtimeInfo);
}

static void CalcOtherAxisIndex(const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    for (int i = 0; i < shapeInfo.reducedPerm.size(); i++) {
        bool borrowed = false;
        for (int j = 0; j < runtimeInfo.borrowInfo.srcNum; j++) {
            if (i == runtimeInfo.borrowInfo.srcIndexIn[j].idx_in) {
                borrowed = true;
            }
        }
        for (int j = 0; j < runtimeInfo.borrowInfo.dstNum; j++) {
            if (i == runtimeInfo.borrowInfo.dstIndexIn[j].idx_in) {
                borrowed = true;
            }
        }
        if (!borrowed) {
            if (i != shapeInfo.dim - 1) {
                int otherNum = runtimeInfo.borrowInfo.otherNum;
                runtimeInfo.borrowInfo.otherIndex[otherNum].idx_in = i;
                runtimeInfo.borrowInfo.otherIndex[otherNum].idx_out = GetPermIndex(shapeInfo.reducedPerm, i);
                runtimeInfo.borrowInfo.otherIndex[otherNum].loop = shapeInfo.reducedInShape[i];
                runtimeInfo.borrowInfo.otherNum++;
            }
        }
    }
}

/*
 * four axis
 * perm           valid
 * ---------------------
 * 3 2 0 1        y
 * 3 2 1 0        y
 * 3 0 2 1        n
 * 3 0 1 2        n
 * 3 1 2 0        n
 * 3 1 0 2        n
 * 2 3 0 1        y 
 * 2 3 1 0        y 
 * 2 0 3 1        n
 * 2 0 1 3        n
 * 2 1 3 0        n
 * 2 1 0 3        n
 * 0 3 2 1        n
 * 0 3 1 2        n
 * 0 2 3 1        n
 * 0 2 1 3        n
 * 0 1 3 2        n
 * 0 1 2 3        n
 * 1 3 2 0        n
 * 1 3 0 2        n
 * 1 2 3 0        n
 * 1 2 0 3        n
 * 1 0 3 2        n
 * 1 0 2 3        n
 *
 * there axis
 * perm           valid
 * ---------------------
 * 1 0 2          n 
 * 1 2 0          y 
 * 0 1 2          n
 * 0 2 1          n
 * 2 1 0          y
 * 2 0 1          y
 *
 * two axis
 * perm           valid
 * ---------------------
 * 1 0            y
 * 0 1            n
 */
static int64_t GetPermHex(const RuntimeInfo& runtimeInfo) {
    const BorrowInfo& bi = runtimeInfo.borrowInfo;
    if (bi.ubPermNum == 4) {
        if(bi.ubPerm[0] == 3 && bi.ubPerm[1] == 2 && bi.ubPerm[2] == 1 && bi.ubPerm[3] == 0) {
            return 0x3210;
        } else  if(bi.ubPerm[0] == 3 && bi.ubPerm[1] == 2 && bi.ubPerm[2] == 0 && bi.ubPerm[3] == 1) {
            return 0x3201;
        } else  if(bi.ubPerm[0] == 2 && bi.ubPerm[1] == 3 && bi.ubPerm[2] == 0 && bi.ubPerm[3] == 1) {
            return 0x2301;
        } else  if(bi.ubPerm[0] == 2 && bi.ubPerm[1] == 3 && bi.ubPerm[2] == 1 && bi.ubPerm[3] == 0) {
            return 0x2310;
        }
    } else if (bi.ubPermNum == 3) {
        if(bi.ubPerm[0] == 2 && bi.ubPerm[1] == 1 && bi.ubPerm[2] == 0) {
            return 0x210;
        } else if(bi.ubPerm[0] == 2 && bi.ubPerm[1] == 0 && bi.ubPerm[2] == 1) {
            return 0x201;
        } else if(bi.ubPerm[0] == 1 && bi.ubPerm[1] == 2 && bi.ubPerm[2] == 0) {
            return 0x120;
        }
    } else if (bi.ubPermNum == 2) {
        return 0x10;
    }
    return 0;
}

static void SetLRSB(LRSB& lrsb, int64_t p1, int64_t p2, int64_t bl, int epb) {
    if (p1 > 1 && p2 > 1) {
        if (p1 < p2) {
            lrsb.Set(p1, p2, 0, bl * (p1 - 1),  bl, bl * epb * p2,  bl * epb);
        } else {
            lrsb.Set(p2, p1, bl * (p2 - 1), 0, bl, bl * epb, bl * epb * p1);
        }
    }
}

/*
 * 0123 -> 1203 -> 2103 -> 3210
 */
static void RepeatStride3210(const CompilerInfo& compilerInfo,
                             const ShapeInfo& shapeInfo,
                             RuntimeInfo& runtimeInfo,
                             int64_t step[UB_REORDER_COMBINATION][BORROW_MAX_AXIS_NUM]) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    int64_t * perm = bi.ubPerm;
    int64_t bl = shapeInfo.alignElement == 0 ? shapeInfo.lastAxisBurstLen :\
                                               shapeInfo.lastAxisLen * compilerInfo.fp16Times;
    int64_t epb = shapeInfo.elePerBlock;
    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        LRSB * lrsb = bi.lrsb[i];
        int64_t* s = step[i];
        SetLRSB(lrsb[0], s[0], s[1] * s[2], s[3] * bl, epb);
        SetLRSB(lrsb[1], s[1], s[2], s[0] * s[3] * bl, epb);
        SetLRSB(lrsb[2], s[2] * s[1] * s[0], s[3], bl, epb);
    }
}

/*
 * 3201:   0123  ->  2301 -> 3201 
 */
static void RepeatStride3201(const CompilerInfo& compilerInfo,
                             const ShapeInfo& shapeInfo,
                             RuntimeInfo& runtimeInfo,
                             int64_t step[UB_REORDER_COMBINATION][BORROW_MAX_AXIS_NUM]) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    int64_t * perm = bi.ubPerm;
    int64_t bl = shapeInfo.alignElement == 0 ? shapeInfo.lastAxisBurstLen :\
                                               shapeInfo.lastAxisLen * compilerInfo.fp16Times;
    int64_t epb = shapeInfo.elePerBlock;
    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        LRSB * lrsb = bi.lrsb[i];
        int64_t* s = step[i];
        SetLRSB(lrsb[0], s[0] * s[1], s[2] * s[3], bl, epb);
        SetLRSB(lrsb[1], s[2], s[3], s[0] * s[1] * bl, epb);
    }
}

/*
 * 2301:   0123  ->  2301 
 */
static void RepeatStride2301(const CompilerInfo& compilerInfo,
                             const ShapeInfo& shapeInfo,
                             RuntimeInfo& runtimeInfo,
                             int64_t step[UB_REORDER_COMBINATION][BORROW_MAX_AXIS_NUM]) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    int64_t * perm = bi.ubPerm;
    int64_t bl = shapeInfo.alignElement == 0 ? shapeInfo.lastAxisBurstLen :\
                                               shapeInfo.lastAxisLen * compilerInfo.fp16Times;
    int64_t epb = shapeInfo.elePerBlock;
    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        LRSB * lrsb = bi.lrsb[i];
        int64_t* s = step[i];
        SetLRSB(lrsb[0], s[0] * s[1], s[2] * s[3], bl, epb);
    }
}

/*
 * 2310:   0123  ->  1023 -> 2310
 */
static void RepeatStride2310(const CompilerInfo& compilerInfo,
                             const ShapeInfo& shapeInfo,
                             RuntimeInfo& runtimeInfo,
                             int64_t step[UB_REORDER_COMBINATION][BORROW_MAX_AXIS_NUM]) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    int64_t * perm = bi.ubPerm;
    int64_t bl = shapeInfo.alignElement == 0 ? shapeInfo.lastAxisBurstLen :\
                                               shapeInfo.lastAxisLen * compilerInfo.fp16Times;
    int64_t epb = shapeInfo.elePerBlock;
    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        LRSB * lrsb = bi.lrsb[i];
        int64_t* s = step[i];
        SetLRSB(lrsb[0], s[0], s[1], s[2] * s[3] * bl, epb);
        SetLRSB(lrsb[1], s[1] * s[0], s[2] * s[3], bl, epb);
    }
}

/*
 * 210: 012 -> 102 ->  210
 */
static void RepeatStride210(const CompilerInfo& compilerInfo,
                             const ShapeInfo& shapeInfo,
                             RuntimeInfo& runtimeInfo,
                             int64_t step[UB_REORDER_COMBINATION][BORROW_MAX_AXIS_NUM]) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    int64_t * perm = bi.ubPerm;
    int64_t bl = shapeInfo.alignElement == 0 ? shapeInfo.lastAxisBurstLen :\
                                               shapeInfo.lastAxisLen * compilerInfo.fp16Times;
    int64_t epb = shapeInfo.elePerBlock;
    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        LRSB * lrsb = bi.lrsb[i];
        int64_t* s = step[i];
        SetLRSB(lrsb[0], s[0], s[1], s[2] * bl, epb);
        SetLRSB(lrsb[1], s[0] * s[1], s[2], bl, epb);
    }
}

/*
 * 201: 012 -> 201 
 */
static void RepeatStride201(const CompilerInfo& compilerInfo,
                             const ShapeInfo& shapeInfo,
                             RuntimeInfo& runtimeInfo,
                             int64_t step[UB_REORDER_COMBINATION][BORROW_MAX_AXIS_NUM]) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    int64_t * perm = bi.ubPerm;
    int64_t bl = shapeInfo.alignElement == 0 ? shapeInfo.lastAxisBurstLen :\
                                               shapeInfo.lastAxisLen * compilerInfo.fp16Times;
    int64_t epb = shapeInfo.elePerBlock;
    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        LRSB * lrsb = bi.lrsb[i];
        int64_t* s = step[i];
        SetLRSB(lrsb[0], s[0] * s[1], s[2], bl, epb);
    }
}

/*
 * 120: 012 -> 120 
 */
static void RepeatStride120(const CompilerInfo& compilerInfo,
                             const ShapeInfo& shapeInfo,
                             RuntimeInfo& runtimeInfo,
                             int64_t step[UB_REORDER_COMBINATION][BORROW_MAX_AXIS_NUM]) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    int64_t * perm = bi.ubPerm;
    int64_t bl = shapeInfo.alignElement == 0 ? shapeInfo.lastAxisBurstLen :\
                                               shapeInfo.lastAxisLen * compilerInfo.fp16Times;
    int64_t epb = shapeInfo.elePerBlock;
    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        LRSB * lrsb = bi.lrsb[i];
        int64_t* s = step[i];
        SetLRSB(lrsb[0], s[0], s[1] * s[2], bl, epb);
    }
}

/*
 * 01 -> 10
 */
static void RepeatStride10(const CompilerInfo& compilerInfo,
                             const ShapeInfo& shapeInfo,
                             RuntimeInfo& runtimeInfo,
                             int64_t step[UB_REORDER_COMBINATION][BORROW_MAX_AXIS_NUM]) {
    BorrowInfo& bi = runtimeInfo.borrowInfo;
    int64_t * perm = bi.ubPerm;
    int64_t bl = shapeInfo.alignElement == 0 ? shapeInfo.lastAxisBurstLen :\
                                               shapeInfo.lastAxisLen * compilerInfo.fp16Times;
    int64_t epb = shapeInfo.elePerBlock;
    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        LRSB * lrsb = bi.lrsb[i];
        int64_t* s = step[i];
        SetLRSB(lrsb[0], s[0], s[1], bl, epb);
    }
}

/*   0 :  major_dst_major_src
 *   1 :  major_dst_tail_src 
 *   2 :  major_src_tail_dst
 *   3 :  tail_src_tail_dst
 *
 */
#define SET_P4(d0, p0, p1, p2, p3) step[d0][0] = p0, step[d0][1] = p1, step[d0][2] = p2, step[d0][3] = p3;

static void ConstructStep(const RuntimeInfo& runtimeInfo, int64_t step[][BORROW_MAX_AXIS_NUM]) {
    const BorrowInfo& bi = runtimeInfo.borrowInfo; 

    if (bi.srcNum == 1 && bi.dstNum == 1) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.srcIndexIn[0].step, 0, 0);
        SET_P4(1, bi.dstIndexIn[0].step, bi.srcIndexIn[0].tail, 0, 0);
        SET_P4(2, bi.dstIndexIn[0].tail, bi.srcIndexIn[0].step, 0, 0);
        SET_P4(3, bi.dstIndexIn[0].tail, bi.srcIndexIn[0].tail, 0, 0);
    }

    if (bi.srcNum == 2 && bi.dstNum == 1 && bi.srcNumNoDup == 2 && bi.dstNumNoDup == 1) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step, 0);
        SET_P4(1, bi.dstIndexIn[0].step, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step, 0);
        SET_P4(2, bi.dstIndexIn[0].tail, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step, 0);
        SET_P4(3, bi.dstIndexIn[0].tail, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step, 0);
    }

    if (bi.srcNum == 1 && bi.dstNum == 2 && bi.srcNumNoDup == 1 && bi.dstNumNoDup == 2 && bi.dstAxisPerm == 0x10) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].step, 0);
        SET_P4(1, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].tail, 0);
        SET_P4(2, bi.dstIndexIn[0].step, bi.dstIndexIn[1].tail, bi.srcIndexIn[0].step, 0);
        SET_P4(3, bi.dstIndexIn[0].step, bi.dstIndexIn[1].tail, bi.srcIndexIn[0].tail, 0);
    }

    if (bi.srcNum == 1 && bi.dstNum == 2 && bi.srcNumNoDup == 1 && bi.dstNumNoDup == 2 && bi.dstAxisPerm == 0x01) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].step, 0);
        SET_P4(1, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].tail, 0);
        SET_P4(2, bi.dstIndexIn[0].tail, bi.dstIndexIn[1].step, bi.srcIndexIn[0].step, 0);
        SET_P4(3, bi.dstIndexIn[0].tail, bi.dstIndexIn[1].step, bi.srcIndexIn[0].tail, 0);
    }

    if (bi.srcNum == 2 && bi.dstNum == 2 && bi.srcNumNoDup == 2 && bi.dstNumNoDup == 2 && bi.dstAxisPerm == 0x10) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step);
        SET_P4(1, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step);
        SET_P4(2, bi.dstIndexIn[0].step, bi.dstIndexIn[1].tail, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step);
        SET_P4(3, bi.dstIndexIn[0].step, bi.dstIndexIn[1].tail, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step);
    }

    if (bi.srcNum == 2 && bi.dstNum == 2 && bi.srcNumNoDup == 2 && bi.dstNumNoDup == 2 && bi.dstAxisPerm == 0x01) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step);
        SET_P4(1, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step);
        SET_P4(2, bi.dstIndexIn[0].tail, bi.dstIndexIn[1].step, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step);
        SET_P4(3, bi.dstIndexIn[0].tail, bi.dstIndexIn[1].step, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step);
    }

    if (bi.srcNum == 2 && bi.dstNum == 2 && bi.srcNumNoDup == 1 && bi.dstNumNoDup == 1 && bi.axisPerm == 0x210) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step, 0);
        SET_P4(3, bi.dstIndexIn[0].step, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step, 0);
    }

    if (bi.srcNum == 2 && bi.dstNum == 2 && bi.srcNumNoDup == 1 && bi.dstNumNoDup == 1 && bi.axisPerm == 0x120) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step, 0);
        SET_P4(1, bi.dstIndexIn[0].step, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step, 0);
        SET_P4(2, bi.dstIndexIn[0].tail, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step, 0);
        SET_P4(3, bi.dstIndexIn[0].tail, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step, 0);
    }

    if (bi.srcNum == 2 && bi.dstNum == 2 && bi.srcNumNoDup == 1 && bi.dstNumNoDup == 1 && bi.axisPerm == 0x201) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step, 0);
        SET_P4(1, bi.dstIndexIn[0].step, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step, 0);
        SET_P4(2, bi.dstIndexIn[0].tail, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step, 0);
        SET_P4(3, bi.dstIndexIn[0].tail, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step, 0);
    }

    if (bi.srcNum == 2 && bi.dstNum == 2 && bi.srcNumNoDup == 0 && bi.dstNumNoDup == 0) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, 0, 0)
        SET_P4(1, bi.dstIndexIn[0].step, bi.dstIndexIn[1].tail, 0, 0)
        SET_P4(2, bi.dstIndexIn[0].tail, bi.dstIndexIn[1].step, 0, 0)
        SET_P4(3, bi.dstIndexIn[0].tail, bi.dstIndexIn[1].tail, 0, 0)
    }

    if (bi.srcNum == 2 && bi.dstNum == 1 && bi.dstNumNoDup == 0 && bi.srcNumNoDup == 1) {
        SET_P4(0, bi.srcIndexIn[0].step, bi.srcIndexIn[1].step, 0, 0);
        SET_P4(1, bi.srcIndexIn[0].tail, bi.srcIndexIn[1].step, 0, 0);
    }

    if (bi.srcNum == 1 && bi.dstNum == 2 && bi.dstNumNoDup == 1 && bi.srcNumNoDup == 0) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.srcIndexIn[0].step, 0, 0);
        SET_P4(1, bi.dstIndexIn[0].step, bi.dstIndexIn[1].tail, 0, 0);
    }

    if(bi.srcNum == 1 && bi.dstNum == 2 && bi.srcNumNoDup == 1 && bi.dstNumNoDup == 2 && bi.dstAxisPerm == 0x10) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].step, 0)
        SET_P4(1, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].tail, 0)
        SET_P4(2, bi.dstIndexIn[0].step, bi.dstIndexIn[1].tail, bi.srcIndexIn[0].step, 0)
        SET_P4(3, bi.dstIndexIn[0].step, bi.dstIndexIn[1].tail, bi.srcIndexIn[0].tail, 0)
    }

    if(bi.srcNum == 1 && bi.dstNum == 2 && bi.srcNumNoDup == 1 && bi.dstNumNoDup == 2 && bi.dstAxisPerm == 0x01) {
        SET_P4(0, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].step, 0)
        SET_P4(1, bi.dstIndexIn[0].step, bi.dstIndexIn[1].step, bi.srcIndexIn[0].tail, 0)
        SET_P4(2, bi.dstIndexIn[0].tail, bi.dstIndexIn[1].step, bi.srcIndexIn[0].step, 0)
        SET_P4(3, bi.dstIndexIn[0].tail, bi.dstIndexIn[1].step, bi.srcIndexIn[0].tail, 0)
    }
}

static void CalcRepetStride(const CompilerInfo& compilerInfo, const ShapeInfo& shapeInfo, RuntimeInfo& runtimeInfo) {
    int perm = GetPermHex(runtimeInfo);
    runtimeInfo.borrowInfo.axisPerm = perm;

    int64_t step[UB_REORDER_COMBINATION][BORROW_MAX_AXIS_NUM];
    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        for (int j = 0; j < BORROW_MAX_AXIS_NUM; j++) {
            step[i][j] = 0;
        }
    }

    ConstructStep(runtimeInfo, step);

    switch(perm) {
        case 0x3210:
            RepeatStride3210(compilerInfo, shapeInfo, runtimeInfo, step);
            break;
        case 0x03201:
            RepeatStride3201(compilerInfo, shapeInfo, runtimeInfo, step);
            break;
        case 0x2301:
            RepeatStride2301(compilerInfo, shapeInfo, runtimeInfo, step);
            break;
        case 0x2310:
            RepeatStride2310(compilerInfo, shapeInfo, runtimeInfo, step);
            break;
        case 0x210:
            RepeatStride210(compilerInfo, shapeInfo, runtimeInfo, step);
            break;
        case 0x201:
            RepeatStride201(compilerInfo, shapeInfo, runtimeInfo, step);
            break;
        case 0x120:
            RepeatStride120(compilerInfo, shapeInfo, runtimeInfo, step);
            break;
        case 0x10:
            RepeatStride10(compilerInfo, shapeInfo, runtimeInfo, step);
            break;
    }
}

static bool TilingDataScenario4(const CompilerInfo& compilerInfo,
                                const ShapeInfo& shapeInfo,
                                RuntimeInfo& runtimeInfo) {
    CalcLeftVol(compilerInfo, shapeInfo, runtimeInfo);
    CalcSrcBorrowAxisIndex(shapeInfo, runtimeInfo);
    CalcDstBorrowAxisIndex(shapeInfo, runtimeInfo);
    MergeDupAxis(shapeInfo, runtimeInfo);
    ReorderIndexInfo(shapeInfo, runtimeInfo);
    CalcSrcDstPerm(shapeInfo, runtimeInfo);
    CalcBorrowLoop(shapeInfo, runtimeInfo);
    CalcBorrowBurstLen(shapeInfo, runtimeInfo);
    CalcOtherAxisIndex(shapeInfo, runtimeInfo);
    CalcPermInUb(shapeInfo, runtimeInfo);
    CalcRepetStride(compilerInfo, shapeInfo, runtimeInfo);
    SplitCore(compilerInfo, shapeInfo, runtimeInfo);
    CalcStrideS4(shapeInfo, runtimeInfo);
    return true;
}

static bool TilingDataScenario6(const CompilerInfo& compilerInfo,
                                const ShapeInfo& shapeInfo,
                                RuntimeInfo& runtimeInfo) {
    TilingDataScenario2(compilerInfo, shapeInfo, runtimeInfo);
    return true;
}

static int64_t Align32BCeil(int64_t i, int64_t elementNumPerBlock) {
    if (i % elementNumPerBlock == 0) {
        return i;
    }
    return i + (elementNumPerBlock - i % elementNumPerBlock);
}

static void Split(int64_t val,
                  int64_t factor,
                  int64_t elePerBlock,
                  vector<pair<int64_t, int64_t>> & range) {
    if (factor == 1 || factor == 0) {
        range.push_back({0,val});
    } else {
        int64_t perCore = Align32BCeil(ceil(val * 1.0 / factor), elePerBlock);
        for (int64_t i = 0; i < factor - 1; i++) {
            range.push_back({0, perCore});
        }
        range.push_back({0,val - (factor - 1) * perCore});
        for (int i = factor - 2; i >= 0; i--) {
            if ((range[factor - 1].second < elePerBlock) && (range[i].second >= elePerBlock)) {
                range[factor - 1].second += elePerBlock;
                range[i].second -= elePerBlock;
            }
        }
        //update first of the pair
        int64_t base = 0;
        for (int64_t i = 0; i < factor; i++) {
            range[i].first = base;
            base += range[i].second;
        }
    }
}

static void SplitN(int64_t val, int64_t factor, vector<pair<int64_t, int64_t>> & range) {
    if (factor == 1) {
            range.push_back({0,val});
    } else {
        int64_t base = 0;
        int64_t stride1 = 0;
        int64_t stride2 = 0;
        int64_t s1Num = 0;
        int64_t s2Num = 0;
        SplitEvenly(factor, val, s1Num, s2Num, stride1, stride2);
        for (int64_t i = 0; i < s1Num; i++) {
            range.push_back({base, stride1});
            base += stride1;
        }
        for (int64_t i = 0; i < s2Num; i++) {
            range.push_back({base, stride2});
            base += stride2;
        }
        for (int64_t i = 0; i < factor - s1Num - s2Num; i++) {
            range.push_back({0, 0});
        }
    }
}

static void SplitColRowForCores(const CompilerInfo & compilerInfo,
                                const ShapeInfo & shapeInfo,
                                RuntimeInfo & runtimeInfo) {
    priority_queue<shared_ptr<TilingModel>, vector<shared_ptr<TilingModel>>, TMCompare> pqtm = runtimeInfo.pqtm;
    shared_ptr<TilingModel> tm = pqtm.top();

    SplitN(tm->ncr.nVol, tm->sp.nFactor, runtimeInfo.nRange);
    Split(tm->ncr.cVol, tm->sp.colFactor, shapeInfo.elePerBlock, runtimeInfo.colRange);
    Split(tm->ncr.rVol, tm->sp.rowFactor, shapeInfo.elePerBlock, runtimeInfo.rowRange);
    return;
}

static void SplitNByFactor(RuntimeInfo & runtimeInfo, int64_t elePerBlock) {
    priority_queue<shared_ptr<TilingModel>, vector<shared_ptr<TilingModel>>, TMCompare> pqtm = runtimeInfo.pqtm;
    shared_ptr<TilingModel> tm = pqtm.top();
    int64_t vol = tm->ncr.cVol * tm->ncr.rVol;
    int64_t factor = tm->sp.nFactor;
    runtimeInfo.rPartVol = vol;
    runtimeInfo.infoN.resize(factor);
    vector<InfoN> & info = runtimeInfo.infoN;
    for (int64_t i = 0; i < factor; i++) {
        info[i].nOffsetLogic = runtimeInfo.nRange[i].first;
        info[i].nOffsetActual = runtimeInfo.nRange[i].first * vol;
        info[i].loopOnN = runtimeInfo.nRange[i].second;

    }

    for (int64_t i = 1; i < runtimeInfo.nJumpAxisNum; i++) {
        runtimeInfo.nJumpFactorMod[i] *= runtimeInfo.nJumpFactorMod[i - 1] * runtimeInfo.nJumpFactor[i - 1];
    }

    for (int64_t i = 0; i < factor; i++) {
        info[i].initNTuple.resize(runtimeInfo.nJumpAxisNum);
        for (int64_t j = 0; j < runtimeInfo.nJumpAxisNum; j++) {
            info[i].initNTuple[j] = (info[i].nOffsetLogic / runtimeInfo.nJumpFactorMod[j]) % runtimeInfo.nJumpFactor[j];
        }
    }
}

static bool SplitColByFactor(const CompilerInfo & compilerInfo, RuntimeInfo & runtimeInfo, int64_t elePerBlock) {
    priority_queue<shared_ptr<TilingModel>, vector<shared_ptr<TilingModel>>, TMCompare> pqtm = runtimeInfo.pqtm;
    shared_ptr<TilingModel> tm = pqtm.top();
    int64_t maxCol = tm->maxCol;
    int64_t factor = tm->sp.colFactor;
    runtimeInfo.infoCol.resize(factor);
    vector<InfoCol> & info = runtimeInfo.infoCol;

    if (tm->Ist2f()) {
        for (int64_t i = 0; i < factor; i++) {
            info[i].colPerMC = runtimeInfo.colRange[i].second;
        }
    } else {
        for (int64_t i = 0; i < factor; i++) {
            int64_t col = runtimeInfo.colRange[i].second;
            if (col != 0) {
                int64_t k = col % elePerBlock;
                if (k != 0) {
                    col = col - k;
                }
                if (col == 0) {
                    return false;
                }
                info[i].colPerMC = min(maxCol, col);
                info[i].colBlockPerMC = info[i].colPerMC / elePerBlock;
                info[i].loopOnMC = col / info[i].colPerMC;
                info[i].backStepLeft = (k != 0) ? (elePerBlock - k) : 0;
                info[i].colTC = runtimeInfo.colRange[i].second - info[i].loopOnMC * info[i].colPerMC +\
                                info[i].backStepLeft;
                info[i].colBlockTC = info[i].colTC / elePerBlock;
                info[i].colOffset = runtimeInfo.colRange[i].first;
            }
        }

        for (int64_t i = 1; i < runtimeInfo.dstJumpAxisNum; i++) {
            runtimeInfo.dstJumpFactorMod[i] *= runtimeInfo.dstJumpFactorMod[i - 1] * runtimeInfo.dstJumpFactor[i - 1];
        }

        for (int64_t i = 0; i < factor; i++) {
            info[i].initDstTuple.resize(runtimeInfo.dstJumpAxisNum);
            info[i].tailDstTuple.resize(runtimeInfo.dstJumpAxisNum);
            for (int64_t j = 0; j < runtimeInfo.dstJumpAxisNum; j++) {
                info[i].initDstTuple[j] = (info[i].colOffset / runtimeInfo.dstJumpFactorMod[j]) %\
                                          runtimeInfo.dstJumpFactor[j];
                if (info[i].colTC != 0) {
                    int64_t col = info[i].colOffset + info[i].colPerMC * info[i].loopOnMC - info[i].backStepLeft;
                    info[i].tailDstTuple[j] = (col / runtimeInfo.dstJumpFactorMod[j]) % runtimeInfo.dstJumpFactor[j];
                }
            }
        }
    }
    return true;
}

static bool SplitRowByFactor(const CompilerInfo & compilerInfo, RuntimeInfo & runtimeInfo, int64_t elePerBlock) {
    priority_queue<shared_ptr<TilingModel>, vector<shared_ptr<TilingModel>>, TMCompare> pqtm = runtimeInfo.pqtm;
    shared_ptr<TilingModel> tm = pqtm.top();
    int64_t factor = tm->sp.rowFactor;
    int64_t maxRow = tm->maxRow;
    runtimeInfo.infoRow.resize(factor);
    vector<InfoRow> & info = runtimeInfo.infoRow;
    /*
     * vnchwconv max repeat is 255, so max block number is 255 * 16 = 4080
     */
    if (tm->Isf2t()) {
        for (int64_t i = 0; i < factor; i++) {
            info[i].rowPerMR = runtimeInfo.rowRange[i].second;
        }
    } else {
        for (int64_t i = 0; i < factor; i++) {
            int64_t row = runtimeInfo.rowRange[i].second;
            if (row != 0) {
                int64_t k = row % elePerBlock;
                if (k != 0) {
                    row = row - k;
                }
                if (row == 0) {
                    return false;
                }
                info[i].rowPerMR = min(maxRow, row);
                info[i].rowBlockPerMR = info[i].rowPerMR / elePerBlock;
                info[i].loopOnMR = row / info[i].rowPerMR;
                info[i].backStepUp = (k != 0) ? (elePerBlock - k) : 0;
                info[i].rowTR = runtimeInfo.rowRange[i].second - info[i].loopOnMR * info[i].rowPerMR +\
                                info[i].backStepUp;
                info[i].rowBlockTR = info[i].rowTR / elePerBlock;
                info[i].rowOffset = runtimeInfo.rowRange[i].first;
            }
        }

        for (int64_t i = 1; i < runtimeInfo.srcJumpAxisNum; i++) {
            runtimeInfo.srcJumpFactorMod[i] *= runtimeInfo.srcJumpFactorMod[i - 1] * runtimeInfo.srcJumpFactor[i - 1];
        }

        for (int64_t i = 0; i < factor; i++) {
            info[i].initSrcTuple.resize(runtimeInfo.srcJumpAxisNum);
            info[i].tailSrcTuple.resize(runtimeInfo.srcJumpAxisNum);
            for (int64_t j = 0; j < runtimeInfo.srcJumpAxisNum; j++) {
                info[i].initSrcTuple[j] = (info[i].rowOffset / runtimeInfo.srcJumpFactorMod[j]) %\
                                          runtimeInfo.srcJumpFactor[j];
                if (info[i].rowTR != 0) {
                    int64_t row = info[i].rowOffset + info[i].rowPerMR * info[i].loopOnMR - info[i].backStepUp;
                    info[i].tailSrcTuple[j] = (row / runtimeInfo.srcJumpFactorMod[j]) % runtimeInfo.srcJumpFactor[j];
                }
            }
        }
    }
    return true;
}

void CalcJumpInfo(RuntimeInfo & runtimeInfo,
                  int64_t dim,
                  const vector<int64_t> &inShape,
                  const vector<int64_t> &outShape,
                  const vector<int64_t> &perm) {

    priority_queue<shared_ptr<TilingModel>, vector<shared_ptr<TilingModel>>, TMCompare> pqtm = runtimeInfo.pqtm;
    shared_ptr<TilingModel> tm = pqtm.top();

    int64_t nAxisIndex = 0;
    int64_t srcAxisIndex = 0;
    int64_t dstAxisIndex = 0;

    runtimeInfo.nJumpAxisNum = tm->ncr.n.size();
    runtimeInfo.dstJumpAxisNum = tm->ncr.col.size();
    runtimeInfo.srcJumpAxisNum = tm->ncr.row.size();

    // 1. n jump
    for (int64_t i = runtimeInfo.nJumpAxisNum - 1; i >= 0; i--) {
        int64_t p = tm->ncr.n[i];
        runtimeInfo.nJumpFactor[nAxisIndex] = inShape[p];
        runtimeInfo.nJumpStride[nAxisIndex] = CalcStride(inShape, dim, p);
        nAxisIndex++;
    }

    // 2. src jump
    for (int64_t i = runtimeInfo.srcJumpAxisNum - 1; i >= 0; i--) {
        int64_t p = tm->ncr.row[i];
        runtimeInfo.srcJumpFactor[srcAxisIndex] = inShape[p];
        runtimeInfo.srcJumpStride[srcAxisIndex] = CalcStride(inShape, dim, p);
        srcAxisIndex++;
    }

    // 3. dst jump
    vector<int64_t> permDecrease = tm->ncr.col;
    std::sort(permDecrease.begin(), permDecrease.end(), std::greater<int64_t>());
    for (size_t i = 0; i < permDecrease.size(); i++) {
        int64_t p = permDecrease[i];
        int64_t index = GetPermIndex(perm, p);
        runtimeInfo.dstJumpFactor[i] = inShape[p];
        runtimeInfo.dstJumpStride[i] = CalcStride(outShape, dim, index);
        dstAxisIndex++;
    }
}

static bool IsContiguousIndex(const vector<int64_t> & perm, vector<int64_t> partialPerm) {
    vector<int64_t> index;
    for (size_t i = 0; i < partialPerm.size(); i++) {
        index.push_back(GetPermIndex(perm, partialPerm[i]));
    }
    sort(index.begin(), index.end());
    for (size_t i = 0; i < index.size() - 1 ; i++) {
        if (index[i] + 1 != index[i + 1]) {
            return false;
        }
    }
    return true;
}

static bool IsContiguousPerm(const vector<int64_t> & perm, vector<int64_t> partialPerm) {
    sort(partialPerm.begin(), partialPerm.end());
    for (size_t i = 0; i < partialPerm.size() - 1 ; i++) {
        if (partialPerm[i] + 1 != partialPerm[i + 1]) {
            return false;
        }
    }
    return true;
}

static bool IsContainTailAxis(const vector<int64_t> & perm, const vector<int64_t> & col) {
    int lastAxis = perm.size() - 1;
    for(size_t i = 0; i < col.size(); i++) {
        if (lastAxis == col[i]) {
            return true;
        }
    }
    return false;
}

static bool IsCouldBeCol(const vector<int64_t> & perm, const vector<int64_t> & col, const vector<int64_t> & colPerm) {
    if (!IsSubset(colPerm, col)) {
        return false;
    }
    if (!IsContainTailAxis(perm, col)) {
        return false;
    }
    if (!IsContiguousPerm(perm, col)) {
        return false;
    }
    return true;
}

static bool IsCouldBeRow(const vector<int64_t> & row, const vector<int64_t> & rowPerm) {
    if (IsSubset(rowPerm, row)) {
        return true;
    }
    return false;
}

static void FindLongestRowPerm(const ShapeInfo & shapeInfo, RuntimeInfo & runtimeInfo) {
    int64_t dim = shapeInfo.dim;
    int64_t re = dim - 1; //right endian
    int idx = GetPermIndex(shapeInfo.reducedPerm, re);
    for (int i = idx + 1; i < dim; i++) {
        runtimeInfo.rowPerm.push_back(shapeInfo.reducedPerm[i]);
    }
}

static void FindLongestColPerm(const ShapeInfo & shapeInfo,RuntimeInfo & runtimeInfo) {
    int64_t dim = shapeInfo.dim;
    int64_t le = shapeInfo.reducedPerm[dim - 1]; //left endian
    int idx = le;
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
        vector<int64_t> cPerm;
        for (int j = idx + 1; j < dim; j++) {
            cPerm.push_back(j);
        }
        if (IsContiguousIndex(shapeInfo.reducedPerm, cPerm)) {
            runtimeInfo.colPerm = cPerm;
            return;
        }
        idx++;
    }
}

static void GetN(const vector<int64_t> & perm,
                 const vector<int64_t> & row,
                 const vector<int64_t> & col,
                 vector<int64_t> & n) {
    vector<int64_t> rowAndCol;
    VectorAdd(row, col, rowAndCol);
    VectorSub(perm, rowAndCol, n);
}

static void FixNCRSeq(const vector<int64_t> & perm,
                      vector<int64_t> & n,
                      vector<int64_t> & col,
                      vector<int64_t> & row) {
    vector<int64_t> nt;
    vector<int64_t> ct;
    vector<int64_t> rt;
    nt.swap(n);
    ct.swap(col);
    rt.swap(row);
    for (size_t i = 0; i < perm.size(); i++) {
        for(size_t j = 0; j < nt.size(); j++) {
            if (perm[i] == nt[j]) {
                n.push_back(perm[i]);
                continue;
            }
        }
        for (size_t j = 0; j < ct.size(); j++) {
            if (perm[i] == ct[j]) {
                col.push_back(perm[i]);
                continue;
            }
        }
        for (size_t j = 0; j < rt.size(); j++) {
            if (perm[i] == rt[j]) {
                row.push_back(perm[i]);
            }
        }
    }
}

static int64_t CalcVolumeByPartialPerm(const ShapeInfo & shapeInfo,
                                       const vector<int64_t> & partialPerm) {
    int64_t vol = 1;
    for (size_t i = 0; i < partialPerm.size(); i++) {
        vol *= shapeInfo.reducedInShape[partialPerm[i]];
    }
    return vol;
}

static void DispatchNCR(const ShapeInfo & shapeInfo, RuntimeInfo & runtimeInfo) {
    vector<int64_t> row;
    int64_t dim = shapeInfo.dim;

    FindLongestRowPerm(shapeInfo, runtimeInfo);
    FindLongestColPerm(shapeInfo, runtimeInfo);

    for (int64_t i = dim - 1; i >= 0; i--) {
        row.push_back(shapeInfo.reducedPerm[i]);
        if (!IsCouldBeRow(row, runtimeInfo.rowPerm)) {
            break;
        }
        vector<int64_t> col;
        for (int64_t j = i - 1; j >= 0 ; j--) {
            col.push_back(shapeInfo.reducedPerm[j]);
            if (IsCouldBeCol(shapeInfo.reducedPerm, col, runtimeInfo.colPerm)) {
                vector<int64_t> n;
                GetN(shapeInfo.reducedPerm, row, col, n);
                NCR ncr;
                FixNCRSeq(shapeInfo.reducedPerm, n, col, row);
                ncr.n = n;
                ncr.col = col;
                ncr.row = row;
                ncr.nVol = CalcVolumeByPartialPerm(shapeInfo, n);
                ncr.cVol = CalcVolumeByPartialPerm(shapeInfo, col);
                ncr.rVol = CalcVolumeByPartialPerm(shapeInfo, row);
                runtimeInfo.ncrs.push_back(ncr);
            }
        }
    }
}

class Model001 : public TilingModel {
public:
    Model001(int64_t coreNum, int64_t ubBlocks) : TilingModel(1, coreNum, ubBlocks, LAST_AXIS_TR_COMMON, "Model001") {
        maxCol = CalcVnchwconvFullColSize(coreNum, ubBlocks) / 2;
        maxRow = 128;
    }
    ~Model001() {}
    void Decision(const NCR & n, int64_t dim) {
        bool res = ((n.nVol >= coreNum) && (n.cVol >= 64) && (n.rVol >= 64));
        if (res) {
            sp.Set(coreNum, 1, 1);
        } else {
            priority = INVALID_SPLIT;
        }
        ncr = n;
    }
};

class Model002 : public TilingModel {
public:
    Model002(int64_t coreNum, int64_t ubBlocks) : TilingModel(2, coreNum, ubBlocks, LAST_AXIS_TR_COMMON, "Model002") {
        maxCol = CalcVnchwconvFullColSize(coreNum, ubBlocks) / 2;
        maxRow = 128;
    }
    ~Model002() {}
    void Decision(const NCR & n, int64_t dim) {
        bool res = ((n.cVol >= 64) && (n.rVol >= 64 * coreNum));
        if (res) {
            sp.Set(1, 1, coreNum);
        } else {
            priority = INVALID_SPLIT;
        }
        ncr = n;
    }
};

class Model003 : public TilingModel {
public:
    Model003(int64_t coreNum, int64_t ubBlocks) : TilingModel(3, coreNum, ubBlocks, LAST_AXIS_TR_COMMON, "Model003") {
        maxCol = CalcVnchwconvFullColSize(coreNum, ubBlocks) / 2;
        maxRow = 128;
    }
    ~Model003() {}
    void Decision(const NCR & n, int64_t dim) {
        bool res = ((n.cVol >= 64 * coreNum) && (n.rVol >= 64));
        if (res) {
            sp.Set(1, coreNum, 1);
        } else {
            priority = INVALID_SPLIT;
        }
        ncr = n;
    }
};

class Model004 : public TilingModel {
public:
    Model004(int64_t coreNum, int64_t ubBlocks) : TilingModel(4, coreNum, ubBlocks, LAST_AXIS_TR_F2T, "Model004_f2t") {}
    ~Model004() {}
    void Decision(const NCR & n, int64_t dim) {
        ncr = n;
        if (n.col.size() != 1) {
            priority = INVALID_SPLIT;
            return;
        }

        int64_t ubSize = CalcVnchwconvPartialUbSize(coreNum, ubBlocks);
        maxCol = Align16(ubSize, n.rVol, ubSize) / 2;

        if ((n.cVol >= 128 * coreNum) && (n.rVol < F2T_THRESHOLD_B32)) {
            sp.Set(1, coreNum, 1);
        } else if ((n.cVol >= 128) && (n.rVol < F2T_THRESHOLD_B32)) {
            if (n.nVol > coreNum) {
                sp.Set(coreNum, 1, 1);
            } else {
                sp.Set(n.nVol, coreNum / n.nVol, 1);
            }
        } else {
            priority = INVALID_SPLIT;
        }
    }
    bool Isf2t() {
        return true;
    }
};

class Model005 : public TilingModel {
public:
    Model005(int64_t coreNum, int64_t ubBlocks) : TilingModel(5, coreNum, ubBlocks, LAST_AXIS_TR_T2F, "Model005_t2f") {}
    ~Model005() {}
    void Decision(const NCR & n, int64_t dim) {
        ncr = n;
        if (n.row.size() != 1) {
            priority = INVALID_SPLIT;
            return;
        }
        if (IsValid(ncr, dim) == false) {
            priority = INVALID_SPLIT;
            return;
        }

        int64_t ubSize = CalcVnchwconvPartialUbSize(coreNum, ubBlocks);
        maxRow = Align16(ubSize, n.cVol, ubSize) / 2;

        if ((n.cVol < F2T_THRESHOLD_B32) && (n.rVol >= 128 * coreNum)) {
            sp.Set(1, 1, coreNum);
        } else if ((n.cVol < F2T_THRESHOLD_B32) && (n.rVol >= 128)) {
            if (n.nVol > coreNum) {
                sp.Set(coreNum, 1, 1);
            } else {
                sp.Set(n.nVol, 1, coreNum / n.nVol);
            }
        } else {
            priority = INVALID_SPLIT;
        }
    }
    bool Ist2f() {
        return true;
    }
private:
    bool IsValid(const NCR & ncr, int64_t dim) {
        int64_t rowIndex = ncr.row[0];
        if (rowIndex + (int64_t)ncr.col.size() != dim - 1) {
            return false;
        }
        return true;
    }
};

class Model006 : public TilingModel {
public:
    Model006(int64_t coreNum, int64_t ubBlocks) : TilingModel(6, coreNum, ubBlocks, LAST_AXIS_TR_COMMON, "Model006") {
        maxCol = CalcVnchwconvFullColSize(coreNum, ubBlocks) / 2;
        maxRow = 128;
    }
    ~Model006() {}
    void Decision(const NCR & n, int64_t dim) {
        if (n.nVol >= coreNum) {
            if (n.cVol >= 8 && n.rVol >= 8) {
                sp.Set(coreNum, 1, 1);
            } else {
                priority = INVALID_SPLIT;
            }
        } else {
            if (n.cVol < 8 || n.rVol < 8) {
                priority = INVALID_SPLIT;
            } else {
                if (n.cVol > n.rVol) {
                    sp.Set(n.nVol, coreNum / n.nVol, 1);
                } else {
                    sp.Set(n.nVol, 1, coreNum / n.nVol);
                }
            }
        }
        ncr = n;
    }
};

class Model007 : public TilingModel {
public:
    Model007(int64_t coreNum, int64_t ubBlocks) : TilingModel(7, coreNum, ubBlocks, LAST_AXIS_TR_F2T, "Model007_f2t") {}
    ~Model007() {}
    void Decision(const NCR & n, int64_t dim) {
        ncr = n;
        if (n.col.size() != 1) {
            priority = INVALID_SPLIT;
            return;
        }

        int64_t ubSize = CalcVnchwconvPartialUbSize(coreNum, ubBlocks);
        maxCol = Align16(ubSize, n.rVol, ubSize) / 2;

        if ((n.cVol >= 8) && (n.rVol < F2T_THRESHOLD_B32)) {
            sp.Set(coreNum, 1, 1);
        } else {
            priority = INVALID_SPLIT;
        }
    }
    bool Isf2t() {
        return true;
    }
};

class Model008 : public TilingModel {
public:
    Model008(int64_t coreNum, int64_t ubBlocks) : TilingModel(8, coreNum, ubBlocks, LAST_AXIS_TR_T2F, "Model008_t2f") {}
    ~Model008() {}
    void Decision(const NCR & n, int64_t dim) {
        ncr = n;
        if (n.row.size() != 1) {
            priority = INVALID_SPLIT;
            return;
        }
        if (IsValid(ncr, dim) == false) {
            priority = INVALID_SPLIT;
            return;
        }

        int64_t ubSize = CalcVnchwconvPartialUbSize(coreNum, ubBlocks);
        maxRow = Align16(ubSize, n.cVol, ubSize) / 2;

        if ((n.cVol < F2T_THRESHOLD_B32) && (n.rVol >= 8)) {
            sp.Set(coreNum, 1, 1);
        } else {
            priority = INVALID_SPLIT;
        }
    }
    bool Ist2f() {
        return true;
    }
private:
    bool IsValid(const NCR & ncr, int64_t dim) {
        int64_t rowIndex = ncr.row[0];
        if (rowIndex + (int64_t)ncr.col.size() != dim - 1) {
            return false;
        }
        return true;
    }
};

class Model001_b16 : public TilingModel {
public:
    Model001_b16(int64_t coreNum, int64_t ubBlocks) :
                 TilingModel(1, coreNum, ubBlocks, LAST_AXIS_TR_COMMON, "Model001_b16") {
        maxCol = CalcVnchwconvFullColSize(coreNum, ubBlocks);
        maxRow = 128;
    }
    ~Model001_b16() {}
    void Decision(const NCR & n, int64_t dim) {
        bool res = ((n.nVol >= coreNum) && (n.cVol >= 64) && (n.rVol >= 64));
        if (res) {
            sp.Set(coreNum, 1, 1);
        } else {
            priority = INVALID_SPLIT;
        }
        ncr = n;
    }
};

class Model002_b16 : public TilingModel {
public:
    Model002_b16(int64_t coreNum, int64_t ubBlocks) :
                 TilingModel(2, coreNum, ubBlocks, LAST_AXIS_TR_COMMON, "Model002_b16") {
        maxCol = CalcVnchwconvFullColSize(coreNum, ubBlocks);
        maxRow = 128;
    }
    ~Model002_b16() {}
    void Decision(const NCR & n, int64_t dim) {
        bool res = ((n.cVol >= 64) && (n.rVol >= 64 * coreNum));
        if (res) {
            sp.Set(1, 1, coreNum);
        } else {
            priority = INVALID_SPLIT;
        }
        ncr = n;
    }
};

class Model003_b16 : public TilingModel {
public:
    Model003_b16(int64_t coreNum, int64_t ubBlocks) :
                 TilingModel(3, coreNum, ubBlocks, LAST_AXIS_TR_COMMON, "Model003_b16") {
        maxCol = CalcVnchwconvFullColSize(coreNum, ubBlocks);
        maxRow = 128;
    }
    ~Model003_b16() {}
    void Decision(const NCR & n, int64_t dim) {
        bool res = ((n.cVol >= 64 * coreNum) && (n.rVol >= 64));
        if (res) {
            sp.Set(1, coreNum, 1);
        } else {
            priority = INVALID_SPLIT;
        }
        ncr = n;
    }
};

class Model004_b16 : public TilingModel {
public:
    Model004_b16(int64_t coreNum, int64_t ubBlocks) :
                 TilingModel(4, coreNum, ubBlocks, LAST_AXIS_TR_F2T, "Model004_b16_f2t") {}
    ~Model004_b16() {}
    void Decision(const NCR & n, int64_t dim) {
        ncr = n;
        if (n.col.size() != 1) {
            priority = INVALID_SPLIT;
            return;
        }

        int64_t ubSize = CalcVnchwconvPartialUbSize(coreNum, ubBlocks);
        maxCol = Align16(ubSize, n.rVol, ubSize);

        if ((n.cVol >= 256 * coreNum) && (n.rVol <= F2T_THRESHOLD_B16)) {
            sp.Set(1, coreNum, 1);
        } else if ((n.cVol >= 256) && (n.rVol < F2T_THRESHOLD_B16)) {
            if (n.nVol > coreNum) {
                sp.Set(coreNum, 1, 1);
            } else {
                sp.Set(n.nVol, coreNum / n.nVol, 1);
            }
        } else {
            priority = INVALID_SPLIT;
        }
        if (n.col.size() > 1) {
            priority = INVALID_SPLIT;
        }
    }
    bool Isf2t() {
        return true;
    }
};

class Model005_b16 : public TilingModel {
public:
    Model005_b16(int64_t coreNum, int64_t ubBlocks) :
                 TilingModel(5, coreNum, ubBlocks, LAST_AXIS_TR_T2F, "Model005_b16_t2f") {}
    ~Model005_b16() {}
    void Decision(const NCR & n, int64_t dim) {
        ncr = n;
        if (n.row.size() != 1) {
            priority = INVALID_SPLIT;
            return;
        }
        if (IsValid(ncr, dim) == false) {
            priority = INVALID_SPLIT;
            return;
        }

        int64_t ubSize = CalcVnchwconvPartialUbSize(coreNum, ubBlocks);
        maxRow = Align16(ubSize, n.cVol, ubSize);

        if ((n.cVol <= F2T_THRESHOLD_B16) && (n.rVol >= 256 * coreNum)) {
            sp.Set(1, 1, coreNum);
        } else if ((n.cVol < F2T_THRESHOLD_B16) && (n.rVol >= 256)) {
            if (n.nVol > coreNum) {
                sp.Set(coreNum, 1, 1);
            } else {
                sp.Set(n.nVol, 1, coreNum / n.nVol);
            }
        } else {
            priority = INVALID_SPLIT;
        }
    }
    bool Ist2f() {
        return true;
    }
private:
    bool IsValid(const NCR & ncr, int64_t dim) {
        int64_t rowIndex = ncr.row[0];
        if (rowIndex + (int64_t)ncr.col.size() != dim - 1) {
            return false;
        }
        return true;
    }
};

class Model006_b16 : public TilingModel {
public:
    Model006_b16(int64_t coreNum, int64_t ubBlocks) :
                 TilingModel(6, coreNum, ubBlocks, LAST_AXIS_TR_COMMON, "Model006_b16") {
        maxCol = CalcVnchwconvFullColSize(coreNum, ubBlocks);
        maxRow = 128;
    }
    ~Model006_b16() {}
    void Decision(const NCR & n, int64_t dim) {
        if (n.nVol >= coreNum) {
            if (n.cVol >= 16 && n.rVol >= 16) {
                sp.Set(coreNum, 1, 1);
            } else {
                priority = INVALID_SPLIT;
            }
        } else {
            if (n.cVol < 16 || n.rVol < 16) {
                priority = INVALID_SPLIT;
            } else {
                if (n.cVol > n.rVol) {
                    sp.Set(n.nVol, coreNum / n.nVol, 1);
                } else {
                    sp.Set(n.nVol, 1, coreNum / n.nVol);
                }
            }
        }
        ncr = n;
    }
};

class Model007_b16 : public TilingModel {
public:
    Model007_b16(int64_t coreNum, int64_t ubBlocks) :
                 TilingModel(7, coreNum, ubBlocks, LAST_AXIS_TR_F2T, "Model007_b16_f2t") {
    }
    ~Model007_b16() {}
    void Decision(const NCR & n, int64_t dim) {
        ncr = n;
        if (n.col.size() != 1) {
            priority = INVALID_SPLIT;
            return;
        }

        int64_t ubSize = CalcVnchwconvPartialUbSize(coreNum, ubBlocks);
        maxCol = Align16(ubSize, n.rVol, ubSize);

        if ((n.cVol >= 16) && (n.rVol < F2T_THRESHOLD_B16)) {
            sp.Set(coreNum, 1, 1);
        } else {
            priority = INVALID_SPLIT;
        }
    }
    bool Isf2t() {
        return true;
    }
};

class Model008_b16 : public TilingModel {
public:
    Model008_b16(int64_t coreNum, int64_t ubBlocks) :
                 TilingModel(8, coreNum, ubBlocks, LAST_AXIS_TR_T2F, "Model008_b16_t2f") {
    }
    ~Model008_b16() {}
    void Decision(const NCR & n, int64_t dim) {
        ncr = n;
        if (n.row.size() != 1) {
            priority = INVALID_SPLIT;
            return;
        }
        if (IsValid(ncr, dim) == false) {
            priority = INVALID_SPLIT;
            return;
        }

        int64_t ubSize = CalcVnchwconvPartialUbSize(coreNum, ubBlocks);
        maxRow = Align16(ubSize, n.cVol, ubSize);

        if ((n.cVol < F2T_THRESHOLD_B16) && (n.rVol >= 16)) {
            sp.Set(coreNum, 1, 1);
        } else {
            priority = INVALID_SPLIT;
        }
    }
    bool Ist2f() {
        return true;
    }
private:
    bool IsValid(const NCR & ncr, int64_t dim) {
        int64_t rowIndex = ncr.row[0];
        if (rowIndex + (int64_t)ncr.col.size() != dim - 1) {
            return false;
        }
        return true;
    }
};

static void MakeNCRDecision(const CompilerInfo & compilerInfo,
                            const ShapeInfo & shapeInfo,
                            RuntimeInfo & runtimeInfo) {

#define ADD_MODEL(SpecificModel) \
    {\
        auto model = std::make_shared<SpecificModel>(compilerInfo.coreNum, compilerInfo.ubSize);\
        model->Decision(runtimeInfo.ncrs[i], shapeInfo.dim);\
        runtimeInfo.pqtm.push(model);\
    }

    if (compilerInfo.fp16Times == 2) {
        for (size_t i = 0; i < runtimeInfo.ncrs.size(); i++)  {
            ADD_MODEL(Model001);
            ADD_MODEL(Model002);
            ADD_MODEL(Model003);
            ADD_MODEL(Model004);
            ADD_MODEL(Model005);
            ADD_MODEL(Model006);
            ADD_MODEL(Model007);
            ADD_MODEL(Model008);
        }
    } else {
        for (size_t i = 0; i < runtimeInfo.ncrs.size(); i++)  {
            ADD_MODEL(Model001_b16);
            ADD_MODEL(Model002_b16);
            ADD_MODEL(Model003_b16);
            ADD_MODEL(Model004_b16);
            ADD_MODEL(Model005_b16);
            ADD_MODEL(Model006_b16);
            ADD_MODEL(Model007_b16);
            ADD_MODEL(Model008_b16);
        }
    }
}

static void Composite(RuntimeInfo & runtimeInfo, int64_t coreNum) {
    priority_queue<shared_ptr<TilingModel>, vector<shared_ptr<TilingModel>>, TMCompare> pqtm = runtimeInfo.pqtm;
    shared_ptr<TilingModel> tm = pqtm.top();
    int64_t nFactor = tm->sp.nFactor;
    int64_t colFactor = tm->sp.colFactor;
    int64_t rowFactor = tm->sp.rowFactor;
    runtimeInfo.infoPerCore.resize(coreNum);
    for (int64_t i = 0; i < nFactor; i++) {
        for (int64_t j = 0; j < colFactor; j++) {
            for (int64_t k = 0; k < rowFactor; k++) {
                int64_t coreId = i * colFactor * rowFactor + j * rowFactor +  k;
                runtimeInfo.infoPerCore[coreId].infoN = runtimeInfo.infoN[i];
                runtimeInfo.infoPerCore[coreId].infoCol = runtimeInfo.infoCol[j];
                runtimeInfo.infoPerCore[coreId].infoRow = runtimeInfo.infoRow[k];
            }
        }
    }
}

static bool IsScenario7Accept(const RuntimeInfo & runtimeInfo) {
    priority_queue<shared_ptr<TilingModel>, vector<shared_ptr<TilingModel>>, TMCompare> pqtm = runtimeInfo.pqtm;
    if (pqtm.empty()) {
        return false;
    }
    shared_ptr<TilingModel> tm = pqtm.top();
    if (tm->priority == INVALID_SPLIT) {
        return false;
    }
    return true;
}

static bool TilingDataScenario7(const CompilerInfo & compilerInfo,
                                         const ShapeInfo & shapeInfo,
                                         RuntimeInfo & runtimeInfo) {
    bool res = false;

    DispatchNCR(shapeInfo, runtimeInfo);

    MakeNCRDecision(compilerInfo, shapeInfo, runtimeInfo);

    CalcJumpInfo(runtimeInfo, shapeInfo.dim, shapeInfo.reducedInShape,
                 shapeInfo.reducedOutShape, shapeInfo.reducedPerm);

    SplitColRowForCores(compilerInfo, shapeInfo, runtimeInfo);

    SplitNByFactor(runtimeInfo, shapeInfo.elePerBlock);

    res = SplitColByFactor(compilerInfo, runtimeInfo, shapeInfo.elePerBlock);
    TRANSPOSE_CHECK_RET(res);

    res = SplitRowByFactor(compilerInfo, runtimeInfo, shapeInfo.elePerBlock);
    TRANSPOSE_CHECK_RET(res);

    Composite(runtimeInfo, compilerInfo.coreNum);

    return IsScenario7Accept(runtimeInfo);

}

static bool TilingDataScenario8(const CompilerInfo & compilerInfo,
                                         const ShapeInfo & shapeInfo,
                                         RuntimeInfo & runtimeInfo) {
    return true;
}

static bool ScenarioGuaranteed(const CompilerInfo &compilerInfo, ShapeInfo &shapeInfo, RuntimeInfo &runtimeInfo) {
    Reshape(shapeInfo);
    shapeInfo.scenario = SCENARIO_2;
    shapeInfo.isLastAxisHuge = true;
    return TilingDataScenario2(compilerInfo, shapeInfo, runtimeInfo);
}

static bool IsSpecificShape(const ShapeInfo &shapeInfo) {
    return false;
}

bool TransposeCalcTilingData(const string &opType,
                             const CompilerInfo &compilerInfo,
                             ShapeInfo &shapeInfo,
                             RuntimeInfo &runtimeInfo) {
    bool res = true;
    if (IsSpecificShape(shapeInfo)) {
        return ScenarioGuaranteed(compilerInfo, shapeInfo, runtimeInfo);
    }

    switch (shapeInfo.scenario) {
        case SCENARIO_0:
            res = TilingDataScenario0(compilerInfo, shapeInfo, runtimeInfo);
            OP_LOGI(opType.c_str(), "%s", PrintTilingInfoScenario0(compilerInfo, shapeInfo, runtimeInfo).c_str());
            break;
        case SCENARIO_1:
            res = TilingDataScenario1(compilerInfo, shapeInfo, runtimeInfo);
            OP_LOGD(opType.c_str(), "%s", PrintTilingInfoScenario1(compilerInfo, shapeInfo, runtimeInfo).c_str());
            break;
        case SCENARIO_2:
            res = TilingDataScenario2(compilerInfo, shapeInfo, runtimeInfo);
            OP_LOGI(opType.c_str(), "%s", PrintTilingInfoScenario2(compilerInfo, shapeInfo, runtimeInfo).c_str());
            break;
        case SCENARIO_3:
            res = TilingDataScenario3(compilerInfo, shapeInfo, runtimeInfo);
            OP_LOGI(opType.c_str(), "%s", PrintTilingInfoScenario3(compilerInfo, shapeInfo, runtimeInfo).c_str());
            break;
        case SCENARIO_4:
            res = TilingDataScenario4(compilerInfo, shapeInfo, runtimeInfo);
            OP_LOGI(opType.c_str(), "%s", PrintTilingInfoScenario4(compilerInfo, shapeInfo, runtimeInfo).c_str());
            break;
        case SCENARIO_6:
            res = TilingDataScenario6(compilerInfo, shapeInfo, runtimeInfo);
            OP_LOGI(opType.c_str(), "%s", PrintTilingInfoScenario6(compilerInfo, shapeInfo, runtimeInfo).c_str());
            break;
        case SCENARIO_7:
            res = TilingDataScenario7(compilerInfo, shapeInfo, runtimeInfo);
            if (res == false) {
                res = ScenarioGuaranteed(compilerInfo, shapeInfo, runtimeInfo);
                OP_LOGI(opType.c_str(), "%s", PrintTilingInfoScenario2(compilerInfo, shapeInfo, runtimeInfo).c_str());
            } else {
                OP_LOGI(opType.c_str(), "%s", PrintTilingInfoScenario7(compilerInfo, shapeInfo, runtimeInfo).c_str());
            }
            break;
        case SCENARIO_8:
            res = TilingDataScenario8(compilerInfo, shapeInfo, runtimeInfo);
            break;
        default:
            break;
    }
    return res;
}

bool GetCompileParams(const string & opType, const nlohmann::json &opCompileInfoJson, CompilerInfo & info) {
    OP_LOGD(opType.c_str(), "Entering GetCompileParams.");

    using namespace nlohmann;

    if (opCompileInfoJson.count("vars") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get vars error");
        return false;
    }

    auto allVars = opCompileInfoJson["vars"];

    info.opType = opType;

    if (allVars.count("core_num") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get core_num error");
        return false;
    }
    if (allVars.count("ub_size") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get ub_size error");
        return false;
    }
    if (allVars.count("dtype") == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get dtype error");
        return false;
    }

    info.coreNum = allVars["core_num"].get<std::int64_t>();
    info.ubSize = allVars["ub_size"].get<std::int64_t>();
    info.ubSizeCouldUse = info.ubSize - UB_RESERVED_BLOCK_SIZE;
    info.dType = allVars["dtype"].get<std::string>();
    info.fp16Times = (SizeofDType(info.dType) + 1) / 2; //add 1 for int8
    if (info.coreNum == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING(opType, "The core count cannot be zero!");
        return false;
    }

    OP_LOGD(opType.c_str(), "GetCompileParams, coreNum[%d], ubSize[%d] blocks, dType[%s].",
           info.coreNum, info.ubSize, info.dType.c_str());

    // for depthtospace and spacetodepth
    if ((opType == "DepthToSpace") || (opType == "SpaceToDepth")) {
        if (allVars.count("block_size") == 0) {
            VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get block_size error");
            return false;
        }
        info.blockSize = allVars["block_size"].get<std::int64_t>();
        OP_LOGD(opType.c_str(), "GetCompileParams, blockSize[%d].", info.blockSize);
    }

    return true;
}

#define WRITE_DATA_H(v) headVec.push_back(v)
#define WRITE_DATA_F(v) fixedVec.push_back(v)
#define WRITE_DATA_P(v) perCoreVec.push_back(v)

static void SerializeScenario0(OpRunInfo &runInfo,
                               const CompilerInfo &compilerInfo,
                               const ShapeInfo &shapeInfo,
                               const RuntimeInfo &runtimeInfo) {
    vector<int64_t> headVec;
    vector<int64_t> fixedVec;
    vector<int64_t> perCoreVec;

    // part1: head
    WRITE_DATA_H(shapeInfo.scenario);              //0 : scenario
    WRITE_DATA_H(0);                               //1 : fixed_len
    WRITE_DATA_H(0);                               //2 : percore_len
    WRITE_DATA_H(0);                               //3 : subSceanrio

    // part2: fixed
    WRITE_DATA_F(compilerInfo.coreNum);
    WRITE_DATA_F(compilerInfo.ubSize);

    // part3: per core
    int perCoreLen = 0;
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        WRITE_DATA_P(runtimeInfo.infoPerCoreIdentical[i].base);
        WRITE_DATA_P(runtimeInfo.infoPerCoreIdentical[i].eleNum);
        WRITE_DATA_P(runtimeInfo.infoPerCoreIdentical[i].majorLoop);
        WRITE_DATA_P(runtimeInfo.infoPerCoreIdentical[i].majorNum);
        WRITE_DATA_P(runtimeInfo.infoPerCoreIdentical[i].tailNum);
        WRITE_DATA_P(runtimeInfo.infoPerCoreIdentical[i].notAlignEle);
        if (perCoreLen == 0) {
            perCoreLen = perCoreVec.size();
        }
    }

    BlockAlign(fixedVec);
    BlockAlign(perCoreVec);

    headVec[1] = fixedVec.size();
    headVec[2] = perCoreLen;

    for (size_t i = 0; i < headVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, headVec[i]);
    }
    for (size_t i = 0; i < fixedVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, fixedVec[i]);
    }
    for (size_t i = 0; i < perCoreVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, perCoreVec[i]);
    }

    runInfo.block_dim = compilerInfo.coreNum;
    std::vector<int64_t> workspace;
    workspace.push_back(1024);
    runInfo.workspaces = workspace;
}

static void SerializeScenario1(OpRunInfo &runInfo,
                                         const CompilerInfo & compilerInfo,
                                         const ShapeInfo & shapeInfo,
                                         const RuntimeInfo & runtimeInfo) {
    vector<int64_t> headVec;
    vector<int64_t> fixedVec;
    vector<int64_t> perCoreVec;

    // part1: head
    WRITE_DATA_H(shapeInfo.scenario);              //0 : scenario
    WRITE_DATA_H(0);                               //1 : fixed_len
    WRITE_DATA_H(0);                               //2 : percore_len
    WRITE_DATA_H(0);                               //3 : subSceanrio

    // part2: fixed
    WRITE_DATA_F(compilerInfo.coreNum);
    WRITE_DATA_F(compilerInfo.ubSize);
    WRITE_DATA_F(shapeInfo.lastAxisLen);
    WRITE_DATA_F(shapeInfo.lastAxisBurstLen);
    WRITE_DATA_F(shapeInfo.alignElement);
    WRITE_DATA_F(shapeInfo.dim - 1);

    for (int i = 0; i < shapeInfo.dim - 1; i++) {
        WRITE_DATA_F(runtimeInfo.srcJumpStride[i]);
    }
    for (int i = 0; i < shapeInfo.dim - 1; i++) {
        WRITE_DATA_F(runtimeInfo.dstJumpStride[i]);
    }
    for (int i = 0; i < shapeInfo.dim - 1; i++) {
        WRITE_DATA_F(runtimeInfo.dstJumpFactor[i]);
    }

    // part3: per core
    int perCoreLen = 0;
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        WRITE_DATA_P(runtimeInfo.infoPerCoreLastAxisNT[i].num);
        for (int j = 0; j < shapeInfo.dim - 1; j++) {
            WRITE_DATA_P(runtimeInfo.infoPerCoreLastAxisNT[i].initTuple[j]);
        }
        if (perCoreLen == 0) {
            perCoreLen = perCoreVec.size();
        }
    }

    BlockAlign(fixedVec);
    BlockAlign(perCoreVec);

    headVec[1] = fixedVec.size();
    headVec[2] = perCoreLen;

    for (size_t i = 0; i < headVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, headVec[i]);
    }
    for (size_t i = 0; i < fixedVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, fixedVec[i]);
    }
    for (size_t i = 0; i < perCoreVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, perCoreVec[i]);
    }

    runInfo.block_dim = compilerInfo.coreNum;
    std::vector<int64_t> workspace;
    workspace.push_back(1024);
    runInfo.workspaces = workspace;
}

static void SerializeScenario2(OpRunInfo &runInfo,
                                        const CompilerInfo & compilerInfo,
                                        const ShapeInfo & shapeInfo,
                                        const RuntimeInfo & runtimeInfo) {
    vector<int64_t> headVec;
    vector<int64_t> fixedVec;
    vector<int64_t> perCoreVec;

    // part1: head
    WRITE_DATA_H(shapeInfo.scenario);              //0 : scenario
    WRITE_DATA_H(0);                               //1 : fixed_len
    WRITE_DATA_H(0);                               //2 : percore_len
    WRITE_DATA_H(0);                               //3 : subSceanrio

    // part2: fixed
    WRITE_DATA_F(compilerInfo.coreNum);
    WRITE_DATA_F(compilerInfo.ubSize);
    WRITE_DATA_F(shapeInfo.lastAxisLen);
    WRITE_DATA_F(shapeInfo.lastAxisBurstLen);
    WRITE_DATA_F(shapeInfo.alignElement);
    WRITE_DATA_F(shapeInfo.dim - 1);
    if (runtimeInfo.srcStrideLogic * shapeInfo.lastAxisBurstLen <= STRIDE_BOUNDARY) {
        WRITE_DATA_F(runtimeInfo.srcStrideLogic * shapeInfo.lastAxisBurstLen);
    } else {
        WRITE_DATA_F((int64_t)0);
    }
    WRITE_DATA_F(runtimeInfo.backNum);
    WRITE_DATA_F(runtimeInfo.skipEle);

    for (int i = 0; i < shapeInfo.dim - 1; i++) {
        WRITE_DATA_F(runtimeInfo.srcJumpStride[i]);
    }
    for (int i = 0; i < shapeInfo.dim - 1; i++) {
        WRITE_DATA_F(runtimeInfo.dstJumpStride[i]);
    }
    for (int i = 0; i < shapeInfo.dim - 1; i++) {
        WRITE_DATA_F(runtimeInfo.dstJumpFactor[i]);
    }
    for (int i = 0; i < shapeInfo.dim - 1; i++) {
        WRITE_DATA_F(runtimeInfo.dstJumpFactorMod[i]);
    }

    // part3: per core
    int perCoreLen = 0;
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        WRITE_DATA_P(runtimeInfo.infoPerCoreLastAxisNT[i].base);
        WRITE_DATA_P(runtimeInfo.infoPerCoreLastAxisNT[i].num);
        for (int j = 0; j < shapeInfo.dim - 1; j++) {
            WRITE_DATA_P(runtimeInfo.infoPerCoreLastAxisNT[i].initTuple[j]);
        }
        const InfoPerCoreLastAxisNT & infoPerCore = runtimeInfo.infoPerCoreLastAxisNT[i];
        const LastAxisNTLoopInfo & loopInfo = infoPerCore.loopInfo;
        WRITE_DATA_P(loopInfo.headMajorLoop);
        WRITE_DATA_P(loopInfo.headMajorNum);
        WRITE_DATA_P(loopInfo.headTailNum);
        WRITE_DATA_P(loopInfo.bodyLoopNum);
        WRITE_DATA_P(loopInfo.bodyMajorLoop);
        WRITE_DATA_P(loopInfo.bodyMajorNum);
        WRITE_DATA_P(loopInfo.bodyTailNum);
        WRITE_DATA_P(loopInfo.tailMajorLoop);
        WRITE_DATA_P(loopInfo.tailMajorNum);
        WRITE_DATA_P(loopInfo.tailTailNum);

        if (perCoreLen == 0) {
            perCoreLen = perCoreVec.size();
        }
    }

    BlockAlign(fixedVec);
    BlockAlign(perCoreVec);

    headVec[1] = fixedVec.size();
    headVec[2] = perCoreLen;

    for (size_t i = 0; i < headVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, headVec[i]);
    }
    for (size_t i = 0; i < fixedVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, fixedVec[i]);
    }
    for (size_t i = 0; i < perCoreVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, perCoreVec[i]);
    }

    runInfo.block_dim = compilerInfo.coreNum;
    std::vector<int64_t> workspace;
    workspace.push_back(1024);
    runInfo.workspaces = workspace;
}

static void SerializeScenario3(OpRunInfo &runInfo,
                                         const CompilerInfo & compilerInfo,
                                         const ShapeInfo & shapeInfo,
                                         const RuntimeInfo & runtimeInfo) {
    vector<int64_t> headVec;
    vector<int64_t> fixedVec;
    vector<int64_t> perCoreVec;

    // part1: head
    WRITE_DATA_H(shapeInfo.scenario);              //0 : scenario
    WRITE_DATA_H(0);                               //1 : fixed_len
    WRITE_DATA_H(0);                               //2 : percore_len
    WRITE_DATA_H(0);                               //3 : subSceanrio

    // part2: fixed
    WRITE_DATA_F(compilerInfo.coreNum);
    WRITE_DATA_F(compilerInfo.ubSize);
    WRITE_DATA_F(shapeInfo.lastAxisLen);
    WRITE_DATA_F(shapeInfo.lastAxisBurstLen);
    WRITE_DATA_F(shapeInfo.alignElement);
    WRITE_DATA_F(shapeInfo.dim - 1);
    WRITE_DATA_F(runtimeInfo.hugeInfo.majorLoopNum);
    WRITE_DATA_F(runtimeInfo.hugeInfo.majorBlocks);
    WRITE_DATA_F(runtimeInfo.hugeInfo.tailBlocks);
    WRITE_DATA_F(runtimeInfo.hugeInfo.backEle);

    for (int i = 0; i < shapeInfo.dim - 1; i++) {
        WRITE_DATA_F(runtimeInfo.srcJumpStride[i]);
    }
    for (int i = 0; i < shapeInfo.dim - 1; i++) {
        WRITE_DATA_F(runtimeInfo.dstJumpStride[i]);
    }
    for (int i = 0; i < shapeInfo.dim - 1; i++) {
        WRITE_DATA_F(runtimeInfo.dstJumpFactor[i]);
    }

    // part3: per core
    int perCoreLen = 0;
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        WRITE_DATA_P(runtimeInfo.infoPerCoreLastAxisNT[i].num);
        for (int j = 0; j < shapeInfo.dim - 1; j++) {
            WRITE_DATA_P(runtimeInfo.infoPerCoreLastAxisNT[i].initTuple[j]);
        }
        if (perCoreLen == 0) {
            perCoreLen = perCoreVec.size();
        }
    }

    BlockAlign(fixedVec);
    BlockAlign(perCoreVec);

    headVec[1] = fixedVec.size();
    headVec[2] = perCoreLen;

    for (size_t i = 0; i < headVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, headVec[i]);
    }
    for (size_t i = 0; i < fixedVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, fixedVec[i]);
    }
    for (size_t i = 0; i < perCoreVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, perCoreVec[i]);
    }

    runInfo.block_dim = compilerInfo.coreNum;
    std::vector<int64_t> workspace;
    workspace.push_back(1024);
    runInfo.workspaces = workspace;
}

static void SerializeScenario4(OpRunInfo &runInfo,
                               const CompilerInfo & compilerInfo,
                               const ShapeInfo & shapeInfo,
                               const RuntimeInfo & runtimeInfo) {
    vector<int64_t> headVec;
    vector<int64_t> fixedVec;
    vector<int64_t> perCoreVec;
    const BorrowInfo& borrowInfo = runtimeInfo.borrowInfo;

    // part1: head
    WRITE_DATA_H(shapeInfo.scenario);              //0 : scenario
    WRITE_DATA_H(0);                               //1 : fixed_len
    WRITE_DATA_H(0);                               //2 : percore_len
    WRITE_DATA_H(0);                               //3 : subSceanrio

    // part2: fixed
    WRITE_DATA_F(shapeInfo.lastAxisLen);
    WRITE_DATA_F(shapeInfo.lastAxisBurstLen);
    WRITE_DATA_F(shapeInfo.alignElement);
    WRITE_DATA_F(borrowInfo.otherNum + 2);//logic_axis_num
    WRITE_DATA_F(borrowInfo.otherNum);
    WRITE_DATA_F(borrowInfo.srcNumNoDup);
    WRITE_DATA_F(borrowInfo.dstNumNoDup);
    WRITE_DATA_F(borrowInfo.majorBurstLen_in);
    WRITE_DATA_F(borrowInfo.tailBurstLen_in);
    WRITE_DATA_F(borrowInfo.majorBurstLen_out);
    WRITE_DATA_F(borrowInfo.tailBurstLen_out);
    WRITE_DATA_F(borrowInfo.majorDstLoop_in);
    WRITE_DATA_F(borrowInfo.tailDstLoop_in);
    WRITE_DATA_F(borrowInfo.majorSrcLoop_out);
    WRITE_DATA_F(borrowInfo.tailSrcLoop_out);
    WRITE_DATA_F(borrowInfo.majorInEle);
    WRITE_DATA_F(borrowInfo.tailInEle);
    WRITE_DATA_F(borrowInfo.majorInTailEle);
    WRITE_DATA_F(borrowInfo.tailInTailEle);
    WRITE_DATA_F(borrowInfo.majorOutEle);
    WRITE_DATA_F(borrowInfo.tailOutEle);
    WRITE_DATA_F(borrowInfo.majorOutTailEle);
    WRITE_DATA_F(borrowInfo.tailOutTailEle);
    WRITE_DATA_F(borrowInfo.dstIndexOut[borrowInfo.dstNum - 1].step);
    WRITE_DATA_F(borrowInfo.srcIndexIn[0].step);
    WRITE_DATA_F(borrowInfo.dupAxis);
    WRITE_DATA_F(borrowInfo.srcAxisPerm);
    WRITE_DATA_F(borrowInfo.dstAxisPerm);
    WRITE_DATA_F(borrowInfo.axisPerm);

    for (int i = 0; i < UB_REORDER_COMBINATION; i++) {
        const LRSB* lrsb = borrowInfo.lrsb[i];
        for (int j = 0; j < UB_REORDER_LOOP; j++) {
            WRITE_DATA_F(lrsb[j].loop);
        }
        for (int j = 0; j < UB_REORDER_LOOP; j++) {
            WRITE_DATA_F(lrsb[j].repeat);
        }
        for (int j = 0; j < UB_REORDER_LOOP; j++) {
            WRITE_DATA_F(lrsb[j].srcStride);
        }
        for (int j = 0; j < UB_REORDER_LOOP; j++) {
            WRITE_DATA_F(lrsb[j].dstStride);
        }
        for (int j = 0; j < UB_REORDER_LOOP; j++) {
            WRITE_DATA_F(lrsb[j].burstLen);
        }
        for (int j = 0; j < UB_REORDER_LOOP; j++) {
            WRITE_DATA_F(lrsb[j].srcOffset);
        }
        for (int j = 0; j < UB_REORDER_LOOP; j++) {
            WRITE_DATA_F(lrsb[j].dstOffset);
        }
    }

    for (int64_t i = 0 ; i < borrowInfo.dstNumNoDup; i++) {
        WRITE_DATA_F(borrowInfo.dstFactorCopyIn[i]);
    }
    for (int64_t i = 0 ; i < borrowInfo.srcNumNoDup; i++) {
        WRITE_DATA_F(borrowInfo.srcFactorCopyOut[i]);
    }

    WRITE_DATA_F(borrowInfo.srcJumpFactorLogic_in);
    WRITE_DATA_F(borrowInfo.dstJumpFactorLogic_in);

    for (int64_t i = 0 ; i < borrowInfo.otherNum; i++) {
        WRITE_DATA_F(borrowInfo.otherJumpFactor_in[i]);
    }

    for (int64_t i = 0 ; i < borrowInfo.dstNumNoDup; i++) {
        WRITE_DATA_F(borrowInfo.dstStrideCopyIn[i]);
    }
    for (int64_t i = 0 ; i < borrowInfo.srcNumNoDup; i++) {
        WRITE_DATA_F(borrowInfo.srcStrideCopyOut[i]);
    }

    //logicStrideIn,first two is 0 for no use
    for (int i = 0; i < 2; i++) {
        WRITE_DATA_F(0);
    }
    for (int64_t i = 0 ; i < borrowInfo.otherNum; i++) {
        WRITE_DATA_F(borrowInfo.otherJumpStride_in[i]);
    }

    //logicStrideOut, first two is 0 for no use
    for (int i = 0; i < 2; i++) {
        WRITE_DATA_F(0);
    }
    for (int64_t i = 0 ; i < borrowInfo.otherNum; i++) {
        WRITE_DATA_F(borrowInfo.otherJumpStride_out[i]);
    }

    // part3: per core
    int perCoreLen = 0;
    for (int64_t i = 0; i < compilerInfo.coreNum; i++) {
        WRITE_DATA_P(borrowInfo.loopPerCore[i]);
        for (int64_t j = 0; j < borrowInfo.srcNumNoDup; j++) {
            WRITE_DATA_P(borrowInfo.srcAxis_in[i].initTuple[j]);
        }
        for (int64_t j = 0; j < borrowInfo.dstNumNoDup; j++) {
            WRITE_DATA_P(borrowInfo.dstAxis_in[i].initTuple[j]);
        }

        WRITE_DATA_P(borrowInfo.srcAxis_in[i].initTupleLogic);
        WRITE_DATA_P(borrowInfo.dstAxis_in[i].initTupleLogic);

        for(int64_t j = 0; j < borrowInfo.otherNum; j++) {
            WRITE_DATA_P(borrowInfo.otherAxis_in[i].initTuple[j]);
        }
        if (perCoreLen == 0) {
            perCoreLen = perCoreVec.size();
        }
    }

    BlockAlign(fixedVec);
    BlockAlign(perCoreVec);

    headVec[1] = fixedVec.size();
    headVec[2] = perCoreLen;

    for (size_t i = 0; i < headVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, headVec[i]);
    }
    for (size_t i = 0; i < fixedVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, fixedVec[i]);
    }
    for (size_t i = 0; i < perCoreVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, perCoreVec[i]);
    }

    runInfo.block_dim = compilerInfo.coreNum;
    std::vector<int64_t> workspace;
    workspace.push_back(1024);
    runInfo.workspaces = workspace;
}

static void SerializeScenario6(OpRunInfo &runInfo,
                                         const CompilerInfo & compilerInfo,
                                         const ShapeInfo & shapeInfo,
                                         const RuntimeInfo & runtimeInfo) {
    SerializeScenario2(runInfo, compilerInfo, shapeInfo, runtimeInfo);
}

static void SerializeScenario7(OpRunInfo &runInfo,
                                         const CompilerInfo &compilerInfo,
                                         const ShapeInfo &shapeInfo,
                                         const RuntimeInfo &runtimeInfo) {
    priority_queue<shared_ptr<TilingModel>, vector<shared_ptr<TilingModel>>, TMCompare> pqtm = runtimeInfo.pqtm;
    shared_ptr<TilingModel> tm = pqtm.top();

    vector<int64_t> headVec;
    vector<int64_t> fixedVec;
    vector<int64_t> perCoreVec;

    // part1: head
    WRITE_DATA_H(shapeInfo.scenario);              //0 : scenario
    WRITE_DATA_H(0);                               //1 : fixed_len
    WRITE_DATA_H(0);                               //2 : percore_len
    WRITE_DATA_H(tm->subScenario);                 //3 : subSceanrio

    // part2: fixed
    WRITE_DATA_F(compilerInfo.coreNum);
    WRITE_DATA_F(compilerInfo.ubSize);
    WRITE_DATA_F(runtimeInfo.nJumpAxisNum);
    WRITE_DATA_F(runtimeInfo.dstJumpAxisNum);
    WRITE_DATA_F(runtimeInfo.srcJumpAxisNum);
    WRITE_DATA_F(runtimeInfo.rPartVol);

    for (int i = 0; i < runtimeInfo.nJumpAxisNum; i++) {
        WRITE_DATA_F(runtimeInfo.nJumpFactor[i]);
    }
    for (int i = 0; i < runtimeInfo.nJumpAxisNum; i++) {
        WRITE_DATA_F(runtimeInfo.nJumpStride[i]);
    }
    for (int i = 0; i < runtimeInfo.dstJumpAxisNum; i++) {
        WRITE_DATA_F(runtimeInfo.dstJumpFactor[i]);
    }
    for (int i = 0; i < runtimeInfo.dstJumpAxisNum; i++) {
        WRITE_DATA_F(runtimeInfo.dstJumpStride[i]);
    }
    for (int i = 0; i < runtimeInfo.srcJumpAxisNum; i++) {
        WRITE_DATA_F(runtimeInfo.srcJumpFactor[i]);
    }
    for (int i = 0; i < runtimeInfo.srcJumpAxisNum; i++) {
        WRITE_DATA_F(runtimeInfo.srcJumpStride[i]);
    }

    // part3: per core
    int perCoreLen = 0;
    for (int i = 0; i < compilerInfo.coreNum; i++) {
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoN.loopOnN);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoN.nOffsetActual);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoCol.colPerMC);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoCol.loopOnMC);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoCol.colTC);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoCol.colOffset);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoCol.backStepLeft);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoRow.rowPerMR);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoRow.loopOnMR);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoRow.rowTR);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoRow.rowOffset);
        WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoRow.backStepUp);

        for (int j = 0; j < runtimeInfo.nJumpAxisNum; j++) {
            WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoN.initNTuple[j]);
        }
        for (int j = 0; j < runtimeInfo.dstJumpAxisNum; j++) {
            WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoCol.initDstTuple[j]);
        }
        for (int j = 0; j < runtimeInfo.dstJumpAxisNum; j++) {
            WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoCol.tailDstTuple[j]);
        }
        for (int j = 0; j < runtimeInfo.srcJumpAxisNum; j++) {
            WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoRow.initSrcTuple[j]);
        }
        for (int j = 0; j < runtimeInfo.srcJumpAxisNum; j++) {
            WRITE_DATA_P(runtimeInfo.infoPerCore[i].infoRow.tailSrcTuple[j]);
        }
        if (perCoreLen == 0) {
            perCoreLen = perCoreVec.size();
        }
    }

    BlockAlign(fixedVec);
    BlockAlign(perCoreVec);

    headVec[1] = fixedVec.size();
    headVec[2] = perCoreLen;

    for (size_t i = 0; i < headVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, headVec[i]);
    }
    for (size_t i = 0; i < fixedVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, fixedVec[i]);
    }
    for (size_t i = 0; i < perCoreVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, perCoreVec[i]);
    }

    runInfo.block_dim = compilerInfo.coreNum;
    std::vector<int64_t> workspace;
    workspace.push_back(1024);
    runInfo.workspaces = workspace;
}

static void SerializeScenario8(OpRunInfo &runInfo,
                               const CompilerInfo &compilerInfo,
                               const ShapeInfo &shapeInfo,
                               const RuntimeInfo &runtimeInfo) {
    vector<int64_t> headVec;

    // part1: head
    WRITE_DATA_H(shapeInfo.scenario);              //0 : scenario
    WRITE_DATA_H(0);                               //1 : fixed_len
    WRITE_DATA_H(0);                               //2 : percore_len
    WRITE_DATA_H(0);                               //3 : subSceanrio

    for (size_t i = 0; i < headVec.size(); i++) {
        ByteBufferPut(runInfo.tiling_data, headVec[i]);
    }

    runInfo.block_dim = compilerInfo.coreNum;
    std::vector<int64_t> workspace;
    workspace.push_back(1024);
    runInfo.workspaces = workspace;
}

void SerializeTilingData(OpRunInfo &runInfo,
                         const CompilerInfo & compilerInfo,
                         const ShapeInfo & shapeInfo,
                         const RuntimeInfo & runtimeInfo)
{
    switch (shapeInfo.scenario) {
        case SCENARIO_0:
            SerializeScenario0(runInfo, compilerInfo, shapeInfo, runtimeInfo);
            break;
        case SCENARIO_1:
            SerializeScenario1(runInfo, compilerInfo, shapeInfo, runtimeInfo);
            break;
        case SCENARIO_2:
            SerializeScenario2(runInfo, compilerInfo, shapeInfo, runtimeInfo);
            break;
        case SCENARIO_3:
            SerializeScenario3(runInfo, compilerInfo, shapeInfo, runtimeInfo);
            break;
        case SCENARIO_4:
            SerializeScenario4(runInfo, compilerInfo, shapeInfo, runtimeInfo);
            break;
        case SCENARIO_6:
            SerializeScenario6(runInfo, compilerInfo, shapeInfo, runtimeInfo);
            break;
        case SCENARIO_7:
            SerializeScenario7(runInfo, compilerInfo, shapeInfo, runtimeInfo);
            break;
        case SCENARIO_8:
            SerializeScenario8(runInfo, compilerInfo, shapeInfo, runtimeInfo);
            break;
        default:
            break;
    }
}

bool TransposeTiling(const std::string &opType,
                     const TeOpParas &opParas,
                     const nlohmann::json &opInfo,
                     OpRunInfo &runInfo) {
    OP_LOGI(opType.c_str(), "Tiling is running.");

    CompilerInfo compilerInfo;
    ShapeInfo shapeInfo;
    RuntimeInfo runtimeInfo;

    if (GetCompileParams(opType, opInfo, compilerInfo) == false) {
        return false;
    }
    // for depthtospace and spacetodepth
    if ((opType == "DepthToSpace") || (opType == "SpaceToDepth")) {
        if (AddShapePerm(opType, opParas, compilerInfo, shapeInfo) == false) {
            return false;
        }
    } else {
        if (GetShapePerm(opType, opParas, shapeInfo) == false) {
            return false;
        }
    }
    if (CheckTensorShape(opType, shapeInfo) == false) {
        return false;
    }

    ReduceAxis(opType, compilerInfo, shapeInfo);

    if (TransposeCalcTilingData(opType, compilerInfo, shapeInfo, runtimeInfo) == false) {
        return false;
    }

    SerializeTilingData(runInfo, compilerInfo, shapeInfo, runtimeInfo);

    return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(DepthToSpace, TransposeTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(SpaceToDepth, TransposeTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(Transpose, TransposeTiling);

}
