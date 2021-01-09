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
 * \file trans_data.cpp
 * \brief dynamic TransData op tiling
 */
#include <string>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "trans_data_common.h"

namespace optiling {

const int32_t C0_32 = 32;
const int32_t C0_16 = 16;
const int32_t BLOCK_SIZE = 32;
const int32_t STRIDE_LIMIT_MTE = 65535;
const int32_t FRAME_LEVEL = 6;
const int32_t CUBE_SIZE = 16;

struct HeadTilingParam {
  int32_t shapeLoopCnt;
};

struct TransDataMode100Param {
  int32_t tillingParamCount;

  int32_t tilingMode;
  int32_t ubOffset;
  int32_t usedCoreCnt;
  int32_t coreStepIn;
  int32_t coreStepOut;
  int32_t oneLineSize;

  int32_t inLevel2C1LpStepIn;
  int32_t inLevel2C1LpStepOut;
  int32_t inLevel1LastLpStepIn;
  int32_t inLevel1LastLpStepOut;
  int32_t inLevel0LpCnt;
  int32_t inLevel0C0LpStepUb;
  int32_t inLevel0LpStepIn;
  int32_t inLevel0RepeatCnt;
  int32_t inLevel0Nburst;
  int32_t inLevel0SrcStride;
  int32_t inLevel0DstStride;
  int32_t outLevel1LpStepUb;
  int32_t outLevel1LpStepOut;
  int32_t outLevel0LpCnt;
  int32_t outLevel0LpStepUb;
  int32_t outLevel0LpStepOut;
  int32_t outLevel0RepeatCnt;
  int32_t outLevel0Nburst;
  int32_t outLevel0SrcStride;
  int32_t outLevel0DstStride;

  int32_t inLevel2NlcC1LpCnt;
  int32_t inLevel2NlcC1LeftLines;
  int32_t inLevel1NlcLastLpCnt;
  int32_t inLevel1NlcLastLeftLines;
  int32_t inLevel0NlcLpCnt;
  int32_t inLevel0NlcRepeatCnt;
  int32_t inLevel0NlcNburst;
  int32_t outLevel1NlcLpCnt;
  int32_t outLevel0NlcLpCnt;
  int32_t outLevel0NlcRepeatCnt;
  int32_t outLevel0NlcNburst;

  int32_t inLevel2LcC1LpCnt;
  int32_t inLevel2LcC1LeftLines;
  int32_t inLevel1LcLastLpCnt;
  int32_t inLevel1LcLastLeftLines;
  int32_t inLevel0LcLpCnt;
  int32_t inLevel0LcRepeatCnt;
  int32_t inLevel0LcNburst;
  int32_t outLevel1LcLpCnt;
  int32_t outLevel0LcLpCnt;
  int32_t outLevel0LcRepeatCnt;
  int32_t outLevel0LcNburst;

  int32_t inLevelx1LpStepIn;
  int32_t inLevelx1LpStepOut;
  int32_t inLevelx1NlcLpCnt;
  int32_t inLevelx1NlcLeftLines;
  int32_t inLevelx1LcLpCnt;
  int32_t inLevelx1LcLeftLines;
  int32_t inLevelx2LpStepIn;
  int32_t inLevelx2LpStepOut;
  int32_t inLevelx2NlcLpCnt;
  int32_t inLevelx2NlcLeftLines;
  int32_t inLevelx2LcLpCnt;
  int32_t inLevelx2LcLeftLines;
  int32_t inLevelx3LpStepIn;
  int32_t inLevelx3LpStepOut;
  int32_t inLevelx3NlcLpCnt;
  int32_t inLevelx3NlcLeftLines;
  int32_t inLevelx3LcLpCnt;
  int32_t inLevelx3LcLeftLines;

  int32_t nextShapeLpOffsetOut;
  int32_t nextShapeLpOffsetIn;
};

struct TransDataMode101Param {
  int32_t tillingParamCount;

  int32_t tilingMode;
  int32_t ubOffset;
  int32_t usedCoreCnt;
  int32_t coreStepIn;
  int32_t coreStepOut;
  int32_t oneLineSize;

  int32_t inLevel2HLpStepIn;
  int32_t inLevel2HLpStepOut;
  int32_t inLevel1CLpStepIn;
  int32_t inLevel1CLpStepOut;
  int32_t inLevel0LpCnt;
  int32_t inLevel0LpStepUb;
  int32_t inLevel0LpStepIn;
  int32_t inLevel0CCnt;
  int32_t inLevel0SubCSize;
  int32_t inLevel0PerLineData;
  int32_t inLevel0RepeatCnt;
  int32_t inLevel0Nburst;
  int32_t inLevel0SrcStride;
  int32_t inLevel0DstStride;
  int32_t outLevel1C1LpCnt;
  int32_t outLevel1LpStepOut;
  int32_t outLevel0NLpCnt;
  int32_t outLevel0LpStepUb;
  int32_t outLevel0LpStepOut;
  int32_t outLevel0RepeatCnt;
  int32_t outLevel0Nburst;
  int32_t outLevel0SrcStride;
  int32_t outLevel0DstStride;

  int32_t inLevel2NlcHLpCnt;
  int32_t inLevel2NlcHLeftLines;
  int32_t inLevel1NlcCLpCnt;
  int32_t inLevel1NlcLeftLines;
  int32_t inLevel0NlcLpCnt;
  int32_t inLevel0NlcCCnt;
  int32_t inLevel0NlcSubCSize;
  int32_t inLevel0NlcPerLineData;
  int32_t inLevel0NlcRepeatCnt;
  int32_t inLevel0NlcNburstNt;
  int32_t inLevel0NlcNburstT;
  int32_t outLevel1NlcC1LpCnt;
  int32_t outLevel0NlcNLpCnt;
  int32_t outLevel0NlcRepeatCnt;
  int32_t outLevel0NlcNburst;

  int32_t inLevel2LcHLpCnt;
  int32_t inLevel2LcHLeftLines;
  int32_t inLevel1LcCLpCnt;
  int32_t inLevel1LcLeftLines;
  int32_t inLevel0LcLpCnt;
  int32_t inLevel0LcCCnt;
  int32_t inLevel0LcSubCSize;
  int32_t inLevel0LcPerLineData;
  int32_t inLevel0LcRepeatCnt;
  int32_t inLevel0LcNburstNt;
  int32_t inLevel0LcNburstT;
  int32_t outLevel1LcC1LpCnt;
  int32_t outLevel0LcNLpCnt;
  int32_t outLevel0LcRepeatCnt;
  int32_t outLevel0LcNburst;

  int32_t inLevelx1LpStepIn;
  int32_t inLevelx1LpStepOut;
  int32_t inLevelx1NlcLpCnt;
  int32_t inLevelx1NlcLeftLines;
  int32_t inLevelx1LcLpCnt;
  int32_t inLevelx1LcLeftLines;
  int32_t inLevelx2LpStepIn;
  int32_t inLevelx2LpStepOut;
  int32_t inLevelx2NlcLpCnt;
  int32_t inLevelx2NlcLeftLines;
  int32_t inLevelx2LcLpCnt;
  int32_t inLevelx2LcLeftLines;
  int32_t inLevelx3LpStepIn;
  int32_t inLevelx3LpStepOut;
  int32_t inLevelx3NlcLpCnt;
  int32_t inLevelx3NlcLeftLines;
  int32_t inLevelx3LcLpCnt;
  int32_t inLevelx3LcLeftLines;
};

int32_t GetC0Len(std::string& opType) {
    if (opType == "int8" || opType == "uint8" || opType == "bool") {
        return C0_32;
    }
    return C0_16;
}

int32_t GetDTypeLen(std::string& opType) {
    int32_t typeLen;
    if (opType == "int8" || opType == "uint8") {
        typeLen = 1;
    } else if (opType == "float16" || opType == "int16" || opType == "uint16") {
        typeLen = 2;
    } else if (opType == "float32" || opType == "int32" || opType == "uint32") {
        typeLen = 4;
    } else if (opType == "int64" || opType == "uint64") {
        typeLen = 8;
    }

    return typeLen;
}

bool CheckTensorShape(const std::string& opType, int32_t ubSize, int32_t blockDim, std::vector<int64_t> outShape) {
    int32_t outDims = outShape.size();

    if (ubSize < 0) {
       ge::OpsOneInputShapeErrReport(opType.c_str(), "ubSize", "ubSize can not be less than 0");
        OP_LOGE(opType.c_str(), "op [TransDataTiling] : CheckTensorShape, ubSize is invalid.");
        return false;
    }

    if (blockDim < 0) {
        ge::OpsOneInputShapeErrReport(opType.c_str(), "blockDim", "blockDim can not be less than 0");
        OP_LOGE(opType.c_str(), "op [TransDataTiling] : CheckTensorShape, blockDim is invalid.");
        return false;
    }

    if (outDims == 0) {
        ge::OpsOneInputShapeErrReport(opType.c_str(), "outShape", "outShape can not be null");
        OP_LOGE(opType.c_str(), "op [TransDataTiling] : CheckTensorShape, outShape is invalid.");
        return false;
    }

    for (int32_t i = 0; i < outDims; i++) {
        if (outShape[i] <= 0) {
            ge::OpsOneInputShapeErrReport(opType.c_str(), "outShape", "the value of outShape must be large than 0");
            OP_LOGE(opType.c_str(), "op [TransDataTiling] : CheckTensorShape, outShape.shape[i] must be > 0");
            return false;
        }
    }

    return true;
}

bool GetCompileParams(const nlohmann::json& opCompileInfoJson, std::string& srcFormat, std::string& dstFormat,
                      std::string& dType, int32_t& ubSize, int32_t& blockDim, int32_t& inputSize, int32_t& hiddenSize,
                      int32_t& group, const std::string& opType) {

    using namespace nlohmann;

    auto allVars = opCompileInfoJson["vars"];
    if (allVars.count("srcFormat") == 0) {
        OP_LOGE("op [TransDataTiling] : GetCompileParams, get srcFormat error");
        return false;
    }
    srcFormat = allVars["srcFormat"].get<std::string>();

    if (allVars.count("dstFormat") == 0) {
        OP_LOGE("op [TransDataTiling] : GetCompileParams, get dstFormat error");
        return false;
    }
    dstFormat = allVars["dstFormat"].get<std::string>();

    if (allVars.count("dType") == 0) {
        OP_LOGE("op [TransDataTiling] : GetCompileParams, get dType error");
        return false;
    }
    dType = allVars["dType"].get<std::string>();

    if (allVars.count("ubSize") == 0) {
        OP_LOGE("op [TransDataTiling] : GetCompileParams, get ubSize error");
        return false;
    }
    ubSize = allVars["ubSize"].get<std::int32_t>();

    if (allVars.count("blockDim") == 0) {
        OP_LOGE("op [TransDataTiling] : GetCompileParams, get blockDim error");
        return false;
    }
    blockDim = allVars["blockDim"].get<std::int32_t>();

    if (allVars.count("inputSize") == 0) {
        OP_LOGE("op [TransDataTiling] : GetCompileParams, get inputSize error");
        return false;
    }
    inputSize = allVars["inputSize"].get<std::int32_t>();

    if (allVars.count("hiddenSize") == 0) {
        OP_LOGE("op [TransDataTiling] : GetCompileParams, get hiddenSize error");
        return false;
    }
    hiddenSize = allVars["hiddenSize"].get<std::int32_t>();

    if (allVars.count("group") == 0) {
        OP_LOGE("op [TransDataTiling] : GetCompileParams, get group error");
        return false;
    }
    group = allVars["group"].get<std::int32_t>();

    OP_LOGD(opType.c_str(), "GetCompileParams, srcFormat[%s], dstFormat[%s], \
          dType[%s], ubSize[%d], blockDim[%d], inputSize[%d], hiddenSize[%d], group[%d].",
          srcFormat.c_str(), dstFormat.c_str(), dType.c_str(), ubSize, blockDim, inputSize, hiddenSize, group);

    return true;
}

bool GetRenew4Shape(std::vector<int64_t> inShape, std::vector<int64_t> outShape, std::string srcFormat,
                    std::string dstFormat, std::vector<int32_t>& combAxis, int32_t c0Len, int32_t group,
                    std::vector<int64_t>& inShapeNew, std::vector<int64_t>& outShapeNew,
                    std::vector<int64_t>& inShapeP2, std::vector<int64_t>& outShapeP2,
                    std::string& realSrcFormat, std::string& realDstFormat)
{
    return true;
}

bool GetRenew2Shape(std::vector<int64_t> inShape, std::vector<int64_t> outShape, std::string srcFormat,
                    std::string dstFormat, std::vector<int32_t>& combAxis, int32_t c0Len, int32_t group,
                    std::vector<int64_t>& inShapeNew, std::vector<int64_t>& outShapeNew, std::string& realSrcFormat,
                    std::string& realDstFormat)
{
    int32_t combAxisCnt = combAxis.size();
    if (combAxisCnt > 0) {
        OP_LOGE("op [TransDataTiling] : GetRenew2Shape error, combAxisCnt > 0");
        return false;
    }
    if (srcFormat == "ND" && dstFormat == "FRACTAL_ZN_RNN") {
        if (inShape[inShape.size() - 2] == 1) {
            inShapeNew = inShape;
            inShapeNew.erase(inShapeNew.end() - 2);
            int32_t axisC1 = GetCeilDiv(inShape[inShape.size() - 3], c0Len);
            int32_t axisN = GetCeilFill(inShape[inShape.size() - 1], NI_16);
            int32_t axisC0 = c0Len;
            outShapeNew.push_back(axisC1);
            for (int i = 0; i < inShape.size() - 3; i++) {
                outShapeNew.push_back(inShape[i]);
            }
            outShapeNew.push_back(axisN);
            outShapeNew.push_back(axisC0);

            if (inShape.size() == 3) {
                realSrcFormat = "CN";
                realDstFormat = "CNT";
            } else {
                realSrcFormat = "HCN";
                realDstFormat = "CHNT";
            }
        } else if (inShape[inShape.size() - 2] > 1) {
            inShapeNew = inShape;
            int32_t axisC1 = GetCeilDiv(inShape[inShape.size() - 3], c0Len);
            int32_t axisN = GetCeilFill(inShape[inShape.size() - 1], NI_16);
            int32_t axisC0 = c0Len;
            outShapeNew.push_back(axisC1);
            for (int i = 0; i < inShape.size() - 3; i++) {
                outShapeNew.push_back(inShape[i]);
            }
            outShapeNew.push_back(inShape[inShape.size() - 2]);
            outShapeNew.push_back(axisN);
            outShapeNew.push_back(axisC0);
            if (inShape.size() == 3) {
                realSrcFormat = "CDN";
                realDstFormat = "CDNT";
            } else {
                realSrcFormat = "HCDN";
                realDstFormat = "CHDNT";
            }
        }
    }

    if ((srcFormat == "NCHW" || srcFormat == "NHWC") && (dstFormat == "NC1HWC0")) {
        int32_t hwIdx = std::strchr(srcFormat.c_str(), 'H') - srcFormat.c_str();
        int32_t cIdx = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
        realDstFormat = "NCHT";
        if (srcFormat == "NCHW") {
            realSrcFormat = "NCH";
            for (int i = 0; i < inShape.size() - hwIdx; i++) {
                inShapeNew.push_back(inShape[i]);
            }
            int32_t lastSize = GetShapeSize(inShape, hwIdx);
            inShapeNew.push_back(lastSize);
        } else {
            realSrcFormat = "NHC";
            for (int i = 0; i < hwIdx; i++) {
                inShapeNew.push_back(inShape[i]);
            }
            int32_t n = inShape.size() - 1;
            int32_t shapeSize = 1;
            for (int32_t i = hwIdx; i < n; i++) {
                shapeSize *= inShape[i];
            }
            inShapeNew.push_back(shapeSize);
            inShapeNew.push_back(inShape[inShape.size() - 1]);
        }
        int32_t axisC1 = GetCeilDiv(inShape[cIdx], c0Len);
        int32_t axisN = inShape[0];
        int32_t axisH = inShapeNew[hwIdx];
        int32_t axisC0 = c0Len;
        outShapeNew.push_back(axisN);
        outShapeNew.push_back(axisC1);
        outShapeNew.push_back(axisH);
        outShapeNew.push_back(axisC0);
    }

    if (srcFormat == "ND" && dstFormat == "FRACTAL_NZ") {
        realSrcFormat = "HNC";
        realDstFormat = "HCNT";
        if (inShape.size() == 1) {
            inShapeNew.push_back(1);
            inShapeNew.push_back(1);
            inShapeNew.push_back(inShape[0]);
        } else if (inShape.size() == 2) {
            inShapeNew.push_back(1);
            inShapeNew.push_back(inShape[0]);
            inShapeNew.push_back(inShape[1]);
        } else {
            int32_t shapeSize = 1;
            for (int32_t i = 0; i < inShape.size() - 2; i++) {
                shapeSize *= inShape[i];
            }
            inShapeNew.push_back(shapeSize);
            inShapeNew.push_back(inShape[inShape.size() - 2]);
            inShapeNew.push_back(inShape[inShape.size() - 1]);
        }
        outShapeNew = inShapeNew;
        outShapeNew[outShapeNew.size() - 2] = (inShapeNew[inShapeNew.size() - 1] + CUBE_SIZE - 1) / CUBE_SIZE;
        outShapeNew[outShapeNew.size() - 1] = (inShapeNew[inShapeNew.size() - 2] + CUBE_SIZE - 1) / CUBE_SIZE * CUBE_SIZE;
        outShapeNew.push_back(CUBE_SIZE);
    }
    return true;
}

int32_t GetMultiCoreAxis(std::vector<int64_t> inShape, int32_t axisPosC, int32_t blockElemCnt, int32_t c0Len,
                         int32_t coreNum) {
    int32_t shapeLen = inShape.size();
    bool axisCNotLastDim = axisPosC + 1 != shapeLen;
    std::vector<int32_t> coreLpCnt;

    for (int32_t index = 0; index < shapeLen; index++) {
        int32_t tmpFullCycleLoopCnt;
        int32_t leftLoopCnt;
        int32_t fullCycleLoopCnt;
        if (index + 1 == shapeLen) {
            if (GetFloorDiv(inShape[index], 8 * blockElemCnt * coreNum) > 0) {
                tmpFullCycleLoopCnt = coreNum;
            } else {
                tmpFullCycleLoopCnt = 0;
            }
            leftLoopCnt = GetCeilDiv(inShape[index], 8 * blockElemCnt) % coreNum;
        } else if (index == axisPosC && axisCNotLastDim) {
            if (GetFloorDiv(inShape[index], c0Len * coreNum) > 0) {
                tmpFullCycleLoopCnt = coreNum;
            } else {
                tmpFullCycleLoopCnt = 0;
            }
            leftLoopCnt = GetCeilDiv(inShape[index], c0Len) % coreNum;
        } else {
            if (GetFloorDiv(inShape[index], coreNum) > 0) {
                tmpFullCycleLoopCnt = coreNum;
            } else {
                tmpFullCycleLoopCnt = 0;
            }
            leftLoopCnt = inShape[index] % coreNum;
        }

        if (tmpFullCycleLoopCnt > 0 && leftLoopCnt == 0) {
            fullCycleLoopCnt = 2 * tmpFullCycleLoopCnt;
        } else {
            fullCycleLoopCnt = tmpFullCycleLoopCnt;
        }
        coreLpCnt.push_back(fullCycleLoopCnt + leftLoopCnt);
    }

    return max_element(coreLpCnt.begin(), coreLpCnt.end()) - coreLpCnt.begin();
}

bool CalcMcTilingParams(int32_t multiCoreAxisPos, int32_t multiCoreAxisSize, int32_t shapeLen, int32_t axisPosC,
                        int32_t c0Len, int32_t coreNum, vector<int64_t> outShape, std::string dstFormat,
                        std::string srcFormat, int32_t blockElemCnt, vector<int64_t> inShape, int32_t& usedCoreCnt,
                        int32_t& coreStepIn, int32_t& coreStepOut, int32_t& nlcAxisMcSize, int32_t& lcAxisMcSize) {
    if (multiCoreAxisPos == axisPosC && multiCoreAxisPos != shapeLen - 1) {
        int32_t c0CntInC = GetCeilDiv(multiCoreAxisSize, c0Len);
        usedCoreCnt = GetCeilDiv(c0CntInC, GetCeilDiv(c0CntInC, coreNum));
        int32_t nlcC1Cnt = GetCeilDiv(c0CntInC, usedCoreCnt);
        nlcAxisMcSize = nlcC1Cnt * c0Len;
        lcAxisMcSize = multiCoreAxisSize - (usedCoreCnt - 1) * nlcAxisMcSize;
        coreStepIn = GetShapeSize(inShape, axisPosC + 1) * nlcAxisMcSize;
        coreStepOut = GetShapeSize(outShape, std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str() + 1) * nlcC1Cnt;
    } else if (multiCoreAxisPos == shapeLen - 1) {
        int32_t axisLast8BlockCnt = GetCeilDiv(multiCoreAxisSize, 8 * blockElemCnt);
        usedCoreCnt = GetCeilDiv(axisLast8BlockCnt, GetCeilDiv(axisLast8BlockCnt, coreNum));
        int32_t nlc8BlockCnt = GetCeilDiv(axisLast8BlockCnt, usedCoreCnt);

        nlcAxisMcSize = nlc8BlockCnt * 8 * blockElemCnt;
        lcAxisMcSize = multiCoreAxisSize - (usedCoreCnt - 1) * nlcAxisMcSize;
        int32_t axisMcDstPos = std::strchr(dstFormat.c_str(), srcFormat[multiCoreAxisPos]) - dstFormat.c_str();
        coreStepIn = GetShapeSize(inShape, -1) * nlcAxisMcSize;
        if (srcFormat[multiCoreAxisPos] == 'C') {
            coreStepOut = GetShapeSize(outShape, axisMcDstPos + 1) * nlcAxisMcSize / 16;
        } else {
            coreStepOut = GetShapeSize(outShape, axisMcDstPos + 1) * nlcAxisMcSize;
        }
    } else {
        usedCoreCnt = GetCeilDiv(multiCoreAxisSize, GetCeilDiv(multiCoreAxisSize, coreNum));
        nlcAxisMcSize = GetCeilDiv(multiCoreAxisSize, usedCoreCnt);
        lcAxisMcSize = multiCoreAxisSize - (usedCoreCnt - 1) * nlcAxisMcSize;
        int32_t axisMcDstPos = std::strchr(dstFormat.c_str(), srcFormat[multiCoreAxisPos]) - dstFormat.c_str();
        coreStepIn = GetShapeSize(inShape, multiCoreAxisPos + 1) * nlcAxisMcSize;
        coreStepOut = GetShapeSize(outShape, axisMcDstPos + 1) * nlcAxisMcSize;
    }
    return true;
}

bool TillingPositiveMode100(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, int32_t& multiCoreAxisPos, int32_t& axisPosC, int32_t& coreNum,
                            int32_t& blockElemCnt, int32_t& c0Len, int32_t& ubSize, int32_t& shapeLpIdx,
                            TransDataMode100Param& params) {
    int32_t shapeLen = inShape.size();
    inShape.push_back(1);
    outShape.push_back(1);

    int32_t halfUbSize = ubSize / 2;
    int32_t multiCoreAxisSize = inShape[multiCoreAxisPos];

    int32_t nlcAxisMcSize;
    int32_t lcAxisMcSize;
    bool ret = CalcMcTilingParams(multiCoreAxisPos, multiCoreAxisSize, shapeLen, axisPosC, c0Len, coreNum, outShape,
                                  dstFormat, srcFormat, blockElemCnt, inShape, params.usedCoreCnt, params.coreStepIn,
                                  params.coreStepOut, nlcAxisMcSize, lcAxisMcSize);
    if (!ret) {
        OP_LOGE("op TransDataTiling: TillingPositiveMode100 CalcMcTilingParams Failed.");
        return ret;
    }

    params.tillingParamCount = 68;
    params.tilingMode = 100;
    params.ubOffset = GetCeilFill(halfUbSize, blockElemCnt);
    params.oneLineSize = params.ubOffset / VNC_LINES / blockElemCnt * blockElemCnt;

    vector<int64_t> nlcInShape(inShape), lcInShape(inShape);
    nlcInShape[multiCoreAxisPos] = nlcAxisMcSize;
    lcInShape[multiCoreAxisPos] = lcAxisMcSize;
    int32_t gapCBtwLast = 0;

    /*     input loop             */
    /*  level2 tiling parameters      */
    int32_t c1PerLpC0Cnt = c0Len;
    params.inLevel2C1LpStepIn = GetShapeSize(inShape, axisPosC + 1) * c1PerLpC0Cnt;
    params.inLevel2C1LpStepOut = GetShapeSize(outShape, std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str() + 1);
    params.inLevel2NlcC1LpCnt = GetCeilDiv(nlcInShape[axisPosC], c1PerLpC0Cnt);
    params.inLevel2NlcC1LeftLines = nlcInShape[axisPosC] % c1PerLpC0Cnt;
    params.inLevel2LcC1LpCnt =  GetCeilDiv(lcInShape[axisPosC], c1PerLpC0Cnt);
    params.inLevel2LcC1LeftLines = lcInShape[axisPosC] % c1PerLpC0Cnt;

    int32_t cRightSize;
    if (multiCoreAxisPos <= axisPosC) {
        while (gapCBtwLast < shapeLen - (axisPosC + 1)) {
            cRightSize = GetShapeSize(inShape, axisPosC + 1 + gapCBtwLast);
            if (cRightSize > params.oneLineSize) {
                gapCBtwLast += 1;
            } else {
                break;
            }
        }
    } else {
        cRightSize = 0;
        gapCBtwLast = (shapeLen - 1) - (axisPosC + 1);
    }

    /*  level1 tiling parameters      */
    int32_t srcLastInDstPos = std::strchr(dstFormat.c_str(), srcFormat[srcFormat.length() - 1]) - dstFormat.c_str();

    if (cRightSize > 0 && cRightSize <= params.oneLineSize) {
        params.inLevel1LastLpStepIn = cRightSize;
        params.inLevel1LastLpStepOut = 0;
        params.inLevel1NlcLastLpCnt = GetCeilDiv(cRightSize, params.oneLineSize);
        params.inLevel1NlcLastLeftLines = cRightSize % params.oneLineSize;
        params.inLevel1LcLastLpCnt = GetCeilDiv(cRightSize, params.oneLineSize);
        params.inLevel1LcLastLeftLines = cRightSize % params.oneLineSize;
    } else {
        params.inLevel1LastLpStepIn = params.oneLineSize;
        params.inLevel1LastLpStepOut = GetShapeSize(outShape, srcLastInDstPos + 1) * params.oneLineSize;
        params.inLevel1NlcLastLpCnt = GetCeilDiv(nlcInShape[nlcInShape.size() - 2], params.oneLineSize);
        params.inLevel1NlcLastLeftLines = nlcInShape[nlcInShape.size() - 2] % params.oneLineSize;
        params.inLevel1LcLastLpCnt = GetCeilDiv(lcInShape[lcInShape.size() - 2], params.oneLineSize);
        params.inLevel1LcLastLeftLines = lcInShape[lcInShape.size() - 2] % params.oneLineSize;
    }

    /*  level0 tiling parameters      */
    if (cRightSize == params.oneLineSize && axisPosC + 1 == shapeLen - 1) {
        params.inLevel0LpCnt = 1;
        params.inLevel0C0LpStepUb = 0;
        params.inLevel0LpStepIn = 0;
        params.inLevel0RepeatCnt = 1;
        params.inLevel0Nburst = params.oneLineSize * c0Len;
        params.inLevel0SrcStride = 0;
        params.inLevel0DstStride = 0;
        params.inLevel0NlcLpCnt = 1;
        params.inLevel0LcLpCnt = 1;
        params.inLevel0NlcRepeatCnt = 1;
        params.inLevel0LcRepeatCnt = 1;
        params.inLevel0NlcNburst = params.oneLineSize * params.inLevel2NlcC1LeftLines;
        params.inLevel0LcNburst = params.oneLineSize * params.inLevel2LcC1LeftLines;
    } else {
        params.inLevel0LpCnt = c0Len;
        params.inLevel0C0LpStepUb = params.oneLineSize;
        params.inLevel0LpStepIn = cRightSize > 0 ? cRightSize : inShape[inShape.size() - 2];
        params.inLevel0RepeatCnt = 1;
        params.inLevel0Nburst = params.oneLineSize;
        params.inLevel0SrcStride = 0;
        params.inLevel0DstStride = 0;
        params.inLevel0NlcLpCnt = params.inLevel2NlcC1LeftLines;
        params.inLevel0LcLpCnt = params.inLevel2LcC1LeftLines;
        params.inLevel0NlcRepeatCnt = 1;
        params.inLevel0LcRepeatCnt = 1;
        params.inLevel0NlcNburst = params.inLevel1NlcLastLeftLines;
        params.inLevel0LcNburst = params.inLevel1LcLastLeftLines;
    }

    /*         output loop            */
    /*  level0 tiling parameters      */
    if (srcLastInDstPos + 1 == outShape.size() - 2) {
        params.outLevel0LpCnt = 1;
        params.outLevel0LpStepUb = 0;
        params.outLevel0LpStepOut = 0;
        params.outLevel0RepeatCnt = 1;
        params.outLevel0Nburst = params.oneLineSize * outShape[outShape.size() - 2];
        params.outLevel0SrcStride = 0;
        params.outLevel0DstStride = 0;
        params.outLevel0NlcLpCnt = 1;
        params.outLevel0LcLpCnt = 1;
        params.outLevel0NlcRepeatCnt = 1;
        params.outLevel0LcRepeatCnt = 1;

        if (axisPosC + gapCBtwLast + 1 != shapeLen - 1) {
            params.outLevel0NlcNburst = nlcInShape[nlcInShape.size() - 2] * outShape[outShape.size() - 2];
            params.outLevel0LcNburst = lcInShape[lcInShape.size() - 2] * outShape[outShape.size() - 2];
        } else {
            params.outLevel0NlcNburst = params.inLevel1NlcLastLeftLines * outShape[outShape.size() - 2];
            params.outLevel0LcNburst = params.inLevel1LcLastLeftLines * outShape[outShape.size() - 2];
        }
    } else {
        char srcLastChar = srcFormat[srcFormat.length() - 1];
        int32_t tempPos = std::strchr(dstFormat.c_str(), srcLastChar) - dstFormat.c_str() + 1;

        vector<int64_t> tempOutShape(outShape.begin() + tempPos, outShape.end() - 2);
        int32_t srcLastDstLastGap = GetShapeSize(tempOutShape, 0);
        if (srcLastDstLastGap > STRIDE_LIMIT_MTE) {
            params.outLevel0LpCnt = params.oneLineSize;
            params.outLevel0LpStepUb = outShape[outShape.size() - 2];
            params.outLevel0LpStepOut = GetShapeSize(outShape, tempPos);
            params.outLevel0RepeatCnt = 1;
            params.outLevel0Nburst = outShape[outShape.size() - 2];
            params.outLevel0SrcStride = 0;
            params.outLevel0DstStride = 0;
            params.outLevel0NlcLpCnt = params.inLevel1NlcLastLeftLines;
            params.outLevel0LcLpCnt = params.inLevel1LcLastLeftLines;
            params.outLevel0NlcRepeatCnt = 1;
            params.outLevel0LcRepeatCnt = 1;
            params.outLevel0NlcNburst = outShape[outShape.size() - 2];
            params.outLevel0LcNburst = outShape[outShape.size() - 2];
        } else {
            params.outLevel0LpCnt = 1;
            params.outLevel0LpStepUb = 0;
            params.outLevel0LpStepOut = 0;
            params.outLevel0RepeatCnt = params.oneLineSize;
            params.outLevel0Nburst = outShape[outShape.size() - 2];
            params.outLevel0SrcStride = 0;
            if (srcLastDstLastGap == 0) {
                params.outLevel0DstStride = 0;
            } else {
                params.outLevel0DstStride = srcLastDstLastGap - 1;
            }
            params.outLevel0NlcLpCnt = 1;
            params.outLevel0LcLpCnt = 1;
            params.outLevel0NlcRepeatCnt = params.inLevel1NlcLastLeftLines;
            params.outLevel0LcRepeatCnt = params.inLevel1LcLastLeftLines;
            params.outLevel0NlcNburst = outShape[outShape.size() - 2];
            params.outLevel0LcNburst = outShape[outShape.size() - 2];
        }
    }

    /*  level1 tiling parameters      */
    if (axisPosC + 1 + gapCBtwLast == shapeLen - 1) {
        params.outLevel1NlcLpCnt = 1;
        params.outLevel1LcLpCnt = 1;
        params.outLevel1LpStepUb = 0;
        params.outLevel1LpStepOut = 0;
    } else {
        char cRightAxisChar = srcFormat[axisPosC + 1 + gapCBtwLast];
        int32_t tempSrcPos = std::strchr(srcFormat.c_str(), cRightAxisChar) - srcFormat.c_str();
        int32_t tempDstPos = std::strchr(dstFormat.c_str(), cRightAxisChar) - dstFormat.c_str();

        params.outLevel1NlcLpCnt = nlcInShape[tempSrcPos];
        params.outLevel1LcLpCnt = lcInShape[tempSrcPos];
        params.outLevel1LpStepOut = GetShapeSize(outShape, tempDstPos + 1);
        params.outLevel1LpStepUb = GetShapeSize(inShape, tempSrcPos + 1) * outShape[outShape.size() - 2];
    }

    string leftSrcFormat = srcFormat;
    leftSrcFormat.replace(leftSrcFormat.find("C"), 1, "");
    leftSrcFormat.replace(leftSrcFormat.find(srcFormat[axisPosC + 1 + gapCBtwLast]),
                          srcFormat.length() - (axisPosC + 1 + gapCBtwLast), "");

    int32_t noneLevelCnt = FRAME_LEVEL - (leftSrcFormat.length() + 3);
    if (noneLevelCnt > 0) {
        string noneLevelString(noneLevelCnt,'X');
        leftSrcFormat.insert(0, noneLevelString);
    }
    vector<int> inLevelx;
    for (int32_t i = 0; i < leftSrcFormat.length(); i++) {
        if (leftSrcFormat[i] == 'X') {
            inLevelx.push_back(0);
            inLevelx.push_back(0);
            inLevelx.push_back(1);
            inLevelx.push_back(0);
            inLevelx.push_back(1);
            inLevelx.push_back(0);
        } else {
            int32_t curCharSrcPos = std::strchr(srcFormat.c_str(), leftSrcFormat[i]) - srcFormat.c_str();
            int32_t curCharDstPos = std::strchr(dstFormat.c_str(), leftSrcFormat[i]) - dstFormat.c_str();
            inLevelx.push_back(GetShapeSize(inShape, curCharSrcPos + 1));
            inLevelx.push_back(GetShapeSize(outShape, curCharDstPos + 1));
            inLevelx.push_back(nlcInShape[curCharSrcPos]);
            inLevelx.push_back(0);
            inLevelx.push_back(lcInShape[curCharSrcPos]);
            inLevelx.push_back(0);
        }
    }
    params.inLevelx1LpStepIn = inLevelx[0];
    params.inLevelx1LpStepOut = inLevelx[1];
    params.inLevelx1NlcLpCnt = inLevelx[2];
    params.inLevelx1NlcLeftLines = inLevelx[3];
    params.inLevelx1LcLpCnt = inLevelx[4];
    params.inLevelx1LcLeftLines = inLevelx[5];
    params.inLevelx2LpStepIn = inLevelx[6];
    params.inLevelx2LpStepOut = inLevelx[7];
    params.inLevelx2NlcLpCnt = inLevelx[8];
    params.inLevelx2NlcLeftLines = inLevelx[9];
    params.inLevelx2LcLpCnt = inLevelx[10];
    params.inLevelx2LcLeftLines = inLevelx[11];
    params.inLevelx3LpStepIn = inLevelx[12];
    params.inLevelx3LpStepOut = inLevelx[13];
    params.inLevelx3NlcLpCnt = inLevelx[14];
    params.inLevelx3NlcLeftLines = inLevelx[15];
    params.inLevelx3LcLpCnt = inLevelx[16];
    params.inLevelx3LcLeftLines = inLevelx[17];
    if (shapeLpIdx == 0) {
        params.nextShapeLpOffsetOut = GetShapeSize(outShape, 0);
        params.nextShapeLpOffsetIn = GetShapeSize(inShape, axisPosC);
    } else {
        params.nextShapeLpOffsetOut = 0;
        params.nextShapeLpOffsetIn = 0;
    }
    return true;
}


bool TillingPositiveMode101(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, int32_t& multiCoreAxisPos, int32_t& axisPosC, int32_t& coreNum,
                            int32_t& blockElemCnt, int32_t& c0Len, int32_t& ubSize, int32_t& shapeLpIdx,
                            TransDataMode101Param& params) {
    int32_t shapeLen = inShape.size();
    inShape.push_back(1);
    outShape.push_back(1);

    int32_t halfUbSize = ubSize / 2;
    int32_t multiCoreAxisSize = inShape[multiCoreAxisPos];

    int32_t nlcAxisMcSize;
    int32_t lcAxisMcSize;
    bool ret = CalcMcTilingParams(multiCoreAxisPos, multiCoreAxisSize, shapeLen, axisPosC, c0Len, coreNum, outShape,
                                  dstFormat, srcFormat, blockElemCnt, inShape, params.usedCoreCnt, params.coreStepIn,
                                  params.coreStepOut, nlcAxisMcSize, lcAxisMcSize);
    if (!ret) {
        OP_LOGE("op TransDataTiling: TillingPositiveMode101 CalcMcTilingParams Failed.");
        return ret;
    }

    params.tillingParamCount = 77;
    params.tilingMode = 101;
    params.ubOffset = GetCeilFill(halfUbSize, blockElemCnt);
    params.oneLineSize = params.ubOffset / VNC_LINES / blockElemCnt * blockElemCnt;

    vector<int64_t> nlcInShape(inShape), lcInShape(inShape);
    nlcInShape[multiCoreAxisPos] = nlcAxisMcSize;
    lcInShape[multiCoreAxisPos] = lcAxisMcSize;

    /*     input loop             */
    /* --- in-level 2 tiling params, control level 0 loop count    */
    char rvs2ndChar = srcFormat[srcFormat.length() - 2];
    int32_t cCntPerLine;
    if (multiCoreAxisPos == shapeLen - 1) {
        cCntPerLine = 1;
    } else {
        int32_t tmpCnt = params.oneLineSize / GetCeilFill(inShape[inShape.size()-2], c0Len);
        if (tmpCnt > 0) {
            cCntPerLine = tmpCnt;
        } else {
            cCntPerLine = 1;
        }
    }

    int32_t hStepCnt = cCntPerLine * VNC_LINES;
    params.inLevel2HLpStepIn = GetShapeSize(inShape, -2) * hStepCnt;
    params.inLevel2HLpStepOut = GetShapeSize(outShape, std::strchr(dstFormat.c_str(), rvs2ndChar) - dstFormat.c_str()
                                            + 1) * hStepCnt;
    params.inLevel2NlcHLpCnt = GetCeilDiv(nlcInShape[nlcInShape.size() - 3], hStepCnt);
    params.inLevel2NlcHLeftLines = nlcInShape[nlcInShape.size() - 3] % hStepCnt;
    params.inLevel2LcHLpCnt = GetCeilDiv(lcInShape[lcInShape.size() - 3], hStepCnt);
    params.inLevel2LcHLeftLines = lcInShape[lcInShape.size() - 3] % hStepCnt;

    //--- in-level 1 tiling params, control level 0 data move burst
    char rvs1stChar = srcFormat[srcFormat.length() - 1];
    int32_t c0Cnt = params.oneLineSize / c0Len;
    params.inLevel1CLpStepIn = GetShapeSize(inShape, -1) * params.oneLineSize;
    params.inLevel1CLpStepOut = GetShapeSize(outShape, std::strchr(dstFormat.c_str(), rvs1stChar) - dstFormat.c_str()
                                            + 1) * c0Cnt;
    params.inLevel1NlcCLpCnt = GetCeilDiv(nlcInShape[nlcInShape.size() - 2], params.oneLineSize);
    params.inLevel1NlcLeftLines = nlcInShape[nlcInShape.size() - 2] % params.oneLineSize;
    params.inLevel1LcCLpCnt = GetCeilDiv(lcInShape[lcInShape.size() - 2], params.oneLineSize);
    params.inLevel1LcLeftLines = lcInShape[lcInShape.size() - 2] % params.oneLineSize;

    //--- in-level 0 tiling params, data move
    if (multiCoreAxisPos != shapeLen - 1 && inShape[inShape.size() - 2] == params.oneLineSize) {
        params.inLevel0LpCnt = 1;
        params.inLevel0LpStepUb = 0;
        params.inLevel0LpStepIn = 0;
        params.inLevel0RepeatCnt = 1;
        params.inLevel0Nburst = params.oneLineSize * VNC_LINES;
        params.inLevel0CCnt = 1;
        params.inLevel0SubCSize = params.oneLineSize;
        params.inLevel0PerLineData = params.oneLineSize;
        params.inLevel0SrcStride = 0;
        params.inLevel0DstStride = 0;
        params.inLevel0NlcLpCnt = 1;
        params.inLevel0NlcRepeatCnt = 1;
        params.inLevel0NlcNburstNt = params.inLevel2NlcHLeftLines * params.oneLineSize;
        params.inLevel0NlcNburstT = params.inLevel0NlcNburstNt;
        params.inLevel0NlcCCnt = 1;
        params.inLevel0NlcSubCSize = params.oneLineSize;
        params.inLevel0NlcPerLineData = params.oneLineSize;
        params.inLevel0LcLpCnt = 1;
        params.inLevel0LcRepeatCnt = 1;
        params.inLevel0LcNburstNt = params.inLevel2LcHLeftLines * params.oneLineSize;
        params.inLevel0LcNburstT = params.inLevel0LcNburstNt;
        params.inLevel0LcCCnt = 1;
        params.inLevel0LcSubCSize = params.oneLineSize;
        params.inLevel0LcPerLineData = params.oneLineSize;

    } else {
        params.inLevel0LpCnt = VNC_LINES;
        params.inLevel0LpStepUb = params.oneLineSize;
        params.inLevel0LpStepIn = inShape[inShape.size() - 2] * cCntPerLine;
        params.inLevel0RepeatCnt = 1;
        params.inLevel0SrcStride = 0;
        params.inLevel0DstStride = 0;
        if (cCntPerLine == 1) {
            params.inLevel0Nburst = params.oneLineSize;
            params.inLevel0CCnt = 1;
            params.inLevel0SubCSize = params.oneLineSize;
            params.inLevel0PerLineData = params.oneLineSize;
            params.inLevel0NlcLpCnt = params.inLevel2NlcHLeftLines;
            params.inLevel0NlcRepeatCnt = 1;
            params.inLevel0NlcNburstNt = nlcInShape[nlcInShape.size() - 2] % params.oneLineSize;
            params.inLevel0NlcNburstT = params.inLevel0NlcNburstNt;
            params.inLevel0NlcCCnt = 1;
            params.inLevel0NlcSubCSize = nlcInShape[nlcInShape.size() - 2] % params.oneLineSize;
            params.inLevel0NlcPerLineData = GetCeilFill(nlcInShape[nlcInShape.size() - 2] % params.oneLineSize,
                                                        blockElemCnt);
            params.inLevel0LcLpCnt = params.inLevel2LcHLeftLines;
            params.inLevel0LcRepeatCnt = 1;
            params.inLevel0LcNburstNt = lcInShape[lcInShape.size() - 2] % params.oneLineSize;
            params.inLevel0LcNburstT = params.inLevel0LcNburstNt;
            params.inLevel0LcCCnt = 1;
            params.inLevel0LcSubCSize = lcInShape[lcInShape.size() - 2] % params.oneLineSize;
            params.inLevel0LcPerLineData = GetCeilFill(lcInShape[lcInShape.size() - 2] % params.oneLineSize, blockElemCnt);

        } else {
            params.inLevel0Nburst = cCntPerLine * inShape[inShape.size() - 2];
            params.inLevel0CCnt = cCntPerLine;
            params.inLevel0SubCSize = inShape[inShape.size() - 2] % params.oneLineSize;
            params.inLevel0PerLineData = GetCeilFill(cCntPerLine * inShape[inShape.size() - 2], blockElemCnt);
            params.inLevel0NlcLpCnt = GetCeilDiv(params.inLevel2NlcHLeftLines, cCntPerLine);
            params.inLevel0NlcRepeatCnt = 1;
            params.inLevel0NlcNburstNt = cCntPerLine * inShape[inShape.size() - 2];
            params.inLevel0NlcNburstT = params.inLevel2NlcHLeftLines % cCntPerLine * nlcInShape[nlcInShape.size() - 2];
            if (params.inLevel2NlcHLeftLines > cCntPerLine) {
                params.inLevel0NlcCCnt = cCntPerLine;
            } else {
                params.inLevel0NlcCCnt = params.inLevel2NlcHLeftLines;
            }

            params.inLevel0NlcSubCSize = inShape[inShape.size() - 2] % params.oneLineSize;
            params.inLevel0NlcPerLineData = GetCeilFill(cCntPerLine * inShape[inShape.size() - 2], blockElemCnt);
            params.inLevel0LcLpCnt = GetCeilDiv(params.inLevel2LcHLeftLines, cCntPerLine);
            params.inLevel0LcRepeatCnt = 1;
            params.inLevel0LcNburstNt = cCntPerLine * inShape[inShape.size() - 2];
            params.inLevel0LcNburstT = params.inLevel2LcHLeftLines % cCntPerLine * lcInShape[lcInShape.size() - 2];
            if (params.inLevel2LcHLeftLines > cCntPerLine) {
                params.inLevel0LcCCnt = cCntPerLine;
            } else {
                params.inLevel0LcCCnt = params.inLevel2LcHLeftLines;
            }

            params.inLevel0LcSubCSize = inShape[inShape.size() - 2] % params.oneLineSize;
            params.inLevel0LcPerLineData = GetCeilFill(cCntPerLine * inShape[inShape.size() - 2], blockElemCnt);
        }
    }
    //------------------------------- output loop -------------------------------
    //--- out-level 0 tiling params, data move
    int32_t srcRvs2ndInDstPos = std::strchr(dstFormat.c_str(), rvs2ndChar) - dstFormat.c_str();
    // has gap btw source reverse second axis and target last axis
    if (srcRvs2ndInDstPos + 1 < dstFormat.length() - 1) {
        vector<int64_t> tempOutShape(outShape.begin() + srcRvs2ndInDstPos + 1, outShape.end() - 2);
        int32_t srcRvs2ndDstLastGap = GetShapeSize(tempOutShape, 0);
        if (srcRvs2ndDstLastGap > STRIDE_LIMIT_MTE) {
            params.outLevel0NLpCnt = NI_16 * cCntPerLine;
            params.outLevel0LpStepUb = outShape[outShape.size() - 2];
            params.outLevel0LpStepOut = GetShapeSize(outShape, srcRvs2ndInDstPos + 1);
            params.outLevel0RepeatCnt = 1;
            params.outLevel0Nburst = outShape[outShape.size() - 2];
            params.outLevel0SrcStride = 0;
            params.outLevel0DstStride = 0;
            // controlled by in-level 2
            params.outLevel0NlcNLpCnt = params.inLevel2NlcHLeftLines;
            params.outLevel0NlcRepeatCnt = 1;
            params.outLevel0NlcNburst = outShape[outShape.size() - 2];
            params.outLevel0LcNLpCnt = params.inLevel2LcHLeftLines;
            params.outLevel0LcRepeatCnt = 1;
            params.outLevel0LcNburst = outShape[outShape.size() - 2];
        } else {
            params.outLevel0NLpCnt = 1;
            params.outLevel0LpStepUb = 0;
            params.outLevel0LpStepOut = 0;
            params.outLevel0RepeatCnt = NI_16 * cCntPerLine;
            params.outLevel0Nburst = outShape[outShape.size() - 2];
            params.outLevel0SrcStride = 0;
            params.outLevel0DstStride = srcRvs2ndDstLastGap - GetCeilDiv(outShape[outShape.size() - 2], blockElemCnt);
            params.outLevel0NlcNLpCnt = 1;
            params.outLevel0NlcRepeatCnt = params.inLevel2NlcHLeftLines;
            params.outLevel0NlcNburst = outShape[outShape.size() - 2];
            params.outLevel0LcNLpCnt = 1;
            params.outLevel0LcRepeatCnt = params.inLevel2LcHLeftLines;
            params.outLevel0LcNburst = outShape[outShape.size() - 2];
        }
    } else {
        params.outLevel0NLpCnt = 1;
        params.outLevel0LpStepUb = 0;
        params.outLevel0LpStepOut = 0;
        params.outLevel0RepeatCnt = 1;
        params.outLevel0Nburst = NI_16 * cCntPerLine * c0Len;
        params.outLevel0SrcStride = 0;
        params.outLevel0DstStride = 0;
        params.outLevel0NlcNLpCnt = 1;
        params.outLevel0NlcRepeatCnt = 1;
        params.outLevel0NlcNburst = params.inLevel2NlcHLeftLines * outShape[outShape.size() - 2];
        params.outLevel0LcNLpCnt = 1;
        params.outLevel0LcRepeatCnt = 1;
        params.outLevel0LcNburst = params.inLevel2LcHLeftLines * outShape[outShape.size() - 2];
    }
    // --- out-level 1 tiling params, data move
    params.outLevel1C1LpCnt = GetCeilDiv(params.inLevel0SubCSize, c0Len);
    params.outLevel1LpStepOut = GetShapeSize(outShape, std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str() + 1);
    params.outLevel1NlcC1LpCnt = GetCeilDiv(params.inLevel0NlcSubCSize, c0Len);
    params.outLevel1LcC1LpCnt = GetCeilDiv(params.inLevel0LcSubCSize, c0Len);

    // --- left source axises tiling params
    string leftSrcFormat = srcFormat;
    leftSrcFormat.replace(leftSrcFormat.find(srcFormat[srcFormat.length() - 2]), srcFormat.length() - 1, "");

    int32_t noneLevelCnt = FRAME_LEVEL - (leftSrcFormat.length() + 3);
    if (noneLevelCnt > 0) {
        string noneLevelString(noneLevelCnt,'X');
        leftSrcFormat.insert(0, noneLevelString);
    }
    vector<int> inLevelx;
    for (int32_t i = 0; i < leftSrcFormat.length(); i++) {
        if (leftSrcFormat[i] == 'X') {
            inLevelx.push_back(0);
            inLevelx.push_back(0);
            inLevelx.push_back(1);
            inLevelx.push_back(0);
            inLevelx.push_back(1);
            inLevelx.push_back(0);
        } else {
            int32_t curCharSrcPos = std::strchr(srcFormat.c_str(), leftSrcFormat[i]) - srcFormat.c_str();
            int32_t curCharDstPos = std::strchr(dstFormat.c_str(), leftSrcFormat[i]) - dstFormat.c_str();
            inLevelx.push_back(GetShapeSize(inShape, curCharSrcPos + 1));
            inLevelx.push_back(GetShapeSize(outShape, curCharDstPos + 1));
            inLevelx.push_back(nlcInShape[curCharSrcPos]);
            inLevelx.push_back(0);
            inLevelx.push_back(lcInShape[curCharSrcPos]);
            inLevelx.push_back(0);
        }
    }
    params.inLevelx1LpStepIn = inLevelx[0];
    params.inLevelx1LpStepOut = inLevelx[1];
    params.inLevelx1NlcLpCnt = inLevelx[2];
    params.inLevelx1NlcLeftLines = inLevelx[3];
    params.inLevelx1LcLpCnt = inLevelx[4];
    params.inLevelx1LcLeftLines = inLevelx[5];
    params.inLevelx2LpStepIn = inLevelx[6];
    params.inLevelx2LpStepOut = inLevelx[7];
    params.inLevelx2NlcLpCnt = inLevelx[8];
    params.inLevelx2NlcLeftLines = inLevelx[9];
    params.inLevelx2LcLpCnt = inLevelx[10];
    params.inLevelx2LcLeftLines = inLevelx[11];
    params.inLevelx3LpStepIn = inLevelx[12];
    params.inLevelx3LpStepOut = inLevelx[13];
    params.inLevelx3NlcLpCnt = inLevelx[14];
    params.inLevelx3NlcLeftLines = inLevelx[15];
    params.inLevelx3LcLpCnt = inLevelx[16];
    params.inLevelx3LcLeftLines = inLevelx[17];

    return true;
}

void SetRunningMode100Params(const TransDataMode100Param& runParams, OpRunInfo& runInfo) {
    /*   tiling_sub_head and tiling_core      */
    ByteBufferPut(runInfo.tiling_data, runParams.tillingParamCount);
    ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);
    ByteBufferPut(runInfo.tiling_data, runParams.ubOffset);
    ByteBufferPut(runInfo.tiling_data, runParams.usedCoreCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.coreStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.coreStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.oneLineSize);

    /*       com_sub_tiling_params     */
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1LpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2LpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3LpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2C1LpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2C1LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1LastLpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1LastLpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0C0LpStepUb);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0RepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0Nburst);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0SrcStride);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0DstStride);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel1LpStepUb);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel1LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LpStepUb);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0RepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0Nburst);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0SrcStride);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0DstStride);

    /*       nlc_sub_tiling_params     */
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1NlcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1NlcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2NlcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2NlcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3NlcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3NlcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2NlcC1LpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2NlcC1LeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1NlcLastLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1NlcLastLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0NlcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0NlcRepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0NlcNburst);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel1NlcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0NlcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0NlcRepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0NlcNburst);

    /*       lc_sub_tiling_params     */
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1LcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1LcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2LcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2LcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3LcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3LcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2LcC1LpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2LcC1LeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1LcLastLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1LcLastLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LcRepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LcNburst);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel1LcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LcRepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LcNburst);

    ByteBufferPut(runInfo.tiling_data, runParams.nextShapeLpOffsetIn);
    ByteBufferPut(runInfo.tiling_data, runParams.nextShapeLpOffsetOut);
}

void SetRunningMode101Params(const TransDataMode101Param& runParams, OpRunInfo& runInfo) {
    /*   tiling_sub_head and tiling_core      */
    ByteBufferPut(runInfo.tiling_data, runParams.tillingParamCount);
    ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);
    ByteBufferPut(runInfo.tiling_data, runParams.ubOffset);
    ByteBufferPut(runInfo.tiling_data, runParams.usedCoreCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.coreStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.coreStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.oneLineSize);

    /*       com_sub_tiling_params     */
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1LpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2LpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3LpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2HLpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2HLpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1CLpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1CLpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LpStepUb);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0CCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0SubCSize);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0PerLineData);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0RepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0Nburst);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0SrcStride);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0DstStride);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel1C1LpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel1LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0NLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LpStepUb);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0RepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0Nburst);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0SrcStride);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0DstStride);

    /*       nlc_sub_tiling_params     */
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1NlcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1NlcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2NlcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2NlcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3NlcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3NlcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2NlcHLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2NlcHLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1NlcCLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1NlcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0NlcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0NlcCCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0NlcSubCSize);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0NlcPerLineData);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0NlcRepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0NlcNburstNt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0NlcNburstT);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel1NlcC1LpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0NlcNLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0NlcRepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0NlcNburst);

    /*       lc_sub_tiling_params     */
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1LcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx1LcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2LcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx2LcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3LcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevelx3LcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2LcHLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel2LcHLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1LcCLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel1LcLeftLines);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LcLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LcCCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LcSubCSize);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LcPerLineData);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LcRepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LcNburstNt);
    ByteBufferPut(runInfo.tiling_data, runParams.inLevel0LcNburstT);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel1LcC1LpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LcNLpCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LcRepeatCnt);
    ByteBufferPut(runInfo.tiling_data, runParams.outLevel0LcNburst);
}

void PrintTilingMode101Params(const std::string& opType, const TransDataMode101Param& params) {
    OP_LOGD(opType.c_str(), "tillingParamCount=%d", params.tillingParamCount);
    OP_LOGD(opType.c_str(), "tilingMode=%d", params.tilingMode);
    OP_LOGD(opType.c_str(), "ubOffset=%d", params.ubOffset);
    OP_LOGD(opType.c_str(), "usedCoreCnt=%d", params.usedCoreCnt);
    OP_LOGD(opType.c_str(), "coreStepIn=%d", params.coreStepIn);
    OP_LOGD(opType.c_str(), "coreStepOut=%d", params.coreStepOut);
    OP_LOGD(opType.c_str(), "oneLineSize=%d", params.oneLineSize);

    OP_LOGD(opType.c_str(), "inLevelx1LpStepIn=%d", params.inLevelx1LpStepIn);
    OP_LOGD(opType.c_str(), "inLevelx1LpStepOut=%d", params.inLevelx1LpStepOut);
    OP_LOGD(opType.c_str(), "inLevelx2LpStepIn=%d", params.inLevelx2LpStepIn);
    OP_LOGD(opType.c_str(), "inLevelx2LpStepOut=%d", params.inLevelx2LpStepOut);
    OP_LOGD(opType.c_str(), "inLevelx3LpStepIn=%d", params.inLevelx3LpStepIn);
    OP_LOGD(opType.c_str(), "inLevelx3LpStepOut=%d", params.inLevelx3LpStepOut);
    OP_LOGD(opType.c_str(), "inLevel2HLpStepIn=%d", params.inLevel2HLpStepIn);
    OP_LOGD(opType.c_str(), "inLevel2HLpStepOut=%d", params.inLevel2HLpStepOut);
    OP_LOGD(opType.c_str(), "inLevel1CLpStepIn=%d", params.inLevel1CLpStepIn);
    OP_LOGD(opType.c_str(), "inLevel1CLpStepOut=%d", params.inLevel1CLpStepOut);
    OP_LOGD(opType.c_str(), "inLevel0LpCnt=%d", params.inLevel0LpCnt);
    OP_LOGD(opType.c_str(), "inLevel0LpStepUb=%d", params.inLevel0LpStepUb);
    OP_LOGD(opType.c_str(), "inLevel0LpStepIn=%d", params.inLevel0LpStepIn);
    OP_LOGD(opType.c_str(), "inLevel0CCnt=%d", params.inLevel0CCnt);
    OP_LOGD(opType.c_str(), "inLevel0SubCSize=%d", params.inLevel0SubCSize);
    OP_LOGD(opType.c_str(), "inLevel0PerLineData=%d", params.inLevel0PerLineData);
    OP_LOGD(opType.c_str(), "inLevel0RepeatCnt=%d", params.inLevel0RepeatCnt);
    OP_LOGD(opType.c_str(), "inLevel0Nburst=%d", params.inLevel0Nburst);
    OP_LOGD(opType.c_str(), "inLevel0SrcStride=%d", params.inLevel0SrcStride);
    OP_LOGD(opType.c_str(), "inLevel0DstStride=%d", params.inLevel0DstStride);
    OP_LOGD(opType.c_str(), "outLevel1C1LpCnt=%d", params.outLevel1C1LpCnt);
    OP_LOGD(opType.c_str(), "outLevel1LpStepOut=%d", params.outLevel1LpStepOut);
    OP_LOGD(opType.c_str(), "outLevel0NLpCnt=%d", params.outLevel0NLpCnt);
    OP_LOGD(opType.c_str(), "outLevel0LpStepUb=%d", params.outLevel0LpStepUb);
    OP_LOGD(opType.c_str(), "outLevel0LpStepOut=%d", params.outLevel0LpStepOut);
    OP_LOGD(opType.c_str(), "outLevel0RepeatCnt=%d", params.outLevel0RepeatCnt);
    OP_LOGD(opType.c_str(), "outLevel0Nburst=%d", params.outLevel0Nburst);
    OP_LOGD(opType.c_str(), "outLevel0SrcStride=%d", params.outLevel0SrcStride);
    OP_LOGD(opType.c_str(), "outLevel0DstStride=%d", params.outLevel0DstStride);

    OP_LOGD(opType.c_str(), "inLevelx1NlcLpCnt=%d", params.inLevelx1NlcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx1NlcLeftLines=%d", params.inLevelx1NlcLeftLines);
    OP_LOGD(opType.c_str(), "inLevelx2NlcLpCnt=%d", params.inLevelx2NlcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx2NlcLeftLines=%d", params.inLevelx2NlcLeftLines);
    OP_LOGD(opType.c_str(), "inLevelx3NlcLpCnt=%d", params.inLevelx3NlcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx3NlcLeftLines=%d", params.inLevelx3NlcLeftLines);
    OP_LOGD(opType.c_str(), "inLevel2NlcHLpCnt=%d", params.inLevel2NlcHLpCnt);
    OP_LOGD(opType.c_str(), "inLevel2NlcHLeftLines=%d", params.inLevel2NlcHLeftLines);
    OP_LOGD(opType.c_str(), "inLevel1NlcCLpCnt=%d", params.inLevel1NlcCLpCnt);
    OP_LOGD(opType.c_str(), "inLevel1NlcLeftLines=%d", params.inLevel1NlcLeftLines);
    OP_LOGD(opType.c_str(), "inLevel0NlcLpCnt=%d", params.inLevel0NlcLpCnt);
    OP_LOGD(opType.c_str(), "inLevel0NlcCCnt=%d", params.inLevel0NlcCCnt);
    OP_LOGD(opType.c_str(), "inLevel0NlcSubCSize=%d", params.inLevel0NlcSubCSize);
    OP_LOGD(opType.c_str(), "inLevel0NlcPerLineData=%d", params.inLevel0NlcPerLineData);
    OP_LOGD(opType.c_str(), "inLevel0NlcRepeatCnt=%d", params.inLevel0NlcRepeatCnt);
    OP_LOGD(opType.c_str(), "inLevel0NlcNburstNt=%d", params.inLevel0NlcNburstNt);
    OP_LOGD(opType.c_str(), "inLevel0NlcNburstT=%d", params.inLevel0NlcNburstT);
    OP_LOGD(opType.c_str(), "outLevel1NlcC1LpCnt=%d", params.outLevel1NlcC1LpCnt);
    OP_LOGD(opType.c_str(), "outLevel0NlcNLpCnt=%d", params.outLevel0NlcNLpCnt);
    OP_LOGD(opType.c_str(), "outLevel0NlcRepeatCnt=%d", params.outLevel0NlcRepeatCnt);
    OP_LOGD(opType.c_str(), "outLevel0NlcNburst=%d", params.outLevel0NlcNburst);

    OP_LOGD(opType.c_str(), "inLevelx1LcLpCnt=%d", params.inLevelx1LcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx1LcLeftLines=%d", params.inLevelx1LcLeftLines);
    OP_LOGD(opType.c_str(), "inLevelx2LcLpCnt=%d", params.inLevelx2LcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx2LcLeftLines=%d", params.inLevelx2LcLeftLines);
    OP_LOGD(opType.c_str(), "inLevelx3LcLpCnt=%d", params.inLevelx3LcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx3LcLeftLines=%d", params.inLevelx3LcLeftLines);
    OP_LOGD(opType.c_str(), "inLevel2LcHLpCnt=%d", params.inLevel2LcHLpCnt);
    OP_LOGD(opType.c_str(), "inLevel2LcHLeftLines=%d", params.inLevel2LcHLeftLines);
    OP_LOGD(opType.c_str(), "inLevel1LcCLpCnt=%d", params.inLevel1LcCLpCnt);
    OP_LOGD(opType.c_str(), "inLevel1LcLeftLines=%d", params.inLevel1LcLeftLines);
    OP_LOGD(opType.c_str(), "inLevel0LcLpCnt=%d", params.inLevel0LcLpCnt);
    OP_LOGD(opType.c_str(), "inLevel0LcCCnt=%d", params.inLevel0LcCCnt);
    OP_LOGD(opType.c_str(), "inLevel0LcSubCSize=%d", params.inLevel0LcSubCSize);
    OP_LOGD(opType.c_str(), "inLevel0LcPerLineData=%d", params.inLevel0LcPerLineData);
    OP_LOGD(opType.c_str(), "inLevel0LcRepeatCnt=%d", params.inLevel0LcRepeatCnt);
    OP_LOGD(opType.c_str(), "inLevel0LcNburstNt=%d", params.inLevel0LcNburstNt);
    OP_LOGD(opType.c_str(), "inLevel0LcNburstT=%d", params.inLevel0LcNburstT);
    OP_LOGD(opType.c_str(), "outLevel1LcC1LpCnt=%d", params.outLevel1LcC1LpCnt);
    OP_LOGD(opType.c_str(), "outLevel0LcNLpCnt=%d", params.outLevel0LcNLpCnt);
    OP_LOGD(opType.c_str(), "outLevel0LcRepeatCnt=%d", params.outLevel0LcRepeatCnt);
    OP_LOGD(opType.c_str(), "outLevel0LcNburst=%d", params.outLevel0LcNburst);
}

void PrintTilingMode100Params(const std::string& opType, const TransDataMode100Param& params) {
    OP_LOGD(opType.c_str(), "tillingParamCount=%d", params.tillingParamCount);

    OP_LOGD(opType.c_str(), "tilingMode=%d", params.tilingMode);
    OP_LOGD(opType.c_str(), "ubOffset=%d", params.ubOffset);
    OP_LOGD(opType.c_str(), "usedCoreCnt=%d", params.usedCoreCnt);
    OP_LOGD(opType.c_str(), "coreStepIn=%d", params.coreStepIn);
    OP_LOGD(opType.c_str(), "coreStepOut=%d", params.coreStepOut);
    OP_LOGD(opType.c_str(), "oneLineSize=%d", params.oneLineSize);

    OP_LOGD(opType.c_str(), "inLevelx1LpStepIn=%d", params.inLevelx1LpStepIn);
    OP_LOGD(opType.c_str(), "inLevelx1LpStepOut=%d", params.inLevelx1LpStepOut);
    OP_LOGD(opType.c_str(), "inLevelx2LpStepIn=%d", params.inLevelx2LpStepIn);
    OP_LOGD(opType.c_str(), "inLevelx2LpStepOut=%d", params.inLevelx2LpStepOut);
    OP_LOGD(opType.c_str(), "inLevelx3LpStepIn=%d", params.inLevelx3LpStepIn);
    OP_LOGD(opType.c_str(), "inLevelx3LpStepOut=%d", params.inLevelx3LpStepOut);
    OP_LOGD(opType.c_str(), "inLevel2C1LpStepIn=%d", params.inLevel2C1LpStepIn);
    OP_LOGD(opType.c_str(), "inLevel2C1LpStepOut=%d", params.inLevel2C1LpStepOut);
    OP_LOGD(opType.c_str(), "inLevel1LastLpStepIn=%d", params.inLevel1LastLpStepIn);
    OP_LOGD(opType.c_str(), "inLevel1LastLpStepOut=%d", params.inLevel1LastLpStepOut);
    OP_LOGD(opType.c_str(), "inLevel0LpCnt=%d", params.inLevel0LpCnt);
    OP_LOGD(opType.c_str(), "inLevel0C0LpStepUb=%d", params.inLevel0C0LpStepUb);
    OP_LOGD(opType.c_str(), "inLevel0LpStepIn=%d", params.inLevel0LpStepIn);
    OP_LOGD(opType.c_str(), "inLevel0RepeatCnt=%d", params.inLevel0RepeatCnt);
    OP_LOGD(opType.c_str(), "inLevel0Nburst=%d", params.inLevel0Nburst);
    OP_LOGD(opType.c_str(), "inLevel0SrcStride=%d", params.inLevel0SrcStride);
    OP_LOGD(opType.c_str(), "inLevel0DstStride=%d", params.inLevel0DstStride);
    OP_LOGD(opType.c_str(), "outLevel1LpStepUb=%d", params.outLevel1LpStepUb);
    OP_LOGD(opType.c_str(), "outLevel1LpStepOut=%d", params.outLevel1LpStepOut);
    OP_LOGD(opType.c_str(), "outLevel0LpCnt=%d", params.outLevel0LpCnt);
    OP_LOGD(opType.c_str(), "outLevel0LpStepUb=%d", params.outLevel0LpStepUb);
    OP_LOGD(opType.c_str(), "outLevel0LpStepOut=%d", params.outLevel0LpStepOut);
    OP_LOGD(opType.c_str(), "outLevel0RepeatCnt=%d", params.outLevel0RepeatCnt);
    OP_LOGD(opType.c_str(), "outLevel0Nburst=%d", params.outLevel0Nburst);
    OP_LOGD(opType.c_str(), "outLevel0SrcStride=%d", params.outLevel0SrcStride);
    OP_LOGD(opType.c_str(), "outLevel0DstStride=%d", params.outLevel0DstStride);

    OP_LOGD(opType.c_str(), "inLevelx1NlcLpCnt=%d", params.inLevelx1NlcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx1NlcLeftLines=%d", params.inLevelx1NlcLeftLines);
    OP_LOGD(opType.c_str(), "inLevelx2NlcLpCnt=%d", params.inLevelx2NlcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx2NlcLeftLines=%d", params.inLevelx2NlcLeftLines);
    OP_LOGD(opType.c_str(), "inLevelx3NlcLpCnt=%d", params.inLevelx3NlcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx3NlcLeftLines=%d", params.inLevelx3NlcLeftLines);
    OP_LOGD(opType.c_str(), "inLevel2NlcC1LpCnt=%d", params.inLevel2NlcC1LpCnt);
    OP_LOGD(opType.c_str(), "inLevel2NlcC1LeftLines=%d", params.inLevel2NlcC1LeftLines);
    OP_LOGD(opType.c_str(), "inLevel1NlcLastLpCnt=%d", params.inLevel1NlcLastLpCnt);
    OP_LOGD(opType.c_str(), "inLevel1NlcLastLeftLines=%d", params.inLevel1NlcLastLeftLines);
    OP_LOGD(opType.c_str(), "inLevel0NlcLpCnt=%d", params.inLevel0NlcLpCnt);
    OP_LOGD(opType.c_str(), "inLevel0NlcRepeatCnt=%d", params.inLevel0NlcRepeatCnt);
    OP_LOGD(opType.c_str(), "inLevel0NlcNburst=%d", params.inLevel0NlcNburst);
    OP_LOGD(opType.c_str(), "outLevel1NlcLpCnt=%d", params.outLevel1NlcLpCnt);
    OP_LOGD(opType.c_str(), "outLevel0NlcLpCnt=%d", params.outLevel0NlcLpCnt);
    OP_LOGD(opType.c_str(), "outLevel0NlcRepeatCnt=%d", params.outLevel0NlcRepeatCnt);
    OP_LOGD(opType.c_str(), "outLevel0NlcNburst=%d", params.outLevel0NlcNburst);

    OP_LOGD(opType.c_str(), "inLevelx1LcLpCnt=%d", params.inLevelx1LcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx1LcLeftLines=%d", params.inLevelx1LcLeftLines);
    OP_LOGD(opType.c_str(), "inLevelx2LcLpCnt=%d", params.inLevelx2LcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx2LcLeftLines=%d", params.inLevelx2LcLeftLines);
    OP_LOGD(opType.c_str(), "inLevelx3LcLpCnt=%d", params.inLevelx3LcLpCnt);
    OP_LOGD(opType.c_str(), "inLevelx3LcLeftLines=%d", params.inLevelx3LcLeftLines);
    OP_LOGD(opType.c_str(), "inLevel2LcC1LpCnt=%d", params.inLevel2LcC1LpCnt);
    OP_LOGD(opType.c_str(), "inLevel2LcC1LeftLines=%d", params.inLevel2LcC1LeftLines);
    OP_LOGD(opType.c_str(), "inLevel1LcLastLpCnt=%d", params.inLevel1LcLastLpCnt);
    OP_LOGD(opType.c_str(), "inLevel1LcLastLeftLines=%d", params.inLevel1LcLastLeftLines);
    OP_LOGD(opType.c_str(), "inLevel0LcLpCnt=%d", params.inLevel0LcLpCnt);
    OP_LOGD(opType.c_str(), "inLevel0LcRepeatCnt=%d", params.inLevel0LcRepeatCnt);
    OP_LOGD(opType.c_str(), "inLevel0LcNburst=%d", params.inLevel0LcNburst);
    OP_LOGD(opType.c_str(), "outLevel1LcLpCnt=%d", params.outLevel1LcLpCnt);
    OP_LOGD(opType.c_str(), "outLevel0LcLpCnt=%d", params.outLevel0LcLpCnt);
    OP_LOGD(opType.c_str(), "outLevel0LcRepeatCnt=%d", params.outLevel0LcRepeatCnt);
    OP_LOGD(opType.c_str(), "outLevel0LcNburst=%d", params.outLevel0LcNburst);

    OP_LOGD(opType.c_str(), "nextShapeLpOffsetIn=%d", params.nextShapeLpOffsetIn);
    OP_LOGD(opType.c_str(), "nextShapeLpOffsetOut=%d", params.nextShapeLpOffsetOut);
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool TransDataTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& op_info,
                     OpRunInfo& runInfo) {
    OP_LOGI(opType.c_str(), "Tiling is running.");
    if (op_info == nullptr) {
        OP_LOGE(opType.c_str(), "op TransDataTiling: op_info json error.");
        return false;
    }
    if (opParas.inputs.empty() || opParas.inputs.size() < 1 || opParas.inputs[0].tensor.empty()) {
        ge::OpsOneInputShapeErrReport(opType.c_str(), "src",
                                      "The length of inputs is less than 1 or the inputs is empty");
        OP_LOGE(opType.c_str(), "op TransDataTiling: input shape error.");
        return false;
    }
    if (opParas.outputs.empty() || opParas.outputs.size() < 1 || opParas.outputs[0].tensor.empty()) {
        ge::OpsOneOutputShapeErrReport(opType.c_str(), "dst",
                                      "The length of outputs is less than 1 or the outputs is empty");
        OP_LOGE(opType.c_str(), "op TransDataTiling: output shape error.");
        return false;
    }

    std::vector<int64_t> inShape = opParas.inputs[0].tensor[0].shape;
    std::vector<int64_t> outShape = opParas.outputs[0].tensor[0].shape;
    std::string srcFormat;
    std::string dstFormat;
    std::string realSrcFormat;
    std::string realDstFormat;
    std::string dType;
    int32_t ubSize = 0;
    int32_t blockDim = 0;
    std::vector<int64_t> inShapeNew;
    std::vector<int64_t> outShapeNew;
    std::vector<int32_t> combAxis;
    int32_t inputSize = 0;
    int32_t hiddenSize = 0;
    int32_t group = 1;
    int32_t c0Len = GetC0Len(dType);

    bool flag = GetCompileParams(op_info, srcFormat, dstFormat, dType, ubSize, blockDim, inputSize, hiddenSize, group,
                                 opType);
    if (!flag) {
        OP_LOGE("op[%s] TransDataTiling: GetCompileParams error.", opType.c_str());
        return false;
    }

    bool ret = CheckTensorShape(opType, ubSize, blockDim, outShape);
    if (!ret) {
        OP_LOGE(opType.c_str(), "op TransDataTiling: CheckTensor Failed.");
        return ret;
    }

    if (srcFormat == "ND" && dstFormat == "FRACTAL_ZN_RNN") {
        int32_t fuseShape = GetShapeSize(inShape, -2);
        int32_t hiddenCnt = inShape[inShape.size() - 1] / hiddenSize;
        std::vector<int64_t> inShapeUp = {fuseShape, inShape[inShape.size() - 2], hiddenCnt, hiddenSize};
        if (inShape[inShape.size()-2] > max(inputSize, hiddenSize)) {
            combAxis.push_back(-3);
            combAxis.push_back(inputSize);
            combAxis.push_back(hiddenSize);
            std::vector<int64_t> inShapeP2;
            std::vector<int64_t> outShapeP2;
            flag = GetRenew4Shape(inShapeUp, outShape, srcFormat, dstFormat, combAxis, c0Len, group, inShapeNew,
                                  outShapeNew, inShapeP2, outShapeP2, realSrcFormat, realDstFormat);
        } else {
            flag = GetRenew2Shape(inShapeUp, outShape, srcFormat, dstFormat, combAxis, c0Len, group, inShapeNew,
                                  outShapeNew, realSrcFormat, realDstFormat);
        }
        if (!flag) {
            OP_LOGE(opType.c_str(), "TransDataTiling: get Renew Shape tiling params error");
            return false;
        }

    } else if (srcFormat == "ND" && dstFormat == "ND_RNN_BIAS") {
        int32_t hiddenCnt = inShape[inShape.size() - 1] / hiddenSize;
        std::vector<int64_t> inShapeUp = {hiddenCnt, hiddenSize};
        flag = GetRenew2Shape(inShapeUp, outShape, srcFormat, dstFormat, combAxis, c0Len, group, inShapeNew,
                              outShapeNew, realSrcFormat, realDstFormat);
        if (!flag) {
            OP_LOGE(opType.c_str(), "TransDataTiling: GetRenew2Shape tiling params error");
            return false;
        }

    } else if (srcFormat == "FRACTAL_ZN_RNN" && dstFormat == "ND") {
        int32_t hiddenCnt = outShape[outShape.size() - 1] / hiddenSize;
        int32_t fuseShape = GetShapeSize(outShape, -2);
        std::vector<int64_t> outShapeUp = {fuseShape, outShape[outShape.size() - 2], hiddenCnt, hiddenSize};
        if (outShape[outShape.size() - 2] > max(inputSize, hiddenSize)) {
            combAxis.push_back(-3);
            combAxis.push_back(inputSize);
            combAxis.push_back(hiddenSize);
            std::vector<int64_t> inShapeP2;
            std::vector<int64_t> outShapeP2;
            flag = GetRenew4Shape(inShape, outShapeUp, srcFormat, dstFormat, combAxis, c0Len, group, inShapeNew,
                                  outShapeNew, inShapeP2, outShapeP2, realSrcFormat, realDstFormat);
        } else {
            flag = GetRenew2Shape(inShape, outShapeUp, srcFormat, dstFormat, combAxis, c0Len, group, inShapeNew,
                                  outShapeNew, realSrcFormat, realDstFormat);
        }
        if (!flag) {
            OP_LOGE(opType.c_str(), "TransDataTiling: get Renew Shape tiling params error");
            return false;
        }

    } else if (srcFormat == "ND_RNN_BIAS" && dstFormat == "ND") {
        int32_t hiddenCnt = outShape[outShape.size() - 1] / hiddenSize;
        std::vector<int64_t> outShapeUp = {hiddenCnt, hiddenSize};
        flag = GetRenew2Shape(inShape, outShapeUp, srcFormat, dstFormat, combAxis, c0Len, group, inShapeNew,
                              outShapeNew, realSrcFormat, realDstFormat);
        if (!flag) {
            OP_LOGE(opType.c_str(), "TransDataTiling: GetRenew2Shape tiling params error");
            return false;
        }

    } else {
        flag = GetRenew2Shape(inShape, outShape, srcFormat, dstFormat, combAxis, c0Len, group, inShapeNew,
                              outShapeNew, realSrcFormat, realDstFormat);
        if (!flag) {
            OP_LOGE(opType.c_str(), "TransDataTiling: GetRenew2Shape tiling params error");
            return false;
        }
    }

    HeadTilingParam headParams;
    int32_t blockElemCnt = BLOCK_BYTE_SIZE / GetDTypeLen(dType);
    int32_t axisPosC = std::strchr(realSrcFormat.c_str(), 'C') - realSrcFormat.c_str();
    headParams.shapeLoopCnt = 1;
    int32_t shapeLpIdx = 0;

    /*      get part1 multiple core axis       */
    int32_t multiCoreAxisPosPart1 = GetMultiCoreAxis(inShapeNew, axisPosC, blockElemCnt, c0Len, blockDim);

    if (realSrcFormat[realSrcFormat.length() - 1] != 'C' && realDstFormat[realDstFormat.length() - 1] == 'T') {
        TransDataMode100Param runParamsPart1;
        flag = TillingPositiveMode100(inShapeNew, outShapeNew, realSrcFormat, realDstFormat, multiCoreAxisPosPart1,
                                      axisPosC, blockDim, blockElemCnt, c0Len, ubSize, shapeLpIdx, runParamsPart1);
        if (!flag) {
            OP_LOGE(opType.c_str(), "TransDataTiling: get TransDataMode100Param tiling params error");
            return false;
        }
        ByteBufferPut(runInfo.tiling_data, headParams.shapeLoopCnt);
        OP_LOGD(opType.c_str(), "shapeLoopCnt=%d", headParams.shapeLoopCnt);
        SetRunningMode100Params(runParamsPart1, runInfo);
        OP_LOGD(opType.c_str(), "start print runParams");
        PrintTilingMode100Params(opType, runParamsPart1);
    } else if (realSrcFormat[realSrcFormat.length() - 1] == 'C' && realDstFormat[realDstFormat.length() - 1] == 'T') {
        TransDataMode101Param runParamsPart1;
        flag = TillingPositiveMode101(inShapeNew, outShapeNew, realSrcFormat, realDstFormat, multiCoreAxisPosPart1,
                                      axisPosC, blockDim, blockElemCnt, c0Len, ubSize, shapeLpIdx, runParamsPart1);
        if (!flag) {
            OP_LOGE(opType.c_str(), "TransDataTiling: get TransDataMode101Param tiling params error");
            return false;
        }
        ByteBufferPut(runInfo.tiling_data, headParams.shapeLoopCnt);
        OP_LOGD(opType.c_str(), "shapeLoopCnt=%d", headParams.shapeLoopCnt);
        SetRunningMode101Params(runParamsPart1, runInfo);
        OP_LOGD(opType.c_str(), "start print runParams");
        PrintTilingMode101Params(opType, runParamsPart1);
    } else if ((srcFormat == "NC1HWC0" && dstFormat == "NHWC") || (srcFormat == "FRACTAL_NZ" && dstFormat == "ND") ||
               (srcFormat == "FRACTAL_Z_3D" && dstFormat == "NDHWC")) {
        TransDataMode201Param runParams201;
        flag = TillingNegativeMode201(inShape, outShape, srcFormat, dstFormat, blockDim, blockElemCnt, ubSize, runParams201);
        if (!flag) {
            OP_LOGE(opType.c_str(), "TransDataTiling: get TransDataMode201Param tiling params error");
            return false;
        }
        OP_LOGD(opType.c_str(), "***start to put mode 201 tiling parameters");
        SetRunningMode201Params(runParams201, runInfo);
        PrintTilingMode201Params(opType, runParams201);
    }

    // block_dim, core num used in tik op
    runInfo.block_dim = blockDim;
    // workspace, null for tik op
    std::vector<int64_t> workspace;
    runInfo.workspaces = workspace;

    OP_LOGI(opType.c_str(), "tiling run success.");

    return true;
}

// register tiling interface of the TransData op
REGISTER_OP_TILING_FUNC_BUFFERED(TransData, TransDataTiling);

}  // namespace optiling
