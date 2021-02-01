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
 * \file trans_data_negative_target_tc.cpp
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

const int64_t FRAME_LEVEL = 3;

void GenNewShape(const std::vector<int64_t>& tempInShape, const std::vector<int64_t>& tempOutShape, std::vector<int64_t>& inShapeNew,
                 std::vector<int64_t>& outShapeNew) {
    for (auto i : tempInShape) {
        inShapeNew.push_back(i);
    }

    for (auto j : tempOutShape) {
        outShapeNew.push_back(j);
    }
}

bool RenewInputOutputShapeFormat(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                                 std::string& inFormat, std::string& outFormat, std::vector<int64_t>& inShapeNew,
                                 std::vector<int64_t>& outShapeNew, std::string& inFormatNew,
                                 std::string& outFormatNew) {
    int64_t axisN = 1;
    int64_t axisC1 = 1;
    int64_t axisH = 1;
    int64_t axisW = 1;
    int64_t axisC0 = 1;
    int64_t axisHW = 1;
    int64_t axisC = 1;
    int64_t axisD = 1;
    int64_t axisNo = 1;
    int64_t axisDC1HW = 1;

    transform(inFormat.begin(), inFormat.end(), inFormat.begin(), ::toupper);
    transform(outFormat.begin(), outFormat.end(), outFormat.begin(), ::toupper);

    if (inFormat == "NC1HWC0" && outFormat == "NHWC") {
        if (inShape.size() != 5 || outShape.size() != 4) {
            OP_LOGE("trans_data", "The input shape dimension size should be 5 and output shape dimension should be 4!");
            return false;
        }
        inFormatNew = "NCHT";
        outFormatNew = "NHC";
        axisN = inShape[0];
        axisC1 = inShape[1];
        axisH = inShape[2];
        axisW = inShape[3];
        axisC0 = inShape[4];
        axisC = outShape[outShape.size() - 1];
        axisHW = axisH * axisW;

        int64_t tempInArray[4] = {axisN, axisC1, axisHW, axisC0};
        int64_t tempOutArray[3] = {axisN, axisHW, axisC};
        std::vector<int64_t> tempInShape(tempInArray, tempInArray+4);
        std::vector<int64_t> tempOutShape(tempOutArray, tempOutArray+3);
        GenNewShape(tempInShape, tempOutShape, inShapeNew, outShapeNew);

        return true;
    }

    if (inFormat == "FRACTAL_NZ" && outFormat == "ND") {
        if (outShape.size() == 0) {
            OP_LOGE("trans_data", "The input shape dimension size cannot be 0!");
            return false;
        }
        inFormatNew = "HCNT";
        outFormatNew = "HNC";

        if (outShape.size() == 1) {
          axisH = 1;
          axisN = 1;
          axisC = outShape[0];
        } else if (outShape.size() == 2) {
          axisH = 1;
          axisN = outShape[0];
          axisC = outShape[1];
        } else {
          for (int32_t i = 0; i < outShape.size() - 2; i++) {
            axisH *= outShape[i];
          }
          axisN = outShape[outShape.size() - 2];
          axisC = outShape[outShape.size() - 1];
        }

        axisC0 = inShape[inShape.size() - 1];
        axisC1 = GetCeilDiv(axisC, axisC0);
        axisNo = GetCeilDiv(axisN, NI_16);

        int64_t tempInArray[4] = {axisH, axisC1, axisNo * NI_16, axisC0};
        int64_t tempOutArray[3] = {axisH, axisN, axisC};
        std::vector<int64_t> tempInShape(tempInArray, tempInArray+4);
        std::vector<int64_t> tempOutShape(tempOutArray, tempOutArray+3);
        GenNewShape(tempInShape, tempOutShape, inShapeNew, outShapeNew);

        return true;
    }

    if (inFormat == "FRACTAL_Z_3D" && outFormat == "NDHWC") {
        inFormatNew = "DCHNT";
        outFormatNew = "NDHC";

        if (inShape.size() != 4 || outShape.size() != 5) {
            OP_LOGE("trans_data", "The input or output shape dimension size is not correct!");
            return false;
        }
        axisDC1HW = inShape[0];
        axisNo = inShape[1];
        axisC0 = inShape[3];
        axisN = outShape[0];
        axisD = outShape[1];
        axisH = outShape[2];
        axisW = outShape[3];
        axisC = outShape[4];
        axisHW = axisH * axisW;
        axisC1 = axisDC1HW / (axisD * axisHW);

        int64_t tempInArray[5] = {axisD, axisC1, axisHW, axisNo * NI_16, axisC0};
        int64_t tempOutArray[4] = {axisN, axisD, axisHW, axisC};
        std::vector<int64_t> tempInShape(tempInArray, tempInArray+5);
        std::vector<int64_t> tempOutShape(tempOutArray, tempOutArray+4);
        GenNewShape(tempInShape, tempOutShape, inShapeNew, outShapeNew);

        return true;
    }
}

bool GetMcInfoNegative201(const std::vector<int64_t>& r2ndArgs, const std::vector<int64_t>& c1Args,
                          const int64_t leftLpCnt, const int64_t coreNum, std::vector<int64_t>& mcParams) {
    int64_t r2ndLpCnt = r2ndArgs[0];
    int64_t r2ndLpStepIn = r2ndArgs[1];
    int64_t r2ndLpStepOut = r2ndArgs[2];
    int64_t r2ndSize = r2ndArgs[3];
    int64_t r2ndLpUnit = r2ndArgs[4];
    int64_t c1LpCnt = c1Args[0];
    int64_t c1LpStepIn = c1Args[1];
    int64_t c1LpStepOut = c1Args[2];
    int64_t c1Size = c1Args[3];
    int64_t c1LpUnit = c1Args[4];

    int64_t tmpFullLpCntR2nd = (r2ndLpCnt / coreNum > 0) ? coreNum : 0;
    int64_t reminderLpCntR2nd = r2ndLpCnt % coreNum;
    tmpFullLpCntR2nd = (reminderLpCntR2nd == 0) ? tmpFullLpCntR2nd + coreNum : tmpFullLpCntR2nd;
    int64_t fullLpCntR2nd = tmpFullLpCntR2nd + reminderLpCntR2nd;

    int64_t tmpFullLpCntC1 = (c1LpCnt / coreNum > 0) ? coreNum : 0;
    int64_t reminderLpCntC1 = c1LpCnt % coreNum;
    tmpFullLpCntC1 = (reminderLpCntC1 == 0) ? tmpFullLpCntC1 + coreNum : tmpFullLpCntC1;
    int64_t fullLpCntC1 = tmpFullLpCntC1 + reminderLpCntC1;

    int64_t tmpFullLpCntLeft = (leftLpCnt / coreNum > 0) ? coreNum : 0;
    int64_t reminderLpCntLeft = leftLpCnt % coreNum;
    tmpFullLpCntLeft = (reminderLpCntLeft == 0) ? tmpFullLpCntLeft + coreNum : tmpFullLpCntLeft;
    int64_t fullLpCntLeft = tmpFullLpCntLeft + reminderLpCntLeft;

    if (fullLpCntLeft >= fullLpCntC1 && fullLpCntLeft >= fullLpCntR2nd) {
        int64_t usedCoreCnt = GetCeilDiv(leftLpCnt, GetCeilDiv(leftLpCnt, coreNum));
        int64_t nlcLeftLpCnt = GetCeilDiv(leftLpCnt, usedCoreCnt);
        int64_t lcLeftLpCnt = leftLpCnt - (usedCoreCnt - 1) * nlcLeftLpCnt;
        mcParams.push_back(1);  // mcMode
        mcParams.push_back(usedCoreCnt);  //usedCoreCnt
        mcParams.push_back(0);  // coreStepIn
        mcParams.push_back(0);  // coreStepOut
        mcParams.push_back(r2ndLpCnt);  // nlcR2ndLpCnt
        mcParams.push_back(c1LpCnt);  // nlcC1LpCnt
        mcParams.push_back(nlcLeftLpCnt);  // nlcLeftLpCnt
        mcParams.push_back(r2ndSize % r2ndLpUnit);  // nlcR2ndLeft
        mcParams.push_back(c1Size % c1LpUnit);  // nlcC1Left
        mcParams.push_back(r2ndLpCnt);  // lcR2ndLpCnt
        mcParams.push_back(c1LpCnt);  // lcC1LpCnt
        mcParams.push_back(lcLeftLpCnt);  // lcLeftLpCnt
        mcParams.push_back(r2ndSize % r2ndLpUnit);  // lcR2ndLeft
        mcParams.push_back(c1Size % c1LpUnit);  // lcC1Left

        return true;
  }

    if (fullLpCntC1 >= fullLpCntLeft && fullLpCntC1 >= fullLpCntR2nd) {
        int64_t usedCoreCnt = GetCeilDiv(c1LpCnt, GetCeilDiv(c1LpCnt, coreNum));
        int64_t nlcC1LpCnt = GetCeilDiv(c1LpCnt, usedCoreCnt);
        int64_t lcC1LpCnt = c1LpCnt - (usedCoreCnt - 1) * nlcC1LpCnt;
        mcParams.push_back(0);  // mcMode
        mcParams.push_back(usedCoreCnt);  // usedCoreCnt
        mcParams.push_back(nlcC1LpCnt * c1LpStepIn);  // coreStepIn
        mcParams.push_back(nlcC1LpCnt * c1LpStepOut);  // coreStepOut
        mcParams.push_back(r2ndLpCnt);  // nlcR2ndLpCnt
        mcParams.push_back(nlcC1LpCnt);  // nlcC1LpCnt
        mcParams.push_back(leftLpCnt); // nlcLeftLpCnt
        mcParams.push_back(r2ndSize % r2ndLpUnit);  // nlcR2ndLeft
        mcParams.push_back(0);  // nlcC1Left
        mcParams.push_back(r2ndLpCnt);  // lcR2ndLpCnt
        mcParams.push_back(lcC1LpCnt);  // lcC1LpCnt
        mcParams.push_back(leftLpCnt);  // lcLeftLpCnt
        mcParams.push_back(r2ndSize % r2ndLpUnit);  // lcR2ndLeft
        mcParams.push_back(c1Size % c1LpUnit);  // lcC1Left
 
        return true;
    }

    if (fullLpCntR2nd >= fullLpCntLeft && fullLpCntR2nd >= fullLpCntC1) {
        int64_t usedCoreCnt = GetCeilDiv(r2ndLpCnt, GetCeilDiv(r2ndLpCnt, coreNum));
        int64_t nlcR2ndLpCnt = GetCeilDiv(r2ndLpCnt, usedCoreCnt);
        int64_t lcR2ndLpCnt = r2ndLpCnt - (usedCoreCnt - 1) * nlcR2ndLpCnt;
        mcParams.push_back(0);  // mcMode
        mcParams.push_back(usedCoreCnt);  // usedCoreCnt
        mcParams.push_back(nlcR2ndLpCnt * r2ndLpStepIn);  // coreStepIn
        mcParams.push_back(nlcR2ndLpCnt * r2ndLpStepOut);  // coreStepOut
        mcParams.push_back(nlcR2ndLpCnt);  // nlcR2ndLpCnt
        mcParams.push_back(c1LpCnt);  // nlcC1LpCnt
        mcParams.push_back(leftLpCnt); // nlcLeftLpCnt
        mcParams.push_back(0);  // nlcR2ndLeft
        mcParams.push_back(c1Size % c1LpUnit);  // nlcC1Left
        mcParams.push_back(lcR2ndLpCnt);  // lcR2ndLpCnt
        mcParams.push_back(c1LpCnt);  // lcC1LpCnt
        mcParams.push_back(leftLpCnt);  // lcLeftLpCnt
        mcParams.push_back(r2ndSize % r2ndLpUnit);  // lcR2ndLeft
        mcParams.push_back(c1Size % c1LpUnit);  // lcC1Left
 
        return true;
    }
}

void GenIOIndex(std::string& tmpDstFormat, std::string& fullFormat, const std::vector<int64_t>& fullShape,
                const std::vector<int64_t>& subShape, std::vector<int64_t>& ioOrderParams) {
    int32_t chrPos = 0;
    for (int j=1; j <= tmpDstFormat.length(); j++) {
        chrPos = fullFormat.find(*(tmpDstFormat.end() - j));
        ioOrderParams.push_back(fullShape[chrPos]);
        ioOrderParams.push_back(GetShapeSize(subShape, -j));
        ioOrderParams.push_back(GetShapeSize(fullShape, chrPos + 1));
    }
}

void PadIOIndex(const int32_t padAxisCnt, std::vector<int64_t>& ioOrderParams) {
    if (padAxisCnt > 0) {
        for (int32_t k=0; k < padAxisCnt; k++) {
            ioOrderParams.push_back(int64_t(0));
            ioOrderParams.push_back(int64_t(0));
            ioOrderParams.push_back(int64_t(0));
        }
    }
}

bool TillingNegativeMode201(std::vector<int64_t>& inShape, std::vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, const int64_t coreNum, const int64_t blockElemCnt, const int64_t ubSize,
                            TransDataMode201Param& params) {
    std::vector<int64_t> inShapeNew, outShapeNew;
    std::string srcFormatNew, dstFormatNew;
    RenewInputOutputShapeFormat(inShape, outShape, srcFormat, dstFormat, inShapeNew, outShapeNew, srcFormatNew, dstFormatNew);

    int64_t shapeLen = inShapeNew.size();
    int64_t c0Len = inShapeNew[srcFormatNew.length() - 1];
    int64_t halfUbSize = ubSize / 2;
    params.tilingMode = 201;
    params.ubOffset = halfUbSize / blockElemCnt * blockElemCnt;

    // axis -2 tiling parameters
    int32_t srcR2ndInDstPos = dstFormatNew.find(*(srcFormatNew.end() - 2));
    int64_t axisSrcR2ndSize = outShapeNew[srcR2ndInDstPos];
    params.srcR2ndLpUnit = NI_16;
    int64_t srcR2ndLpCnt = GetCeilDiv(axisSrcR2ndSize, params.srcR2ndLpUnit);
    params.srcR2ndLpStepIn = params.srcR2ndLpUnit * c0Len;
    params.srcR2ndLpStepOut = GetShapeSize(outShapeNew, srcR2ndInDstPos + 1);

    // axis c1 tiling parameters
    int32_t srcC1Pos = srcFormatNew.find('C');
    int64_t axisSrcC1Size = inShapeNew[srcC1Pos];
    params.srcC1LpUnit = 8;
    int64_t srcC1LpCnt = GetCeilDiv(axisSrcC1Size, params.srcC1LpUnit);
    int64_t srcC1Left = axisSrcC1Size % params.srcC1LpUnit;
    params.srcC1LpStepIn = GetShapeSize(inShapeNew, srcC1Pos + 1) * params.srcC1LpUnit;
    params.srcC1LpStepOut = params.srcC1LpUnit * c0Len;
    params.srcC1StepIn = GetShapeSize(inShapeNew, srcC1Pos + 1);
    params.perLineDstCCount = (axisSrcC1Size < params.srcC1LpUnit) ? params.srcC1LpUnit / srcC1Left : 1;
    params.cModC0 = outShapeNew[dstFormatNew.find('C')] % c0Len;

    // axis left tiling parameters
    std::string tmpSrcFormat = srcFormatNew;
    std::string tmpDstFormat = dstFormatNew;
    int64_t srcLeftAxisSize = 1;
    std::vector<int64_t> tmpLeftInShape;
    std::vector<int64_t> tmpLeftOutShape;
    tmpSrcFormat.replace(srcFormatNew.length()-2, 2, "");
    tmpSrcFormat.replace(tmpSrcFormat.find('C'), 1, "");
    tmpDstFormat.replace(srcR2ndInDstPos, 1, "");
    tmpDstFormat.replace(tmpDstFormat.find('C'), 1, "");
    for (int i=0; i < tmpSrcFormat.length(); i++) {
        srcLeftAxisSize *= inShapeNew[srcFormatNew.find(*(tmpSrcFormat.begin() + i))];
        tmpLeftInShape.push_back(inShapeNew[srcFormatNew.find(*(tmpSrcFormat.begin() + i))]);
    }
    for (int i=0; i < tmpDstFormat.length(); i++) {
        tmpLeftOutShape.push_back(outShapeNew[dstFormatNew.find(*(tmpDstFormat.begin() + i))]);
    }
    tmpLeftOutShape.push_back(1);  // for convenient calculation
    int32_t padAxisCnt = FRAME_LEVEL - tmpDstFormat.length();
    GenIOIndex(tmpDstFormat, srcFormatNew, inShapeNew, tmpLeftOutShape, params.ioOrderParams);
    PadIOIndex(padAxisCnt, params.ioOrderParams);
    GenIOIndex(tmpDstFormat, dstFormatNew, outShapeNew, tmpLeftOutShape, params.ioOrderParams);
    PadIOIndex(padAxisCnt, params.ioOrderParams);

    // multiple core tiling parameters
    int64_t tmpR2ndArray[5] = {srcR2ndLpCnt, params.srcR2ndLpStepIn,
                              params.srcR2ndLpStepOut * params.srcR2ndLpUnit, axisSrcR2ndSize, params.srcR2ndLpUnit};
    int64_t tmpC1Array[5] = {srcC1LpCnt, params.srcC1LpStepIn, params.srcC1LpStepOut, axisSrcC1Size, params.srcC1LpUnit};
    std::vector<int64_t> r2ndArgs(tmpR2ndArray, tmpR2ndArray + 5);
    std::vector<int64_t> c1Args(tmpC1Array, tmpC1Array + 5);
    GetMcInfoNegative201(r2ndArgs, c1Args, srcLeftAxisSize, coreNum, params.mcParams);

    // vnchwconv tiling parameters
    params.src2DstFlag = (*(srcFormatNew.end() - 2) == *(dstFormatNew.end() - 2)) ? 0 : 1;

    return true;
}

void SetRunningMode201Params(const TransDataMode201Param& runParams, OpRunInfo& runInfo) {
    ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);
    ByteBufferPut(runInfo.tiling_data, runParams.ubOffset);
    for (auto i : runParams.mcParams) {
        ByteBufferPut(runInfo.tiling_data, int64_t(i));
    }
    ByteBufferPut(runInfo.tiling_data, runParams.srcR2ndLpUnit);
    ByteBufferPut(runInfo.tiling_data, runParams.srcR2ndLpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.srcR2ndLpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.srcC1StepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.srcC1LpUnit);
    ByteBufferPut(runInfo.tiling_data, runParams.srcC1LpStepIn);
    ByteBufferPut(runInfo.tiling_data, runParams.srcC1LpStepOut);
    ByteBufferPut(runInfo.tiling_data, runParams.perLineDstCCount);
    ByteBufferPut(runInfo.tiling_data, runParams.cModC0);
    for (auto i : runParams.ioOrderParams) {
        ByteBufferPut(runInfo.tiling_data, int64_t(i));
    }
    ByteBufferPut(runInfo.tiling_data, runParams.src2DstFlag);
}

void PrintTilingMode201Params(const std::string& opType, const TransDataMode201Param& params) {
    OP_LOGD(opType.c_str(), "***Begin to print mode 201 tiling parameters:");
    OP_LOGD(opType.c_str(), "tillingParamCount=%d", params.tilingMode);
    OP_LOGD(opType.c_str(), "ubOffset=%d", params.ubOffset);
    string tmp_str;
    tmp_str = "";
    for (auto i : params.mcParams) {
        tmp_str += std::to_string(i);
        tmp_str += ",";
    }
    OP_LOGD(opType.c_str(), "mcParams order is: mcFlag, usedCoreCnt, coreStepIn, coreStepOut, nlcR2ndLpCnt, nlcC1LpCnt, "
                            "nlcLeftLpCnt, nlcR2ndLeft, nlcC1Left, lcR2ndLpCnt, lcC1LpCnt, lcLeftLpCnt, lcR2ndLeft, lcC1Left");
    OP_LOGD(opType.c_str(), "mcParams=%s", tmp_str.c_str());
    OP_LOGD(opType.c_str(), "srcR2ndLpUnit=%d", params.srcR2ndLpUnit);
    OP_LOGD(opType.c_str(), "srcR2ndLpStepIn=%d", params.srcR2ndLpStepIn);
    OP_LOGD(opType.c_str(), "srcR2ndLpStepOut=%d", params.srcR2ndLpStepOut);
    OP_LOGD(opType.c_str(), "srcC1StepIn=%d", params.srcC1StepIn);
    OP_LOGD(opType.c_str(), "srcC1LpUnit=%d", params.srcC1LpUnit);
    OP_LOGD(opType.c_str(), "srcC1LpStepIn=%d", params.srcC1LpStepIn);
    OP_LOGD(opType.c_str(), "srcC1LpStepOut=%d", params.srcC1LpStepOut);
    OP_LOGD(opType.c_str(), "perLineDstCCount=%d", params.perLineDstCCount);
    OP_LOGD(opType.c_str(), "cModC0=%d", params.cModC0);
    tmp_str = "";
    for (auto i : params.ioOrderParams) {
        tmp_str += std::to_string(i);
        tmp_str += ",";
    }
    OP_LOGD(opType.c_str(), "inIdx0Size, inIdx0DstRSize, inIdx0SrcASize, inIdx1Size, inIdx1DstRSize, inIdx1SrcASize, "
                            "inIdx2Size, inIdx2DstRSize, inIdx2SrcASize, outIdx0Size, outIdx0DstRSize, outIdx0DstASize, "
                            "outIdx1Size, outIdx1DstRSize, outIdx1DstASize, outIdx2Size, outIdx2DstRSize, outIdx2DstASize");
    OP_LOGD(opType.c_str(), "ioOrderParams=%s", tmp_str.c_str());
    OP_LOGD(opType.c_str(), "src2DstFlag=%d", params.src2DstFlag);
}

}  // namespace optiling

