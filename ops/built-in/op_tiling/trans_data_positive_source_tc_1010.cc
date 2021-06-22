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
 * \file trans_data_positive_source_tc_1010.cc
 * \brief dynamic TransData op tiling
 */
#include <string>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "trans_data_common.h"
#include "error_log.h"

namespace optiling {

int64_t GetCeilFillB(int64_t uValue, int64_t dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    return uValue;
  }

  resValue = (uValue + dValue - 1) / dValue * dValue;

  return resValue;
}

bool GetMcInfoPositive1010(int64_t& axisDstClSize, int64_t& cLpCnt, int64_t& cLeft, int64_t& dstCrLpCnt,
                           int64_t& vncRowLeft, int64_t& llDstCrLeft, int64_t& coreNum,
                           TransDataMode1010Param& params) {
  int64_t tmpFullLoopCntCr;
  tmpFullLoopCntCr = GetFloorDiv(dstCrLpCnt, coreNum) > 0 ? coreNum : 0;
  
  int64_t reminderLoopCntCr = dstCrLpCnt % coreNum;
  if (reminderLoopCntCr == 0) {
    tmpFullLoopCntCr += coreNum;
  }
  int64_t fullLoopCntCr = tmpFullLoopCntCr + reminderLoopCntCr;

  int64_t tmpFullLoopCntC;
  tmpFullLoopCntC = GetFloorDiv(cLpCnt, coreNum) > 0 ? coreNum : 0;

  int64_t reminderLoopCntC = cLpCnt % coreNum;
  if (reminderLoopCntC == 0) {
    tmpFullLoopCntC += coreNum;
  }
  int64_t fullLoopCntC = tmpFullLoopCntC + reminderLoopCntC;

  int64_t tmpFullLoopCntLeft;
  tmpFullLoopCntLeft = GetFloorDiv(axisDstClSize, coreNum) > 0 ? coreNum : 0;
  int64_t reminderLoopCntLeft = axisDstClSize % coreNum;
  if (reminderLoopCntLeft == 0) {
    tmpFullLoopCntLeft += coreNum;
  }
  int64_t fullLoopCntLeft = tmpFullLoopCntLeft + reminderLoopCntLeft;

  vector<int64_t> loopCntList = {fullLoopCntLeft, fullLoopCntCr, fullLoopCntC};
  if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 0) {
    params.usedCoreCnt = GetCeilDiv(axisDstClSize, GetCeilDiv(axisDstClSize, coreNum));
    params.nlcDstClLpCnt = GetCeilDiv(axisDstClSize, params.usedCoreCnt);
    params.lcDstClLpCnt = axisDstClSize - params.nlcDstClLpCnt * (params.usedCoreCnt - 1);
    params.coreStepIn = params.nlcDstClLpCnt * params.dstClLpStepIn;
    params.coreStepOut = params.nlcDstClLpCnt * params.dstClLpStepOut;
    params.nlcCLpCnt = cLpCnt;
    params.lcCLpCnt = cLpCnt;
    params.nlcCLeft = cLeft;
    params.lcCLeft = cLeft;
    params.nlcDstCrLpCnt = dstCrLpCnt;
    params.lcDstCrLpCnt = dstCrLpCnt;
    params.nlcVncRowLeft = vncRowLeft;
    params.lcVncRowLeft = vncRowLeft;
    params.nlcLastLineCrCnt = llDstCrLeft;
    params.lcLastLineCrCnt = llDstCrLeft;
  } else if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 1){
    params.usedCoreCnt = GetCeilDiv(dstCrLpCnt, GetCeilDiv(dstCrLpCnt, coreNum));
    params.nlcDstCrLpCnt = GetCeilDiv(dstCrLpCnt, params.usedCoreCnt);
    params.lcDstCrLpCnt = dstCrLpCnt - params.nlcDstCrLpCnt * (params.usedCoreCnt - 1);
    params.coreStepIn = params.nlcDstCrLpCnt * params.dstCrLpStepIn;
    params.coreStepOut = params.nlcDstCrLpCnt * params.dstCrLpStepOut;
    params.nlcVncRowLeft = 0;
    params.lcVncRowLeft = vncRowLeft;
    params.nlcLastLineCrCnt = params.plnDstCrSize;
    params.lcLastLineCrCnt = llDstCrLeft;
    params.nlcCLpCnt = cLpCnt;
    params.lcCLpCnt = cLpCnt;
    params.nlcCLeft = cLeft;
    params.lcCLeft = cLeft;
    params.nlcDstClLpCnt = axisDstClSize;
    params.lcDstClLpCnt = axisDstClSize;
  } else {
    params.usedCoreCnt = GetCeilDiv(cLpCnt, GetCeilDiv(cLpCnt, coreNum));
    params.nlcCLpCnt = GetCeilDiv(cLpCnt, params.usedCoreCnt);
    params.lcCLpCnt = cLpCnt - params.nlcCLpCnt * (params.usedCoreCnt - 1);
    params.coreStepIn = params.nlcCLpCnt * params.cLpStepIn;
    params.coreStepOut = params.nlcCLpCnt * params.cLpStepOut;
    params.nlcCLeft = 0;
    params.lcCLeft = cLeft;
    params.nlcDstClLpCnt = axisDstClSize;
    params.lcDstClLpCnt = axisDstClSize;
    params.nlcDstCrLpCnt = dstCrLpCnt;
    params.lcDstCrLpCnt = dstCrLpCnt;
    params.nlcVncRowLeft = vncRowLeft;
    params.lcVncRowLeft = vncRowLeft;
    params.nlcLastLineCrCnt = llDstCrLeft;
    params.lcLastLineCrCnt = llDstCrLeft;
  }
  return true;
}
bool GetCommonParam(int64_t& ubSize, int64_t& blockElemCnt, int64_t& c0Len, int64_t& axisCSize,
                    TransDataMode1010Param& params) {
  int64_t halfUbSize;
  if (c0Len == C0_16) {
    halfUbSize = ubSize / 2;
  } else {
    halfUbSize = ubSize / 4;
  }
  params.vncLineSize = halfUbSize / VNC_LINES / blockElemCnt * blockElemCnt;
  int64_t tempUbOffset = params.vncLineSize * VNC_LINES;
  if (c0Len == C0_16) {
    params.ubOffset = tempUbOffset;
  } else {
    params.ubOffset = tempUbOffset * 2;
  }
  params.cModC0 = axisCSize % c0Len;
  params.c0Size = c0Len;
  return true;
}

bool TillingPositiveMode1010(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                             std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt,
                             int64_t& ubSize, TransDataMode1010Param& params) {
  if ((srcFormat.length() != inShape.size()) || (dstFormat.length() != outShape.size())) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TillingPositiveMode1010 Failed.");
    return false;
  }
  int64_t axisCSize = inShape[inShape.size() - 1];
  int64_t c0Len = outShape[outShape.size() - 1];
  bool ret = GetCommonParam(ubSize, blockElemCnt, c0Len, axisCSize, params);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TillingPositiveMode1010 GetCommonParam Failed.");
    return ret;
  }

  params.tilingMode = 1010;
  params.vncLineSize = params.vncLineSize / c0Len * c0Len;
  // target axis c-left tiling parameters
  int32_t dstAxisPosC = std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str();
  int64_t axisDstClSize = 1;
  for (int32_t i = 0; i < dstAxisPosC; i++) {
    axisDstClSize *= outShape[i];
  }
  char dstClChar = dstFormat[dstAxisPosC - 1];

  params.dstClLpUnit = 1;
  params.dstClLpStepIn = GetShapeSize(inShape, std::strchr(srcFormat.c_str(), dstClChar) - srcFormat.c_str() + 1);
  params.dstClLpStepOut = GetShapeSize(outShape, dstAxisPosC);
  
  // source axis c tiling parameters
  if (axisCSize < params.vncLineSize) {
    params.cLpUnit = axisCSize;
  } else {
    params.cLpUnit = params.vncLineSize;
  }
  params.cLpStepIn = params.cLpUnit;
  int64_t lpC1Cnt = GetCeilDiv(params.cLpUnit, c0Len);
  params.cLpStepOut = lpC1Cnt * GetShapeSize(outShape, dstAxisPosC + 1);
  params.cStepOut = GetShapeSize(outShape, dstAxisPosC + 1);
  int64_t cLpCnt = GetCeilDiv(axisCSize, params.cLpUnit);
  int64_t cLeft = axisCSize % params.cLpUnit;

  // arget axis c-right tiling parameters
  int32_t tmpSrcPos = std::strchr(srcFormat.c_str(), dstFormat[dstFormat.length() - 2]) - srcFormat.c_str();
  int64_t axisDstCrSize = GetShapeSize(inShape, tmpSrcPos) / inShape[inShape.size() - 1];

  params.plnDstCrSize = params.vncLineSize / GetCeilFillB(params.cLpUnit, c0Len);
  params.vncRowSize = VNC_LINES;
  int64_t perVncDstCrCnt = params.plnDstCrSize * params.vncRowSize;
  int64_t dstCrLpCnt = GetCeilDiv(axisDstCrSize, perVncDstCrCnt);
  int64_t dstCrLeft = axisDstCrSize % perVncDstCrCnt;
  int64_t vncRowLeft = GetCeilDiv(dstCrLeft, params.plnDstCrSize);
  int64_t tmpDstCrLeft = dstCrLeft % params.plnDstCrSize;
  int64_t llDstCrLeft;
  if (tmpDstCrLeft > 0) {
    llDstCrLeft = tmpDstCrLeft;
  } else {
    llDstCrLeft = params.plnDstCrSize;
  }

  params.dstCrLpStepIn = inShape[inShape.size() - 1] * perVncDstCrCnt;
  int32_t tmpDstPos = std::strchr(dstFormat.c_str(), srcFormat[srcFormat.length() - 2]) - dstFormat.c_str();
  params.dstCrLpStepOut = GetShapeSize(outShape, tmpDstPos + 1) * perVncDstCrCnt;
  params.dstCrStepIn = GetShapeSize(inShape, -1);

  ret = GetMcInfoPositive1010(axisDstClSize, cLpCnt, cLeft, dstCrLpCnt, vncRowLeft, llDstCrLeft, coreNum, params);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetMcInfoPositive1010 Failed.");
    return ret;
  }
  return true;
}

void SetRunningMode1010Params(const TransDataMode1010Param& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);
  ByteBufferPut(runInfo.tiling_data, runParams.ubOffset);
  ByteBufferPut(runInfo.tiling_data, runParams.usedCoreCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepOut);

  ByteBufferPut(runInfo.tiling_data, runParams.dstClLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstClLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstClLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.vncLineSize);

  ByteBufferPut(runInfo.tiling_data, runParams.plnDstCrSize);
  ByteBufferPut(runInfo.tiling_data, runParams.vncRowSize);
  ByteBufferPut(runInfo.tiling_data, runParams.cLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.cLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.cStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.c0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.cModC0);
  ByteBufferPut(runInfo.tiling_data, runParams.cLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcDstClLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcDstCrLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcVncRowLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcLastLineCrCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcDstClLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcDstCrLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcVncRowLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcLastLineCrCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCLeft);
}

void PrintTilingMode1010Params(const std::string& opType, const TransDataMode1010Param& params) {
  OP_LOGD(opType.c_str(), "tilingMode=%d", params.tilingMode);
  OP_LOGD(opType.c_str(), "ubOffset=%d", params.ubOffset);
  OP_LOGD(opType.c_str(), "usedCoreCnt=%d", params.usedCoreCnt);
  OP_LOGD(opType.c_str(), "coreStepIn=%d", params.coreStepIn);
  OP_LOGD(opType.c_str(), "coreStepOut=%d", params.coreStepOut);

  OP_LOGD(opType.c_str(), "dstClLpStepIn=%d", params.dstClLpStepIn);
  OP_LOGD(opType.c_str(), "dstClLpStepOut=%d", params.dstClLpStepOut);
  OP_LOGD(opType.c_str(), "dstClLpUnit=%d", params.dstClLpUnit);
  OP_LOGD(opType.c_str(), "dstCrLpStepIn=%d", params.dstCrLpStepIn);
  OP_LOGD(opType.c_str(), "dstCrLpStepOut=%d", params.dstCrLpStepOut);
  OP_LOGD(opType.c_str(), "dstCrStepIn=%d", params.dstCrStepIn);
  OP_LOGD(opType.c_str(), "vncLineSize=%d", params.vncLineSize);


  OP_LOGD(opType.c_str(), "plnDstCrSize=%d", params.plnDstCrSize);
  OP_LOGD(opType.c_str(), "vncRowSize=%d", params.vncRowSize);
  OP_LOGD(opType.c_str(), "cLpStepIn=%d", params.cLpStepIn);
  OP_LOGD(opType.c_str(), "cLpStepOut=%d", params.cLpStepOut);
  OP_LOGD(opType.c_str(), "cStepOut=%d", params.cStepOut);
  OP_LOGD(opType.c_str(), "c0Size=%d", params.c0Size);
  OP_LOGD(opType.c_str(), "cModC0=%d", params.cModC0);
  OP_LOGD(opType.c_str(), "cLpUnit=%d", params.cLpUnit);
  OP_LOGD(opType.c_str(), "nlcDstClLpCnt=%d", params.nlcDstClLpCnt);
  OP_LOGD(opType.c_str(), "nlcDstCrLpCnt=%d", params.nlcDstCrLpCnt);
  OP_LOGD(opType.c_str(), "nlcVncRowLeft=%d", params.nlcVncRowLeft);
  OP_LOGD(opType.c_str(), "nlcLastLineCrCnt=%d", params.nlcLastLineCrCnt);

  OP_LOGD(opType.c_str(), "nlcCLpCnt=%d", params.nlcCLpCnt);
  OP_LOGD(opType.c_str(), "nlcCLeft=%d", params.nlcCLeft);
  OP_LOGD(opType.c_str(), "lcDstClLpCnt=%d", params.lcDstClLpCnt);
  OP_LOGD(opType.c_str(), "lcDstCrLpCnt=%d", params.lcDstCrLpCnt);
  OP_LOGD(opType.c_str(), "lcVncRowLeft=%d", params.lcVncRowLeft);
  OP_LOGD(opType.c_str(), "lcLastLineCrCnt=%d", params.lcLastLineCrCnt);
  OP_LOGD(opType.c_str(), "lcCLpCnt=%d", params.lcCLpCnt);
  OP_LOGD(opType.c_str(), "lcCLeft=%d", params.lcCLeft);
}

}  // namespace optiling
