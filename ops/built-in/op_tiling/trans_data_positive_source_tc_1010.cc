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

bool GetMcInfoPositive1010(int64_t& dstClLpCnt, int64_t& vncRowClLeft, int64_t& llDstClLeft, int64_t& cLpCnt,
                           int64_t& cLeft, int64_t& dstCrLpCnt, int64_t& vncRowLeft, int64_t& llDstCrLeft,
                           int64_t& coreNum, TransDataMode1010Param& params) {
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
  tmpFullLoopCntLeft = GetFloorDiv(dstClLpCnt, coreNum) > 0 ? coreNum : 0;
  int64_t reminderLoopCntLeft = dstClLpCnt % coreNum;
  if (reminderLoopCntLeft == 0) {
    tmpFullLoopCntLeft += coreNum;
  }
  int64_t fullLoopCntLeft = tmpFullLoopCntLeft + reminderLoopCntLeft;

  vector<int64_t> loopCntList = {fullLoopCntLeft, fullLoopCntCr, fullLoopCntC};
  if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 0) {
    params.usedCoreCnt = GetCeilDiv(dstClLpCnt, GetCeilDiv(dstClLpCnt, coreNum));
    params.nlcDstClLpCnt = GetCeilDiv(dstClLpCnt, params.usedCoreCnt);
    params.lcDstClLpCnt = dstClLpCnt - params.nlcDstClLpCnt * (params.usedCoreCnt - 1);
    params.coreStepIn = params.nlcDstClLpCnt * params.dstClLpStepIn;
    params.coreStepOut = params.nlcDstClLpCnt * params.dstClLpStepOut;
    params.nlcVncRowClLeft = 0;
    params.lcVncRowClLeft = vncRowClLeft;
    params.nlcLastLineClCnt = llDstClLeft;
    params.lcLastLineClCnt = llDstClLeft;
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
    params.nlcDstClLpCnt = dstClLpCnt;
    params.lcDstClLpCnt = dstClLpCnt;
    params.nlcVncRowClLeft = vncRowClLeft;
    params.lcVncRowClLeft = vncRowClLeft;
    params.nlcLastLineClCnt = llDstClLeft;
    params.lcLastLineClCnt = llDstClLeft;
  } else {
    params.usedCoreCnt = GetCeilDiv(cLpCnt, GetCeilDiv(cLpCnt, coreNum));
    params.nlcCLpCnt = GetCeilDiv(cLpCnt, params.usedCoreCnt);
    params.lcCLpCnt = cLpCnt - params.nlcCLpCnt * (params.usedCoreCnt - 1);
    params.coreStepIn = params.nlcCLpCnt * params.cLpStepIn;
    params.coreStepOut = params.nlcCLpCnt * params.cLpStepOut;
    params.nlcCLeft = 0;
    params.lcCLeft = cLeft;
    params.nlcDstClLpCnt = dstClLpCnt;
    params.lcDstClLpCnt = dstClLpCnt;
    params.nlcVncRowClLeft = vncRowClLeft;
    params.lcVncRowClLeft = vncRowClLeft;
    params.nlcLastLineClCnt = llDstClLeft;
    params.lcLastLineClCnt = llDstClLeft;
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

  // source axis c tiling parameters
  int32_t dstAxisPosC = std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str();
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

  // target axis c-right tiling parameters
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

  // target axis c-left tiling parameters
  int64_t axisDstClSize = 1;
  int64_t perVncDstClCnt = 1;
  int64_t dstClLpCnt = 1;
  int64_t dstClLeft = 0;
  int64_t vncRowClLeft = 0;
  int64_t tmpDstClLeft = 0;
  int64_t llDstClLeft = 0;
  for (int32_t i = 0; i < dstAxisPosC; i++) {
    axisDstClSize *= outShape[i];
  }
  char dstClChar = dstFormat[dstAxisPosC - 1];

  if (dstCrLpCnt == 1 && params.cLpUnit == axisCSize && vncRowLeft <= GetFloorDiv(VNC_LINES, 2)) {
    // nc is less than vnchwconv col size
    if (vncRowLeft == 1) {
      params.ncLeVCol = 1;
      params.plnDstClSize = GetFloorDiv(params.plnDstCrSize, axisDstCrSize);
    } else {
      params.ncLeVCol = 2;
      params.plnDstClSize = 1;
      // adjust c-right parameters
      dstCrLpCnt = GetCeilDiv(axisDstCrSize, params.plnDstCrSize);
      vncRowLeft = axisDstCrSize % params.plnDstCrSize;
      if (vncRowLeft > 0) {
        llDstCrLeft = vncRowLeft;
      } else {
        llDstCrLeft = params.plnDstCrSize;
      }
      params.dstCrLpStepIn = inShape[inShape.size() - 1] * params.plnDstCrSize;
      params.dstCrLpStepOut = GetShapeSize(outShape, tmpDstPos + 1) * params.plnDstCrSize;
    }

    perVncDstClCnt = params.plnDstClSize * params.vncRowSize;
    dstClLpCnt = GetCeilDiv(axisDstClSize, perVncDstClCnt);
    // adjust c-left parameters
    int64_t fourInCoreCnt = 4;
    int64_t plnClGate = 64;
    if ((dstClLpCnt < GetFloorDiv(coreNum, fourInCoreCnt)) && (params.plnDstClSize > plnClGate)) {
      params.plnDstClSize = GetFloorDiv(params.plnDstClSize, plnClGate);
      perVncDstClCnt = params.plnDstClSize * params.vncRowSize;
      dstClLpCnt = GetCeilDiv(axisDstClSize, perVncDstClCnt);
    }
    dstClLeft = axisDstClSize % perVncDstClCnt;
    vncRowClLeft = GetCeilDiv(dstClLeft, params.plnDstClSize);
    tmpDstClLeft = dstClLeft % params.plnDstClSize;
    if (tmpDstClLeft > 0) {
      llDstClLeft = tmpDstClLeft;
    } else {
      llDstClLeft = params.plnDstClSize;
    }

  } else {
    params.ncLeVCol = 0;
    params.plnDstClSize = 1;
    dstClLpCnt = axisDstClSize;
    vncRowClLeft = params.plnDstClSize;
    llDstClLeft = params.plnDstClSize;
  }
  params.dstClStepIn = GetShapeSize(inShape, std::strchr(srcFormat.c_str(), dstClChar) - srcFormat.c_str() + 1);
  params.dstClStepOut = GetShapeSize(outShape, dstAxisPosC);
  if (params.ncLeVCol == 0) {
    params.dstClLpStepIn = params.dstClStepIn;
    params.dstClLpStepOut = params.dstClStepOut;
  } else {
    params.dstClLpStepIn = params.dstClStepIn * perVncDstClCnt;
    params.dstClLpStepOut = params.dstClStepOut * perVncDstClCnt;
  }

  ret = GetMcInfoPositive1010(dstClLpCnt, vncRowClLeft, llDstClLeft, cLpCnt, cLeft,
                              dstCrLpCnt, vncRowLeft, llDstCrLeft, coreNum, params);
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
  ByteBufferPut(runInfo.tiling_data, runParams.dstClStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstClStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.ncLeVCol);
  ByteBufferPut(runInfo.tiling_data, runParams.vncLineSize);

  ByteBufferPut(runInfo.tiling_data, runParams.plnDstClSize);
  ByteBufferPut(runInfo.tiling_data, runParams.plnDstCrSize);
  ByteBufferPut(runInfo.tiling_data, runParams.vncRowSize);
  ByteBufferPut(runInfo.tiling_data, runParams.cLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.cLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.cStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.c0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.cModC0);
  ByteBufferPut(runInfo.tiling_data, runParams.cLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcDstClLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcVncRowClLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcLastLineClCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcDstCrLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcVncRowLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcLastLineCrCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcDstClLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcVncRowClLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcLastLineClCnt);
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
  OP_LOGD(opType.c_str(), "dstClStepIn=%d", params.dstClStepIn);
  OP_LOGD(opType.c_str(), "dstClStepOut=%d", params.dstClStepOut);
  OP_LOGD(opType.c_str(), "dstCrLpStepIn=%d", params.dstCrLpStepIn);
  OP_LOGD(opType.c_str(), "dstCrLpStepOut=%d", params.dstCrLpStepOut);
  OP_LOGD(opType.c_str(), "dstCrStepIn=%d", params.dstCrStepIn);
  OP_LOGD(opType.c_str(), "ncLeVCol=%d", params.ncLeVCol);
  OP_LOGD(opType.c_str(), "vncLineSize=%d", params.vncLineSize);

  OP_LOGD(opType.c_str(), "plnDstClSize=%d", params.plnDstClSize);
  OP_LOGD(opType.c_str(), "plnDstCrSize=%d", params.plnDstCrSize);
  OP_LOGD(opType.c_str(), "vncRowSize=%d", params.vncRowSize);
  OP_LOGD(opType.c_str(), "cLpStepIn=%d", params.cLpStepIn);
  OP_LOGD(opType.c_str(), "cLpStepOut=%d", params.cLpStepOut);
  OP_LOGD(opType.c_str(), "cStepOut=%d", params.cStepOut);
  OP_LOGD(opType.c_str(), "c0Size=%d", params.c0Size);
  OP_LOGD(opType.c_str(), "cModC0=%d", params.cModC0);
  OP_LOGD(opType.c_str(), "cLpUnit=%d", params.cLpUnit);

  OP_LOGD(opType.c_str(), "nlcDstClLpCnt=%d", params.nlcDstClLpCnt);
  OP_LOGD(opType.c_str(), "nlcVncRowClLeft=%d", params.nlcVncRowClLeft);
  OP_LOGD(opType.c_str(), "nlcLastLineClCnt=%d", params.nlcLastLineClCnt);
  OP_LOGD(opType.c_str(), "nlcDstCrLpCnt=%d", params.nlcDstCrLpCnt);
  OP_LOGD(opType.c_str(), "nlcVncRowLeft=%d", params.nlcVncRowLeft);
  OP_LOGD(opType.c_str(), "nlcLastLineCrCnt=%d", params.nlcLastLineCrCnt);
  OP_LOGD(opType.c_str(), "nlcCLpCnt=%d", params.nlcCLpCnt);
  OP_LOGD(opType.c_str(), "nlcCLeft=%d", params.nlcCLeft);

  OP_LOGD(opType.c_str(), "lcDstClLpCnt=%d", params.lcDstClLpCnt);
  OP_LOGD(opType.c_str(), "lcVncRowClLeft=%d", params.lcVncRowClLeft);
  OP_LOGD(opType.c_str(), "lcLastLineClCnt=%d", params.lcLastLineClCnt);
  OP_LOGD(opType.c_str(), "lcDstCrLpCnt=%d", params.lcDstCrLpCnt);
  OP_LOGD(opType.c_str(), "lcVncRowLeft=%d", params.lcVncRowLeft);
  OP_LOGD(opType.c_str(), "lcLastLineCrCnt=%d", params.lcLastLineCrCnt);
  OP_LOGD(opType.c_str(), "lcCLpCnt=%d", params.lcCLpCnt);
  OP_LOGD(opType.c_str(), "lcCLeft=%d", params.lcCLeft);
}

}  // namespace optiling
