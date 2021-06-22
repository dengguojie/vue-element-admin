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
 * \file trans_data_negative_target_ntc.cc
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

const int32_t FRAME_LEVEL = 2;

int64_t GetCeilFillA(int64_t uValue, int64_t dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    return uValue;
  }

  resValue = (uValue + dValue - 1) / dValue * dValue;

  return resValue;
}

bool GetMcInfoNegative200(int64_t& dstCrLpCnt, int64_t& dstCrLeft, int64_t& srcCLpCnt, int64_t& srcCLeft,
                          int64_t& dstClLpCnt, int64_t& dstClLeft, int64_t& coreNum, TransDataNtc200Param& params) {
  int64_t tmpFullLoopCntCr;
  if (GetFloorDiv(dstCrLpCnt, coreNum) > 0) {
    tmpFullLoopCntCr = coreNum;
  } else {
    tmpFullLoopCntCr = 0;
  }
  int64_t reminderLoopCntCr = dstCrLpCnt % coreNum;
  if (reminderLoopCntCr == 0 && dstCrLeft > params.dstCrLpUnit / 2) {
    tmpFullLoopCntCr += coreNum;
  }
  int64_t fullLoopCntCr = tmpFullLoopCntCr + reminderLoopCntCr;

  int64_t tmpFullLoopCntC;
  if (GetFloorDiv(srcCLpCnt, coreNum) > 0) {
    tmpFullLoopCntC = coreNum;
  } else {
    tmpFullLoopCntC = 0;
  }
  int64_t reminderLoopCntC = srcCLpCnt % coreNum;
  if (reminderLoopCntC == 0) {
    tmpFullLoopCntC += coreNum;
  }
  int64_t fullLoopCntC = tmpFullLoopCntC + reminderLoopCntC;

  int64_t tmpFullLoopCntCl;
  if (GetFloorDiv(dstClLpCnt, coreNum) > 0) {
    tmpFullLoopCntCl = coreNum;
  } else {
    tmpFullLoopCntCl = 0;
  }
  int64_t reminderLoopCntCl = dstClLpCnt % coreNum;
  if (reminderLoopCntCl == 0) {
    tmpFullLoopCntCl += coreNum;
  }
  int64_t fullLoopCntCl = tmpFullLoopCntCl + reminderLoopCntCl;
  vector<int64_t> loopCntList = {fullLoopCntCl, fullLoopCntC, fullLoopCntCr};

  if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 0) {
    params.mcPos = 0;
    params.isMcCl = 1;
    params.isMcCr = 0;
    params.usedCoreCnt = GetCeilDiv(dstClLpCnt, GetCeilDiv(dstClLpCnt, coreNum));
    params.nlcClLpCnt = GetCeilDiv(dstClLpCnt, params.usedCoreCnt);
    params.lcClLpCnt = dstClLpCnt - params.nlcClLpCnt * (params.usedCoreCnt - 1);
    params.coreStepIn = params.nlcClLpCnt * params.dstClLpStepIn;
    params.coreStepOut = params.nlcClLpCnt * params.dstClLpStepOut;
    params.nlcClLeft = 0;
    params.lcClLeft = dstClLeft;
    params.nlcCLpCnt = srcCLpCnt;
    params.lcCLpCnt = srcCLpCnt;
    params.nlcCLeft = srcCLeft;
    params.lcCLeft = srcCLeft;
    params.nlcCrLpCnt = dstCrLpCnt;
    params.lcCrLpCnt = dstCrLpCnt;
    params.nlcCrLeft = dstCrLeft;
    params.lcCrLeft = dstCrLeft;
  } else if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 1) {
    params.mcPos = 1;
    params.isMcCl = 0;
    params.isMcCr = 0;
    params.usedCoreCnt = GetCeilDiv(srcCLpCnt, GetCeilDiv(srcCLpCnt, coreNum));
    params.nlcCLpCnt = GetCeilDiv(srcCLpCnt, params.usedCoreCnt);
    params.lcCLpCnt = srcCLpCnt - params.nlcCLpCnt * (params.usedCoreCnt - 1);
    params.nlcCLeft = 0;
    params.lcCLeft = srcCLeft;
    params.coreStepIn = params.nlcCLpCnt * params.srcCLpStepIn;
    params.coreStepOut = params.nlcCLpCnt * params.srcCLpStepOut;
    params.nlcCrLpCnt = dstCrLpCnt;
    params.lcCrLpCnt = dstCrLpCnt;
    params.nlcCrLeft = dstCrLeft;
    params.lcCrLeft = dstCrLeft;
    params.nlcClLpCnt = dstClLpCnt;
    params.lcClLpCnt = dstClLpCnt;

    params.nlcClLeft = dstClLeft;
    params.lcClLeft = dstClLeft;
  } else {
    params.mcPos = 2;
    params.isMcCl = 0;
    params.isMcCr = 1;
    params.usedCoreCnt = GetCeilDiv(dstCrLpCnt, GetCeilDiv(dstCrLpCnt, coreNum));
    params.nlcCrLpCnt = GetCeilDiv(dstCrLpCnt, params.usedCoreCnt);
    params.lcCrLpCnt = dstCrLpCnt - params.nlcCrLpCnt * (params.usedCoreCnt - 1);
    params.nlcCrLeft = 0;
    params.lcCrLeft = dstCrLeft;
    params.coreStepIn = params.nlcCrLpCnt * params.dstCrLpStepIn;;
    params.coreStepOut = params.nlcCrLpCnt * params.dstCrLpStepOut;
    params.nlcCLpCnt = srcCLpCnt;
    params.lcCLpCnt = srcCLpCnt;
    params.nlcCLeft = srcCLeft;
    params.lcCLeft = srcCLeft;
    params.nlcClLpCnt = dstClLpCnt;
    params.lcClLpCnt = dstClLpCnt;
    params.nlcClLeft = dstClLeft;
    params.lcClLeft = dstClLeft;
  }
  return true;
}

bool TilingNegativeNtc200(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt, std::string& dtype,
                            int64_t& ubSize, TransDataNtc200Param& params) {
  if (srcFormat.length() < 2 || dstFormat.length() < 1) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TilingNegativeNtc200 Failed.");
    return false;
  }

  int64_t c0Len = inShape[inShape.size() - 1];
  params.c0Len = c0Len;

  if (srcFormat[srcFormat.length() - 2] == dstFormat[dstFormat.length() - 1]) {
    params.srcR2ndDstR1stSame = 1;
  } else {
    params.srcR2ndDstR1stSame = 0;
  }
  int64_t halfUbSize = ubSize / 2 / blockElemCnt * blockElemCnt;
  int64_t vncColSize = halfUbSize / VNC_LINES / blockElemCnt * blockElemCnt;
  params.ubOffset = halfUbSize;

  // dst axis C-RIGHT tiling parameters
  params.dstCrDims = 2;
  int32_t srcAxisPosC = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
  int32_t dstAxisPosC = std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str();
  int64_t axisDstCrSize = GetShapeSize(outShape, dstAxisPosC + 1);
  // once vnchwconv flow
  int64_t tmpDstCrLpUnit;
  int64_t crGate = 3 * c0Len;
  if ((dtype == "float16" || ((c0Len == C0_32) && (dtype == "int8" || dtype == "uint8"))) && (axisDstCrSize >= crGate)) {
    tmpDstCrLpUnit = halfUbSize / c0Len / blockElemCnt * blockElemCnt;
  } else {
    // twice vnchwconv flow
    if (dtype == "int8" || dtype == "uint8") {
      tmpDstCrLpUnit = vncColSize / 2 / c0Len / blockElemCnt * blockElemCnt;
    } else {
      tmpDstCrLpUnit = vncColSize / c0Len / blockElemCnt * blockElemCnt;
    }
  }

  if (axisDstCrSize > tmpDstCrLpUnit) {
    params.dstCrLpUnit = tmpDstCrLpUnit;
  } else {
    params.dstCrLpUnit = axisDstCrSize;
  }
  int64_t dstCrLpCnt = GetCeilDiv(axisDstCrSize, params.dstCrLpUnit);
  int64_t dstCrLeft = axisDstCrSize % params.dstCrLpUnit;
  string tmpDstCrFormat = dstFormat.substr(dstAxisPosC + 1, dstFormat.length() - dstAxisPosC - 1);
  vector<int64_t> tmpDstCrShape;
  for (size_t i = dstAxisPosC + 1; i < outShape.size(); i++) {
    tmpDstCrShape.push_back(outShape[i]);
  }
  tmpDstCrShape.push_back(1);
  reverse(tmpDstCrFormat.begin(), tmpDstCrFormat.end());
  for (size_t i = 0; i < tmpDstCrFormat.length(); i++) {
    char chr = tmpDstCrFormat[i];
    int32_t srcChrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    int32_t dstChrPos = std::strchr(dstFormat.c_str(), chr) - dstFormat.c_str();
    if (i == 0) {
      params.crInIdx0Size = outShape[dstChrPos];
      params.crInIdx0DstRsize = GetShapeSize(tmpDstCrShape, -1 - i);
      params.crInIdx0SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
    } else if (i == 1) {
      params.crInIdx1Size = outShape[dstChrPos];
      params.crInIdx1DstRsize = GetShapeSize(tmpDstCrShape, -1 - i);
      params.crInIdx1SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
    }
  }
  // suppose there are 2 axises
  int32_t padAxisCnt = FRAME_LEVEL - tmpDstCrFormat.length();
  if (padAxisCnt != 0) {
    params.dstCrDims = 1;
    if (tmpDstCrFormat.length() == 0) {
      params.crInIdx0Size = 1;
      params.crInIdx0DstRsize = 1;
      params.crInIdx0SrcAsize = 0;
      params.crInIdx1Size = 1;
      params.crInIdx1DstRsize = 1;
      params.crInIdx1SrcAsize = 0;
    } else if (tmpDstCrFormat.length() == 1) {
      params.crInIdx1Size = 1;
      params.crInIdx1DstRsize = 1;
      params.crInIdx1SrcAsize = 0;
    }
  }
  params.dstCrStepOut = 1;
  params.dstCrLpStepOut = params.dstCrLpUnit * params.dstCrStepOut;
  if (params.dstCrDims == 2) {
    params.dstCrStepIn = 0;
  } else {
    char dstCrChr = dstFormat[dstFormat.length() - 1];
    int32_t dstCrInSrc = std::strchr(srcFormat.c_str(), dstCrChr) - srcFormat.c_str();
    params.dstCrStepIn = GetShapeSize(inShape, dstCrInSrc + 1);
  }
  params.dstCrLpStepIn = params.dstCrLpUnit * params.dstCrStepIn;
  // axis C tiling parameters
  int64_t axisSrcCSize = inShape[srcAxisPosC];
  int64_t axisDstCSize = outShape[dstAxisPosC];
  int64_t tmpSrcCLpUnit;
  if (dstCrLpCnt > 1 || axisSrcCSize == 1) {
    params.srcCLpUnit = 1;
  } else {
    tmpSrcCLpUnit = tmpDstCrLpUnit / GetCeilFillA(axisDstCrSize, blockElemCnt);
    if (axisSrcCSize > tmpSrcCLpUnit) {
      params.srcCLpUnit = tmpSrcCLpUnit;
    } else {
      params.srcCLpUnit = axisSrcCSize;
    }
  }
  int64_t srcCLpCnt = GetCeilDiv(axisSrcCSize, params.srcCLpUnit);
  int64_t srcCLeft = axisSrcCSize % params.srcCLpUnit;
  params.srcCStepIn = GetShapeSize(inShape, srcAxisPosC + 1);
  params.srcCStepOut = GetShapeSize(outShape, dstAxisPosC + 1);
  params.srcCLpStepIn = params.srcCLpUnit * params.srcCStepIn;
  params.srcCLpStepOut = params.srcCLpUnit * c0Len * params.srcCStepOut;
  params.cModC0 = axisDstCSize % c0Len;
  params.dstCSize = axisDstCSize;

  // dst axis C-LEFT tiling parameters
  params.dstClDims = 2;
  int64_t axisDstClSize = 1;
  for (int32_t i = 0; i < dstAxisPosC; i++) {
    axisDstClSize *= outShape[i];
  }
  int64_t srcCDstCrSize = axisSrcCSize * axisDstCrSize;
  if ((dtype == "float16" || ((c0Len == C0_32) && (dtype == "int8" || dtype == "uint8"))) && (axisDstCrSize >= crGate)) {
    params.tilingMode = 2001;
    int64_t tmpDstClLpUnit = halfUbSize / (params.srcCLpUnit * GetCeilFillA(params.dstCrLpUnit, blockElemCnt) * c0Len);
    if (axisDstClSize > tmpDstClLpUnit) {
      params.dstClLpUnit = tmpDstClLpUnit;
    } else {
      params.dstClLpUnit = axisDstClSize;
    }
  } else if (axisSrcCSize > params.srcCLpUnit || srcCDstCrSize > tmpDstCrLpUnit) {
    // c and c-right cannot move out one time or one vnc line cannot save c0_size c * c-right
    params.tilingMode = 2002;
    if (axisDstClSize > VNC_LINES) {
      params.dstClLpUnit = VNC_LINES;
    } else {
      params.dstClLpUnit = axisDstClSize;
    }
  } else {
    params.tilingMode = 2003;
    int64_t supposedLpUnit = 4 * blockElemCnt;
    int64_t tmpDstClLpUnit = tmpDstCrLpUnit / (params.srcCLpUnit * params.dstCrLpUnit);
    if (tmpDstClLpUnit < supposedLpUnit) {
      params.dstClLpUnit = tmpDstClLpUnit;
    } else {
      params.dstClLpUnit = supposedLpUnit;
    }
  }
  int64_t dstClLpCnt = GetCeilDiv(axisDstClSize, params.dstClLpUnit);
  int64_t dstClLeft = axisDstClSize % params.dstClLpUnit;
  // for tiling mode 2003
  params.leftClCCrSize = dstClLeft * axisDstCSize * axisDstCrSize;
  string tmpDstClFormat = dstFormat.substr(0, dstAxisPosC);
  vector<int64_t> tmpCLeftShape;
  for (int32_t i = 0; i < dstAxisPosC; i++) {
    tmpCLeftShape.push_back(outShape[i]);
  }
  tmpCLeftShape.push_back(1);

  reverse(tmpDstClFormat.begin(), tmpDstClFormat.end());
  for (size_t i = 0; i < tmpDstClFormat.length(); i++) {
    char chr = tmpDstClFormat[i];
    int32_t srcChrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    int32_t dstChrPos = std::strchr(dstFormat.c_str(), chr) - dstFormat.c_str();
    if (i == 0) {
      params.clInIdx0Size = outShape[dstChrPos];
      params.clInIdx0DstRsize = GetShapeSize(tmpCLeftShape, -1 - i);
      params.clInIdx0SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
    } else if (i == 1) {
      params.clInIdx1Size = outShape[dstChrPos];
      params.clInIdx1DstRsize = GetShapeSize(tmpCLeftShape, -1 - i);
      params.clInIdx1SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
    }
  }
  // suppose there are 2 axises
  padAxisCnt = FRAME_LEVEL - tmpDstClFormat.length();
  if (padAxisCnt != 0) {
    params.dstClDims = 1;
    if (tmpDstClFormat.length() == 0) {
      params.clInIdx0Size = 1;
      params.clInIdx0DstRsize = 1;
      params.clInIdx0SrcAsize = 0;
      params.clInIdx1Size = 1;
      params.clInIdx1DstRsize = 1;
      params.clInIdx1SrcAsize = 0;
    } else if (tmpDstClFormat.length() == 1) {
      params.clInIdx1Size = 1;
      params.clInIdx1DstRsize = 1;
      params.clInIdx1SrcAsize = 0;
    }
  }

  params.dstClStepOut = GetShapeSize(outShape, dstAxisPosC);
  params.dstClLpStepOut = params.dstClLpUnit * params.dstClStepOut;
  if (params.dstClDims == 2) {
    params.dstClStepIn = 0;
  } else {
    char dstClChr = dstFormat[0];
    params.dstClStepIn = GetShapeSize(inShape, std::strchr(srcFormat.c_str(), dstClChr) - srcFormat.c_str() + 1);
  }
  params.dstClLpStepIn = params.dstClLpUnit * params.dstClStepIn;

  bool ret = GetMcInfoNegative200(dstCrLpCnt, dstCrLeft, srcCLpCnt, srcCLeft, dstClLpCnt, dstClLeft, coreNum, params);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetMcInfoNegative200 Failed.");
    return ret;
  }
  return true;
}

void SetRunningNtc200Params(const TransDataNtc200Param& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);
  ByteBufferPut(runInfo.tiling_data, runParams.ubOffset);
  ByteBufferPut(runInfo.tiling_data, runParams.mcPos);
  ByteBufferPut(runInfo.tiling_data, runParams.usedCoreCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.c0Len);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepOut);

  ByteBufferPut(runInfo.tiling_data, runParams.nlcCrLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcClLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCrLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcClLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCrLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcClLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCrLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcClLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.dstClLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCSize);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstClStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstClStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstClLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstClLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.cModC0);
  ByteBufferPut(runInfo.tiling_data, runParams.dstCrDims);
  ByteBufferPut(runInfo.tiling_data, runParams.dstClDims);
  ByteBufferPut(runInfo.tiling_data, runParams.isMcCr);
  ByteBufferPut(runInfo.tiling_data, runParams.isMcCl);
  ByteBufferPut(runInfo.tiling_data, runParams.srcR2ndDstR1stSame);
  ByteBufferPut(runInfo.tiling_data, runParams.leftClCCrSize);

  ByteBufferPut(runInfo.tiling_data, runParams.clInIdx0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.clInIdx0DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.clInIdx0SrcAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.clInIdx1Size);
  ByteBufferPut(runInfo.tiling_data, runParams.clInIdx1DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.clInIdx1SrcAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx0DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx0SrcAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx1Size);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx1DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx1SrcAsize);
}

void PrintTilingModeNtc200Params(const std::string& opType, const TransDataNtc200Param& params) {
  OP_LOGD(opType.c_str(), "tilingMode=%d", params.tilingMode);
  OP_LOGD(opType.c_str(), "ubOffset=%d", params.ubOffset);
  OP_LOGD(opType.c_str(), "mcPos=%d", params.mcPos);
  OP_LOGD(opType.c_str(), "usedCoreCnt=%d", params.usedCoreCnt);
  OP_LOGD(opType.c_str(), "c0Len=%d", params.c0Len);
  OP_LOGD(opType.c_str(), "coreStepIn=%d", params.coreStepIn);
  OP_LOGD(opType.c_str(), "coreStepOut=%d", params.coreStepOut);

  OP_LOGD(opType.c_str(), "nlcCrLpCnt=%d", params.nlcCrLpCnt);
  OP_LOGD(opType.c_str(), "nlcCLpCnt=%d", params.nlcCLpCnt);
  OP_LOGD(opType.c_str(), "nlcClLpCnt=%d", params.nlcClLpCnt);
  OP_LOGD(opType.c_str(), "nlcCrLeft=%d", params.nlcCrLeft);
  OP_LOGD(opType.c_str(), "nlcCLeft=%d", params.nlcCLeft);
  OP_LOGD(opType.c_str(), "nlcClLeft=%d", params.nlcClLeft);
  OP_LOGD(opType.c_str(), "lcCrLpCnt=%d", params.lcCrLpCnt);
  OP_LOGD(opType.c_str(), "lcCLpCnt=%d", params.lcCLpCnt);
  OP_LOGD(opType.c_str(), "lcClLpCnt=%d", params.lcClLpCnt);
  OP_LOGD(opType.c_str(), "lcCrLeft=%d", params.lcCrLeft);
  OP_LOGD(opType.c_str(), "lcCLeft=%d", params.lcCLeft);
  OP_LOGD(opType.c_str(), "lcClLeft=%d", params.lcClLeft);
  OP_LOGD(opType.c_str(), "dstCrLpUnit=%d", params.dstCrLpUnit);
  OP_LOGD(opType.c_str(), "srcCLpUnit=%d", params.srcCLpUnit);
  OP_LOGD(opType.c_str(), "dstClLpUnit=%d", params.dstClLpUnit);
  OP_LOGD(opType.c_str(), "dstCrStepIn=%d", params.dstCrStepIn);
  OP_LOGD(opType.c_str(), "dstCrStepOut=%d", params.dstCrStepOut);
  OP_LOGD(opType.c_str(), "dstCrLpStepIn=%d", params.dstCrLpStepIn);
  OP_LOGD(opType.c_str(), "dstCrLpStepOut=%d", params.dstCrLpStepOut);
  OP_LOGD(opType.c_str(), "dstCSize=%d", params.dstCSize);
  OP_LOGD(opType.c_str(), "srcCStepIn=%d", params.srcCStepIn);
  OP_LOGD(opType.c_str(), "srcCStepOut=%d", params.srcCStepOut);
  OP_LOGD(opType.c_str(), "srcCLpStepIn=%d", params.srcCLpStepIn);
  OP_LOGD(opType.c_str(), "srcCLpStepOut=%d", params.srcCLpStepOut);
  OP_LOGD(opType.c_str(), "dstClStepIn=%d", params.dstClStepIn);
  OP_LOGD(opType.c_str(), "dstClStepOut=%d", params.dstClStepOut);
  OP_LOGD(opType.c_str(), "dstClLpStepIn=%d", params.dstClLpStepIn);
  OP_LOGD(opType.c_str(), "dstClLpStepOut=%d", params.dstClLpStepOut);
  OP_LOGD(opType.c_str(), "cModC0=%d", params.cModC0);
  OP_LOGD(opType.c_str(), "dstCrDims=%d", params.dstCrDims);
  OP_LOGD(opType.c_str(), "dstClDims=%d", params.dstClDims);
  OP_LOGD(opType.c_str(), "isMcCr=%d", params.isMcCr);
  OP_LOGD(opType.c_str(), "isMcCl=%d", params.isMcCl);

  OP_LOGD(opType.c_str(), "srcR2ndDstR1stSame=%d", params.srcR2ndDstR1stSame);
  OP_LOGD(opType.c_str(), "leftClCCrSize=%d", params.leftClCCrSize);
  OP_LOGD(opType.c_str(), "clInIdx0Size=%d", params.clInIdx0Size);
  OP_LOGD(opType.c_str(), "clInIdx0DstRsize=%d", params.clInIdx0DstRsize);
  OP_LOGD(opType.c_str(), "clInIdx0SrcAsize=%d", params.clInIdx0SrcAsize);
  OP_LOGD(opType.c_str(), "clInIdx1Size=%d", params.clInIdx1Size);
  OP_LOGD(opType.c_str(), "clInIdx1DstRsize=%d", params.clInIdx1DstRsize);
  OP_LOGD(opType.c_str(), "clInIdx1SrcAsize=%d", params.clInIdx1SrcAsize);

  OP_LOGD(opType.c_str(), "crInIdx0Size=%d", params.crInIdx0Size);
  OP_LOGD(opType.c_str(), "crInIdx0DstRsize=%d", params.crInIdx0DstRsize);
  OP_LOGD(opType.c_str(), "crInIdx0SrcAsize=%d", params.crInIdx0SrcAsize);
  OP_LOGD(opType.c_str(), "crInIdx1Size=%d", params.crInIdx1Size);
  OP_LOGD(opType.c_str(), "crInIdx1DstRsize=%d", params.crInIdx1DstRsize);
  OP_LOGD(opType.c_str(), "crInIdx1SrcAsize=%d", params.crInIdx1SrcAsize);
}

}  // namespace optiling
