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
 * \file trans_data_positive_source_tc_1011.cc
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

int64_t GetCeilFillC(int64_t uValue, int64_t dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    return uValue;
  }

  resValue = (uValue + dValue - 1) / dValue * dValue;

  return resValue;
}

bool GetMcInfoPositive1011(int64_t& axisDstR2ndLpCnt, int64_t& axisDstR2ndLeft, int64_t& cLpCnt, int64_t& cLeft,
                           int64_t& axisSrcClLpCnt, int64_t& axisSrcClLeft, int64_t& coreNum,
                           TransDataMode1011Param& params) {
  int64_t tmpFullLoopCntR2nd;
  tmpFullLoopCntR2nd = GetFloorDiv(axisDstR2ndLpCnt, coreNum) > 0 ? coreNum : 0;
  
  int64_t reminderLoopCntR2nd = axisDstR2ndLpCnt % coreNum;
  if (reminderLoopCntR2nd == 0) {
    tmpFullLoopCntR2nd += coreNum;
  }
  int64_t fullLoopCntR2nd = tmpFullLoopCntR2nd + reminderLoopCntR2nd;

  int64_t tmpFullLoopCntC;
  tmpFullLoopCntC = GetFloorDiv(cLpCnt, coreNum) > 0 ? coreNum : 0;
  int64_t reminderLoopCntC = cLpCnt % coreNum;
  if (reminderLoopCntC == 0) {
    tmpFullLoopCntC += coreNum;
  }
  int64_t fullLoopCntC = tmpFullLoopCntC + reminderLoopCntC;

  int64_t tmpFullLoopCntLeft;
  tmpFullLoopCntLeft = GetFloorDiv(axisSrcClLpCnt, coreNum) > 0 ? coreNum : 0;
  int64_t reminderLoopCntLeft = axisSrcClLpCnt % coreNum;
  if (reminderLoopCntLeft == 0) {
    tmpFullLoopCntLeft += coreNum;
  }
  int64_t fullLoopCntLeft = tmpFullLoopCntLeft + reminderLoopCntLeft;

  vector<int64_t> loopCntList = {fullLoopCntR2nd, fullLoopCntLeft, fullLoopCntC};
  if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 0) {
    params.mcOnCl = 0;
    params.usedCoreCnt = GetCeilDiv(axisDstR2ndLpCnt, GetCeilDiv(axisDstR2ndLpCnt, coreNum));
    params.nlcDstR2ndLpCnt = GetCeilDiv(axisDstR2ndLpCnt, params.usedCoreCnt);
    params.lcDstR2ndLpCnt = axisDstR2ndLpCnt - params.nlcDstR2ndLpCnt * (params.usedCoreCnt - 1);
    params.coreStepIn = params.nlcDstR2ndLpCnt * params.dstR2ndLpStepIn;
    params.coreStepOut = params.nlcDstR2ndLpCnt * params.dstR2ndLpStepOut;
    params.nlcDstR2ndLeft = 0;
    params.lcDstR2ndLeft = axisDstR2ndLeft;
    params.nlcCLpCnt = cLpCnt;
    params.lcCLpCnt = cLpCnt;
    params.nlcCLeft = cLeft;
    params.lcCLeft = cLeft;
    params.nlcSrcClLpCnt = axisSrcClLpCnt;
    params.lcSrcClLpCnt = axisSrcClLpCnt;
    params.nlcSrcClLeft = axisSrcClLeft;
    params.lcSrcClLeft = axisSrcClLeft;
  } else if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 1){
    params.mcOnCl = 1;
    params.usedCoreCnt = GetCeilDiv(axisSrcClLpCnt, GetCeilDiv(axisSrcClLpCnt, coreNum));
    params.nlcSrcClLpCnt = GetCeilDiv(axisSrcClLpCnt, params.usedCoreCnt);
    params.lcSrcClLpCnt = axisSrcClLpCnt - params.nlcSrcClLpCnt * (params.usedCoreCnt - 1);
    params.coreStepIn = params.nlcSrcClLpCnt * params.srcClLpStepIn;
    params.coreStepOut = params.nlcSrcClLpCnt * params.srcClLpStepOut;
    params.nlcSrcClLeft = 0;
    params.lcSrcClLeft = axisSrcClLeft;
    params.nlcCLpCnt = cLpCnt;
    params.lcCLpCnt = cLpCnt;
    params.nlcCLeft = cLeft;
    params.lcCLeft = cLeft;
    params.nlcDstR2ndLpCnt = axisDstR2ndLpCnt;
    params.lcDstR2ndLpCnt = axisDstR2ndLpCnt;
    params.nlcDstR2ndLeft = axisDstR2ndLeft;
    params.lcDstR2ndLeft = axisDstR2ndLeft;
  } else {
    params.mcOnCl = 0;
    params.usedCoreCnt = GetCeilDiv(cLpCnt, GetCeilDiv(cLpCnt, coreNum));
    params.nlcCLpCnt = GetCeilDiv(cLpCnt, params.usedCoreCnt);
    params.lcCLpCnt = cLpCnt - params.nlcCLpCnt * (params.usedCoreCnt - 1);
    params.coreStepIn = params.nlcCLpCnt * params.cLpStepIn;
    params.coreStepOut = params.nlcCLpCnt * params.cLpStepOut;
    params.nlcCLeft = 0;
    params.lcCLeft = cLeft;
    params.nlcSrcClLpCnt = axisSrcClLpCnt;
    params.lcSrcClLpCnt = axisSrcClLpCnt;
    params.nlcSrcClLeft = axisSrcClLeft;
    params.lcSrcClLeft = axisSrcClLeft;
    params.nlcDstR2ndLpCnt = axisDstR2ndLpCnt;
    params.lcDstR2ndLpCnt = axisDstR2ndLpCnt;
    params.nlcDstR2ndLeft = axisDstR2ndLeft;
    params.lcDstR2ndLeft = axisDstR2ndLeft;
  }
  return true;
}
bool GetCommonParam(int64_t& ubSize, int64_t& blockElemCnt, int64_t& c0Len, int64_t& axisCSize,
                    TransDataMode1011Param& params) {
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

bool TillingPositiveMode1011(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                             std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt,
                             int64_t& ubSize, TransDataMode1011Param& params) {
  if ((srcFormat.length() != inShape.size()) || (dstFormat.length() != outShape.size())) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TillingPositiveMode1011 Failed.");
    return false;
  }
  int64_t axisCSize = inShape[inShape.size() - 1];
  int64_t c0Len = outShape[outShape.size() - 1];
  bool ret = GetCommonParam(ubSize, blockElemCnt, c0Len, axisCSize, params);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TillingPositiveMode1011 GetCommonParam Failed.");
    return ret;
  }

  params.tilingMode = 1011;

  // target axis -2 tiling parameters
  int32_t dstAxisPosC = std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str();
  int32_t srcAxisPosC = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
  int32_t dstR2ndInSrcIdx = std::strchr(srcFormat.c_str(), dstFormat[dstFormat.length() - 2]) - srcFormat.c_str();
  int64_t axisDstR2ndSize = inShape[dstR2ndInSrcIdx];
  if (axisDstR2ndSize < VNC_LINES) {
    params.dstR2ndLpUnit = axisDstR2ndSize;
  } else {
    params.dstR2ndLpUnit = VNC_LINES;
  }
  int64_t axisDstR2ndLpCnt = GetCeilDiv(axisDstR2ndSize, params.dstR2ndLpUnit);
  int64_t axisDstR2ndLeft = axisDstR2ndSize % params.dstR2ndLpUnit;
  params.dstR2ndLpStepIn = GetShapeSize(inShape, dstR2ndInSrcIdx + 1) * params.dstR2ndLpUnit;
  params.dstR2ndLpStepOut = GetShapeSize(outShape, -1) * params.dstR2ndLpUnit;
  params.dstR2ndStepIn = GetShapeSize(inShape, dstR2ndInSrcIdx + 1);

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

  // source axis left tiling parameters
  string srcFormatLeft = srcFormat;
  srcFormatLeft.replace(srcAxisPosC, 1, "");
  int32_t chrPos = std::strchr(srcFormatLeft.c_str(), dstFormat[dstFormat.length() - 2]) - srcFormatLeft.c_str();
  srcFormatLeft.replace(chrPos, 1, "");
  vector<int64_t> srcLeftShape;
  for (size_t i = 0; i < srcFormatLeft.length(); i++) {
    char curChar = srcFormatLeft[i];
    int32_t curPos = std::strchr(srcFormat.c_str(), curChar) - srcFormat.c_str();
    srcLeftShape.push_back(inShape[curPos]);
  }
  srcLeftShape.push_back(1);
  int64_t axisSrcClSize = GetShapeSize(srcLeftShape, 0);
  int64_t plnSrcClCnt = params.vncLineSize / GetCeilFillC(params.cLpUnit, c0Len);
  if (axisSrcClSize < plnSrcClCnt) {
    params.srcClLpUnit = axisSrcClSize;
  } else {
    params.srcClLpUnit = plnSrcClCnt;
  }
  int64_t axisSrcClLpCnt = GetCeilDiv(axisSrcClSize, params.srcClLpUnit);
  int64_t axisSrcClLeft = axisSrcClSize % params.srcClLpUnit;
  params.srcClLpStepIn = GetShapeSize(inShape, -1) * params.srcClLpUnit;
  params.srcClLpStepOut = 0;

  // parameters for output data
   reverse(srcFormatLeft.begin(), srcFormatLeft.end());
   for (size_t i = 0; i < srcFormatLeft.length(); i++) {
    char chr = srcFormatLeft[i];
    int32_t srcChrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    int32_t dstChrPos = std::strchr(dstFormat.c_str(), chr) - dstFormat.c_str();
    if (i == 0) {
      params.clOut0Size = inShape[srcChrPos];
      params.clOut0SrcRsize = GetShapeSize(srcLeftShape, -1 - i);
      params.clOut0DstAsize = GetShapeSize(outShape, dstChrPos + 1);
    } else if (i == 1) {
      params.clOut1Size = inShape[srcChrPos];
      params.clOut1SrcRsize = GetShapeSize(srcLeftShape, -1 - i);
      params.clOut1DstAsize = GetShapeSize(outShape, dstChrPos + 1);
    }
  }

  ret = GetMcInfoPositive1011(axisDstR2ndLpCnt, axisDstR2ndLeft, cLpCnt, cLeft, axisSrcClLpCnt, axisSrcClLeft,
                              coreNum, params);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetMcInfoPositive1011 Failed.");
    return ret;
  }
  return true;
}

void SetRunningMode1011Params(const TransDataMode1011Param& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);
  ByteBufferPut(runInfo.tiling_data, runParams.ubOffset);
  ByteBufferPut(runInfo.tiling_data, runParams.usedCoreCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.mcOnCl);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.srcClLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.vncLineSize);
  ByteBufferPut(runInfo.tiling_data, runParams.srcClLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.srcClLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.cLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.cLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.cStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.c0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.cModC0);
  ByteBufferPut(runInfo.tiling_data, runParams.cLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcDstR2ndLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcDstR2ndLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcSrcClLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcSrcClLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcDstR2ndLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcDstR2ndLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcSrcClLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcSrcClLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.clOut0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.clOut0SrcRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.clOut0DstAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.clOut1Size);
  ByteBufferPut(runInfo.tiling_data, runParams.clOut1SrcRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.clOut1DstAsize);
}

void PrintTilingMode1011Params(const std::string& opType, const TransDataMode1011Param& params) {
  OP_LOGD(opType.c_str(), "tilingMode=%d", params.tilingMode);
  OP_LOGD(opType.c_str(), "ubOffset=%d", params.ubOffset);
  OP_LOGD(opType.c_str(), "usedCoreCnt=%d", params.usedCoreCnt);
  OP_LOGD(opType.c_str(), "mcOnCl=%d", params.mcOnCl);
  OP_LOGD(opType.c_str(), "coreStepIn=%d", params.coreStepIn);
  OP_LOGD(opType.c_str(), "coreStepOut=%d", params.coreStepOut);
  OP_LOGD(opType.c_str(), "dstR2ndLpStepIn=%d", params.dstR2ndLpStepIn);
  OP_LOGD(opType.c_str(), "dstR2ndLpStepOut=%d", params.dstR2ndLpStepOut);
  OP_LOGD(opType.c_str(), "dstR2ndStepIn=%d", params.dstR2ndStepIn);
  OP_LOGD(opType.c_str(), "dstR2ndLpUnit=%d", params.dstR2ndLpUnit);
  OP_LOGD(opType.c_str(), "srcClLpStepIn=%d", params.srcClLpStepIn);
  OP_LOGD(opType.c_str(), "vncLineSize=%d", params.vncLineSize);
  OP_LOGD(opType.c_str(), "srcClLpUnit=%d", params.srcClLpUnit);
  OP_LOGD(opType.c_str(), "srcClLpStepOut=%d", params.srcClLpStepOut);
  OP_LOGD(opType.c_str(), "cLpStepIn=%d", params.cLpStepIn);
  OP_LOGD(opType.c_str(), "cLpStepOut=%d", params.cLpStepOut);
  OP_LOGD(opType.c_str(), "cStepOut=%d", params.cStepOut);
  OP_LOGD(opType.c_str(), "c0Size=%d", params.c0Size);
  OP_LOGD(opType.c_str(), "cModC0=%d", params.cModC0);
  OP_LOGD(opType.c_str(), "cLpUnit=%d", params.cLpUnit);
  OP_LOGD(opType.c_str(), "nlcDstR2ndLpCnt=%d", params.nlcDstR2ndLpCnt);
  OP_LOGD(opType.c_str(), "nlcDstR2ndLeft=%d", params.nlcDstR2ndLeft);
  OP_LOGD(opType.c_str(), "nlcSrcClLpCnt=%d", params.nlcSrcClLpCnt);
  OP_LOGD(opType.c_str(), "nlcSrcClLeft=%d", params.nlcSrcClLeft);
  OP_LOGD(opType.c_str(), "nlcCLpCnt=%d", params.nlcCLpCnt);
  OP_LOGD(opType.c_str(), "nlcCLeft=%d", params.nlcCLeft);
  OP_LOGD(opType.c_str(), "lcDstR2ndLpCnt=%d", params.lcDstR2ndLpCnt);
  OP_LOGD(opType.c_str(), "lcDstR2ndLeft=%d", params.lcDstR2ndLeft);
  OP_LOGD(opType.c_str(), "lcSrcClLpCnt=%d", params.lcSrcClLpCnt);
  OP_LOGD(opType.c_str(), "lcSrcClLeft=%d", params.lcSrcClLeft);
  OP_LOGD(opType.c_str(), "lcCLpCnt=%d", params.lcCLpCnt);
  OP_LOGD(opType.c_str(), "lcCLeft=%d", params.lcCLeft);
  OP_LOGD(opType.c_str(), "clOut0Size=%d", params.clOut0Size);
  OP_LOGD(opType.c_str(), "clOut0SrcRsize=%d", params.clOut0SrcRsize);
  OP_LOGD(opType.c_str(), "clOut0DstAsize=%d", params.clOut0DstAsize);
  OP_LOGD(opType.c_str(), "clOut1Size=%d", params.clOut1Size);
  OP_LOGD(opType.c_str(), "clOut1SrcRsize=%d", params.clOut1SrcRsize);
  OP_LOGD(opType.c_str(), "clOut1DstAsize=%d", params.clOut1DstAsize);
}

}  // namespace optiling
