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
 * \file trans_data_positive_target_t_mode100.cpp
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

const int32_t FRAME_LEVEL = 2;

bool GetMcInfoNegative200(int64_t& srcCrLpCnt, int64_t& axisSrcCrSize, int64_t& srcCrLpUnit, int64_t& srcCLpCnt,
                          int64_t& axisSrcCSize, int64_t& srcCLpUnit, int64_t& srcCLpStepIn,
                          int64_t& srcCLpStepOut, int64_t& srcC1LpCnt, int64_t& axisSrcC1Size,
                          int64_t& srcC1LpUnit, int64_t& mcPos, int64_t& isMcCr,
                          int64_t& isMcC1, int64_t& usedCoreCnt, int64_t& coreStepIn,
                          int64_t& coreStepOut, int64_t& nlcC1LpCnt, int64_t& nlcCLpCnt,
                          int64_t& nlcCrLpCnt, int64_t& nlcC1Left, int64_t& nlcCLeft,
                          int64_t& nlcCrLeft, int64_t& lcC1LpCnt, int64_t& lcCLpCnt,
                          int64_t& lcCrLpCnt, int64_t& lcC1Left, int64_t& lcCLeft,
                          int64_t& lcCrLeft, int64_t& coreNum) {
  int64_t tmpFullLoopCntCr;
  if (GetFloorDiv(srcCrLpCnt, coreNum) > 0) {
    tmpFullLoopCntCr = coreNum;
  } else {
    tmpFullLoopCntCr = 0;
  }
  int64_t reminderLoopCntCr = srcCrLpCnt % coreNum;
  if (reminderLoopCntCr == 0 && axisSrcCrSize % srcCrLpUnit > srcCrLpUnit / 2) {
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


  int64_t tmpFullLoopCntC1;
  if (GetFloorDiv(srcC1LpCnt, coreNum) > 0) {
    tmpFullLoopCntC1 = coreNum;
  } else {
    tmpFullLoopCntC1 = 0;
  }
  int64_t reminderLoopCntC1 = srcC1LpCnt % coreNum;
  if (reminderLoopCntC1 == 0) {
    tmpFullLoopCntC1 += coreNum;
  }
  int64_t fullLoopCntC1 = tmpFullLoopCntC1 + reminderLoopCntC1;
  vector<int64_t> loopCntList = {fullLoopCntC1, fullLoopCntC, fullLoopCntCr};

  if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 0) {
    mcPos = 0;
    isMcC1 = 1;
    isMcCr = 0;
    usedCoreCnt = GetCeilDiv(srcC1LpCnt, GetCeilDiv(srcC1LpCnt, coreNum));
    nlcC1LpCnt = GetCeilDiv(srcC1LpCnt, usedCoreCnt);
    lcC1LpCnt = srcC1LpCnt - nlcC1LpCnt * (usedCoreCnt - 1);
    coreStepIn = 0;
    coreStepOut = 0;
    nlcC1Left = 0;
    lcC1Left = axisSrcC1Size % srcC1LpUnit;
    nlcCLpCnt = srcCLpCnt;
    lcCLpCnt = srcCLpCnt;
    nlcCLeft = axisSrcCSize % srcCLpUnit;
    lcCLeft = axisSrcCSize % srcCLpUnit;
    nlcCrLpCnt = srcCrLpCnt;
    lcCrLpCnt = srcCrLpCnt;
    nlcCrLeft = axisSrcCrSize % srcCrLpUnit;
    lcCrLeft = axisSrcCrSize % srcCrLpUnit;
  } else if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 1) {
    mcPos = 1;
    isMcC1 = 0;
    isMcCr = 0;
    usedCoreCnt = GetCeilDiv(srcCLpCnt, GetCeilDiv(srcCLpCnt, coreNum));
    nlcCLpCnt = GetCeilDiv(srcCLpCnt, usedCoreCnt);
    lcCLpCnt = srcCLpCnt - nlcCLpCnt * (usedCoreCnt - 1);
    nlcCLeft = 0;
    lcCLeft = axisSrcCSize % srcCLpUnit;
    coreStepIn = nlcCLpCnt * srcCLpStepIn;
    coreStepOut = nlcCLpCnt * srcCLpStepOut;
    nlcCrLpCnt = srcCrLpCnt;
    lcCrLpCnt = srcCrLpCnt;
    nlcCrLeft = axisSrcCrSize % srcCrLpUnit;
    lcCrLeft = axisSrcCrSize % srcCrLpUnit;
    nlcC1LpCnt = srcC1LpCnt;
    lcC1LpCnt = srcC1LpCnt;
    nlcC1Left = axisSrcC1Size % srcC1LpUnit;
    lcC1Left = axisSrcC1Size % srcC1LpUnit;
  } else {
    mcPos = 2;
    isMcC1 = 0;
    isMcCr = 1;
    usedCoreCnt = GetCeilDiv(srcCrLpCnt, GetCeilDiv(srcCrLpCnt, coreNum));
    nlcCrLpCnt = GetCeilDiv(srcCrLpCnt, usedCoreCnt);
    lcCrLpCnt = srcCrLpCnt - nlcCrLpCnt * (usedCoreCnt - 1);
    nlcCrLeft = 0;
    lcCrLeft = axisSrcCrSize % srcCrLpUnit;
    coreStepIn = 0;
    coreStepOut = 0;
    nlcCLpCnt = srcCLpCnt;
    lcCLpCnt = srcCLpCnt;
    nlcCLeft = axisSrcCSize % srcCLpUnit;
    lcCLeft = axisSrcCSize % srcCLpUnit;
    nlcC1LpCnt = srcC1LpCnt;
    lcC1LpCnt = srcC1LpCnt;
    nlcC1Left = axisSrcC1Size % srcC1LpUnit;
    lcC1Left = axisSrcC1Size % srcC1LpUnit;
  }
  return true;
}

bool TillingPositiveMode200(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt, int64_t& c0Len,
                            int64_t& ubSize, TransDataMode200Param& params) {
  params.tilingMode = 200;
  if (srcFormat.length() < 2 || dstFormat.length() < 1) {
    OP_LOGE("op TransDataTiling: TillingPositiveMode200 Failed.");
    return false;
  }
  if (srcFormat[srcFormat.length() - 2] == dstFormat[dstFormat.length() - 1]) {
    params.src2dstFlag = 0;
    params.tmpUbOffset = 0;
  } else {
    params.src2dstFlag = 1;
    params.tmpUbOffset = NI_16 * c0Len;
  }
  params.oneLineSize = (ubSize - params.tmpUbOffset) / 2 / VNC_LINES / blockElemCnt * blockElemCnt;
  params.ubOffset = params.tmpUbOffset + params.oneLineSize * VNC_LINES;

  int32_t srcAxisPosC = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
  int32_t dstAxisPosC = std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str();
  int64_t axisSrcCrSize = GetShapeSize(outShape, dstAxisPosC + 1);
  params.srcCrLpUnit = params.oneLineSize / c0Len / blockElemCnt * blockElemCnt;
  int64_t srcCrLpCnt = GetCeilDiv(axisSrcCrSize, params.srcCrLpUnit);

  string tmpDstCrFormat = dstFormat;
  tmpDstCrFormat.replace(0, dstAxisPosC + 1, "");
  vector<int64_t> tmpDstCrShape;
  for (int32_t i = dstAxisPosC + 1; i < outShape.size(); i++) {
    tmpDstCrShape.push_back(outShape[i]);
  }
  tmpDstCrShape.push_back(1);
  reverse(tmpDstCrFormat.begin(),tmpDstCrFormat.end());

  for (size_t i = 0; i < tmpDstCrFormat.length(); i++) {
    char chr = tmpDstCrFormat[i];
    int32_t srcChrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    params.crInIdx0Size = outShape[std::strchr(dstFormat.c_str(), chr) - dstFormat.c_str()];
    params.crInIdx0DstRsize = GetShapeSize(tmpDstCrShape, -1 - i);
    params.crInIdx0SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
  }

  int32_t padAxisCnt = FRAME_LEVEL - tmpDstCrFormat.length();

  if (padAxisCnt != 0) {
    params.crInIdx1Size = 1;
    params.crInIdx1DstRsize = 0;
    params.crInIdx1SrcAsize = 0;
  }

  params.srcCrStepOut = 1;
  int64_t axisSrcCSize = inShape[srcAxisPosC];
  int64_t axisDstCSize = outShape[dstAxisPosC];
  if (srcCrLpCnt > 1) {
    params.srcCLpUnit = 1;
  } else {
    params.srcCLpUnit = params.srcCrLpUnit / GetCeilFill(axisSrcCrSize, blockElemCnt);
  }
  int64_t srcCLpCnt = GetCeilDiv(axisSrcCSize, params.srcCLpUnit);
  params.srcCStepIn = GetShapeSize(inShape, srcAxisPosC + 1);
  params.srcCStepOut = GetShapeSize(outShape, dstAxisPosC + 1);
  params.srcCLpStepIn = params.srcCLpUnit * params.srcCStepIn;
  params.srcCLpStepOut = params.srcCLpUnit * c0Len * params.srcCStepOut;
  params.cModC0 = axisDstCSize % c0Len;

  int64_t axisSrcC1Size = 1;
  for (int32_t i = 0; i <= dstAxisPosC - 1; i++) {
    axisSrcC1Size *= outShape[i];
  }
  params.srcC1LpUnit = NI_16;
  params.srcC1StepOut = GetShapeSize(outShape, dstAxisPosC);
  int64_t srcC1LpCnt = GetCeilDiv(axisSrcC1Size, params.srcC1LpUnit);

  string tmpDstC1Format = dstFormat;
  tmpDstC1Format.replace(dstAxisPosC, dstFormat.length() - dstAxisPosC, "");
  
  vector<int64_t> tmpCLeftShape;
  for (int32_t i = 0; i <= dstAxisPosC - 1; i++) {
    tmpCLeftShape.push_back(outShape[i]);
  }
  tmpCLeftShape.push_back(1);
  reverse(tmpDstC1Format.begin(),tmpDstC1Format.end());

  for (size_t i = 0; i < tmpDstC1Format.length(); i++) {
    char chr = tmpDstC1Format[i];
    int32_t srcChrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    if (i == 0) {
      params.inIdx0Size = outShape[std::strchr(dstFormat.c_str(), chr) - dstFormat.c_str()];
      params.inIdx0DstRsize = GetShapeSize(tmpCLeftShape, -1 - i);
      params.inIdx0SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
    } else {
      params.inIdx1Size = outShape[std::strchr(dstFormat.c_str(), chr) - dstFormat.c_str()];
      params.inIdx1DstRsize = GetShapeSize(tmpCLeftShape, -1 - i);
      params.inIdx1SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
    }
  }
  padAxisCnt = FRAME_LEVEL - tmpDstC1Format.length();

  if (padAxisCnt != 0) {
    params.inIdx1Size = 1;
    params.inIdx1DstRsize = 0;
    params.inIdx1SrcAsize = 0;
  }
  bool ret = GetMcInfoNegative200(srcCrLpCnt, axisSrcCrSize, params.srcCrLpUnit, srcCLpCnt, axisSrcCSize,
                                  params.srcCLpUnit, params.srcCLpStepIn, params.srcCLpStepOut, srcC1LpCnt,
                                  axisSrcC1Size, params.srcC1LpUnit, params.mcPos, params.isMcCr, params.isMcC1,
                                  params.usedCoreCnt, params.coreStepIn, params.coreStepOut, params.nlcC1LpCnt,
                                  params.nlcCLpCnt, params.nlcCrLpCnt, params.nlcC1Left, params.nlcCLeft,
                                  params.nlcCrLeft, params.lcC1LpCnt, params.lcCLpCnt, params.lcCrLpCnt,
                                  params.lcC1Left, params.lcCLeft, params.lcCrLeft, coreNum);
  if (!ret) {
    OP_LOGE("op TransDataTiling: GetMcInfoNegative200 Failed.");
    return ret;
  }
  return true;
}

void SetRunningMode200Params(const TransDataMode200Param& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);
  ByteBufferPut(runInfo.tiling_data, runParams.ubOffset);
  ByteBufferPut(runInfo.tiling_data, runParams.tmpUbOffset);
  ByteBufferPut(runInfo.tiling_data, runParams.mcPos);
  ByteBufferPut(runInfo.tiling_data, runParams.usedCoreCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCrLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcC1LpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCrLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcC1Left);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCrLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcC1LpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCrLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcC1Left);

  ByteBufferPut(runInfo.tiling_data, runParams.srcCrLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.srcC1LpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCrStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.srcC1StepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.cModC0);
  ByteBufferPut(runInfo.tiling_data, runParams.isMcCr);
  ByteBufferPut(runInfo.tiling_data, runParams.isMcC1);
  ByteBufferPut(runInfo.tiling_data, runParams.src2dstFlag);
  ByteBufferPut(runInfo.tiling_data, runParams.oneLineSize);

  ByteBufferPut(runInfo.tiling_data, runParams.inIdx0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.inIdx0DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.inIdx0SrcAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.inIdx1Size);
  ByteBufferPut(runInfo.tiling_data, runParams.inIdx1DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.inIdx1SrcAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx0DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx0SrcAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx1Size);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx1DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crInIdx1SrcAsize);
}

void PrintTilingMode200Params(const std::string& opType, const TransDataMode200Param& params) {
  OP_LOGD(opType.c_str(), "tilingMode=%d", params.tilingMode);
  OP_LOGD(opType.c_str(), "ubOffset=%d", params.ubOffset);
  OP_LOGD(opType.c_str(), "tmpUbOffset=%d", params.tmpUbOffset);
  OP_LOGD(opType.c_str(), "mcPos=%d", params.mcPos);
  OP_LOGD(opType.c_str(), "usedCoreCnt=%d", params.usedCoreCnt);
  OP_LOGD(opType.c_str(), "coreStepIn=%d", params.coreStepIn);
  OP_LOGD(opType.c_str(), "coreStepOut=%d", params.coreStepOut);
  OP_LOGD(opType.c_str(), "nlcCrLpCnt=%d", params.nlcCrLpCnt);
  OP_LOGD(opType.c_str(), "nlcCLpCnt=%d", params.nlcCLpCnt);
  OP_LOGD(opType.c_str(), "nlcC1LpCnt=%d", params.nlcC1LpCnt);
  OP_LOGD(opType.c_str(), "nlcCrLeft=%d", params.nlcCrLeft);
  OP_LOGD(opType.c_str(), "nlcCLeft=%d", params.nlcCLeft);
  OP_LOGD(opType.c_str(), "nlcC1Left=%d", params.nlcC1Left);
  OP_LOGD(opType.c_str(), "lcCrLpCnt=%d", params.lcCrLpCnt);
  OP_LOGD(opType.c_str(), "lcCLpCnt=%d", params.lcCLpCnt);
  OP_LOGD(opType.c_str(), "lcC1LpCnt=%d", params.lcC1LpCnt);
  OP_LOGD(opType.c_str(), "lcCrLeft=%d", params.lcCrLeft);

  OP_LOGD(opType.c_str(), "lcCLeft=%d", params.lcCLeft);
  OP_LOGD(opType.c_str(), "lcC1Left=%d", params.lcC1Left);
  OP_LOGD(opType.c_str(), "srcCrLpUnit=%d", params.srcCrLpUnit);
  OP_LOGD(opType.c_str(), "srcCLpUnit=%d", params.srcCLpUnit);
  OP_LOGD(opType.c_str(), "srcC1LpUnit=%d", params.srcC1LpUnit);
  OP_LOGD(opType.c_str(), "srcCrStepOut=%d", params.srcCrStepOut);
  OP_LOGD(opType.c_str(), "srcCStepIn=%d", params.srcCStepIn);
  OP_LOGD(opType.c_str(), "srcCStepOut=%d", params.srcCStepOut);
  OP_LOGD(opType.c_str(), "srcC1StepOut=%d", params.srcC1StepOut);
  OP_LOGD(opType.c_str(), "srcCLpStepIn=%d", params.srcCLpStepIn);
  OP_LOGD(opType.c_str(), "srcCLpStepOut=%d", params.srcCLpStepOut);
  OP_LOGD(opType.c_str(), "cModC0=%d", params.cModC0);
  OP_LOGD(opType.c_str(), "isMcCr=%d", params.isMcCr);
  OP_LOGD(opType.c_str(), "isMcC1=%d", params.isMcC1);
  OP_LOGD(opType.c_str(), "src2dstFlag=%d", params.src2dstFlag);
  OP_LOGD(opType.c_str(), "oneLineSize=%d", params.oneLineSize);
  OP_LOGD(opType.c_str(), "inIdx0Size=%d", params.inIdx0Size);
  OP_LOGD(opType.c_str(), "inIdx0DstRsize=%d", params.inIdx0DstRsize);
  OP_LOGD(opType.c_str(), "inIdx0SrcAsize=%d", params.inIdx0SrcAsize);
  OP_LOGD(opType.c_str(), "inIdx1Size=%d", params.inIdx1Size);
  OP_LOGD(opType.c_str(), "inIdx1DstRsize=%d", params.inIdx1DstRsize);
  OP_LOGD(opType.c_str(), "inIdx1SrcAsize=%d", params.inIdx1SrcAsize);
  OP_LOGD(opType.c_str(), "crInIdx0Size=%d", params.crInIdx0Size);
  OP_LOGD(opType.c_str(), "crInIdx0DstRsize=%d", params.crInIdx0DstRsize);
  OP_LOGD(opType.c_str(), "crInIdx0SrcAsize=%d", params.crInIdx0SrcAsize);
  OP_LOGD(opType.c_str(), "crInIdx1Size=%d", params.crInIdx1Size);
  OP_LOGD(opType.c_str(), "crInIdx1DstRsize=%d", params.crInIdx1DstRsize);
  OP_LOGD(opType.c_str(), "crInIdx1SrcAsize=%d", params.crInIdx1SrcAsize);
}

}  // namespace optiling