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

bool GetMcInfoPositive100(int64_t& srcCrLpCnt, int64_t& srcCrSize,  int64_t& srcCLpCnt, int64_t& srcCSize,
                          int64_t& leftLpCnt, int64_t& coreNum, TransDataMode100Param& params) {
  int64_t tmpFullLoopCntCr;
  if (GetFloorDiv(srcCrLpCnt, coreNum) > 0) {
    tmpFullLoopCntCr = coreNum;
  } else {
    tmpFullLoopCntCr = 0;
  }
  int64_t reminderLoopCntCr = srcCrLpCnt % coreNum;
  if (reminderLoopCntCr == 0) {
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

  int64_t tmpFullLoopCntLeft;
  if (GetFloorDiv(leftLpCnt, coreNum) > 0) {
    tmpFullLoopCntLeft = coreNum;
  } else {
    tmpFullLoopCntLeft = 0;
  }
  int64_t reminderLoopCntLeft = leftLpCnt % coreNum;
  if (reminderLoopCntLeft == 0) {
    tmpFullLoopCntLeft += coreNum;
  }
  int64_t fullLoopCntLeft = tmpFullLoopCntLeft + reminderLoopCntLeft;

  vector<int64_t> loopCntList = {fullLoopCntLeft, fullLoopCntC, fullLoopCntCr};
  if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 0) {
    params.mcFlag = 1;
    params.usedCoreCnt = GetCeilDiv(leftLpCnt, GetCeilDiv(leftLpCnt, coreNum));
    params.nlcLeftLpCnt = GetCeilDiv(leftLpCnt, params.usedCoreCnt);
    params.lcLeftLpCnt = leftLpCnt - params.nlcLeftLpCnt * (params.usedCoreCnt - 1);
    params.coreStepIn = 0;
    params.coreStepOut = 0;
    params.nlcCLpCnt = srcCLpCnt;
    params.lcCLpCnt = srcCLpCnt;
    params.nlcCLeft = srcCSize % params.srcCLpUnit;
    params.lcCLeft = srcCSize % params.srcCLpUnit;
    params.nlcCrLpCnt = srcCrLpCnt;
    params.lcCrLpCnt = srcCrLpCnt;
    params.nlcCrLeft = srcCrSize % params.srcCrLpUnit;
    params.lcCrLeft = srcCrSize % params.srcCrLpUnit;
  } else {
    params.mcFlag = 0;
    params.nlcLeftLpCnt = leftLpCnt;
    params.lcLeftLpCnt = leftLpCnt;
    if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 1) {
      params.usedCoreCnt = GetCeilDiv(srcCLpCnt, GetCeilDiv(srcCLpCnt, coreNum));
      params.nlcCLpCnt = GetCeilDiv(srcCLpCnt, params.usedCoreCnt);
      params.lcCLpCnt = srcCLpCnt - params.nlcCLpCnt * (params.usedCoreCnt - 1);
      params.nlcCLeft = 0;
      params.lcCLeft = srcCSize % params.srcCLpUnit;
      params.coreStepIn = params.nlcCLpCnt * params.srcCLpStepIn;
      params.coreStepOut = params.nlcCLpCnt * params.srcCLpStepOut;
      params.nlcCrLpCnt = srcCrLpCnt;
      params.lcCrLpCnt = srcCrLpCnt;
      params.nlcCrLeft = srcCrSize % params.srcCrLpUnit;
      params.lcCrLeft = srcCrSize % params.srcCrLpUnit;
    } else {
      params.usedCoreCnt = GetCeilDiv(srcCrLpCnt, GetCeilDiv(srcCrLpCnt, coreNum));
      params.nlcCrLpCnt = GetCeilDiv(srcCrLpCnt, params.usedCoreCnt);
      params.lcCrLpCnt = srcCrLpCnt - params.nlcCrLpCnt * (params.usedCoreCnt - 1);
      params.nlcCrLeft = 0;
      params.lcCrLeft = srcCrSize % params.srcCrLpUnit;
      params.coreStepIn = params.nlcCrLpCnt * params.srcCrLpUnit;
      params.coreStepOut = 0;
      params.nlcCLpCnt = srcCLpCnt;
      params.lcCLpCnt = srcCLpCnt;
      params.nlcCLeft = srcCSize % params.srcCLpUnit;
      params.lcCLeft = srcCSize % params.srcCLpUnit;
    }
  }
  return true;
}

bool TillingPositiveMode100(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt, int64_t& c0Len,
                            int64_t& ubSize, TransDataMode100Param& params) {
  if (srcFormat.length() != inShape.size()) {
    OP_LOGE("op TransDataTiling: TillingPositiveMode100 Failed.");
    return false;
  }

  int64_t halfUbSize = ubSize / 2;
  params.tilingMode = 100;
  params.oneLineSize = halfUbSize / VNC_LINES / blockElemCnt * blockElemCnt;
  params.ubOffset = params.oneLineSize * VNC_LINES;

  int32_t srcAxisPosC = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
  int32_t dstAxisPosC = std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str();
  int64_t axisSrcCrSize = GetShapeSize(inShape, srcAxisPosC + 1);
  params.srcCrLpUnit = params.oneLineSize / C0_16 / blockElemCnt * blockElemCnt;
  int64_t srcCrLpCnt = GetCeilDiv(axisSrcCrSize, params.srcCrLpUnit);
  params.srcCrLpStepIn = params.srcCrLpUnit;
  params.srcCrLpStepOut = 0;

  string tmpSrcClFormat = srcFormat;
  tmpSrcClFormat.replace(srcAxisPosC, srcFormat.length() - srcAxisPosC, "");
  string tmpDstCrFormat = dstFormat.substr(0, dstFormat.length() - 1);
  tmpDstCrFormat.replace(dstAxisPosC, 1, "");

  for (int32_t i = 0; i < tmpSrcClFormat.length(); i++) {
    char chr = tmpSrcClFormat[i];
    int32_t chrPos = std::strchr(tmpDstCrFormat.c_str(), chr) - tmpDstCrFormat.c_str();
    tmpDstCrFormat.replace(chrPos, 1, "");
  }

  vector<int64_t> tmpDstCrShape;
  for (int32_t i = 0; i < tmpDstCrFormat.length(); i++) {
    char chr = tmpDstCrFormat[i];
    int32_t chrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    tmpDstCrShape.push_back(inShape[chrPos]);
  }
  tmpDstCrShape.push_back(1);
  reverse(tmpDstCrFormat.begin(), tmpDstCrFormat.end());

  for (size_t i = 0; i < tmpDstCrFormat.length(); i++) {
    char chr = tmpDstCrFormat[i];
    int32_t srcChrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    int32_t dstChrPos = std::strchr(dstFormat.c_str(), chr) - dstFormat.c_str();
    if (i == 0) {
      params.crOutIdx0Size = inShape[srcChrPos];
      params.crOutIdx0DstRsize = GetShapeSize(tmpDstCrShape, -1 - i);
      params.crOutIdx0DstAsize = GetShapeSize(outShape, dstChrPos + 1);
    } else if (i == 1) {
      params.crOutIdx1Size = inShape[srcChrPos];
      params.crOutIdx1DstRsize = GetShapeSize(tmpDstCrShape, -1 - i);
      params.crOutIdx1DstAsize = GetShapeSize(outShape, dstChrPos + 1);
    }
  }

  int32_t padAxisCnt = FRAME_LEVEL - tmpDstCrFormat.length();
  if (padAxisCnt != 0) {
    if (tmpDstCrFormat.length() == 0) {
      params.crOutIdx0Size = 0;
      params.crOutIdx0DstRsize = 0;
      params.crOutIdx0DstAsize = 0;
      params.crOutIdx1Size = 0;
      params.crOutIdx1DstRsize = 0;
      params.crOutIdx1DstAsize = 0;
    } else if (tmpDstCrFormat.length() == 1) {
      params.crOutIdx1Size = 0;
      params.crOutIdx1DstRsize = 0;
      params.crOutIdx1DstAsize = 0;
    }
  }

  int64_t axisSrcCSize = inShape[srcAxisPosC];
  params.srcCLpUnit = c0Len;
  int64_t srcCLpCnt = GetCeilDiv(axisSrcCSize, params.srcCLpUnit);
  params.srcCLpStepIn = params.srcCLpUnit * GetShapeSize(inShape, srcAxisPosC + 1);
  params.srcCLpStepOut = GetShapeSize(outShape, dstAxisPosC + 1);
  params.srcCStepIn = GetShapeSize(inShape, srcAxisPosC + 1);
  params.cModC0 = axisSrcCSize % c0Len;

  string tmpLeftSrcFormat = srcFormat;
  tmpLeftSrcFormat.replace(srcAxisPosC, srcFormat.length() - srcAxisPosC, "");
  vector<int64_t> tmpLeftInShape;
  int64_t axisSrcLeftAxisSize = 1;

  for (size_t i = 0; i < tmpLeftSrcFormat.length(); i++) {
    char chr = tmpLeftSrcFormat[i];
    int32_t srcChrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    axisSrcLeftAxisSize *= inShape[srcChrPos];
    tmpLeftInShape.push_back(inShape[srcChrPos]);
  }

  bool ret = GetMcInfoPositive100(srcCrLpCnt, axisSrcCrSize, srcCLpCnt, axisSrcCSize, axisSrcLeftAxisSize, coreNum,
                                  params);
  if (!ret) {
    OP_LOGE("op TransDataTiling: GetMcInfoPositive100 Failed.");
    return ret;
  }

  string tmpSrcCrFormat = srcFormat;
  tmpSrcCrFormat.replace(0, srcAxisPosC, "");
  string tmpDstLeftFormat = dstFormat.substr(0, dstFormat.length() - 1);
  tmpDstLeftFormat.replace(dstAxisPosC, 1, "");
  vector<int64_t> tmpLeftOutShape;

  for (int32_t i = 0; i < tmpSrcCrFormat.length(); i++) {
    char chr = tmpSrcCrFormat[i];
    int32_t chrPos = std::strchr(tmpDstLeftFormat.c_str(), chr) - tmpDstLeftFormat.c_str();
    if (chrPos >= 0 && chrPos < tmpDstLeftFormat.length()) {
      tmpDstLeftFormat.replace(chrPos, 1, "");
    }
  }

  for (int32_t i = 0; i < tmpDstLeftFormat.length(); i++) {
    char chr = tmpDstLeftFormat[i];
    int32_t chrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    if (chrPos >= 0 && chrPos < srcFormat.length()) {
      tmpLeftOutShape.push_back(inShape[chrPos]);
    }
  }
  inShape.push_back(1);
  tmpLeftOutShape.push_back(1);
  reverse(tmpDstLeftFormat.begin(), tmpDstLeftFormat.end());

  for (size_t i = 0; i < tmpDstLeftFormat.length(); i++) {
    char chr = tmpDstLeftFormat[i];
    int32_t srcChrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    int32_t dstChrPos = std::strchr(dstFormat.c_str(), chr) - dstFormat.c_str();
    if (i == 0) {
      params.inIdx0Size = inShape[srcChrPos];
      params.inIdx0DstRsize = GetShapeSize(tmpLeftOutShape, -1 - i);
      params.inIdx0SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
      params.outIdx0Size = inShape[srcChrPos];
      params.outIdx0DstRsize = GetShapeSize(tmpLeftOutShape, -1 - i);
      params.outIdx0DstAsize = GetShapeSize(outShape, dstChrPos + 1);
    } else if (i == 1) {
      params.inIdx1Size = inShape[srcChrPos];
      params.inIdx1DstRsize = GetShapeSize(tmpLeftOutShape, -1 - i);
      params.inIdx1SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
      params.outIdx1Size = inShape[srcChrPos];
      params.outIdx1DstRsize = GetShapeSize(tmpLeftOutShape, -1 - i);
      params.outIdx1DstAsize = GetShapeSize(outShape, dstChrPos + 1);
    }
  }
  padAxisCnt = FRAME_LEVEL - tmpDstLeftFormat.length();
  if (padAxisCnt != 0) {
    if (tmpDstLeftFormat.length() == 0) {
      params.inIdx0Size = 0;
      params.inIdx0DstRsize = 0;
      params.inIdx0SrcAsize = 0;
      params.outIdx0Size = 0;
      params.outIdx0DstRsize = 0;
      params.outIdx0DstAsize = 0;
      params.inIdx1Size = 0;
      params.inIdx1DstRsize = 0;
      params.inIdx1SrcAsize = 0;
      params.outIdx1Size = 0;
      params.outIdx1DstRsize = 0;
      params.outIdx1DstAsize = 0;
    } else if (tmpDstLeftFormat.length() == 1) {
      params.inIdx1Size = 0;
      params.inIdx1DstRsize = 0;
      params.inIdx1SrcAsize = 0;
      params.outIdx1Size = 0;
      params.outIdx1DstRsize = 0;
      params.outIdx1DstAsize = 0;
    }
  }

  if (srcFormat[srcFormat.length() - 1] == dstFormat[dstFormat.length() - 2]) {
    params.src2dstFlag = 0;
  } else {
    params.src2dstFlag = 1;
  }
  return true;
}

void SetRunningMode100Params(const TransDataMode100Param& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);
  ByteBufferPut(runInfo.tiling_data, runParams.ubOffset);
  ByteBufferPut(runInfo.tiling_data, runParams.mcFlag);
  ByteBufferPut(runInfo.tiling_data, runParams.usedCoreCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepOut);

  ByteBufferPut(runInfo.tiling_data, runParams.nlcCrLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcLeftLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCrLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcCLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCrLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcLeftLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCrLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcCLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCrLpUnit);

  ByteBufferPut(runInfo.tiling_data, runParams.srcCrLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCrLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcCLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.cModC0);
  ByteBufferPut(runInfo.tiling_data, runParams.inIdx0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.inIdx0DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.inIdx0SrcAsize);

  ByteBufferPut(runInfo.tiling_data, runParams.inIdx1Size);
  ByteBufferPut(runInfo.tiling_data, runParams.inIdx1DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.inIdx1SrcAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.outIdx0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.outIdx0DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.outIdx0DstAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.outIdx1Size);
  ByteBufferPut(runInfo.tiling_data, runParams.outIdx1DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.outIdx1DstAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx0DstRsize);

  ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx0DstAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx1Size);
  ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx1DstRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.crOutIdx1DstAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.src2dstFlag);
  ByteBufferPut(runInfo.tiling_data, runParams.oneLineSize);
}

void PrintTilingMode100Params(const std::string& opType, const TransDataMode100Param& params) {
  OP_LOGD(opType.c_str(), "tilingMode=%d", params.tilingMode);
  OP_LOGD(opType.c_str(), "ubOffset=%d", params.ubOffset);
  OP_LOGD(opType.c_str(), "mcFlag=%d", params.mcFlag);
  OP_LOGD(opType.c_str(), "usedCoreCnt=%d", params.usedCoreCnt);
  OP_LOGD(opType.c_str(), "coreStepIn=%d", params.coreStepIn);
  OP_LOGD(opType.c_str(), "coreStepOut=%d", params.coreStepOut);
  OP_LOGD(opType.c_str(), "nlcCrLpCnt=%d", params.nlcCrLpCnt);
  OP_LOGD(opType.c_str(), "nlcCLpCnt=%d", params.nlcCLpCnt);
  OP_LOGD(opType.c_str(), "nlcLeftLpCnt=%d", params.nlcLeftLpCnt);
  OP_LOGD(opType.c_str(), "nlcCrLeft=%d", params.nlcCrLeft);
  OP_LOGD(opType.c_str(), "nlcCLeft=%d", params.nlcCLeft);
  OP_LOGD(opType.c_str(), "lcCrLpCnt=%d", params.lcCrLpCnt);
  OP_LOGD(opType.c_str(), "lcCLpCnt=%d", params.lcCLpCnt);
  OP_LOGD(opType.c_str(), "lcLeftLpCnt=%d", params.lcLeftLpCnt);
  OP_LOGD(opType.c_str(), "lcCrLeft=%d", params.lcCrLeft);
  OP_LOGD(opType.c_str(), "lcCLeft=%d", params.lcCLeft);
  OP_LOGD(opType.c_str(), "srcCrLpUnit=%d", params.srcCrLpUnit);

  OP_LOGD(opType.c_str(), "srcCrLpStepIn=%d", params.srcCrLpStepIn);
  OP_LOGD(opType.c_str(), "srcCrLpStepOut=%d", params.srcCrLpStepOut);
  OP_LOGD(opType.c_str(), "srcCStepIn=%d", params.srcCStepIn);
  OP_LOGD(opType.c_str(), "srcCLpUnit=%d", params.srcCLpUnit);
  OP_LOGD(opType.c_str(), "srcCLpStepIn=%d", params.srcCLpStepIn);
  OP_LOGD(opType.c_str(), "srcCLpStepOut=%d", params.srcCLpStepOut);
  OP_LOGD(opType.c_str(), "cModC0=%d", params.cModC0);
  OP_LOGD(opType.c_str(), "inIdx0Size=%d", params.inIdx0Size);

  OP_LOGD(opType.c_str(), "inIdx0DstRsize=%d", params.inIdx0DstRsize);
  OP_LOGD(opType.c_str(), "inIdx0SrcAsize=%d", params.inIdx0SrcAsize);
  OP_LOGD(opType.c_str(), "inIdx1Size=%d", params.inIdx1Size);
  OP_LOGD(opType.c_str(), "inIdx1DstRsize=%d", params.inIdx1DstRsize);
  OP_LOGD(opType.c_str(), "inIdx1SrcAsize=%d", params.inIdx1SrcAsize);
  OP_LOGD(opType.c_str(), "outIdx0Size=%d", params.outIdx0Size);
  OP_LOGD(opType.c_str(), "outIdx0DstRsize=%d", params.outIdx0DstRsize);
  OP_LOGD(opType.c_str(), "outIdx0DstAsize=%d", params.outIdx0DstAsize);
  OP_LOGD(opType.c_str(), "outIdx1Size=%d", params.outIdx1Size);
  OP_LOGD(opType.c_str(), "outIdx1DstRsize=%d", params.outIdx1DstRsize);
  OP_LOGD(opType.c_str(), "outIdx1DstAsize=%d", params.outIdx1DstAsize);
  OP_LOGD(opType.c_str(), "crOutIdx0Size=%d", params.crOutIdx0Size);
  OP_LOGD(opType.c_str(), "crOutIdx0DstRsize=%d", params.crOutIdx0DstRsize);

  OP_LOGD(opType.c_str(), "crOutIdx0DstAsize=%d", params.crOutIdx0DstAsize);
  OP_LOGD(opType.c_str(), "crOutIdx1Size=%d", params.crOutIdx1Size);
  OP_LOGD(opType.c_str(), "crOutIdx1DstRsize=%d", params.crOutIdx1DstRsize);
  OP_LOGD(opType.c_str(), "crOutIdx1DstAsize=%d", params.crOutIdx1DstAsize);
  OP_LOGD(opType.c_str(), "src2dstFlag=%d", params.src2dstFlag);
  OP_LOGD(opType.c_str(), "oneLineSize=%d", params.oneLineSize);

}

}  // namespace optiling