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
 * \file trans_data_negative_target_tc_201.cc
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

bool GetMcInfoNegative201(int64_t& dstR2ndLpCnt, int64_t& dstR2ndLeft, int64_t& srcClLpCnt, int64_t& srcClLeft,
                          int64_t& srcLeftLpCnt, int64_t& srcLeftLeft, int64_t& coreNum, TransDataTc201Param& params) {
  int64_t tmpFullLoopCntR2nd;
  if (GetFloorDiv(dstR2ndLpCnt, coreNum) > 0) {
    tmpFullLoopCntR2nd = coreNum;
  } else {
    tmpFullLoopCntR2nd = 0;
  }
  int64_t reminderLoopCntR2nd = dstR2ndLpCnt % coreNum;
  if (reminderLoopCntR2nd == 0) {
    tmpFullLoopCntR2nd += coreNum;
  }
  int64_t fullLoopCntR2nd = tmpFullLoopCntR2nd + reminderLoopCntR2nd;

  int64_t tmpFullLoopCntC1;
  if (GetFloorDiv(srcClLpCnt, coreNum) > 0) {
    tmpFullLoopCntC1 = coreNum;
  } else {
    tmpFullLoopCntC1 = 0;
  }
  int64_t reminderLoopCntC1 = srcClLpCnt % coreNum;
  if (reminderLoopCntC1 == 0) {
    tmpFullLoopCntC1 += coreNum;
  }
  int64_t fullLoopCntC1 = tmpFullLoopCntC1 + reminderLoopCntC1;

  int64_t tmpFullLoopCntLeft;
  if (GetFloorDiv(srcLeftLpCnt, coreNum) > 0) {
    tmpFullLoopCntLeft = coreNum;
  } else {
    tmpFullLoopCntLeft = 0;
  }
  int64_t reminderLoopCntLeft = srcLeftLpCnt % coreNum;
  if (reminderLoopCntLeft == 0) {
    tmpFullLoopCntLeft += coreNum;
  }
  int64_t fullLoopCntLeft = tmpFullLoopCntLeft + reminderLoopCntLeft;
  vector<int64_t> loopCntList = {fullLoopCntLeft, fullLoopCntC1, fullLoopCntR2nd};

  if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 0) {
    params.mcPos = 0;
    params.usedCoreCnt = GetCeilDiv(srcLeftLpCnt, GetCeilDiv(srcLeftLpCnt, coreNum));
    params.nlcSrcLeftLpCnt = GetCeilDiv(srcLeftLpCnt, params.usedCoreCnt);
    params.lcSrcLeftLpCnt = srcLeftLpCnt - params.nlcSrcLeftLpCnt * (params.usedCoreCnt - 1);
    params.nlcSrcLeftLeft = 0;
    params.lcSrcLeftLeft = srcLeftLeft;
    params.coreStepIn = params.nlcSrcLeftLpCnt * params.srcLeftLpStepIn;
    params.coreStepOut = params.nlcSrcLeftLpCnt * params.srcLeftLpStepOut;
    params.nlcSrcClLpCnt = srcClLpCnt;
    params.lcSrcClLpCnt = srcClLpCnt;
    params.nlcSrcClLeft = srcClLeft;
    params.lcSrcClLeft = srcClLeft;
    params.nlcDstR2ndLpCnt = dstR2ndLpCnt;
    params.lcDstR2ndLpCnt = dstR2ndLpCnt;
    params.nlcDstR2ndLeft = dstR2ndLeft;
    params.lcDstR2ndLeft = dstR2ndLeft;
  } else if (max_element(loopCntList.begin(), loopCntList.end()) - loopCntList.begin() == 1) {
    params.mcPos = 1;
    params.usedCoreCnt = GetCeilDiv(srcClLpCnt, GetCeilDiv(srcClLpCnt, coreNum));
    params.nlcSrcClLpCnt = GetCeilDiv(srcClLpCnt, params.usedCoreCnt);
    params.lcSrcClLpCnt = srcClLpCnt - params.nlcSrcClLpCnt * (params.usedCoreCnt - 1);
    params.nlcSrcClLeft = 0;
    params.lcSrcClLeft = srcClLeft;
    params.coreStepIn = params.nlcSrcClLpCnt * params.srcClLpStepIn;
    params.coreStepOut = params.nlcSrcClLpCnt * params.srcClLpStepOut;
    params.nlcDstR2ndLpCnt = dstR2ndLpCnt;
    params.lcDstR2ndLpCnt = dstR2ndLpCnt;
    params.nlcDstR2ndLeft = dstR2ndLeft;
    params.lcDstR2ndLeft = dstR2ndLeft;
    params.nlcSrcLeftLpCnt = srcLeftLpCnt;
    params.lcSrcLeftLpCnt = srcLeftLpCnt;
    params.nlcSrcLeftLeft = srcLeftLeft;
    params.lcSrcLeftLeft = srcLeftLeft;
  } else {
    params.mcPos = 2;
    params.usedCoreCnt = GetCeilDiv(dstR2ndLpCnt, GetCeilDiv(dstR2ndLpCnt, coreNum));
    params.nlcDstR2ndLpCnt = GetCeilDiv(dstR2ndLpCnt, params.usedCoreCnt);
    params.lcDstR2ndLpCnt = dstR2ndLpCnt - params.nlcDstR2ndLpCnt * (params.usedCoreCnt - 1);
    params.nlcDstR2ndLeft = 0;
    params.lcDstR2ndLeft = dstR2ndLeft;
    params.coreStepIn = params.nlcDstR2ndLpCnt * params.dstR2ndLpStepIn;;
    params.coreStepOut = params.nlcDstR2ndLpCnt * params.dstR2ndLpStepOut;
    params.nlcSrcLeftLpCnt = srcLeftLpCnt;
    params.lcSrcLeftLpCnt = srcLeftLpCnt;
    params.nlcSrcLeftLeft = srcLeftLeft;
    params.lcSrcLeftLeft = srcLeftLeft;
    params.nlcSrcClLpCnt = srcClLpCnt;
    params.lcSrcClLpCnt = srcClLpCnt;
    params.nlcSrcClLeft = srcClLeft;
    params.lcSrcClLeft = srcClLeft;
  }
  return true;
}

bool TilingNegativeTc201(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                         std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt, std::string& dtype,
                         int64_t& ubSize, TransDataTc201Param& params) {
  if (srcFormat.length() < 2 || dstFormat.length() < 2) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "TilingNegativeTc201 Failed.");
    return false;
  }

  int64_t c0Len = inShape[inShape.size() - 1];
  params.c0Len = c0Len;

  int32_t srcAxisPosC = std::strchr(srcFormat.c_str(), 'C') - srcFormat.c_str();
  int32_t dstAxisPosC = std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str();
  int64_t axisDstCSize = outShape[dstAxisPosC];
  int64_t axisSrcC1Size = inShape[srcAxisPosC];
  vector<int64_t> dstR2ndShape;
  string dstR2ndFormat = "";
  int64_t axisDstR2ndSize;
  int64_t axisSrcLeftSize;
  string srcLeftFormat = "";
  if (srcFormat[srcFormat.length() - 2] == dstFormat[dstFormat.length() - 2]) {
    params.srcR2ndDstR2ndSame = 1;
    dstR2ndFormat += dstFormat[dstFormat.length() - 2];
    dstR2ndShape.push_back(outShape[outShape.size() - 2]);
    axisDstR2ndSize = outShape[outShape.size() - 2];
    srcLeftFormat += srcFormat[0];
    axisSrcLeftSize = outShape[std::strchr(dstFormat.c_str(), srcFormat[0]) - dstFormat.c_str()];
  } else {
    params.srcR2ndDstR2ndSame = 0;
    srcLeftFormat += srcFormat[srcFormat.length() - 2];
    axisSrcLeftSize = outShape[std::strchr(dstFormat.c_str(), srcFormat[srcFormat.length() - 2]) - dstFormat.c_str()];
    dstR2ndFormat = srcFormat.substr(0, srcFormat.length() - 2);
    auto chrCPos = dstR2ndFormat.find('C');
    if (chrCPos != std::string::npos) {
      dstR2ndFormat.replace(chrCPos, 1, "");
    }
    axisDstR2ndSize = 1;
    for (size_t i = 0; i < dstR2ndFormat.length(); i++) {
      char chr = dstR2ndFormat[i];
      int32_t srcChrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
      axisDstR2ndSize *= inShape[srcChrPos];
      dstR2ndShape.push_back(inShape[srcChrPos]);
    }
  }
  dstR2ndShape.push_back(1);

  // output ub offset
  params.ubOffset = ubSize / 2 / blockElemCnt * blockElemCnt;
  // axis c1 tiling parameters
  int64_t vncColBlockCnt = GetFloorDiv(params.ubOffset / VNC_LINES, blockElemCnt);
  if (vncColBlockCnt % 2 == 0) {
    vncColBlockCnt -= 1;
  }
  int64_t vncColSize = vncColBlockCnt * blockElemCnt;
  params.vncColSize = vncColSize;
  int64_t tmpSrcClLpUnit;
  int64_t cGate = 0;
  if (axisDstCSize % params.c0Len == 0) {
    cGate = 16 * params.c0Len;
  } else {
    cGate = 56 * params.c0Len;
  }

  if (axisSrcC1Size * c0Len >= cGate || axisDstCSize == c0Len) {
    params.tilingMode = 2010;
    if (axisDstR2ndSize < NI_16) {
      tmpSrcClLpUnit = GetFloorDiv(params.ubOffset, axisDstR2ndSize * params.c0Len);
    } else {
      tmpSrcClLpUnit = GetFloorDiv(params.ubOffset, NI_16 * params.c0Len);
    }

  } else if (dtype != "int8" && dtype != "uint8") {
    if (axisDstCSize * axisDstR2ndSize >= vncColSize / VNC_LINES) {
      params.tilingMode = 2011;
    } else {
      params.tilingMode = 2012;
    }
    tmpSrcClLpUnit = vncColSize / c0Len / blockElemCnt * blockElemCnt;
  } else {
    if (axisDstCSize * axisDstR2ndSize >= vncColSize / 2 / VNC_LINES) {
      params.tilingMode = 2011;
    } else {
      params.tilingMode = 2012;
    }
    tmpSrcClLpUnit = vncColSize / 2 / c0Len / blockElemCnt * blockElemCnt;
  }

  params.srcClLpUnit = axisSrcC1Size > tmpSrcClLpUnit ? tmpSrcClLpUnit : axisSrcC1Size;
  int64_t srcClLpCnt = GetCeilDiv(axisSrcC1Size, params.srcClLpUnit);
  int64_t srcClLeft = axisSrcC1Size % params.srcClLpUnit;
  params.srcClLpStepIn = params.srcClLpUnit * GetShapeSize(inShape, srcAxisPosC + 1);
  params.srcClLpStepOut = params.srcClLpUnit * c0Len;
  params.srcClStepIn = GetShapeSize(inShape, srcAxisPosC + 1);
  params.srcClStepOut = 1;
  params.cModC0 = axisDstCSize % c0Len;
  if (srcClLpCnt == 1) {
    params.allCIn = 1;
  } else {
    params.allCIn = 0;
  }

  // axis -2 tiling parameters
  params.dstR2ndDims = 2;
  int64_t tmpDstR2ndLpUnit;
  int64_t maxR2ndLpSize = 63;
  int64_t dtypeFactor = 1;
  // to make sure the rep_stride of vor is less than limit
  if (params.tilingMode == 2010) {
    if (dtype == "float32" || dtype == "int32" || dtype == "uint32") {
      if (axisDstCSize == params.c0Len and axisSrcLeftSize <= C0_16) {
        // for vor in copy data in
        maxR2ndLpSize = 63;
      } else {
        // for vor in reorder
        maxR2ndLpSize = 31;
      }
      dtypeFactor = 2;
    } else if (axisDstCSize == params.c0Len and axisSrcLeftSize <= C0_16) {
      maxR2ndLpSize = 127;
    }
    tmpDstR2ndLpUnit = GetFloorDiv(params.ubOffset, params.srcClLpUnit * c0Len);
    if (tmpDstR2ndLpUnit > maxR2ndLpSize) {
      tmpDstR2ndLpUnit = maxR2ndLpSize;
    }
  } else if (dtype != "int8" && dtype != "uint8") {
    tmpDstR2ndLpUnit = vncColSize / (params.srcClLpUnit * c0Len);
  } else {
    tmpDstR2ndLpUnit = vncColSize / 2 / (params.srcClLpUnit * c0Len);
  }
  params.dstR2ndLpUnit = axisDstR2ndSize > tmpDstR2ndLpUnit ? tmpDstR2ndLpUnit : axisDstR2ndSize;
  // to avoid bank conflict
  if (params.tilingMode == 2010 && params.dstR2ndLpUnit*dtypeFactor % NI_16 == 0 &&
      (params.dstR2ndLpUnit < params.srcClLpUnit || params.srcClLpUnit*dtypeFactor % NI_16 == 0)) {
    params.dstR2ndLpUnit -= 1;
  }
  int64_t dstR2ndLpCnt = GetCeilDiv(axisDstR2ndSize, params.dstR2ndLpUnit);
  int64_t dstR2ndLeft = axisDstR2ndSize % params.dstR2ndLpUnit;
  if (dstR2ndLpCnt == 1) {
    params.allR2ndIn = 1;
  } else {
    params.allR2ndIn = 0;
  }

  reverse(dstR2ndFormat.begin(), dstR2ndFormat.end());
  for (size_t i = 0; i < dstR2ndFormat.length(); i++) {
    char chr = dstR2ndFormat[i];
    int32_t srcChrPos = std::strchr(srcFormat.c_str(), chr) - srcFormat.c_str();
    if (i == 0) {
      params.dstR2ndIn0Size = inShape[srcChrPos];
      params.dstR2ndIn0SrcRsize = GetShapeSize(dstR2ndShape, -1 - i);
      params.dstR2ndIn0SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
    } else if (i == 1) {
      params.dstR2ndIn1Size = inShape[srcChrPos];
      params.dstR2ndIn1SrcRsize = GetShapeSize(dstR2ndShape, -1 - i);
      params.dstR2ndIn1SrcAsize = GetShapeSize(inShape, srcChrPos + 1);
    }
  }
  int32_t padAxisCnt = FRAME_LEVEL - dstR2ndFormat.length();
  if (padAxisCnt != 0) {
    params.dstR2ndDims = 1;
    if (dstR2ndFormat.length() == 0) {
      params.dstR2ndIn0Size = 1;
      params.dstR2ndIn0SrcRsize = 1;
      params.dstR2ndIn0SrcAsize = 0;
      params.dstR2ndIn1Size = 1;
      params.dstR2ndIn1SrcRsize = 1;
      params.dstR2ndIn1SrcAsize = 0;
    } else if (dstR2ndFormat.length() == 1) {
      params.dstR2ndIn1Size = 1;
      params.dstR2ndIn1SrcRsize = 1;
      params.dstR2ndIn1SrcAsize = 0;
    }
  }
  if (params.dstR2ndDims == 2) {
    params.dstR2ndStepIn = 0;
  } else {
    params.dstR2ndStepIn = c0Len;
  }
  params.dstR2ndLpStepIn = params.dstR2ndLpUnit * params.dstR2ndStepIn;
  params.dstR2ndStepOut = axisDstCSize;
  params.dstR2ndLpStepOut = params.dstR2ndLpUnit * params.dstR2ndStepOut;

  int64_t tmpSrcLeftLpUnit;
  if (params.tilingMode == 2010) {
    tmpSrcLeftLpUnit = params.ubOffset / (params.srcClLpUnit * params.dstR2ndLpUnit * c0Len);
  } else if (dtype != "int8" && dtype != "uint8") {
    tmpSrcLeftLpUnit = vncColSize / (params.srcClLpUnit * params.dstR2ndLpUnit * c0Len);
  } else {
    tmpSrcLeftLpUnit = vncColSize / 2 / (params.srcClLpUnit * params.dstR2ndLpUnit * c0Len);
  }
  if (params.tilingMode == 2011) {
    tmpSrcLeftLpUnit = NI_16;
  }
  params.srcLeftLpUnit = axisSrcLeftSize > tmpSrcLeftLpUnit ? tmpSrcLeftLpUnit : axisSrcLeftSize;
  int64_t srcLeftLpCnt = GetCeilDiv(axisSrcLeftSize, params.srcLeftLpUnit);
  int64_t srcLeftLeft = axisSrcLeftSize % params.srcLeftLpUnit;
  params.srcLeftStepIn = GetShapeSize(inShape, srcFormat.find(srcLeftFormat) + 1);
  params.srcLeftLpStepIn = params.srcLeftLpUnit * params.srcLeftStepIn;
  params.srcLeftStepOut = GetShapeSize(outShape, dstFormat.find(srcLeftFormat) + 1);
  params.srcLeftLpStepOut = params.srcLeftLpUnit * params.srcLeftStepOut;

  bool ret = GetMcInfoNegative201(dstR2ndLpCnt, dstR2ndLeft, srcClLpCnt, srcClLeft, srcLeftLpCnt, srcLeftLeft, coreNum, params);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransDataTiling", "GetMcInfoNegative201 Failed.");
    return ret;
  }
  return true;
}

void SetRunningTc201Params(const TransDataTc201Param& runParams, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, runParams.tilingMode);
  ByteBufferPut(runInfo.tiling_data, runParams.ubOffset);
  ByteBufferPut(runInfo.tiling_data, runParams.mcPos);
  ByteBufferPut(runInfo.tiling_data, runParams.usedCoreCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.srcR2ndDstR2ndSame);
  ByteBufferPut(runInfo.tiling_data, runParams.c0Len);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.coreStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcDstR2ndLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcSrcClLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcSrcLeftLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcDstR2ndLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcSrcClLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.nlcSrcLeftLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcDstR2ndLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcSrcClLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcSrcLeftLpCnt);
  ByteBufferPut(runInfo.tiling_data, runParams.lcDstR2ndLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcSrcClLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.lcSrcLeftLeft);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.srcClLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.allCIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcClStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcClStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.srcClLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcClLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.cModC0);
  ByteBufferPut(runInfo.tiling_data, runParams.srcLeftLpUnit);
  ByteBufferPut(runInfo.tiling_data, runParams.srcLeftStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcLeftStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.srcLeftLpStepIn);
  ByteBufferPut(runInfo.tiling_data, runParams.srcLeftLpStepOut);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndIn0Size);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndIn0SrcRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndIn0SrcAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndIn1Size);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndIn1SrcRsize);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndIn1SrcAsize);
  ByteBufferPut(runInfo.tiling_data, runParams.dstR2ndDims);
  ByteBufferPut(runInfo.tiling_data, runParams.vncColSize);
  ByteBufferPut(runInfo.tiling_data, runParams.allR2ndIn);
}

void PrintTilingModeTc201Params(const std::string& opType, const TransDataTc201Param& params) {
  OP_LOGD(opType.c_str(), "tilingMode=%d", params.tilingMode);
  OP_LOGD(opType.c_str(), "ubOffset=%d", params.ubOffset);
  OP_LOGD(opType.c_str(), "mcPos=%d", params.mcPos);
  OP_LOGD(opType.c_str(), "usedCoreCnt=%d", params.usedCoreCnt);
  OP_LOGD(opType.c_str(), "srcR2ndDstR2ndSame=%d", params.srcR2ndDstR2ndSame);
  OP_LOGD(opType.c_str(), "c0Len=%d", params.c0Len);
  OP_LOGD(opType.c_str(), "coreStepIn=%d", params.coreStepIn);
  OP_LOGD(opType.c_str(), "coreStepOut=%d", params.coreStepOut);
  OP_LOGD(opType.c_str(), "nlcDstR2ndLpCnt=%d", params.nlcDstR2ndLpCnt);
  OP_LOGD(opType.c_str(), "nlcSrcClLpCnt=%d", params.nlcSrcClLpCnt);
  OP_LOGD(opType.c_str(), "nlcSrcLeftLpCnt=%d", params.nlcSrcLeftLpCnt);
  OP_LOGD(opType.c_str(), "nlcDstR2ndLeft=%d", params.nlcDstR2ndLeft);
  OP_LOGD(opType.c_str(), "nlcSrcClLeft=%d", params.nlcSrcClLeft);
  OP_LOGD(opType.c_str(), "nlcSrcLeftLeft=%d", params.nlcSrcLeftLeft);
  OP_LOGD(opType.c_str(), "lcDstR2ndLpCnt=%d", params.lcDstR2ndLpCnt);
  OP_LOGD(opType.c_str(), "lcSrcClLpCnt=%d", params.lcSrcClLpCnt);
  OP_LOGD(opType.c_str(), "lcSrcLeftLpCnt=%d", params.lcSrcLeftLpCnt);
  OP_LOGD(opType.c_str(), "lcDstR2ndLeft=%d", params.lcDstR2ndLeft);
  OP_LOGD(opType.c_str(), "lcSrcClLeft=%d", params.lcSrcClLeft);
  OP_LOGD(opType.c_str(), "lcSrcLeftLeft=%d", params.lcSrcLeftLeft);
  OP_LOGD(opType.c_str(), "dstR2ndLpUnit=%d", params.dstR2ndLpUnit);
  OP_LOGD(opType.c_str(), "dstR2ndStepIn=%d", params.dstR2ndStepIn);
  OP_LOGD(opType.c_str(), "dstR2ndStepOut=%d", params.dstR2ndStepOut);
  OP_LOGD(opType.c_str(), "dstR2ndLpStepIn=%d", params.dstR2ndLpStepIn);
  OP_LOGD(opType.c_str(), "dstR2ndLpStepOut=%d", params.dstR2ndLpStepOut);
  OP_LOGD(opType.c_str(), "srcClLpUnit=%d", params.srcClLpUnit);
  OP_LOGD(opType.c_str(), "allCIn=%d", params.allCIn);
  OP_LOGD(opType.c_str(), "srcClStepIn=%d", params.srcClStepIn);
  OP_LOGD(opType.c_str(), "srcClStepOut=%d", params.srcClStepOut);
  OP_LOGD(opType.c_str(), "srcClLpStepIn=%d", params.srcClLpStepIn);
  OP_LOGD(opType.c_str(), "srcClLpStepOut=%d", params.srcClLpStepOut);
  OP_LOGD(opType.c_str(), "cModC0=%d", params.cModC0);
  OP_LOGD(opType.c_str(), "srcLeftLpUnit=%d", params.srcLeftLpUnit);
  OP_LOGD(opType.c_str(), "srcLeftStepIn=%d", params.srcLeftStepIn);
  OP_LOGD(opType.c_str(), "srcLeftStepOut=%d", params.srcLeftStepOut);
  OP_LOGD(opType.c_str(), "srcLeftLpStepIn=%d", params.srcLeftLpStepIn);
  OP_LOGD(opType.c_str(), "srcLeftLpStepOut=%d", params.srcLeftLpStepOut);
  OP_LOGD(opType.c_str(), "dstR2ndIn0Size=%d", params.dstR2ndIn0Size);
  OP_LOGD(opType.c_str(), "dstR2ndIn0SrcRsize=%d", params.dstR2ndIn0SrcRsize);
  OP_LOGD(opType.c_str(), "dstR2ndIn0SrcAsize=%d", params.dstR2ndIn0SrcAsize);
  OP_LOGD(opType.c_str(), "dstR2ndIn1Size=%d", params.dstR2ndIn1Size);
  OP_LOGD(opType.c_str(), "dstR2ndIn1SrcRsize=%d", params.dstR2ndIn1SrcRsize);
  OP_LOGD(opType.c_str(), "dstR2ndIn1SrcAsize=%d", params.dstR2ndIn1SrcAsize);
  OP_LOGD(opType.c_str(), "dstR2ndDims=%d", params.dstR2ndDims);
  OP_LOGD(opType.c_str(), "vncColSize=%d", params.vncColSize);
  OP_LOGD(opType.c_str(), "allR2ndIn=%d", params.allR2ndIn);
}

}  // namespace optiling
