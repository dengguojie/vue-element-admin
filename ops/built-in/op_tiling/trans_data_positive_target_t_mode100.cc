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

const int32_t FRAME_LEVEL = 6;

bool TillingPositiveMode100(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, int32_t& multiCoreAxisPos, int32_t& axisPosC, int64_t& coreNum,
                            int64_t& blockElemCnt, int64_t& c0Len, int64_t& ubSize, TransDataMode100Param& params) {
  if (srcFormat.length() < 1 || inShape.size() < 2 || outShape.size() < 2) {
    OP_LOGE("op TransDataTiling: TillingPositiveMode100 Failed.");
    return false;
  }
  int32_t shapeLen = inShape.size();
  inShape.push_back(1);
  outShape.push_back(1);

  int64_t halfUbSize = ubSize / 2;
  if (multiCoreAxisPos < 0 || multiCoreAxisPos > inShape.size() - 1) {
    OP_LOGE("op TransDataTiling: TillingPositiveMode100 Failed.");
    return false;
  }
  int64_t multiCoreAxisSize = inShape[multiCoreAxisPos];

  int64_t nlcAxisMcSize;
  int64_t lcAxisMcSize;
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
  int64_t c1PerLpC0Cnt = c0Len;
  params.inLevel2C1LpStepIn = GetShapeSize(inShape, axisPosC + 1) * c1PerLpC0Cnt;
  params.inLevel2C1LpStepOut = GetShapeSize(outShape, std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str() + 1);
  params.inLevel2NlcC1LpCnt = GetCeilDiv(nlcInShape[axisPosC], c1PerLpC0Cnt);
  params.inLevel2NlcC1LeftLines = nlcInShape[axisPosC] % c1PerLpC0Cnt;
  params.inLevel2LcC1LpCnt =  GetCeilDiv(lcInShape[axisPosC], c1PerLpC0Cnt);
  params.inLevel2LcC1LeftLines = lcInShape[axisPosC] % c1PerLpC0Cnt;

  int64_t cRightSize;
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
    int64_t srcLastDstLastGap = GetShapeSize(tempOutShape, 0);
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

  params.nextShapeLpOffsetOut = GetShapeSize(outShape, 0);
  params.nextShapeLpOffsetIn = GetShapeSize(inShape, axisPosC);

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

}  // namespace optiling