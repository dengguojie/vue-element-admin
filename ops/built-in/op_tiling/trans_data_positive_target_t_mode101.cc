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
 * \file trans_data_positive_target_t_mode101.cpp
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

bool TillingPositiveMode101(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, int32_t& multiCoreAxisPos, int32_t& axisPosC, int64_t& coreNum,
                            int64_t& blockElemCnt, int64_t& c0Len, int64_t& ubSize, TransDataMode101Param& params) {
  if (srcFormat.length() < 2 || inShape.size() < 2 || outShape.size() < 2) {
    OP_LOGE("op TransDataTiling: TillingPositiveMode101 Failed.");
    return false;
  }
  int32_t shapeLen = inShape.size();
  inShape.push_back(1);
  outShape.push_back(1);

  int64_t halfUbSize = ubSize / 2;
  if (multiCoreAxisPos < 0 || multiCoreAxisPos > inShape.size() - 1) {
    OP_LOGE("op TransDataTiling: TillingPositiveMode101 Failed.");
    return false;
  }
  int64_t multiCoreAxisSize = inShape[multiCoreAxisPos];

  int64_t nlcAxisMcSize;
  int64_t lcAxisMcSize;
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
  int64_t cCntPerLine;
  if (multiCoreAxisPos == shapeLen - 1) {
    cCntPerLine = 1;
  } else {
    int64_t tmpCnt = params.oneLineSize / GetCeilFill(inShape[inShape.size() - 2], c0Len);
    if (tmpCnt > 0) {
      cCntPerLine = tmpCnt;
    } else {
      cCntPerLine = 1;
    }
  }

  int64_t hStepCnt = cCntPerLine * VNC_LINES;
  params.inLevel2HLpStepIn = GetShapeSize(inShape, -2) * hStepCnt;
  params.inLevel2HLpStepOut = GetShapeSize(outShape, std::strchr(dstFormat.c_str(), rvs2ndChar) - dstFormat.c_str()
                                          + 1) * hStepCnt;
  params.inLevel2NlcHLpCnt = GetCeilDiv(nlcInShape[nlcInShape.size() - 3], hStepCnt);
  params.inLevel2NlcHLeftLines = nlcInShape[nlcInShape.size() - 3] % hStepCnt;
  params.inLevel2LcHLpCnt = GetCeilDiv(lcInShape[lcInShape.size() - 3], hStepCnt);
  params.inLevel2LcHLeftLines = lcInShape[lcInShape.size() - 3] % hStepCnt;

  //--- in-level 1 tiling params, control level 0 data move burst
  char rvs1stChar = srcFormat[srcFormat.length() - 1];
  int64_t c0Cnt = params.oneLineSize / c0Len;
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
    int64_t srcRvs2ndDstLastGap = GetShapeSize(tempOutShape, 0);
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

}  // namespace optiling