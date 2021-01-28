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
 * \file trans_data_common.cc
 * \brief dynamic TransData common function
 */

#ifndef __TRANSDATA_H__
#define __TRANSDATA_H__

#include <string>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"

namespace optiling {

const int64_t BLOCK_BYTE_SIZE = 32;
const int64_t VNC_LINES = 16;
const int64_t NI_16 = 16;
const int64_t STRIDE_LIMIT_MTE = 65535;


struct HeadTilingParam {
  int64_t shapeLoopCnt;
};

struct TransDataMode100Param {
  int64_t tillingParamCount;

  int64_t tilingMode;
  int64_t ubOffset;
  int64_t usedCoreCnt;
  int64_t coreStepIn;
  int64_t coreStepOut;
  int64_t oneLineSize;

  int64_t inLevel2C1LpStepIn;
  int64_t inLevel2C1LpStepOut;
  int64_t inLevel1LastLpStepIn;
  int64_t inLevel1LastLpStepOut;
  int64_t inLevel0LpCnt;
  int64_t inLevel0C0LpStepUb;
  int64_t inLevel0LpStepIn;
  int64_t inLevel0RepeatCnt;
  int64_t inLevel0Nburst;
  int64_t inLevel0SrcStride;
  int64_t inLevel0DstStride;
  int64_t outLevel1LpStepUb;
  int64_t outLevel1LpStepOut;
  int64_t outLevel0LpCnt;
  int64_t outLevel0LpStepUb;
  int64_t outLevel0LpStepOut;
  int64_t outLevel0RepeatCnt;
  int64_t outLevel0Nburst;
  int64_t outLevel0SrcStride;
  int64_t outLevel0DstStride;

  int64_t inLevel2NlcC1LpCnt;
  int64_t inLevel2NlcC1LeftLines;
  int64_t inLevel1NlcLastLpCnt;
  int64_t inLevel1NlcLastLeftLines;
  int64_t inLevel0NlcLpCnt;
  int64_t inLevel0NlcRepeatCnt;
  int64_t inLevel0NlcNburst;
  int64_t outLevel1NlcLpCnt;
  int64_t outLevel0NlcLpCnt;
  int64_t outLevel0NlcRepeatCnt;
  int64_t outLevel0NlcNburst;

  int64_t inLevel2LcC1LpCnt;
  int64_t inLevel2LcC1LeftLines;
  int64_t inLevel1LcLastLpCnt;
  int64_t inLevel1LcLastLeftLines;
  int64_t inLevel0LcLpCnt;
  int64_t inLevel0LcRepeatCnt;
  int64_t inLevel0LcNburst;
  int64_t outLevel1LcLpCnt;
  int64_t outLevel0LcLpCnt;
  int64_t outLevel0LcRepeatCnt;
  int64_t outLevel0LcNburst;

  int64_t inLevelx1LpStepIn;
  int64_t inLevelx1LpStepOut;
  int64_t inLevelx1NlcLpCnt;
  int64_t inLevelx1NlcLeftLines;
  int64_t inLevelx1LcLpCnt;
  int64_t inLevelx1LcLeftLines;
  int64_t inLevelx2LpStepIn;
  int64_t inLevelx2LpStepOut;
  int64_t inLevelx2NlcLpCnt;
  int64_t inLevelx2NlcLeftLines;
  int64_t inLevelx2LcLpCnt;
  int64_t inLevelx2LcLeftLines;
  int64_t inLevelx3LpStepIn;
  int64_t inLevelx3LpStepOut;
  int64_t inLevelx3NlcLpCnt;
  int64_t inLevelx3NlcLeftLines;
  int64_t inLevelx3LcLpCnt;
  int64_t inLevelx3LcLeftLines;

  int64_t nextShapeLpOffsetOut;
  int64_t nextShapeLpOffsetIn;
};

struct TransDataMode101Param {
  int64_t tillingParamCount;

  int64_t tilingMode;
  int64_t ubOffset;
  int64_t usedCoreCnt;
  int64_t coreStepIn;
  int64_t coreStepOut;
  int64_t oneLineSize;

  int64_t inLevel2HLpStepIn;
  int64_t inLevel2HLpStepOut;
  int64_t inLevel1CLpStepIn;
  int64_t inLevel1CLpStepOut;
  int64_t inLevel0LpCnt;
  int64_t inLevel0LpStepUb;
  int64_t inLevel0LpStepIn;
  int64_t inLevel0CCnt;
  int64_t inLevel0SubCSize;
  int64_t inLevel0PerLineData;
  int64_t inLevel0RepeatCnt;
  int64_t inLevel0Nburst;
  int64_t inLevel0SrcStride;
  int64_t inLevel0DstStride;
  int64_t outLevel1C1LpCnt;
  int64_t outLevel1LpStepOut;
  int64_t outLevel0NLpCnt;
  int64_t outLevel0LpStepUb;
  int64_t outLevel0LpStepOut;
  int64_t outLevel0RepeatCnt;
  int64_t outLevel0Nburst;
  int64_t outLevel0SrcStride;
  int64_t outLevel0DstStride;

  int64_t inLevel2NlcHLpCnt;
  int64_t inLevel2NlcHLeftLines;
  int64_t inLevel1NlcCLpCnt;
  int64_t inLevel1NlcLeftLines;
  int64_t inLevel0NlcLpCnt;
  int64_t inLevel0NlcCCnt;
  int64_t inLevel0NlcSubCSize;
  int64_t inLevel0NlcPerLineData;
  int64_t inLevel0NlcRepeatCnt;
  int64_t inLevel0NlcNburstNt;
  int64_t inLevel0NlcNburstT;
  int64_t outLevel1NlcC1LpCnt;
  int64_t outLevel0NlcNLpCnt;
  int64_t outLevel0NlcRepeatCnt;
  int64_t outLevel0NlcNburst;

  int64_t inLevel2LcHLpCnt;
  int64_t inLevel2LcHLeftLines;
  int64_t inLevel1LcCLpCnt;
  int64_t inLevel1LcLeftLines;
  int64_t inLevel0LcLpCnt;
  int64_t inLevel0LcCCnt;
  int64_t inLevel0LcSubCSize;
  int64_t inLevel0LcPerLineData;
  int64_t inLevel0LcRepeatCnt;
  int64_t inLevel0LcNburstNt;
  int64_t inLevel0LcNburstT;
  int64_t outLevel1LcC1LpCnt;
  int64_t outLevel0LcNLpCnt;
  int64_t outLevel0LcRepeatCnt;
  int64_t outLevel0LcNburst;

  int64_t inLevelx1LpStepIn;
  int64_t inLevelx1LpStepOut;
  int64_t inLevelx1NlcLpCnt;
  int64_t inLevelx1NlcLeftLines;
  int64_t inLevelx1LcLpCnt;
  int64_t inLevelx1LcLeftLines;
  int64_t inLevelx2LpStepIn;
  int64_t inLevelx2LpStepOut;
  int64_t inLevelx2NlcLpCnt;
  int64_t inLevelx2NlcLeftLines;
  int64_t inLevelx2LcLpCnt;
  int64_t inLevelx2LcLeftLines;
  int64_t inLevelx3LpStepIn;
  int64_t inLevelx3LpStepOut;
  int64_t inLevelx3NlcLpCnt;
  int64_t inLevelx3NlcLeftLines;
  int64_t inLevelx3LcLpCnt;
  int64_t inLevelx3LcLeftLines;
};

                    
                     
struct TransDataMode200Param {
  int64_t tilingMode;
  int64_t ubOffset;
  int64_t tmpUbOffset;
  int64_t mcPos;
  int64_t usedCoreCnt;
  int64_t coreStepIn;
  int64_t coreStepOut;
  int64_t nlcCrLpCnt;
  int64_t nlcCLpCnt;
  int64_t nlcC1LpCnt;
  int64_t nlcCrLeft;
  int64_t nlcCLeft;
  int64_t nlcC1Left;
  int64_t lcCrLpCnt;
  int64_t lcCLpCnt;
  int64_t lcC1LpCnt;
  int64_t lcCrLeft;
  int64_t lcCLeft;
  int64_t lcC1Left;
  int64_t srcCrLpUnit;
  int64_t srcCLpUnit;
  int64_t srcC1LpUnit;
  int64_t srcCrStepOut;
  int64_t srcCStepIn;
  int64_t srcCStepOut;
  int64_t srcC1StepOut;
  int64_t srcCLpStepIn;
  int64_t srcCLpStepOut;
  int64_t cModC0;
  int64_t isMcCr;
  int64_t isMcC1;
  int64_t src2dstFlag;
  int64_t oneLineSize;

  int64_t inIdx0Size;
  int64_t inIdx0dstRsize;
  int64_t inIdx0SrcAsize;
  int64_t inIdx1Size;
  int64_t inIdx1dstRsize;
  int64_t inIdx1SrcAsize;

  int64_t crInIdx0Size;
  int64_t crInIdx0dstRsize;
  int64_t crInIdx0SrcAsize;
  int64_t crInIdx1Size;
  int64_t crInIdx1dstRsize;
  int64_t crInIdx1SrcAsize;
};

struct TransDataMode201Param {
  int64_t tilingMode;
  int64_t ubOffset;
  /**
   * mcFlag, usedCoreCnt, coreStepIn, coreStepOut,
   * nlcR2ndLpCnt, nlcC1LpCnt, nlcLeftLpCnt, nlcR2ndLeft, nlcC1Left,
   * lcR2ndLpCnt, lcC1LpCnt, lcLeftLpCnt, lcR2ndLeft, lcC1Left
  **/
  std::vector<int64_t> mcParams;
  int64_t srcR2ndLpUnit;
  int64_t srcR2ndLpStepIn;
  int64_t srcR2ndLpStepOut;
  int64_t srcC1StepIn;
  int64_t srcC1LpUnit;
  int64_t srcC1LpStepIn;
  int64_t srcC1LpStepOut;
  int64_t perLineDstCCount;
  int64_t cModC0;
  /**
   * inIdx0Size, inIdx0DstRSize, inIdx0SrcASize, inIdx1Size, inIdx1DstRSize, inIdx1SrcASize, inIdx2Size,
   * inIdx2DstRSize, inIdx2SrcASize, outIdx0Size, outIdx0DstRSize, outIdx0DstASize, outIdx1Size, outIdx1DstRSize,
   * outIdx1DstASize, outIdx2Size, outIdx2DstRSize, outIdx2DstASize
  **/
  std::vector<int64_t> ioOrderParams;
  int64_t src2DstFlag;
};

static int64_t GetFloorDiv(const int64_t uValue, const int64_t dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    return uValue;
  }

  resValue = uValue / dValue;

  return resValue;
}

static int64_t GetCeilFill(const int64_t uValue, const int64_t dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    return uValue;
  }

  resValue = (uValue + dValue - 1) / dValue * dValue;

  return resValue;
}

static int64_t GetCeilDiv(const int64_t uValue, const int64_t dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    return uValue;
  }

  resValue = (uValue + dValue - 1) / dValue;

  return resValue;
}

static int64_t GetShapeSize(std::vector<int64_t> inShape, int32_t pos) {
  int32_t n = inShape.size();
  int64_t shapeSize = 1;
  if (pos < 0) {
    pos = n + pos;
  }
  for (int32_t i = pos; i < n; i++) {
    shapeSize *= inShape[i];
  }
  return shapeSize;
}

static bool CalcMcTilingParams(int32_t multiCoreAxisPos, int64_t multiCoreAxisSize, int32_t shapeLen, int32_t axisPosC,
                        int64_t c0Len, int64_t coreNum, vector<int64_t> outShape, std::string dstFormat,
                        std::string srcFormat, int64_t blockElemCnt, vector<int64_t> inShape, int64_t& usedCoreCnt,
                        int64_t& coreStepIn, int64_t& coreStepOut, int64_t& nlcAxisMcSize, int64_t& lcAxisMcSize) {
  if (multiCoreAxisPos == axisPosC && multiCoreAxisPos != shapeLen - 1) {
    int64_t c0CntInC = GetCeilDiv(multiCoreAxisSize, c0Len);
    usedCoreCnt = GetCeilDiv(c0CntInC, GetCeilDiv(c0CntInC, coreNum));
    int64_t nlcC1Cnt = GetCeilDiv(c0CntInC, usedCoreCnt);
    nlcAxisMcSize = nlcC1Cnt * c0Len;
    lcAxisMcSize = multiCoreAxisSize - (usedCoreCnt - 1) * nlcAxisMcSize;
    coreStepIn = GetShapeSize(inShape, axisPosC + 1) * nlcAxisMcSize;
    coreStepOut = GetShapeSize(outShape, std::strchr(dstFormat.c_str(), 'C') - dstFormat.c_str() + 1) * nlcC1Cnt;
  } else if (multiCoreAxisPos == shapeLen - 1) {
    int64_t axisLast8BlockCnt = GetCeilDiv(multiCoreAxisSize, 8 * blockElemCnt);
    usedCoreCnt = GetCeilDiv(axisLast8BlockCnt, GetCeilDiv(axisLast8BlockCnt, coreNum));
    int64_t nlc8BlockCnt = GetCeilDiv(axisLast8BlockCnt, usedCoreCnt);

    nlcAxisMcSize = nlc8BlockCnt * 8 * blockElemCnt;
    lcAxisMcSize = multiCoreAxisSize - (usedCoreCnt - 1) * nlcAxisMcSize;
    int32_t axisMcDstPos = std::strchr(dstFormat.c_str(), srcFormat[multiCoreAxisPos]) - dstFormat.c_str();
    coreStepIn = GetShapeSize(inShape, -1) * nlcAxisMcSize;
    if (srcFormat[multiCoreAxisPos] == 'C') {
      coreStepOut = GetShapeSize(outShape, axisMcDstPos + 1) * nlcAxisMcSize / 16;
    } else {
      coreStepOut = GetShapeSize(outShape, axisMcDstPos + 1) * nlcAxisMcSize;
    }
  } else {
    usedCoreCnt = GetCeilDiv(multiCoreAxisSize, GetCeilDiv(multiCoreAxisSize, coreNum));
    nlcAxisMcSize = GetCeilDiv(multiCoreAxisSize, usedCoreCnt);
    lcAxisMcSize = multiCoreAxisSize - (usedCoreCnt - 1) * nlcAxisMcSize;
    int32_t axisMcDstPos = std::strchr(dstFormat.c_str(), srcFormat[multiCoreAxisPos]) - dstFormat.c_str();
    coreStepIn = GetShapeSize(inShape, multiCoreAxisPos + 1) * nlcAxisMcSize;
    coreStepOut = GetShapeSize(outShape, axisMcDstPos + 1) * nlcAxisMcSize;
  }
  return true;
}
bool TillingPositiveMode100(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, int32_t& multiCoreAxisPos, int32_t& axisPosC, int64_t& coreNum,
                            int64_t& blockElemCnt, int64_t& c0Len, int64_t& ubSize, TransDataMode100Param& params);

bool TillingPositiveMode101(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, int32_t& multiCoreAxisPos, int32_t& axisPosC, int64_t& coreNum,
                            int64_t& blockElemCnt, int64_t& c0Len, int64_t& ubSize, TransDataMode101Param& params);

bool TillingPositiveMode200(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt, int64_t& c0Len,
                            int64_t& ubSize, TransDataMode200Param& params);

bool TillingNegativeMode201(std::vector<int64_t>& inShape, std::vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, const int64_t coreNum, const int64_t blockElemCnt, const int64_t ubSize,
                            TransDataMode201Param& params);

void SetRunningMode100Params(const TransDataMode100Param& runParams, OpRunInfo& runInfo);
void SetRunningMode101Params(const TransDataMode101Param& runParams, OpRunInfo& runInfo);
void SetRunningMode200Params(const TransDataMode200Param& runParams, OpRunInfo& runInfo);
void SetRunningMode201Params(const TransDataMode201Param& runParams, OpRunInfo& runInfo);

void PrintTilingMode100Params(const std::string& opType, const TransDataMode100Param& params);
void PrintTilingMode101Params(const std::string& opType, const TransDataMode101Param& params);
void PrintTilingMode200Params(const std::string& opType, const TransDataMode200Param& params);
void PrintTilingMode201Params(const std::string& opType, const TransDataMode201Param& params);


}  // namespace optiling

#endif  // __TRANS_DATA_H__