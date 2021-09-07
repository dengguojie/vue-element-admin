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
 * \file trans_data_common.h
 * \brief dynamic TransData common function
 */

#ifndef __TRANSDATA_H__
#define __TRANSDATA_H__

#include <string>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"

namespace optiling {
using namespace ge;

const int64_t BLOCK_BYTE_SIZE = 32;
const int64_t VNC_LINES = 16;
const int64_t NI_16 = 16;
const int64_t C0_16 = 16;
const int64_t C0_32 = 32;
const int64_t CUBE_SIZE = 16;
const int64_t STRIDE_LIMIT_MTE = 65535;
const vector<int64_t> PAD_IDX_LIST = {0, 1};

const std::map<ge::Format, int64_t> HW_IDX_MAP = {{FORMAT_NCHW, 2}, {FORMAT_NHWC, 1},  {FORMAT_NCDHW, 3},
                                                  {FORMAT_HWCN, 0}, {FORMAT_DHWCN, 1}, {FORMAT_NDHWC, 2}};
const std::map<ge::Format, int64_t> C_IDX_MAP = {{FORMAT_NCHW, 1}, {FORMAT_NHWC, 3},  {FORMAT_NCDHW, 1},
                                                 {FORMAT_HWCN, 2}, {FORMAT_DHWCN, 3}, {FORMAT_NDHWC, 4}};

const std::map<ge::Format, int64_t> N_IDX_MAP = {{FORMAT_NCHW, 0}, {FORMAT_NHWC, 0},  {FORMAT_NCDHW, 0},
                                                 {FORMAT_HWCN, 3}, {FORMAT_DHWCN, 4}, {FORMAT_NDHWC, 0}};
struct HeadTilingParam {
  int64_t shapeLoopCnt;
};

struct TransDataMode100Param {
  int64_t tilingMode;
  int64_t ubOffset;
  int64_t mcFlag;
  int64_t usedCoreCnt;
  int64_t coreStepIn;
  int64_t coreStepOut;
  int64_t nlcCrLpCnt;
  int64_t nlcCLpCnt;
  int64_t nlcLeftLpCnt;
  int64_t nlcCrLeft;
  int64_t nlcCLeft;
  int64_t lcCrLpCnt;
  int64_t lcCLpCnt;
  int64_t lcLeftLpCnt;
  int64_t lcCrLeft;
  int64_t lcCLeft;

  int64_t srcCrLpUnit;
  int64_t srcCrLpStepIn;
  int64_t srcCrLpStepOut;
  int64_t srcCStepIn;
  int64_t srcCLpUnit;
  int64_t srcCLpStepIn;
  int64_t srcCLpStepOut;
  int64_t cModC0;
  int64_t inIdx0Size;
  int64_t inIdx0DstRsize;
  int64_t inIdx0SrcAsize;
  int64_t inIdx1Size;
  int64_t inIdx1DstRsize;
  int64_t inIdx1SrcAsize;
  int64_t outIdx0Size;
  int64_t outIdx0DstRsize;
  int64_t outIdx0DstAsize;
  int64_t outIdx1Size;
  int64_t outIdx1DstRsize;
  int64_t outIdx1DstAsize;
  int64_t crOutIdx0Size;
  int64_t crOutIdx0DstRsize;
  int64_t crOutIdx0DstAsize;
  int64_t crOutIdx1Size;
  int64_t crOutIdx1DstRsize;
  int64_t crOutIdx1DstAsize;

  int64_t src2dstFlag;
  int64_t oneLineSize;
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

struct TransDataMode1010Param {
  int64_t tilingMode;
  int64_t ubOffset;
  int64_t usedCoreCnt;
  int64_t coreStepIn;
  int64_t coreStepOut;
  int64_t dstClLpStepIn;
  int64_t dstClLpStepOut;
  int64_t dstClStepIn;
  int64_t dstClStepOut;
  int64_t dstCrLpStepIn;
  int64_t dstCrLpStepOut;
  int64_t dstCrStepIn;
  int64_t ncLeVCol;
  int64_t vncLineSize;
  int64_t plnDstClSize;
  int64_t plnDstCrSize;
  int64_t vncRowSize;
  int64_t cLpStepIn;
  int64_t cLpStepOut;
  int64_t cStepOut;
  int64_t c0Size;
  int64_t cModC0;
  int64_t cLpUnit;
  int64_t nlcDstClLpCnt;
  int64_t nlcVncRowClLeft;
  int64_t nlcLastLineClCnt;
  int64_t nlcDstCrLpCnt;
  int64_t nlcVncRowLeft;
  int64_t nlcLastLineCrCnt;
  int64_t nlcCLpCnt;
  int64_t nlcCLeft;
  int64_t lcDstClLpCnt;
  int64_t lcVncRowClLeft;
  int64_t lcLastLineClCnt;
  int64_t lcDstCrLpCnt;
  int64_t lcVncRowLeft;
  int64_t lcLastLineCrCnt;
  int64_t lcCLpCnt;
  int64_t lcCLeft;
};

struct TransDataMode1011Param {
  int64_t tilingMode;
  int64_t ubOffset;
  int64_t usedCoreCnt;
  int64_t mcOnCl;
  int64_t coreStepIn;
  int64_t coreStepOut;
  int64_t dstR2ndLpStepIn;
  int64_t dstR2ndLpStepOut;
  int64_t dstR2ndStepIn;
  int64_t dstR2ndLpUnit;
  int64_t srcClLpStepIn;
  int64_t vncLineSize;
  int64_t srcClLpUnit;
  int64_t srcClLpStepOut;
  int64_t cLpStepIn;
  int64_t cLpStepOut;
  int64_t cStepOut;
  int64_t c0Size;
  int64_t cModC0;
  int64_t cLpUnit;
  int64_t nlcDstR2ndLpCnt;
  int64_t nlcDstR2ndLeft;
  int64_t nlcSrcClLpCnt;
  int64_t nlcSrcClLeft;
  int64_t nlcCLpCnt;
  int64_t nlcCLeft;
  int64_t lcDstR2ndLpCnt;
  int64_t lcDstR2ndLeft;
  int64_t lcSrcClLpCnt;
  int64_t lcSrcClLeft;
  int64_t lcCLpCnt;
  int64_t lcCLeft;
  int64_t clOut0Size;
  int64_t clOut0SrcRsize;
  int64_t clOut0DstAsize;
  int64_t clOut1Size;
  int64_t clOut1SrcRsize;
  int64_t clOut1DstAsize;
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
  int64_t inIdx0DstRsize;
  int64_t inIdx0SrcAsize;
  int64_t inIdx1Size;
  int64_t inIdx1DstRsize;
  int64_t inIdx1SrcAsize;

  int64_t crInIdx0Size;
  int64_t crInIdx0DstRsize;
  int64_t crInIdx0SrcAsize;
  int64_t crInIdx1Size;
  int64_t crInIdx1DstRsize;
  int64_t crInIdx1SrcAsize;
};

struct TransDataNtc200Param {
  int64_t tilingMode;
  int64_t ubOffset;
  int64_t mcPos;
  int64_t usedCoreCnt;
  int64_t c0Len;
  int64_t coreStepIn;
  int64_t coreStepOut;
  int64_t nlcCrLpCnt;
  int64_t nlcCLpCnt;
  int64_t nlcClLpCnt;
  int64_t nlcCrLeft;
  int64_t nlcCLeft;
  int64_t nlcClLeft;
  int64_t lcCrLpCnt;
  int64_t lcCLpCnt;
  int64_t lcClLpCnt;
  int64_t lcCrLeft;
  int64_t lcCLeft;
  int64_t lcClLeft;
  int64_t dstCrLpUnit;
  int64_t srcCLpUnit;
  int64_t dstClLpUnit;
  int64_t dstCrStepIn;
  int64_t dstCrStepOut;
  int64_t dstCrLpStepIn;
  int64_t dstCrLpStepOut;
  int64_t dstCSize;
  int64_t srcCStepIn;
  int64_t srcCStepOut;
  int64_t srcCLpStepIn;
  int64_t srcCLpStepOut;
  int64_t dstClStepIn;
  int64_t dstClStepOut;
  int64_t dstClLpStepIn;
  int64_t dstClLpStepOut;
  int64_t cModC0;
  int64_t dstCrDims;
  int64_t dstClDims;
  int64_t isMcCr;
  int64_t isMcCl;
  int64_t srcR2ndDstR1stSame;
  int64_t leftClCCrSize;

  int64_t clInIdx0Size;
  int64_t clInIdx0DstRsize;
  int64_t clInIdx0SrcAsize;
  int64_t clInIdx1Size;
  int64_t clInIdx1DstRsize;
  int64_t clInIdx1SrcAsize;
  int64_t crInIdx0Size;
  int64_t crInIdx0DstRsize;
  int64_t crInIdx0SrcAsize;
  int64_t crInIdx1Size;
  int64_t crInIdx1DstRsize;
  int64_t crInIdx1SrcAsize;
};

struct TransDataTc201Param {
  int64_t tilingMode;
  int64_t ubOffset;
  int64_t mcPos;
  int64_t usedCoreCnt;
  int64_t srcR2ndDstR2ndSame;
  int64_t c0Len;
  int64_t coreStepIn;
  int64_t coreStepOut;
  int64_t nlcDstR2ndLpCnt;
  int64_t nlcSrcClLpCnt;
  int64_t nlcSrcLeftLpCnt;
  int64_t nlcDstR2ndLeft;
  int64_t nlcSrcClLeft;
  int64_t nlcSrcLeftLeft;
  int64_t lcDstR2ndLpCnt;
  int64_t lcSrcClLpCnt;
  int64_t lcSrcLeftLpCnt;
  int64_t lcDstR2ndLeft;
  int64_t lcSrcClLeft;
  int64_t lcSrcLeftLeft;
  int64_t dstR2ndLpUnit;
  int64_t dstR2ndStepIn;
  int64_t dstR2ndStepOut;
  int64_t dstR2ndLpStepIn;
  int64_t dstR2ndLpStepOut;
  int64_t srcClLpUnit;
  int64_t allCIn;
  int64_t srcClStepIn;
  int64_t srcClStepOut;
  int64_t srcClLpStepIn;
  int64_t srcClLpStepOut;
  int64_t cModC0;
  int64_t srcLeftLpUnit;
  int64_t srcLeftStepIn;
  int64_t srcLeftStepOut;
  int64_t srcLeftLpStepIn;
  int64_t srcLeftLpStepOut;
  int64_t dstR2ndIn0Size;
  int64_t dstR2ndIn0SrcRsize;
  int64_t dstR2ndIn0SrcAsize;
  int64_t dstR2ndIn1Size;
  int64_t dstR2ndIn1SrcRsize;
  int64_t dstR2ndIn1SrcAsize;
  int64_t dstR2ndDims;
  int64_t vncColSize;
  int64_t allR2ndIn;
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

struct TransDataNtc100Param {
  int64_t tilingMode;
  int64_t ubOffset;
  /**
   * mcPos, usedCoreCnt, coreStepIn, coreStepOut
   **/
  std::vector<int64_t> coreParams;
  int64_t vncLineSize;
  int64_t cModC0;
  int64_t c0Size;
  int64_t clDims;
  int64_t crDims;
  int64_t r1stSrcR2ndDstSame;
  int64_t srcClStepIn;
  int64_t srcClStepOut;
  int64_t srcClLpUnit;
  int64_t srcClLpStepIn;
  int64_t srcClLpStepOut;
  int64_t srcCStepIn;
  int64_t srcCLpUnit;
  int64_t srcCLpStepIn;
  int64_t srcCLpStepOut;
  int64_t srcCrStepIn;
  int64_t srcCrStepOut;
  int64_t srcCrLpUnit;
  int64_t srcCrLpStepIn;
  int64_t srcCrLpStepOut;
  /**
   * nlcClLpCnt, nlcClLeft, nlcCLpCnt,nlcCLeft,nlcCrLpCnt, nlcCrLeft,
   * lcClLpCnt, lcClLeft, lcCLpCnt,lcCLeft, lcCrLpCnt,lcCrLeft
   **/
  std::vector<int64_t> lcParams;
  int64_t clOutIdx0Size;
  int64_t clOutIdx0DstRSize;
  int64_t clOutIdx0DstASize;
  int64_t clOutIdx1Size;
  int64_t clOutIdx1DstRSize;
  int64_t clOutIdx1DstASize;
  int64_t crOutIdx0Size;
  int64_t crOutIdx0DstRSize;
  int64_t crOutIdx0DstASize;
  int64_t crOutIdx1Size;
  int64_t crOutIdx1DstRSize;
  int64_t crOutIdx1DstASize;

  std::string to_string() const {
    std::string result = "tilingMode:" + std::to_string(tilingMode);
    result += " ubOffset:" + std::to_string(ubOffset);
    result += " mcPos:" + std::to_string(coreParams[0]);
    result += " usedCoreCnt:" + std::to_string(coreParams[1]);
    result += " coreStepIn:" + std::to_string(coreParams[2]);
    result += " coreStepOut:" + std::to_string(coreParams[3]);
    result += " vncLineSize:" + std::to_string(vncLineSize);
    result += " cModC0:" + std::to_string(cModC0);
    result += " c0Size:" + std::to_string(c0Size);
    result += " clDims:" + std::to_string(clDims);
    result += " crDims:" + std::to_string(crDims);
    result += " r1stSrcR2ndDstSame:" + std::to_string(r1stSrcR2ndDstSame);
    result += " srcClStepIn:" + std::to_string(srcClStepIn);
    result += " srcClStepOut:" + std::to_string(srcClStepOut);
    result += " srcClLpUnit:" + std::to_string(srcClLpUnit);
    result += " srcClLpStepIn:" + std::to_string(srcClLpStepIn);
    result += " srcClLpStepOut:" + std::to_string(srcClLpStepOut);
    result += " srcCStepIn:" + std::to_string(srcCStepIn);
    result += " srcCLpUnit:" + std::to_string(srcCLpUnit);
    result += " srcCLpStepIn:" + std::to_string(srcCLpStepIn);
    result += " srcCLpStepOut:" + std::to_string(srcCLpStepOut);
    result += " srcCrStepIn:" + std::to_string(srcCrStepIn);
    result += " srcCrStepOut:" + std::to_string(srcCrStepOut);
    result += " srcCrLpUnit:" + std::to_string(srcCrLpUnit);
    result += " srcCrLpStepIn:" + std::to_string(srcCrLpStepIn);
    result += " srcCrLpStepOut:" + std::to_string(srcCrLpStepOut);
    result += " nlcClLpCnt:" + std::to_string(lcParams[0]);
    result += " nlcClLeft:" + std::to_string(lcParams[1]);
    result += " nlcCLpCnt:" + std::to_string(lcParams[2]);
    result += " nlcCLeft:" + std::to_string(lcParams[3]);
    result += " nlcCrLpCnt:" + std::to_string(lcParams[4]);
    result += " nlcCrLeft:" + std::to_string(lcParams[5]);
    result += " lcClLpCnt:" + std::to_string(lcParams[6]);
    result += " lcClLeft:" + std::to_string(lcParams[7]);
    result += " lcCLpCnt:" + std::to_string(lcParams[8]);
    result += " lcCLeft:" + std::to_string(lcParams[9]);
    result += " lcCrLpCnt:" + std::to_string(lcParams[10]);
    result += " lcCrLeft:" + std::to_string(lcParams[11]);
    result += " clOutIdx0Size:" + std::to_string(clOutIdx0Size);
    result += " clOutIdx0DstRSize:" + std::to_string(clOutIdx0DstRSize);
    result += " clOutIdx0DstASize:" + std::to_string(clOutIdx0DstASize);
    result += " clOutIdx1Size:" + std::to_string(clOutIdx1Size);
    result += " clOutIdx1DstRSize:" + std::to_string(clOutIdx1DstRSize);
    result += " clOutIdx1DstASize:" + std::to_string(clOutIdx1DstASize);
    result += " crOutIdx0Size:" + std::to_string(crOutIdx0Size);
    result += " crOutIdx0DstRSize:" + std::to_string(crOutIdx0DstRSize);
    result += " crOutIdx0DstASize:" + std::to_string(crOutIdx0DstASize);
    result += " crOutIdx1Size:" + std::to_string(crOutIdx1Size);
    result += " crOutIdx1DstRSize:" + std::to_string(crOutIdx1DstRSize);
    result += " crOutIdx1DstASize:" + std::to_string(crOutIdx1DstASize);
    return result;
  }
};

static int64_t GetFloorDiv(int64_t uValue, int64_t dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    return uValue;
  }

  resValue = uValue / dValue;

  return resValue;
}

static int64_t GetCeilDiv(int64_t uValue, int64_t dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    return uValue;
  }

  resValue = (uValue + dValue - 1) / dValue;

  return resValue;
}

static int64_t GetIdxFromFormat(const std::map<ge::Format, int64_t> format_map, const ge::Format data_format) {
  auto find_foramt_it = format_map.find(data_format);
  if (find_foramt_it != format_map.end()) {
    return find_foramt_it->second;
  }

  return -1;
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

bool TillingPositiveMode1010(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                             std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt, int64_t& ubSize,
                             TransDataMode1010Param& params);

bool TillingPositiveMode1011(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                             std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt, int64_t& ubSize,
                             TransDataMode1011Param& params);

bool TilingNegativeNtc200(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                          std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt, DataType& dtype,
                          int64_t& ubSize, TransDataNtc200Param& params);

bool TilingNegativeTc201(vector<int64_t>& inShape, vector<int64_t>& outShape, std::string& srcFormat,
                         std::string& dstFormat, int64_t& coreNum, int64_t& blockElemCnt, DataType& dtype,
                         int64_t& ubSize, TransDataTc201Param& params);

bool TilingPositiveSourceNtc100(const vector<int64_t>& in_shape, const vector<int64_t>& out_shape,
                                const ge::Format& src_format, const ge::Format& dst_format, const int64_t& core_num,
                                const int64_t& block_elem_cnt, const int64_t& ub_size, const int64_t& c0Len,
                                const DataType& dType, TransDataNtc100Param& params);

void SetRunningMode1010Params(const TransDataMode1010Param& runParams, utils::OpRunInfo& runInfo);
void SetRunningMode1011Params(const TransDataMode1011Param& runParams, utils::OpRunInfo& runInfo);
void SetRunningNtc200Params(const TransDataNtc200Param& runParams, utils::OpRunInfo& runInfo);
void SetRunningTc201Params(const TransDataTc201Param& runParams, utils::OpRunInfo& runInfo);
void SetRunningNtc100Params(const TransDataNtc100Param& runParams, utils::OpRunInfo& runInfo);

void PrintTilingMode1010Params(const std::string& opType, const TransDataMode1010Param& params);
void PrintTilingMode1011Params(const std::string& opType, const TransDataMode1011Param& params);
void PrintTilingModeNtc200Params(const std::string& opType, const TransDataNtc200Param& params);
void PrintTilingModeTc201Params(const std::string& opType, const TransDataTc201Param& params);
void PrintTilingNtc100Params(const std::string& opType, const TransDataNtc100Param& params);

}  // namespace optiling

#endif  // __TRANS_DATA_H__
