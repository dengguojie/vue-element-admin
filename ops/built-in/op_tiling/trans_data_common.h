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
bool TillingNegativeMode201(std::vector<int64_t>& inShape, std::vector<int64_t>& outShape, std::string& srcFormat,
                            std::string& dstFormat, const int64_t coreNum, const int64_t blockElemCnt, const int64_t ubSize,
                            TransDataMode201Param& params);

void SetRunningMode201Params(const TransDataMode201Param& runParams, OpRunInfo& runInfo);
void PrintTilingMode201Params(const std::string& opType, const TransDataMode201Param& params);


}  // namespace optiling

#endif  // __TRANS_DATA_H__