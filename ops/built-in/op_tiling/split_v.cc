/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file split_v.cc
 * \brief
 */
#include <string>
#include <securec.h>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

// x is empty tensor
const int32_t TILING_MODE_0 = 0;
// num_split is 1
const int32_t TILING_MODE_1 = 1;
// split axis 0, or shape_before is 1
const int32_t TILING_MODE_2 = 2;
// split axis 0, or shape_before is 1, and shape_dim is greater than core num
const int32_t TILING_MODE_8 = 8;
// split axis is 1, tiling with shape_before
const int32_t TILING_MODE_3 = 3;
// only support fp16, num_split <= 16, and size_splits[i] is 1
const int32_t TILING_MODE_4 = 4;
// size_splits[i] is smaller than 32B, e.g [187264,33], 33->[5,6,7,4,3,2,6]
const int32_t TILING_MODE_5 = 5;
// only split_v, e.g int16,[48000,256], 256->[80,80,80,1,1,1,13]
const int32_t TILING_MODE_6 = 6;
// only split_v, only support fp16, e.g [2028,85], 85->[2,2,1,80]
const int32_t TILING_MODE_7 = 7;

const int32_t TRANSPOSE_SIZE = 256;
constexpr int32_t SPLIT_NUM = 16;
constexpr int32_t ALLIGN_NUM_16 = 16;
constexpr int32_t ALLIGN_NUM_15 = 15;

namespace optiling {
struct SplitVTilingParams {
  int64_t tilingMode;
  int64_t needCoreNum;
  int64_t inputElems;
  int64_t shapeDim;
  int64_t dataEachCore;
  int64_t dataLastCore;
  int64_t loopNum;
  int64_t lastNum;
  int64_t oneLoopElems;
  int64_t loopNumLast;
  int64_t lastNumLast;
  int64_t oneLoopElemsLast;

  int64_t shapeAfterDim;
  int64_t shapeBefore;
  int64_t shapeAfter;
  int64_t multiMove;

  int64_t tailEle;
  int64_t oneCoreSeg;
  int64_t segLoopNum;
  int64_t lastSeg;
  int64_t lastCoreSeg;
  int64_t segLoopNumLastCore;
  int64_t lastSegLastCore;

  // use in split
  int64_t sizeValueSplit;
};

void InitSplitVRunningParams(SplitVTilingParams& params) {
  params.tilingMode = 0;
  params.needCoreNum = 0;
  params.inputElems = 0;
  params.shapeDim = 0;
  params.dataEachCore = 0;
  params.dataLastCore = 0;
  params.loopNum = 0;
  params.lastNum = 0;
  params.oneLoopElems = 0;
  params.loopNumLast = 0;
  params.lastNumLast = 0;
  params.oneLoopElemsLast = 0;

  params.shapeAfterDim = 0;
  params.shapeBefore = 0;
  params.shapeAfter = 0;
  params.multiMove = 0;

  params.tailEle = 0;
  params.oneCoreSeg = 0;
  params.segLoopNum = 0;
  params.lastSeg = 0;
  params.lastCoreSeg = 0;
  params.segLoopNumLastCore = 0;
  params.lastSegLastCore = 0;

  params.sizeValueSplit = 0;
}

int64_t CeilDivCal(const int64_t& uValue, const int64_t& dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitVTiling", "CeilDivCal error, dValue is zero");
    return resValue;
  }

  resValue = (uValue + dValue - 1) / dValue;
  return resValue;
}

bool CheckSizeSplitsSmall(std::vector<int64_t> sizeSplitsVec, int64_t dataBlock, int64_t shapeAfterDim, bool isSplitV) {
  bool ret = true;
  if (isSplitV) {
    for (size_t i = 0; i < sizeSplitsVec.size(); ++i) {
      if (sizeSplitsVec[i] * shapeAfterDim >= dataBlock) {
        ret = false;
        break;
      }
    }
  } else {
    if (sizeSplitsVec[0] * shapeAfterDim >= dataBlock) {
      ret = false;
    }
  }

  return ret;
}

bool CheckSplitVAttr(int64_t splitDim, int64_t numSplit, std::vector<int64_t> inputShape,
                     std::vector<int64_t> sizeSplitsVec) {
  if (sizeSplitsVec.size() != static_cast<size_t>(numSplit)) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitVTiling", "num_split must be equal to the size_splits size");

    return false;
  }

  int64_t sizeSplitsSum = 0;
  int64_t dim = inputShape[splitDim];
  if (std::find(sizeSplitsVec.begin(), sizeSplitsVec.end(), -1) == sizeSplitsVec.end()) {
    sizeSplitsSum = std::accumulate(sizeSplitsVec.begin(), sizeSplitsVec.end(), 0);
    GELOGD("op [SplitVTiling] : CheckSplitVAttr  sizeSplitsSum=%ld, splitDim=%ld, inputShape[%ld]=%ld", sizeSplitsSum,
           splitDim, splitDim, dim);
    if (dim != sizeSplitsSum) {
      VECTOR_INNER_ERR_REPORT_TILIING("SplitVTiling", "The sum of size_splits must be equal to the x shape[split_dim]");
      return false;
    }
  } else {
    int64_t tempIndex = -1;
    for (size_t i = 0; i < sizeSplitsVec.size(); ++i) {
      if (sizeSplitsVec[i] != -1) {
        sizeSplitsSum += sizeSplitsVec[i];
      } else {
        tempIndex = i;
      }
    }
    if (dim != sizeSplitsSum && tempIndex != -1) {
      sizeSplitsVec[tempIndex] = dim - sizeSplitsSum;
    }
  }
  return true;
}

bool CheckMode6(std::vector<int64_t> sizeSplitsVec, int64_t dataBlock, int64_t shapeAfterDim, int64_t shapeAfter) {
  if (shapeAfter % dataBlock != 0 || shapeAfter > 300) {
    return false;
  }

  bool ret = true;
  bool flag = false;
  for (size_t i = 0; i < sizeSplitsVec.size(); ++i) {
    int64_t splitSize = sizeSplitsVec[i] * shapeAfterDim;
    if (splitSize >= dataBlock && splitSize % dataBlock != 0) {
      ret = false;
      break;
    }
    if (splitSize < dataBlock) {
      flag = true;
    }
    if (flag && splitSize >= dataBlock) {
      ret = false;
      break;
    }
  }
  return ret;
}

bool CheckMode7(std::vector<int64_t> sizeSplitsVec, int64_t dataBlock, int64_t shapeAfterDim, int64_t shapeAfter,
                int64_t shapeBefore) {
  if (shapeAfter > 128 || shapeBefore < 256) {
    return false;
  }
  int64_t num = sizeSplitsVec.size();
  if (sizeSplitsVec[num - 1] * shapeAfterDim % dataBlock != 0) {
    return false;
  }

  bool ret = true;
  int64_t splitSizeSum = 0;
  for (int64_t i = 0; i < num - 1; ++i) {
    splitSizeSum += sizeSplitsVec[i] * shapeAfterDim;
  }
  if (splitSizeSum >= dataBlock) {
    ret = false;
  }
  return ret;
}

void GetFrontLoopParams(SplitVTilingParams& runParams, int64_t ubElems, int64_t dataNum, int64_t dataBlock) {
  if (dataNum < ubElems) {
    runParams.oneLoopElems = dataNum;
    runParams.loopNum = 0;
    runParams.lastNum = dataNum;
  } else {
    runParams.oneLoopElems = ubElems;
    runParams.loopNum = dataNum / ubElems;
    runParams.lastNum = dataNum % ubElems;
    if ((runParams.lastNum > 0) && (runParams.lastNum < dataBlock)) {
      runParams.oneLoopElems = ubElems - dataBlock;
      runParams.loopNum = dataNum / runParams.oneLoopElems;
      runParams.lastNum = dataNum % runParams.oneLoopElems;
    }
  }
}

void GetLastLoopParams(SplitVTilingParams& runParams, int64_t ubElems, int64_t dataBlock) {
  if (runParams.dataLastCore < ubElems) {
    runParams.oneLoopElemsLast = runParams.dataLastCore;
    runParams.loopNumLast = 0;
    runParams.lastNumLast = runParams.dataLastCore;
  } else {
    runParams.oneLoopElemsLast = ubElems;
    runParams.loopNumLast = runParams.dataLastCore / ubElems;
    runParams.lastNumLast = runParams.dataLastCore % ubElems;
    if ((runParams.lastNumLast > 0) && (runParams.lastNumLast < dataBlock)) {
      runParams.oneLoopElemsLast = ubElems - dataBlock;
      runParams.loopNumLast = runParams.dataLastCore / runParams.oneLoopElemsLast;
      runParams.lastNumLast = runParams.dataLastCore % runParams.oneLoopElemsLast;
    }
  }
}

bool CheckShapeDim(std::vector<int64_t> sizeSplitsVec, int64_t dataBlock, int64_t coreNum, int64_t shapeAfterDim,
                   int64_t shapeDim, bool isSplitV) {
  if (shapeDim < coreNum || shapeAfterDim / coreNum > (4 * dataBlock)) {
    return false;
  }

  bool ret = true;
  if (isSplitV) {
    int64_t splitSize0 = sizeSplitsVec[0];
    for (size_t i = 0; i < sizeSplitsVec.size(); ++i) {
      if (sizeSplitsVec[i] != splitSize0) {
        ret = false;
        break;
      }
    }
  }

  return ret;
}

void CalSpecialParams(SplitVTilingParams& runParams, int64_t coreNum, int64_t dataBlock, int64_t shapeBefore) {
  int64_t unit = dataBlock;
  int64_t maxRows = 8 * unit;
  if (runParams.tilingMode == TILING_MODE_7) {
    unit = 16 * dataBlock;
    maxRows = unit;
  }

  int64_t unitCount = (shapeBefore + unit - 1) / unit;
  int64_t unitOneCore = (unitCount + coreNum - 1) / coreNum;
  int64_t rowsOneCore = unitOneCore * unit;
  runParams.needCoreNum = (shapeBefore + rowsOneCore - 1) / rowsOneCore;
  int64_t rowsLastCore = shapeBefore - (runParams.needCoreNum - 1) * rowsOneCore;
  runParams.dataEachCore = rowsOneCore;
  runParams.dataLastCore = rowsLastCore;

  runParams.oneLoopElems = maxRows;
  runParams.loopNum = rowsOneCore / maxRows;
  // lastNum is always 0 in mode 7
  runParams.lastNum = rowsOneCore - runParams.loopNum * maxRows;
  runParams.oneLoopElemsLast = maxRows;
  runParams.loopNumLast = rowsLastCore / maxRows;
  runParams.lastNumLast = rowsLastCore - runParams.loopNumLast * maxRows;
}

bool CalSplitVRunningParams(SplitVTilingParams& runParams, int64_t inputElems, std::vector<int64_t> inputShape,
                            int64_t ubElems, int64_t coreNum, int64_t splitDim, int64_t numSplit, int64_t dataBlock,
                            std::vector<int64_t> sizeSplitsVec, ge::DataType inputDType, bool isSplitV) {
  int64_t shapeBefore = 1;
  int64_t shapeAfter = 1;
  int64_t shapeAfterDim = 1;
  int64_t shapeDim = inputShape[splitDim];
  int64_t inputSize = inputShape.size();
  for (int64_t i = 0; i < splitDim; i++) {
    shapeBefore = inputShape[i] * shapeBefore;
  }
  for (int64_t j = splitDim + 1; j < inputSize; j++) {
    shapeAfterDim = inputShape[j] * shapeAfterDim;
  }
  shapeAfter = shapeDim * shapeAfterDim;

  runParams.shapeBefore = shapeBefore;
  runParams.shapeAfter = shapeAfter;
  runParams.shapeAfterDim = shapeAfterDim;
  runParams.inputElems = inputElems;
  runParams.shapeDim = shapeDim;

  if (shapeBefore == 0 || shapeAfter == 0 || shapeAfterDim == 0 || shapeDim == 0) {
    GELOGD("op [SplitVTiling] : mode 0");
    runParams.tilingMode = TILING_MODE_0;
    runParams.needCoreNum = 1;
  } else if (numSplit == 1) {
    GELOGD("op [SplitVTiling] : mode 1");
    runParams.tilingMode = TILING_MODE_1;

    runParams.dataEachCore = CeilDivCal(runParams.inputElems, coreNum);
    runParams.needCoreNum = CeilDivCal(runParams.inputElems, runParams.dataEachCore);
    runParams.dataLastCore = runParams.inputElems - (runParams.needCoreNum - 1) * runParams.dataEachCore;
    if (runParams.dataEachCore < dataBlock) {
      runParams.needCoreNum = 1;
      runParams.dataEachCore = runParams.inputElems;
      runParams.dataLastCore = 0;
    }
    if ((runParams.dataLastCore > 0) && (runParams.dataLastCore < dataBlock)) {
      runParams.needCoreNum = runParams.needCoreNum - 1;
      runParams.dataLastCore = runParams.dataLastCore + runParams.dataEachCore;
    }
    GetFrontLoopParams(runParams, ubElems, runParams.dataEachCore, dataBlock);

    runParams.oneLoopElemsLast = runParams.oneLoopElems;
    runParams.loopNumLast = runParams.loopNum;
    runParams.lastNumLast = runParams.lastNum;
    if (runParams.dataLastCore > 0) {
      GetLastLoopParams(runParams, ubElems, dataBlock);
    }
  } else if (shapeBefore == 1 || splitDim == 0) {
    GELOGD("op [SplitVTiling] : mode 2");
    if (CheckShapeDim(sizeSplitsVec, dataBlock, coreNum, shapeAfterDim, shapeDim, isSplitV)) {
      runParams.tilingMode = TILING_MODE_8;
    } else {
      runParams.tilingMode = TILING_MODE_2;
    }
    runParams.needCoreNum = coreNum;
  } else {
    if (inputDType == ge::DT_FLOAT16 && numSplit <= SPLIT_NUM && shapeAfter == numSplit &&
        inputElems >= TRANSPOSE_SIZE * numSplit) {
      GELOGD("op [SplitVTiling] : mode 4");
      runParams.tilingMode = TILING_MODE_4;

      int64_t alignSeg = (shapeBefore + TRANSPOSE_SIZE - 1) / TRANSPOSE_SIZE;
      int64_t tailEle = alignSeg * TRANSPOSE_SIZE - shapeBefore;
      int64_t oneCoreSeg = (alignSeg + coreNum - 1) / coreNum;
      int64_t actCoreNum = alignSeg / oneCoreSeg;
      if (alignSeg % oneCoreSeg != 0) {
        actCoreNum = actCoreNum + 1;
      }
      int64_t lastCoreSeg = alignSeg - (actCoreNum - 1) * oneCoreSeg;

      int64_t maxSeg = (ubElems / 2) / (TRANSPOSE_SIZE * (2 * numSplit + 2));
      if (maxSeg == 0) {
        VECTOR_INNER_ERR_REPORT_TILIING("SplitVTiling", "mode 4 error, maxSeg is zero");
        return false;
      }
      int64_t segLoopNum = oneCoreSeg / maxSeg;
      int64_t lastSeg = oneCoreSeg % maxSeg;

      if (tailEle != 0) {
        lastCoreSeg = lastCoreSeg - 1;
      }
      if (lastCoreSeg < 0) {
        lastCoreSeg = 0;
      }
      int64_t segLoopNumLastCore = lastCoreSeg / maxSeg;
      int64_t lastSegLastCore = lastCoreSeg % maxSeg;

      runParams.needCoreNum = actCoreNum;
      runParams.tailEle = tailEle;
      runParams.oneCoreSeg = oneCoreSeg;
      runParams.segLoopNum = segLoopNum;
      runParams.lastSeg = lastSeg;
      runParams.lastCoreSeg = lastCoreSeg;
      runParams.segLoopNumLastCore = segLoopNumLastCore;
      runParams.lastSegLastCore = lastSegLastCore;
    } else if (CheckSizeSplitsSmall(sizeSplitsVec, dataBlock, shapeAfterDim, isSplitV)) {
      GELOGD("op [SplitVTiling] : mode 5");
      runParams.tilingMode = TILING_MODE_5;

      runParams.dataEachCore = CeilDivCal(shapeBefore, coreNum);
      if (runParams.dataEachCore < dataBlock) {
        // smaller than 16 row
        runParams.needCoreNum = 1;
        runParams.dataEachCore = 0;
        runParams.dataLastCore = shapeBefore;
      } else {
        // row align
        runParams.dataEachCore = CeilDivCal(runParams.dataEachCore, dataBlock) * dataBlock;
        runParams.needCoreNum = CeilDivCal(shapeBefore, runParams.dataEachCore);
        runParams.dataLastCore = shapeBefore - (runParams.needCoreNum - 1) * runParams.dataEachCore;
      }
      if (runParams.dataLastCore > 0 && runParams.dataLastCore < dataBlock && runParams.needCoreNum > 1) {
        runParams.needCoreNum = runParams.needCoreNum - 1;
        runParams.dataLastCore = runParams.dataLastCore + runParams.dataEachCore;
      }

      int64_t maxOneLoopRows = (ubElems / 2) / shapeAfter;
      // align
      maxOneLoopRows = maxOneLoopRows / dataBlock * dataBlock;
      GetFrontLoopParams(runParams, maxOneLoopRows, runParams.dataEachCore, dataBlock);

      runParams.oneLoopElemsLast = runParams.oneLoopElems;
      runParams.loopNumLast = runParams.loopNum;
      runParams.lastNumLast = runParams.lastNum;
      if (runParams.dataLastCore > 0) {
        GetLastLoopParams(runParams, maxOneLoopRows, dataBlock);
      }
    } else if (isSplitV && CheckMode6(sizeSplitsVec, dataBlock, shapeAfterDim, shapeAfter)) {
      GELOGD("op [SplitVTiling] : mode 6");
      runParams.tilingMode = TILING_MODE_6;

      CalSpecialParams(runParams, coreNum, dataBlock, shapeBefore);
    } else if (isSplitV && inputDType == ge::DT_FLOAT16 && numSplit <= SPLIT_NUM &&
               CheckMode7(sizeSplitsVec, dataBlock, shapeAfterDim, shapeAfter, shapeBefore)) {
      // 16 is the max of numSplit, size_split[-1] is 32B align, sum of size_split[0:15] cannot exceed 32B
      GELOGD("op [SplitVTiling] : mode 7");
      runParams.tilingMode = TILING_MODE_7;

      CalSpecialParams(runParams, coreNum, dataBlock, shapeBefore);
    } else {
      GELOGD("op [SplitVTiling] : mode 3");
      // shapeBefore > 1: mode 3
      runParams.tilingMode = TILING_MODE_3;

      runParams.needCoreNum = coreNum;

      runParams.multiMove = 0;
      // 65535 is the max src_stride
      if ((runParams.shapeAfter % dataBlock == 0) && (runParams.shapeAfter / dataBlock < 65536)) {
        runParams.multiMove = 1;
      }
    }
  }

  return true;
}

void SetSplitVRuningParams(const SplitVTilingParams& params, utils::OpRunInfo& runInfo) {
  runInfo.AddTilingData(params.tilingMode);
  runInfo.AddTilingData(params.needCoreNum);
  runInfo.AddTilingData(params.inputElems);
  runInfo.AddTilingData(params.shapeDim);
  runInfo.AddTilingData(params.dataEachCore);
  runInfo.AddTilingData(params.dataLastCore);
  runInfo.AddTilingData(params.loopNum);
  runInfo.AddTilingData(params.lastNum);
  runInfo.AddTilingData(params.oneLoopElems);
  runInfo.AddTilingData(params.loopNumLast);
  runInfo.AddTilingData(params.lastNumLast);
  runInfo.AddTilingData(params.oneLoopElemsLast);

  runInfo.AddTilingData(params.shapeAfterDim);
  runInfo.AddTilingData(params.shapeBefore);
  runInfo.AddTilingData(params.shapeAfter);
  runInfo.AddTilingData(params.multiMove);

  runInfo.AddTilingData(params.tailEle);
  runInfo.AddTilingData(params.oneCoreSeg);
  runInfo.AddTilingData(params.segLoopNum);
  runInfo.AddTilingData(params.lastSeg);
  runInfo.AddTilingData(params.lastCoreSeg);
  runInfo.AddTilingData(params.segLoopNumLastCore);
  runInfo.AddTilingData(params.lastSegLastCore);

  runInfo.AddTilingData(params.sizeValueSplit);
}

void PrintSplitVTilingParams(const SplitVTilingParams& params) {
  GELOGD("op [SplitVTiling] : tilingMode=%d.", params.tilingMode);
  GELOGD("op [SplitVTiling] : needCoreNum=%d.", params.needCoreNum);
  GELOGD("op [SplitVTiling] : inputElems=%d.", params.inputElems);
  GELOGD("op [SplitVTiling] : shapeDim=%d.", params.shapeDim);
  GELOGD("op [SplitVTiling] : dataEachCore=%d.", params.dataEachCore);
  GELOGD("op [SplitVTiling] : dataLastCore=%d.", params.dataLastCore);
  GELOGD("op [SplitVTiling] : loopNum=%d.", params.loopNum);
  GELOGD("op [SplitVTiling] : lastNum=%d.", params.lastNum);
  GELOGD("op [SplitVTiling] : oneLoopElems=%d.", params.oneLoopElems);
  GELOGD("op [SplitVTiling] : loopNumLast=%d.", params.loopNumLast);
  GELOGD("op [SplitVTiling] : lastNumLast=%d.", params.lastNumLast);
  GELOGD("op [SplitVTiling] : oneLoopElemsLast=%d.", params.oneLoopElemsLast);

  GELOGD("op [SplitVTiling] : shapeAfterDim=%d.", params.shapeAfterDim);
  GELOGD("op [SplitVTiling] : shapeBefore=%d.", params.shapeBefore);
  GELOGD("op [SplitVTiling] : shapeAfter=%d.", params.shapeAfter);
  GELOGD("op [SplitVTiling] : multiMove=%d.", params.multiMove);

  GELOGD("op [SplitVTiling] : tailEle=%d.", params.tailEle);
  GELOGD("op [SplitVTiling] : oneCoreSeg=%d.", params.oneCoreSeg);
  GELOGD("op [SplitVTiling] : segLoopNum=%d.", params.segLoopNum);
  GELOGD("op [SplitVTiling] : lastSeg=%d.", params.lastSeg);
  GELOGD("op [SplitVTiling] : lastCoreSeg=%d.", params.lastCoreSeg);
  GELOGD("op [SplitVTiling] : segLoopNumLastCore=%d.", params.segLoopNumLastCore);
  GELOGD("op [SplitVTiling] : lastSegLastCore=%d.", params.lastSegLastCore);

  GELOGD("op [SplitVTiling] : sizeValueSplit=%d.", params.sizeValueSplit);
}

bool GetSplitVCompileParams(const nlohmann::json& opCompileInfo, int64_t& coreNum, int64_t& ubElems,
                            int64_t& numSplit) {
  using namespace nlohmann;
  auto allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitVTiling", "GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  if (allVars.count("ub_elems") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitVTiling", "GetCompileParams, get ub_elems error");
    return false;
  }
  ubElems = allVars["ub_elems"].get<std::int64_t>();

  if (allVars.count("num_split") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitVTiling", "GetCompileParams, get num_split error");
    return false;
  }
  numSplit = allVars["num_split"].get<std::int64_t>();

  return true;
}

bool SplitVTiling(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                  utils::OpRunInfo& runInfo) {
  using namespace ge;
  using namespace std;
  PROFILING_TILING_INIT(opType.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(opParas);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info failed.");
    return false;
  }

  GELOGI("op[%s] SplitVTiling running.", opType.c_str());
  auto input0_desc = operator_info->MutableInputDesc(0);
  auto input1_desc = operator_info->MutableInputDesc(1);
  if (input0_desc == nullptr || input1_desc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input0_desc or input1_desc failed.");
    return false;
  }

  bool isSplitV = false;  // split
  int64_t splitDimInputIndex = 0;
  auto xInput_desc = input1_desc;
  auto splitDimInput_desc = input0_desc;

  if (opType == "SplitV") {  // splitv
    isSplitV = true;
    splitDimInputIndex = 2;
    xInput_desc = input0_desc;
    splitDimInput_desc = operator_info->MutableInputDesc(2);
    if (splitDimInput_desc == nullptr) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "get splitDimInput_desc failed.");
      return false;
    }
  }
  ge::Format input_format = input0_desc->GetFormat();

  const std::vector<int64_t>& inputShape = xInput_desc->MutableShape().GetDims();
  int64_t shapeSize = inputShape.size();
  ge::DataType inputDType = xInput_desc->GetDataType();
  int64_t dataBlock = GetDataBlockElems(inputDType);
  if (dataBlock == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitVTiling", "get data block elements error, dataBlock is zero");
    return false;
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  // get compile info
  int64_t coreNum = 0;
  int64_t ubElems = 0;
  int64_t numSplit = 0;
  bool can_get_params = GetSplitVCompileParams(opCompileInfo, coreNum, ubElems, numSplit);
  if (!can_get_params || coreNum == 0 || ubElems == 0 || numSplit == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SplitVTiling: GetSplitVCompileParams error.");
    return false;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  // get split_dim
  std::vector<int64_t> splitDimVec;
  GELOGD("op SplitVTiling : splitDimInputIndex=%d.", splitDimInputIndex);

  if (isSplitV) {
    // input split_dim index is 2
    OP_TILING_CHECK(!ops::GetConstIntData(opParas, 2, splitDimVec),
                    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SplitVTiling: Get split_dim value failed."), return false);
  } else {
    // input split_dim index is 0
    OP_TILING_CHECK(!ops::GetConstIntData(opParas, 0, splitDimVec),
                    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SplitVTiling: Get split_dim value failed."), return false);
  }

  int64_t splitDim = splitDimVec[0];
  if (splitDim == 1 and input_format == ge::FORMAT_FRACTAL_NZ) {
    splitDim = 0;
  }
  if (splitDim < -shapeSize || splitDim >= shapeSize) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitVTiling", "split_dim is invalid");

    return false;
  }
  if (splitDim < 0) {
    splitDim = splitDim + shapeSize;
  }

  // get size_splits, int64
  int64_t splitSizeValue = 0;
  std::vector<int64_t> sizeSplitsVec;
  if (isSplitV) {
    //input size_splits index is 1
    if (!ops::GetConstIntData(opParas, 1, sizeSplitsVec)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "SplitVTiling: Get size_splits value failed.");
      return false;
    }
    int64_t size = sizeSplitsVec.size();
    if (input_format == ge::FORMAT_FRACTAL_NZ) {
      for (int in = 0; in < size; in++) {
        sizeSplitsVec[in] = (sizeSplitsVec[in]  +  ALLIGN_NUM_15) / ALLIGN_NUM_16;
      }
    }
    if (!CheckSplitVAttr(splitDim, numSplit, inputShape, sizeSplitsVec)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "SplitVTiling: CheckSplitVAttr failed.");
      return false;
    }
  } else {
    int64_t dim = inputShape[splitDim];
    if (dim % numSplit != 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                      "SplitVTiling: The num_split must be divisible by the x.shape[split_dim]");

      return false;
    }
    splitSizeValue = dim / numSplit;
    sizeSplitsVec.insert(sizeSplitsVec.end(), numSplit, splitSizeValue);
  }

  SplitVTilingParams runParams;
  InitSplitVRunningParams(runParams);
  runParams.sizeValueSplit = splitSizeValue;

  int64_t inputElems = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<int>());
  GELOGD("op [SplitVTiling] : inputElems=%d.", inputElems);

  if (!CalSplitVRunningParams(runParams, inputElems, inputShape, ubElems, coreNum, splitDim, numSplit, dataBlock,
                              sizeSplitsVec, inputDType, isSplitV)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SplitVTiling: CalSplitVRunningParams failed.");
    return false;
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  SetSplitVRuningParams(runParams, runInfo);
  PrintSplitVTilingParams(runParams);

  runInfo.SetBlockDim(runParams.needCoreNum);
  PROFILING_TILING_END();
  GELOGI("op[%s] SplitVTiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the SplitV op
REGISTER_OP_TILING_FUNC_BUFFERED_V2(SplitV, SplitVTiling);
// register tiling interface of the Split op
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Split, SplitVTiling);
}  // namespace optiling
