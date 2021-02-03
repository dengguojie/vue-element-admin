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
 * \file split_v.cc
 * \brief
 */
#include <string>
#include <securec.h>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"


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
// e.g int16,[48000,256], 256->[80,80,80,1,1,1,13]
const int32_t TILING_MODE_6 = 6;
// e.g only support fp16, [2028,85], 85->[2,2,1,80]
const int32_t TILING_MODE_7 = 7;

const int32_t TRANSPOSE_SIZE = 256;

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
}

int64_t CeilDivCal(const int64_t& uValue, const int64_t& dValue) {
  int64_t resValue = 0;
  if (dValue == 0) {
    OP_LOGE("op [SplitVTiling] : CeilDivCal error, dValue is zero");
    return resValue;
  }

  resValue = (uValue + dValue - 1) / dValue;
  return resValue;
}

int64_t GetDataBlockElems(const std::string& dtype) {
  int64_t dataBlock = 0;
  if (dtype == "float32" || dtype == "int32" || dtype == "uint32") {
    dataBlock = 8;
  } else if (dtype == "float16" || dtype == "int16" || dtype == "uint16") {
    dataBlock = 16;
  } else if (dtype == "int8" || dtype == "uint8") {
    dataBlock = 32;
  } else if (dtype == "int64" || dtype == "uint64") {
    dataBlock = 4;
  }
  return dataBlock;
}

bool CheckSplitVAttr(int64_t splitDim, int64_t numSplit, std::vector<int64_t> inputShape,
                     std::vector<int64_t> sizeSplitsVec) {
  if (sizeSplitsVec.size() != numSplit) {
    OP_LOGE("op [SplitVTiling] : num_split must be equal to the size_splits size");
    ge::OpsInputShapeErrReport("SplitVTiling", "The shape dim of size_splits must be equal to num_split",
                               "size_splits", to_string(sizeSplitsVec.size()));
    return false;
  }

  int64_t sizeSplitsSum = 0;
  int64_t dim = inputShape[splitDim];
  if (std::find(sizeSplitsVec.begin(), sizeSplitsVec.end(), -1) == sizeSplitsVec.end()) {
    sizeSplitsSum = std::accumulate(sizeSplitsVec.begin(), sizeSplitsVec.end(), 0);
    GELOGD("op [SplitVTiling] : CheckSplitVAttr  sizeSplitsSum=%ld, splitDim=%ld, inputShape[%ld]=%ld",
           sizeSplitsSum, splitDim, splitDim, dim);
    if (dim != sizeSplitsSum) {
      OP_LOGE("op [SplitVTiling] : The sum of size_splits must be equal to the x shape[split_dim]");
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

bool CheckSizeSplitsSmall(std::vector<int64_t> sizeSplitsVec, int64_t dataBlock, int64_t shapeAfterDim) {
  bool ret = true;
  for (size_t i = 0; i < sizeSplitsVec.size(); ++i) {
    if (sizeSplitsVec[i] * shapeAfterDim >= dataBlock) {
      ret = false;
      break;
    }
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

bool CheckMode6(std::vector<int64_t> sizeSplitsVec, int64_t dataBlock, int64_t shapeAfterDim, int64_t shapeAfter) {
  if (shapeAfter % dataBlock != 0 || shapeAfter > 300) {
    return false;
  }

  bool ret = true;
  bool flag = false;
  int64_t splitSize = 0;
  for (size_t i = 0; i < sizeSplitsVec.size(); ++i) {
    splitSize = sizeSplitsVec[i] * shapeAfterDim;
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
  for (size_t i = 0; i < num - 1; ++i) {
    splitSizeSum += sizeSplitsVec[i] * shapeAfterDim;
  }
  if (splitSizeSum >= dataBlock) {
    ret = false;
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

bool CheckShapeDim(std::vector<int64_t> sizeSplitsVec, int64_t dataBlock, int64_t coreNum, int64_t shapeAfterDim,
                   int64_t shapeDim) {
  if (shapeDim < coreNum || shapeAfterDim / coreNum > 4*dataBlock) {
    return false;
  }

  bool ret = true;
  int64_t splitSize0 = sizeSplitsVec[0];
  for (size_t i = 0; i < sizeSplitsVec.size(); ++i) {
    if (sizeSplitsVec[i] != splitSize0) {
      ret = false;
      break;
    }
  }
  return ret;
}

bool CalSplitVRunningParams(SplitVTilingParams& runParams, int64_t inputElems, std::vector<int64_t> inputShape,
                            int64_t ubElems, int64_t coreNum, int64_t splitDim, int64_t numSplit, int64_t dataBlock,
                            std::vector<int64_t> sizeSplitsVec, std::string inputDType) {
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
  if (shapeBefore == 0 || shapeAfter == 0 || shapeAfterDim == 0 || shapeDim == 0) {
    OP_LOGE("op [SplitVTiling] : error, shape has zero");
    return false;
  }

  runParams.shapeBefore = shapeBefore;
  runParams.shapeAfter = shapeAfter;
  runParams.shapeAfterDim = shapeAfterDim;
  runParams.inputElems = inputElems;
  runParams.shapeDim = shapeDim;

  if (numSplit == 1) {
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
    if (CheckShapeDim(sizeSplitsVec, dataBlock, coreNum, shapeAfterDim, shapeDim)) {
      runParams.tilingMode = TILING_MODE_8;
    } else {
      runParams.tilingMode = TILING_MODE_2;
    }
    runParams.needCoreNum = coreNum;
  } else {
    if (inputDType == "float16" && numSplit <= 16 && shapeAfter == numSplit
        && inputElems >= TRANSPOSE_SIZE * numSplit) {
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
        OP_LOGE("op [SplitVTiling] : mode 4 error, maxSeg is zero");
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

    } else if (CheckSizeSplitsSmall(sizeSplitsVec, dataBlock, shapeAfterDim)) {
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

    } else if (CheckMode6(sizeSplitsVec, dataBlock, shapeAfterDim, shapeAfter)) {
      GELOGD("op [SplitVTiling] : mode 6");
      runParams.tilingMode = TILING_MODE_6;

      CalSpecialParams(runParams, coreNum, dataBlock, shapeBefore);

    } else if (inputDType == "float16" && numSplit <= 16 &&
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

void SetSplitVRuningParams(const SplitVTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.needCoreNum);
  ByteBufferPut(runInfo.tiling_data, params.inputElems);
  ByteBufferPut(runInfo.tiling_data, params.shapeDim);
  ByteBufferPut(runInfo.tiling_data, params.dataEachCore);
  ByteBufferPut(runInfo.tiling_data, params.dataLastCore);
  ByteBufferPut(runInfo.tiling_data, params.loopNum);
  ByteBufferPut(runInfo.tiling_data, params.lastNum);
  ByteBufferPut(runInfo.tiling_data, params.oneLoopElems);
  ByteBufferPut(runInfo.tiling_data, params.loopNumLast);
  ByteBufferPut(runInfo.tiling_data, params.lastNumLast);
  ByteBufferPut(runInfo.tiling_data, params.oneLoopElemsLast);

  ByteBufferPut(runInfo.tiling_data, params.shapeAfterDim);
  ByteBufferPut(runInfo.tiling_data, params.shapeBefore);
  ByteBufferPut(runInfo.tiling_data, params.shapeAfter);
  ByteBufferPut(runInfo.tiling_data, params.multiMove);

  ByteBufferPut(runInfo.tiling_data, params.tailEle);
  ByteBufferPut(runInfo.tiling_data, params.oneCoreSeg);
  ByteBufferPut(runInfo.tiling_data, params.segLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.lastSeg);
  ByteBufferPut(runInfo.tiling_data, params.lastCoreSeg);
  ByteBufferPut(runInfo.tiling_data, params.segLoopNumLastCore);
  ByteBufferPut(runInfo.tiling_data, params.lastSegLastCore);
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
}

bool GetSplitVCompileParams(const nlohmann::json& opCompileInfo, int64_t& coreNum, int64_t& ubElems,
                            int64_t& numSplit) {
  using namespace nlohmann;
  auto allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    OP_LOGE("op [SplitVTiling] : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();
  if (allVars.count("ub_elems") == 0) {
    OP_LOGE("op [SplitVTiling] : GetCompileParams, get ub_elems error");
    return false;
  }
  ubElems = allVars["ub_elems"].get<std::int64_t>();
  if (allVars.count("num_split") == 0) {
    OP_LOGE("op [SplitVTiling] : GetCompileParams, get num_split error");
    return false;
  }
  numSplit = allVars["num_split"].get<std::int64_t>();
  return true;
}

static bool GetConstValue(const TeOpParas& paras, const std::string& name, const std::string& dtype,
                          std::vector<int64_t>& values) {
  values.clear();
  if (paras.const_inputs.count(name) == 0) {
    OP_LOGE("op[SplitVTiling] : GetConstValue, name is invalid");
    return false;
  }

  auto size = std::get<1>(paras.const_inputs.at(name));
  if (dtype == "int64") {
    int count = size / sizeof(int64_t);
    values.resize(count);
    if (EOK != memcpy_s(values.data(), count * sizeof(int64_t), std::get<0>(paras.const_inputs.at(name)),
                        std::get<1>(paras.const_inputs.at(name)))) {
      OP_LOGE("op[SplitVTiling]: GetConstValue, int64, memcpy_s failed");
      return false;
    }
  } else if (dtype == "int32") {
    int count = size / sizeof(int32_t);
    std::vector<int32_t> tmp(count, 0);
    if (EOK != memcpy_s(tmp.data(), count * sizeof(int32_t), std::get<0>(paras.const_inputs.at(name)),
                        std::get<1>(paras.const_inputs.at(name)))) {
      OP_LOGE("op[SplitVTiling]: GetConstValue, int32, memcpy_s failed");
      return false;
    }
    values.insert(values.end(), tmp.begin(), tmp.end());
  } else {
    OP_LOGE("op[SplitVTiling]: GetConstValue, data type is invalid");
    return false;
  }

  return true;
}

bool SplitVTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                  OpRunInfo& runInfo) {
  using namespace ge;
  using namespace std;
  GELOGI("op[%s] SplitVTiling running.", opType.c_str());
  if (opParas.inputs.size() < 3 || opParas.inputs[0].tensor.size() == 0 ||
      opParas.inputs[1].tensor.size() == 0 || opParas.inputs[2].tensor.size() == 0) {
    OP_LOGE("op[%s] SplitVTiling : opParas.inputs error");
    return false;
  }
  const std::vector<int64_t>& inputShape = opParas.inputs[0].tensor[0].shape;
  int64_t shapeSize = inputShape.size();
  for (int64_t i = 0; i < shapeSize; ++i) {
    GELOGD("op [SplitVTiling] : inputShape[%d]=%d.", i, inputShape[i]);
  }
  std::string inputDType = opParas.inputs[0].tensor[0].dtype;
  int64_t dataBlock = GetDataBlockElems(inputDType);
  if (dataBlock == 0) {
    OP_LOGE("op [SplitVTiling] : get data block elements error, dataBlock is zero");
    return false;
  }

  // get compile info
  int64_t coreNum = 0;
  int64_t ubElems = 0;
  int64_t numSplit = 0;
  bool can_get_params = GetSplitVCompileParams(opCompileInfo, coreNum, ubElems, numSplit);
  if (!can_get_params || coreNum == 0 || ubElems == 0 || numSplit == 0) {
    OP_LOGE("op[%s] SplitVTiling: GetSplitVCompileParams error.", opType.c_str());
    return false;
  }

  // get size_splits and split_dim
  std::vector<int64_t> sizeSplitsVec;
  if (!GetConstValue(opParas, "size_splits", opParas.inputs[1].tensor[0].dtype, sizeSplitsVec)) {
    OP_LOGE("op[%s] SiplitVTiling: Get size_splits value failed.", opType.c_str());
    return false;
  }
  std::vector<int64_t> splitDimVec;
  if (!GetConstValue(opParas, "split_dim", opParas.inputs[2].tensor[0].dtype, splitDimVec)) {
    OP_LOGE("op[%s] SplitVTiling: Get split_dim value failed.", opType.c_str());
    return false;
  }
  int64_t splitDim = splitDimVec[0];
  if (splitDim < -shapeSize || splitDim >= shapeSize) {
    OP_LOGE("op [SplitVTiling] : split_dim is invalid");
    ge::OpsAttrValueErrReport("SplitVTiling", "split_dim",
                              ConcatString("between ", -shapeSize, " and ", shapeSize - 1), std::to_string(splitDim));
    return false;
  }

  if (splitDim < 0) {
    splitDim = splitDim + shapeSize;
  }
  bool ret = CheckSplitVAttr(splitDim, numSplit, inputShape, sizeSplitsVec);
  if (!ret) {
    OP_LOGE("op[%s] SplitVTiling: CheckSplitVAttr failed.", opType.c_str());
    return false;
  }

  SplitVTilingParams runParams;
  InitSplitVRunningParams(runParams);

  int64_t inputElems = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<int>());
  GELOGD("op [SplitVTiling] : inputElems=%d.", inputElems);

  ret = CalSplitVRunningParams(runParams, inputElems, inputShape, ubElems, coreNum, splitDim, numSplit, dataBlock,
                               sizeSplitsVec, inputDType);
  if (!ret) {
    OP_LOGE("op[%s] SplitVTiling: CalSplitVRunningParams failed.", opType.c_str());
    return false;
  }

  SetSplitVRuningParams(runParams, runInfo);

  PrintSplitVTilingParams(runParams);

  runInfo.block_dim = runParams.needCoreNum;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  GELOGI("op[%s] SplitVTiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the SplitV op
REGISTER_OP_TILING_FUNC_BUFFERED(SplitV, SplitVTiling);
}  // namespace optiling
