/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file split_d.cpp
 * \brief
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "op_log.h"
#include "error_log.h"

const std::string DTYPE_FP32 = "float32";
const std::string DTYPE_FP16 = "float16";
const std::string DTYPE_INT8 = "int8";
const std::string DTYPE_UINT8 = "uint8";
const std::string DTYPE_INT16 = "int16";
const std::string DTYPE_UINT16 = "uint16";
const std::string DTYPE_INT32 = "int32";
const std::string DTYPE_UINT32 = "uint32";
const std::string DTYPE_INT64 = "int64";
const std::string DTYPE_UINT64 = "uint64";

namespace optiling {
struct SplitDTilingParams {
  int64_t tilingMode;
  int64_t inputSizeSplit;
  int64_t outputSizeSplit;
  int64_t coreNum;
  int64_t loopEachCore;
  int64_t loopLastCore;
  int64_t dataEachCore;
  int64_t dataLastCore;
  int64_t loopNum;
  int64_t lastNum;
  int64_t loopNumLast;
  int64_t lastNumLast;
  int64_t inputNum;
  int64_t ubNumber;
  int64_t loopEach;
  int64_t loopLast;
  int64_t loopEachLast;
  int64_t lastLoopLast;
};

void InitRunningParams(SplitDTilingParams& params) {
  params.tilingMode = 0;
  params.inputSizeSplit = 0;
  params.outputSizeSplit = 0;
  params.coreNum = 0;
  params.loopEachCore = 0;
  params.loopLastCore = 0;
  params.dataEachCore = 0;
  params.dataLastCore = 0;
  params.loopNum = 0;
  params.lastNum = 0;
  params.loopNumLast = 0;
  params.lastNumLast = 0;
  params.ubNumber = 0;
  params.inputNum = 0;
  params.loopEach = 0;
  params.loopLast = 0;
  params.loopEachLast = 0;
  params.lastLoopLast = 0;
}

int64_t CalCeilDiv(const int64_t& uValue, const int64_t& dValue) {
  int64_t resValue = 0;

  if (dValue == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitDTiling", "dValue cannot be zero!");
    return uValue;
  }

  resValue = (uValue + dValue - 1) / dValue;

  return resValue;
}

int64_t GetDataOneBlock(const std::string& dtype) {
  int64_t dataBlock = 0;
  if (dtype == "float32") {
    dataBlock = 8;
  } else if (dtype == "float16") {
    dataBlock = 16;
  } else if (dtype == "int8" || dtype == "uint8") {
    dataBlock = 32;
  } else if (dtype == "int16" || dtype == "uint16") {
    dataBlock = 16;
  } else if (dtype == "int32" || dtype == "uint32") {
    dataBlock = 8;
  } else if (dtype == "int64" || dtype == "uint64") {
    dataBlock = 4;
  }
  return dataBlock;
}

bool CheckAttr(int64_t splitDim, int64_t numSplit, std::vector<int64_t> inputShape) {
  if (splitDim < 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitDTiling", "split_dim is error");
    return false;
  }
  if (numSplit == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitDTiling", "numSplit cannot be zero!");
    return false;
  }
  if (inputShape[splitDim] % numSplit != 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitDTiling", "The num_split must be divisible by \
                                    the length of inputShape[split_dim]");
    return false;
  }
  return true;
}

void GetLoopParamsFront(SplitDTilingParams& runParams, int64_t ubSize, int64_t dataNum, int64_t dataBlock) {
  if (dataNum < ubSize) {
    runParams.lastNum = dataNum;
  } else {
    runParams.loopNum = dataNum / ubSize;
    runParams.lastNum = dataNum % ubSize;
    if ((runParams.lastNum > 0) && (runParams.lastNum < dataBlock)) {
      runParams.ubNumber = ubSize - dataBlock;
      runParams.loopNum = dataNum / runParams.ubNumber;
      runParams.lastNum = dataNum % runParams.ubNumber;
    }
  }
}

void GetLoopParamsLast(SplitDTilingParams& runParams, int64_t ubSize, int64_t dataBlock) {
  if (runParams.dataLastCore < ubSize) {
    runParams.lastNumLast = runParams.dataLastCore;
  } else {
    runParams.loopNumLast = runParams.dataLastCore / ubSize;
    runParams.lastNumLast = runParams.dataLastCore % ubSize;
    if ((runParams.lastNumLast > 0) && (runParams.lastNumLast < dataBlock)) {
      runParams.ubNumber = ubSize - dataBlock;
      runParams.loopNumLast = runParams.dataLastCore / runParams.ubNumber;
      runParams.lastNumLast = runParams.dataLastCore % runParams.ubNumber;
    }
  }
}

void GetScalarParamsFront(SplitDTilingParams& runParams, int64_t ubSize, int64_t dataNum, int64_t dataValue,
                          int64_t dataBlock) {
  if (dataNum < ubSize) {
    runParams.loopEach = runParams.loopEachCore;
    runParams.loopLast = runParams.loopEachCore;
    runParams.loopNum = 0;
    runParams.lastNum = dataNum;
  } else {
    runParams.loopNum = dataNum / ubSize;
    runParams.lastNum = dataNum % ubSize;
    if ((runParams.lastNum > 0) && (runParams.lastNum < dataBlock)) {
      runParams.ubNumber = ubSize - dataValue;
      runParams.loopNum = dataNum / runParams.ubNumber;
      runParams.lastNum = dataNum % runParams.ubNumber;
    }
    runParams.loopEach = runParams.ubNumber / runParams.inputSizeSplit;
    runParams.loopLast = runParams.loopEachCore % runParams.loopEach;
  }
}

void GetScalarParamsLast(SplitDTilingParams& runParams, int64_t ubSize, int64_t dataValue, int64_t dataBlock) {
  if (runParams.dataLastCore < ubSize) {
    runParams.loopEachLast = runParams.loopLastCore;
    runParams.lastLoopLast = runParams.loopLastCore;
    runParams.loopNumLast = 0;
    runParams.lastNumLast = runParams.dataLastCore;
  } else {
    runParams.loopNumLast = runParams.dataLastCore / ubSize;
    runParams.lastNumLast = runParams.dataLastCore % ubSize;
    if ((runParams.lastNumLast > 0) && (runParams.lastNumLast < dataBlock)) {
      runParams.ubNumber = ubSize - dataValue;
      runParams.loopNumLast = runParams.dataLastCore / runParams.ubNumber;
      runParams.lastNumLast = runParams.dataLastCore % runParams.ubNumber;
    }
    runParams.loopEachLast = runParams.ubNumber / runParams.inputSizeSplit;
    runParams.lastLoopLast = runParams.loopLastCore % runParams.loopEachLast;
  }
}

void CalRunningParams(SplitDTilingParams& runParams, int64_t inputNum, std::vector<int64_t> inputShape, int64_t ubSize,
                      int64_t coreNum, int64_t splitDim, int64_t numSplit, int64_t dataBlock) {
  int64_t shapeBefore = 1;
  int64_t shapeAfter = 1;
  int64_t inputSize = inputShape.size();
  runParams.ubNumber = ubSize;
  for (int64_t i = 0; i < splitDim; i++) {
    shapeBefore = inputShape[i] * shapeBefore;
  }
  for (int64_t j = splitDim; j < inputSize; j++) {
    shapeAfter = inputShape[j] * shapeAfter;
  }
  runParams.inputNum = inputNum;
  runParams.inputSizeSplit = shapeAfter;
  if (numSplit == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitDTiling", "numSplit cannot be zero!");
    return;
  }
  runParams.outputSizeSplit = shapeAfter / numSplit;
  if (numSplit == 1) {
    runParams.tilingMode = 0;
    runParams.dataEachCore = CalCeilDiv(runParams.inputNum, coreNum);
    runParams.coreNum = CalCeilDiv(runParams.inputNum, runParams.dataEachCore);
    runParams.dataLastCore = runParams.inputNum - (runParams.coreNum - 1) * runParams.dataEachCore;
    if (runParams.dataEachCore < dataBlock) {
      runParams.coreNum = 1;
      runParams.dataEachCore = runParams.inputNum;
      runParams.dataLastCore = 0;
    }
    if ((runParams.dataLastCore > 0) && (runParams.dataLastCore < dataBlock)) {
      runParams.coreNum = runParams.coreNum - 1;
      runParams.dataLastCore = runParams.dataLastCore + runParams.dataEachCore;
    }
    GetLoopParamsFront(runParams, ubSize, runParams.dataEachCore, dataBlock);
    runParams.loopNumLast = runParams.loopNum;
    runParams.lastNumLast = runParams.lastNum;
    if ((runParams.dataEachCore != runParams.dataLastCore) && (runParams.dataLastCore != 0)) {
      GetLoopParamsLast(runParams, ubSize, dataBlock);
    }
  } else if (shapeBefore >= coreNum) {
    runParams.loopEachCore = CalCeilDiv(shapeBefore, coreNum);
    runParams.coreNum = CalCeilDiv(shapeBefore, runParams.loopEachCore);
    runParams.loopLastCore = shapeBefore - (runParams.coreNum - 1) * runParams.loopEachCore;
    if (runParams.outputSizeSplit > dataBlock) {
      runParams.tilingMode = 1;
      GetLoopParamsFront(runParams, ubSize, runParams.outputSizeSplit, dataBlock);
    } else {
      if (runParams.loopEachCore < dataBlock) {
        runParams.tilingMode = 3;
        runParams.coreNum = 1;
        runParams.loopEachCore = shapeBefore;
      } else {
        runParams.tilingMode = 2;
        if (runParams.loopLastCore < dataBlock) {
          runParams.coreNum = runParams.coreNum - 1;
          runParams.loopLastCore = runParams.loopLastCore + runParams.loopEachCore;
        }
        runParams.ubNumber = (runParams.ubNumber / (numSplit + 1)) * numSplit;
        runParams.ubNumber = runParams.ubNumber / runParams.inputSizeSplit * runParams.inputSizeSplit;
        runParams.dataEachCore = runParams.loopEachCore * runParams.inputSizeSplit;
        runParams.dataLastCore = runParams.loopLastCore * runParams.inputSizeSplit;
        int64_t loopCount = CalCeilDiv(dataBlock, runParams.inputSizeSplit);
        int64_t dataValue = CalCeilDiv(dataBlock, (runParams.outputSizeSplit * loopCount));
        dataValue = dataValue * loopCount * runParams.inputSizeSplit;
        GetScalarParamsFront(runParams, runParams.ubNumber, runParams.dataEachCore, dataValue, dataBlock);
        runParams.loopNumLast = runParams.loopNum;
        runParams.lastNumLast = runParams.lastNum;
        runParams.loopEachLast = runParams.loopEach;
        runParams.lastLoopLast = runParams.loopLast;
        if ((runParams.dataEachCore != runParams.dataLastCore) && (runParams.dataLastCore != 0)) {
          GetScalarParamsLast(runParams, runParams.ubNumber, dataValue, dataBlock);
        }
      }
    }
  } else {
    runParams.dataEachCore = CalCeilDiv(runParams.outputSizeSplit, coreNum);
    runParams.coreNum = CalCeilDiv(runParams.outputSizeSplit, runParams.dataEachCore);
    runParams.dataLastCore = runParams.outputSizeSplit - (runParams.coreNum - 1) * runParams.dataEachCore;
    runParams.loopEachCore = shapeBefore;
    if (runParams.dataEachCore < dataBlock) {
      runParams.tilingMode = 3;
      runParams.coreNum = 1;
    } else {
      runParams.tilingMode = 4;
      if ((runParams.dataLastCore > 0) && (runParams.dataLastCore < dataBlock)) {
        runParams.coreNum = runParams.coreNum - 1;
        runParams.dataLastCore = runParams.dataLastCore + runParams.dataEachCore;
      }
      GetLoopParamsFront(runParams, ubSize, runParams.dataEachCore, dataBlock);
      runParams.loopNumLast = runParams.loopNum;
      runParams.lastNumLast = runParams.lastNum;
      if ((runParams.dataEachCore != runParams.dataLastCore) && (runParams.dataLastCore != 0)) {
        GetLoopParamsLast(runParams, ubSize, dataBlock);
      }
    }
  }
}

void SetRuningParams(const SplitDTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.inputSizeSplit);
  ByteBufferPut(runInfo.tiling_data, params.outputSizeSplit);
  ByteBufferPut(runInfo.tiling_data, params.coreNum);
  ByteBufferPut(runInfo.tiling_data, params.loopEachCore);
  ByteBufferPut(runInfo.tiling_data, params.loopLastCore);
  ByteBufferPut(runInfo.tiling_data, params.dataEachCore);
  ByteBufferPut(runInfo.tiling_data, params.dataLastCore);
  ByteBufferPut(runInfo.tiling_data, params.loopNum);
  ByteBufferPut(runInfo.tiling_data, params.lastNum);
  ByteBufferPut(runInfo.tiling_data, params.loopNumLast);
  ByteBufferPut(runInfo.tiling_data, params.lastNumLast);
  ByteBufferPut(runInfo.tiling_data, params.ubNumber);
  ByteBufferPut(runInfo.tiling_data, params.inputNum);
  ByteBufferPut(runInfo.tiling_data, params.loopEach);
  ByteBufferPut(runInfo.tiling_data, params.loopLast);
  ByteBufferPut(runInfo.tiling_data, params.loopEachLast);
  ByteBufferPut(runInfo.tiling_data, params.lastLoopLast);
}

void PrintTilingParams(const SplitDTilingParams& params) {
  GELOGD("op [SplitDTiling] : tilingMode=%d.", params.tilingMode);
  GELOGD("op [SplitDTiling] : inputSizeSplit=%d.", params.inputSizeSplit);
  GELOGD("op [SplitDTiling] : outputSizeSplit=%d.", params.outputSizeSplit);
  GELOGD("op [SplitDTiling] : coreNum=%d.", params.coreNum);
  GELOGD("op [SplitDTiling] : loopEachCore=%d.", params.loopEachCore);
  GELOGD("op [SplitDTiling] : loopLastCore=%d.", params.loopLastCore);
  GELOGD("op [SplitDTiling] : dataEachCore=%d.", params.dataEachCore);
  GELOGD("op [SplitDTiling] : dataLastCore=%d.", params.dataLastCore);
  GELOGD("op [SplitDTiling] : loopNum=%d.", params.loopNum);
  GELOGD("op [SplitDTiling] : lastNum=%d.", params.lastNum);
  GELOGD("op [SplitDTiling] : loopNumLast=%d.", params.loopNumLast);
  GELOGD("op [SplitDTiling] : lastNumLast=%d.", params.lastNumLast);
  GELOGD("op [SplitDTiling] : ubNumber=%d.", params.ubNumber);
  GELOGD("op [SplitDTiling] : inputNum=%d.", params.inputNum);
  GELOGD("op [SplitDTiling] : loopEach=%d.", params.loopEach);
  GELOGD("op [SplitDTiling] : loopLast=%d.", params.loopLast);
  GELOGD("op [SplitDTiling] : loopEachLast=%d.", params.loopEachLast);
  GELOGD("op [SplitDTiling] : lastLoopLast=%d.", params.lastLoopLast);
}

bool GetSplitDCompileParams(const nlohmann::json& opCompileInfo, int64_t& coreNum, int64_t& ubSize, int64_t& splitDim,
                            int64_t& numSplit) {
  using namespace nlohmann;
  auto allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitDTiling", "GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();
  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitDTiling", "GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int64_t>();
  if (allVars.count("split_dim") <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitDTiling", "GetCompileParams, get split_dim error");
    return false;
  }
  splitDim = allVars["split_dim"].get<std::int64_t>();
  if (allVars.count("num_split") <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("SplitDTiling", "GetCompileParams, get num_split error");
    return false;
  }
  numSplit = allVars["num_split"].get<std::int64_t>();
  return true;
}

bool SplitDTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                  OpRunInfo& runInfo) {
  using namespace ge;
  GELOGI("op[%s] SplitDTiling running.", opType.c_str());
  const std::vector<int64_t>& inputShape = opParas.inputs[0].tensor[0].shape;
  int64_t shapeSize = inputShape.size();
  for (int64_t i = 0; i < shapeSize; ++i) {
    GELOGD("op [SplitDTiling] : inputShape=%d.", inputShape[i]);
  }
  std::string input_dtype = opParas.inputs[0].tensor[0].dtype;

  int64_t coreNum = 0;
  int64_t ubSize = 0;
  int64_t splitDim = 0;
  int64_t numSplit = 0;
  int64_t dataBlock = 0;
  bool can_get_params = GetSplitDCompileParams(opCompileInfo, coreNum, ubSize, splitDim, numSplit);
  if (!can_get_params) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SiplitDTiling: GetSplitDCompileParams error.");
    return false;
  }
  if (splitDim < 0) {
    splitDim = splitDim + inputShape.size();
  }
  bool ret = CheckAttr(splitDim, numSplit, inputShape);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "SplitDTiling: CheckAttr failed.");
    return false;
  }
  dataBlock = GetDataOneBlock(input_dtype);

  SplitDTilingParams runParams;
  InitRunningParams(runParams);

  int64_t inputNum = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<int>());
  GELOGD("op [SplitDTiling] : inputNum=%d.", inputNum);

  CalRunningParams(runParams, inputNum, inputShape, ubSize, coreNum, splitDim, numSplit, dataBlock);

  SetRuningParams(runParams, runInfo);

  PrintTilingParams(runParams);

  runInfo.block_dim = runParams.coreNum;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  GELOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the SplitD op.
REGISTER_OP_TILING_FUNC_BUFFERED(SplitD, SplitDTiling);
}  // namespace optiling
