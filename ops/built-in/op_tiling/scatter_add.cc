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
 * \file scatter_add.cpp
 * \brief
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

const int64_t BLOCK_SIZE = 32;
const int64_t OUT_SPECIAL_DIM_0 = 163623;
const int64_t OUT_SPECIAL_DIM_1 = 1;
const int64_t OUT_SPECIAL_DIM_2 = 80;
const int64_t OUT_SPECIAL_DIM_3 = 21340;
const int64_t OUT_SPECIAL_DIM_4 = 12828;
// 32b aligned, ub can store all updatesNum, float32 atomic
const int64_t TILING_MODE_1 = 1;
// 32b aligned, ub can't store all updatesNum, float32 atomic
const int64_t TILING_MODE_2 = 2;
// updateDataNum is less than 1 block, ub can store all updatesNum, float32 atomic
const int64_t TILING_MODE_3 = 3;
// updateDataNum is less than 1 block, ub can't store all updatesNum, float32 atomic
const int64_t TILING_MODE_4 = 4;
// updateDataNum is more than 1 block, float32 atomic
const int64_t TILING_MODE_5 = 5;
// 32b aligned, ub can store all var and updates, not atomic
const int64_t TILING_MODE_6 = 6;
// 32b aligned, ub can store all var, not atomic
const int64_t TILING_MODE_7 = 7;
// 32b aligned, ub can store all updates, not atomic
const int64_t TILING_MODE_8 = 8;
// 32b aligned, ub can't store all var and updates, not atomic
const int64_t TILING_MODE_9 = 9;
// updateDataNum is less than 1 block, ub can store all var and updates, not atomic
const int64_t TILING_MODE_10 = 10;
// updateDataNum is less than 1 block, ub can store all var, not atomic
const int64_t TILING_MODE_11 = 11;
// updateDataNum is less than 1 block, ub can store all updates, not atomic
const int64_t TILING_MODE_12 = 12;
// updateDataNum is less than 1 block, ub can't store all var and updates, not atomic
const int64_t TILING_MODE_13 = 13;
// updateDataNum is more than 1 block, not atomic, and less than updateubnum
const int64_t TILING_MODE_14 = 14;
// updateDataNum is more than 1 block, not atomic, and more than updateubnum
const int64_t TILING_MODE_15 = 15;
// high perf branch, updateDataNum is less than 1 block
const int64_t TILING_MODE_16 = 16;
// high perf branch, updateDataNum is 32b aligned
const int64_t TILING_MODE_17 = 17;
// div 0 check
const int64_t ZERO = 0;

struct ScatterAddTilingParams {
  int64_t tilingMode;
  int64_t indiceStep;
  int64_t coreNum;
  int64_t updatesDataNum;
  int64_t indicesLoopNum;
  int64_t indicesLastNum;
  int64_t updatesNum;
  int64_t updatesLoopNum;
  int64_t updatesLastNum;
  int64_t varNum;
  int64_t varLoopNum;
  int64_t varLastNum;
  int64_t varEachCoreBurstLen;
  int64_t varLastCoreBurstLen;
  int64_t maxIndice;
  int64_t varEachCoreData;
  int64_t indicesEachCoreData;
  int64_t indicesLastCoreData;
  int64_t eachCoreIndicesLoopNum;
  int64_t eachCoreIndicesLastNum;
  int64_t lastCoreIndicesLoopNum;
  int64_t lastCoreIndicesLastNum;
};

void InitRunningParams(ScatterAddTilingParams& params) {
  params.tilingMode = TILING_MODE_1;
  params.indiceStep = 0;
  params.coreNum = 0;
  params.updatesDataNum = 0;
  params.indicesLoopNum = 0;
  params.indicesLastNum = 0;
  params.updatesNum = 0;
  params.updatesLoopNum = 0;
  params.updatesLastNum = 0;
  params.varNum = 0;
  params.varLoopNum = 0;
  params.varLastNum = 0;
  params.varEachCoreBurstLen = 0;
  params.varLastCoreBurstLen = 0;
  params.maxIndice = 0;
  params.varEachCoreData = 0;
  params.indicesEachCoreData = 0;
  params.indicesLastCoreData = 0;
  params.eachCoreIndicesLoopNum = 0;
  params.eachCoreIndicesLastNum = 0;
  params.lastCoreIndicesLoopNum = 0;
  params.lastCoreIndicesLastNum = 0;
}

void CalAtomicBranchRunningParams(ScatterAddTilingParams& runParams, int64_t indicesNum, int64_t updatesNum,
                                  int64_t updateDataNum, int64_t ubSize, int64_t varSize, int64_t indicesSize,
                                  int64_t varDataEachBlock) {
  int64_t updateSizeByte = varSize * updatesNum;
  int64_t halfUbSize = ubSize / 2;
  runParams.updatesLoopNum = updateDataNum / (halfUbSize / varSize);
  runParams.updatesLastNum = updateDataNum % (halfUbSize / varSize);
  runParams.indicesLoopNum = indicesNum / (halfUbSize / indicesSize);
  runParams.indicesLastNum = indicesNum % (halfUbSize / indicesSize);
  runParams.updatesDataNum = updateDataNum;
  runParams.updatesNum = updatesNum;

  if (updateDataNum % varDataEachBlock == 0) {
    if (updateSizeByte <= halfUbSize) {
      runParams.tilingMode = TILING_MODE_1;
    } else {
      runParams.tilingMode = TILING_MODE_2;
    }
  } else {
    if (updateDataNum < varDataEachBlock) {
      if (updateSizeByte <= halfUbSize) {
        runParams.tilingMode = TILING_MODE_4;
        runParams.updatesLoopNum = updatesNum / (halfUbSize / varSize);
        runParams.updatesLastNum = updatesNum % (halfUbSize / varSize);
      } else {
        runParams.tilingMode = TILING_MODE_4;
      }
    } else {
      runParams.tilingMode = TILING_MODE_5;
    }
  }
}

void CalNotAtomicBranchRunningParams(ScatterAddTilingParams& runParams, int64_t varNum, int64_t indicesNum,
                                     int64_t updatesNum, int64_t updateDataNum, int64_t maxIndice, int64_t ubSize,
                                     int64_t coreNum, int64_t varSize, int64_t indicesSize, int64_t varDataEachBlock,
                                     int64_t dataNumOneRepeat) {
  int64_t varAllSizeByte = varSize * varNum;
  int64_t varSizeByte = varSize * runParams.indiceStep * updateDataNum;
  int64_t updateSizeByte = varSize * updatesNum;
  int64_t varUbSize = ubSize / 8 * 3;
  int64_t indicesUbSize = ubSize / 8 * 2;
  runParams.varLoopNum = varNum / (varUbSize / varSize);
  runParams.varLastNum = varNum % (varUbSize / varSize);
  runParams.updatesLoopNum = updateDataNum / (varUbSize / varSize);
  runParams.updatesLastNum = updateDataNum % (varUbSize / varSize);
  runParams.indicesLoopNum = indicesNum / (indicesUbSize / indicesSize);
  runParams.indicesLastNum = indicesNum % (indicesUbSize / indicesSize);
  runParams.updatesDataNum = updateDataNum;
  runParams.updatesNum = updatesNum;
  runParams.varNum = varNum;

  if (updateDataNum % varDataEachBlock == 0) {
    if (updateSizeByte <= varUbSize && varSizeByte <= varUbSize) {
      runParams.tilingMode = TILING_MODE_6;
    } else if (updateSizeByte > varUbSize && varSizeByte <= varUbSize) {
      runParams.tilingMode = TILING_MODE_7;
    } else if (updateSizeByte <= varUbSize && varSizeByte > varUbSize) {
      runParams.tilingMode = TILING_MODE_8;
    } else {
      runParams.tilingMode = TILING_MODE_9;
    }
  } else if (updateDataNum < varDataEachBlock) {
    if (updateSizeByte <= varUbSize && varAllSizeByte <= varUbSize) {
      runParams.tilingMode = TILING_MODE_10;
    } else if (updateSizeByte > varUbSize && varAllSizeByte <= varUbSize) {
      runParams.tilingMode = TILING_MODE_11;
    } else if (updateSizeByte <= varUbSize && varAllSizeByte > varUbSize) {
      runParams.tilingMode = TILING_MODE_12;
    } else {
      runParams.tilingMode = TILING_MODE_13;
    }
  } else {
    if (updateDataNum / (varUbSize / varSize) == 0) {
      runParams.tilingMode = TILING_MODE_14;
    } else {
      runParams.tilingMode = TILING_MODE_15;
    }
  }

  if (runParams.tilingMode == TILING_MODE_6 || runParams.tilingMode == TILING_MODE_7) {
    runParams.varEachCoreData = runParams.indiceStep * runParams.updatesDataNum;
    int64_t varLastCoreData = varNum - runParams.varEachCoreData * (coreNum - 1);
    runParams.varEachCoreBurstLen = runParams.varEachCoreData / varDataEachBlock;
    runParams.varLastCoreBurstLen = varLastCoreData / varDataEachBlock;
  }
  if (runParams.tilingMode == TILING_MODE_9 || runParams.tilingMode == TILING_MODE_14 ||
      runParams.tilingMode == TILING_MODE_15) {
    runParams.varLoopNum = updateDataNum / (varUbSize / varSize);
    runParams.varLastNum = updateDataNum % (varUbSize / varSize);
  }
}

void CalScatterAddHighPerfBranchParams(ScatterAddTilingParams& runParams, int64_t indicesNum, int64_t coreNum,
                                       int64_t ubSize, int64_t updateDataNum, int64_t varDataEachBlock,
                                       int64_t indicesSize) {
  int64_t halfUbSize = ubSize / 2;
  runParams.tilingMode = TILING_MODE_16;
  runParams.updatesDataNum = updateDataNum;
  runParams.indicesEachCoreData = ceil(float(indicesNum) / coreNum);
  runParams.coreNum = ceil(float(indicesNum) / runParams.indicesEachCoreData);
  runParams.indicesLastCoreData = indicesNum - runParams.indicesEachCoreData * (runParams.coreNum - 1);
  runParams.eachCoreIndicesLoopNum = runParams.indicesEachCoreData / (halfUbSize / indicesSize);
  runParams.eachCoreIndicesLastNum = runParams.indicesEachCoreData % (halfUbSize / indicesSize);
  runParams.lastCoreIndicesLoopNum = runParams.indicesLastCoreData / (halfUbSize / indicesSize);
  runParams.lastCoreIndicesLastNum = runParams.indicesLastCoreData % (halfUbSize / indicesSize);

  if (updateDataNum % varDataEachBlock == 0) {
    runParams.tilingMode = TILING_MODE_17;
  }
}

void SetRuningParams(const ScatterAddTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.indiceStep);
  ByteBufferPut(runInfo.tiling_data, params.coreNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesDataNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.varNum);
  ByteBufferPut(runInfo.tiling_data, params.varLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.varLastNum);
  ByteBufferPut(runInfo.tiling_data, params.varEachCoreBurstLen);
  ByteBufferPut(runInfo.tiling_data, params.varLastCoreBurstLen);
  ByteBufferPut(runInfo.tiling_data, params.maxIndice);
  ByteBufferPut(runInfo.tiling_data, params.varEachCoreData);
  ByteBufferPut(runInfo.tiling_data, params.indicesEachCoreData);
  ByteBufferPut(runInfo.tiling_data, params.indicesLastCoreData);
  ByteBufferPut(runInfo.tiling_data, params.eachCoreIndicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.eachCoreIndicesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.lastCoreIndicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.lastCoreIndicesLastNum);
}

void PrintTilingParams(const std::string& opType, const ScatterAddTilingParams& params) {
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : tilingMode=%ld. ", params.tilingMode);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : indiceStep=%ld. ", params.indiceStep);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : coreNum=%ld. ", params.coreNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : updatesDataNum=%ld.", params.updatesDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : indicesLoopNum=%ld.", params.indicesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : indicesLastNum=%ld.", params.indicesLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : updatesNum=%ld.", params.updatesNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : updatesLoopNum=%ld.", params.updatesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : updatesLastNum=%ld.", params.updatesLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : varNum=%ld.", params.varNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : varLoopNum=%ld.", params.varLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : varLastNum=%ld.", params.varLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : varEachCoreBurstLen=%ld.", params.varEachCoreBurstLen);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : varLastCoreBurstLen=%ld.", params.varLastCoreBurstLen);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : maxIndice=%ld.", params.maxIndice);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : varEachCoreData=%ld.", params.varEachCoreData);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : indicesEachCoreData=%ld.", params.indicesEachCoreData);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : indicesLastCoreData=%ld.", params.indicesLastCoreData);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : eachCoreIndicesLoopNum=%ld.", params.eachCoreIndicesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : eachCoreIndicesLastNum=%ld.", params.eachCoreIndicesLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : lastCoreIndicesLoopNum=%ld.", params.lastCoreIndicesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : lastCoreIndicesLastNum=%ld.", params.lastCoreIndicesLastNum);
}

bool CheckScatterAddShape(const std::string& opType, const std::vector<int64_t>& varShape,
                          const std::vector<int64_t>& indicesShape, const std::vector<int64_t>& updatesShape,
                          const std::vector<int64_t>& outShape) {
  if (varShape != outShape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "the length of var must be same as the length of output.");
    return false;
  }

  if (indicesShape.size() == 1 && indicesShape[0] == 1 && varShape.size() - updatesShape.size() == 1) {
    OP_LOGI(opType.c_str(), "Input indices is a scalar.");
    return true;
  }

  std::vector<int64_t> actualUpdatesShape = indicesShape;
  int64_t varSize = varShape.size();
  for (int64_t i = 1; i < varSize; i++) {
    actualUpdatesShape.push_back(varShape[i]);
  }
  if (updatesShape != actualUpdatesShape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "updates does not satisfy the relation expression with actualUpdatesShape.");
    return false;
  }
  return true;
}

bool GetScatterAddCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int64_t& coreNum,
                                int64_t& ubSize, int64_t& varSize, int64_t& indicesSize, int64_t& supportAtomic) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];

  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int64_t>();

  if (allVars.count("var_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get var_size error");
    return false;
  }
  varSize = allVars["var_size"].get<std::int64_t>();

  if (allVars.count("indices_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get indices_size error");
    return false;
  }
  indicesSize = allVars["indices_size"].get<std::int64_t>();

  if (allVars.count("support_atomic") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get support_atomic error");
    return false;
  }
  supportAtomic = allVars["support_atomic"].get<std::int64_t>();

  return true;
}

bool CheckScatterAddHighPerfShape(std::vector<int64_t> varShape, std::vector<int64_t> indicesShape) {
  if ((indicesShape.size() == 1 && varShape.size() == 2) &&
      ((varShape[0] == OUT_SPECIAL_DIM_0 && varShape[1] == OUT_SPECIAL_DIM_1) ||
       (varShape[0] == OUT_SPECIAL_DIM_0 && varShape[1] == OUT_SPECIAL_DIM_2) ||
       (varShape[0] == OUT_SPECIAL_DIM_3 && varShape[1] == OUT_SPECIAL_DIM_1) ||
       (varShape[0] == OUT_SPECIAL_DIM_3 && varShape[1] == OUT_SPECIAL_DIM_2))) {
    return true;
  }
  if ((indicesShape.size() == 1 && varShape.size() == 1) && (varShape[0] == OUT_SPECIAL_DIM_4)) {
    return true;
  }
  return false;
}

bool ScatterAddTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                      OpRunInfo& runInfo) {
  using namespace ge;

  OP_LOGI(opType.c_str(), "ScatterAddTiling running.");
  if (opCompileInfo == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "opCompileInfo json error.");
    return false;
  }

  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty() || opParas.inputs[1].tensor.empty() ||
      opParas.inputs[2].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "input shape error");
    return false;
  }

  if (opParas.outputs.empty() || opParas.outputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "output shape error");
    return false;
  }

  const std::vector<int64_t>& varShape = opParas.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& indicesShape = opParas.inputs[1].tensor[0].shape;
  const std::vector<int64_t>& updatesShape = opParas.inputs[2].tensor[0].shape;
  const std::vector<int64_t>& outShape = opParas.outputs[0].tensor[0].shape;
  std::string input_dtype = opParas.inputs[0].tensor[0].dtype;

  bool is_valid_shape = CheckScatterAddShape(opType, varShape, indicesShape, updatesShape, outShape);
  if (!is_valid_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckScatterAddShape is failed.");
    return false;
  }

  int64_t coreNum = 0;
  int64_t ubSize = 0;
  int64_t varSize = 0;
  int64_t indicesSize = 0;
  int64_t supportAtomic = 0;
  bool can_get_params =
      GetScatterAddCompileParams(opType, opCompileInfo, coreNum, ubSize, varSize, indicesSize, supportAtomic);
  if (!can_get_params) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetScatterAddCompileParams error.");
    return false;
  }
  if (coreNum <= ZERO || ubSize <= ZERO || varSize <= ZERO || indicesSize <= ZERO) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "coreNum, ubSize, varSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum, ubSize,
        varSize, indicesSize);
    return false;
  }

  ScatterAddTilingParams runParams;
  InitRunningParams(runParams);
  int64_t varNum = std::accumulate(varShape.begin(), varShape.end(), 1, std::multiplies<int>());
  int64_t indicesNum = std::accumulate(indicesShape.begin(), indicesShape.end(), 1, std::multiplies<int>());
  int64_t updatesNum = std::accumulate(updatesShape.begin(), updatesShape.end(), 1, std::multiplies<int>());
  int64_t updateDataNum =
      (varShape.size() > 1) ? (std::accumulate(varShape.begin() + 1, varShape.end(), 1, std::multiplies<int>())) : 1;
  int64_t maxIndice = varShape[0];
  runParams.maxIndice = maxIndice;
  int64_t varDataEachBlock = BLOCK_SIZE / varSize;
  int64_t dataNumOneRepeat = 0;

  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : varNum=%ld.", varNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : indicesNum=%ld.", indicesNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : updatesNum=%ld.", updatesNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : updateDataNum=%ld.", updateDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterAddTiling] : maxIndice=%ld.", maxIndice);

  if (updateDataNum < varDataEachBlock) {
    runParams.coreNum = 1;
  } else {
    runParams.indiceStep = ceil(float(maxIndice) / coreNum);
    runParams.coreNum = ceil(float(maxIndice) / runParams.indiceStep);
  }

  if (input_dtype == "float32" || input_dtype == "int32") {
    dataNumOneRepeat = 64;
  } else {
    dataNumOneRepeat = 128;
  }

  if (supportAtomic == 1 && input_dtype == "float32") {
    if (CheckScatterAddHighPerfShape(varShape, indicesShape)) {
      CalScatterAddHighPerfBranchParams(runParams, indicesNum, coreNum, ubSize, updateDataNum, varDataEachBlock,
                                        indicesSize);
    } else {
      runParams.indiceStep = ceil(float(maxIndice) / coreNum);
      runParams.coreNum = ceil(float(maxIndice) / runParams.indiceStep);
      CalAtomicBranchRunningParams(runParams, indicesNum, updatesNum, updateDataNum, ubSize, varSize, indicesSize,
                                   varDataEachBlock);
    }
  } else {
    CalNotAtomicBranchRunningParams(runParams, varNum, indicesNum, updatesNum, updateDataNum, maxIndice, ubSize,
                                    runParams.coreNum, varSize, indicesSize, varDataEachBlock, dataNumOneRepeat);
  }

  SetRuningParams(runParams, runInfo);

  PrintTilingParams(opType, runParams);

  runInfo.block_dim = runParams.coreNum;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  OP_LOGI(opType.c_str(), "ScatterAddTiling run success.");

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(ScatterAdd, ScatterAddTiling);
}  // namespace optiling
