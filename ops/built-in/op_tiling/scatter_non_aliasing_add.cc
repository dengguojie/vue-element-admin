/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file scatter_non_aliasing_add.cpp
 * \brief
 */
#include <math.h>

#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace optiling {
namespace scatternonaliasingadd {
const int64_t BLOCK_SIZE = 32;
// addDataNum is 32b aligned, ub can store all addsNum
const int64_t TILING_MODE_1 = 1;
// addDataNum is 32b aligned, ub can't store all addsNum
const int64_t TILING_MODE_2 = 2;
// addDataNum isn't 32b aligned and less than 1 block
// ub can store all addsNum
const int64_t TILING_MODE_3 = 3;
// addDataNum isn't 32b aligned and less than 1 block
// ub can't store all addsNum
const int64_t TILING_MODE_4 = 4;
// addDataNum isn't 32b aligned and more than 1 block
const int64_t TILING_MODE_5 = 5;
// div 0 check
const int64_t ZERO = 0;

struct ScatterNonAliasingAddTilingParams {
  int64_t tilingMode;
  int64_t indiceStep;
  int64_t coreNum;
  int64_t addsDataNum;
  int64_t indicesLoopNum;
  int64_t indicesLastNum;
  int64_t addsNum;
  int64_t addsLoopNum;
  int64_t addsLastNum;
  std::vector<int64_t> varOffset = {0, 0, 0, 0, 0, 0, 0};
  int64_t indicesLastDim;
  int64_t indicesFrontDim;
  int64_t varNum;
};

void InitRunningParams(ScatterNonAliasingAddTilingParams& params) {
  params.tilingMode = TILING_MODE_1;
  params.indiceStep = 0;
  params.coreNum = 0;
  params.addsDataNum = 0;
  params.indicesLoopNum = 0;
  params.indicesLastNum = 0;
  params.addsNum = 0;
  params.addsLoopNum = 0;
  params.addsLastNum = 0;
  params.varNum = 0;
}

static int64_t GetCeilTwoInt(int64_t value1, int64_t value2) {
  if (value2 == 0) {
    return value1;
  }
  return static_cast<int64_t>((value1 + value2 - 1) / value2);
}

void CalRunningParams(ScatterNonAliasingAddTilingParams& runParams, int64_t indicesNum, int64_t addsNum,
                      int64_t addDataNum, int64_t maxIndice, int64_t ubSize, int64_t coreNum, int64_t varSize,
                      int64_t indicesSize, int64_t varDataEachBlock, const std::string& VarDtype) {
  int64_t addSizeByte = varSize * addsNum;
  int64_t halfUbSize = ubSize / 2;
  int64_t halfUbIndicesNum = halfUbSize / indicesSize;
  runParams.varNum = maxIndice;
  runParams.addsLoopNum = addDataNum / (halfUbSize / varSize);
  runParams.addsLastNum = addDataNum % (halfUbSize / varSize);
  runParams.indicesLoopNum = (indicesNum / runParams.indicesLastDim) / (halfUbIndicesNum / runParams.indicesLastDim);
  runParams.indicesLastNum =
      indicesNum - runParams.indicesLoopNum * (halfUbIndicesNum / runParams.indicesLastDim * runParams.indicesLastDim);
  runParams.addsDataNum = addDataNum;
  runParams.addsNum = addsNum;

  if (addDataNum % varDataEachBlock == 0) {
    if (addSizeByte <= halfUbSize) {
      runParams.tilingMode = TILING_MODE_1;
    } else {
      runParams.tilingMode = TILING_MODE_2;
    }
  } else {
    if (addDataNum < varDataEachBlock) {
      if (addSizeByte <= halfUbSize) {
        runParams.tilingMode = TILING_MODE_3;
        runParams.addsLoopNum = addsNum / (halfUbSize / varSize);
        runParams.addsLastNum = addsNum % (halfUbSize / varSize);
      } else {
        runParams.tilingMode = TILING_MODE_4;
      }
    } else {
      runParams.tilingMode = TILING_MODE_5;
    }
  }
  int64_t VarDtypeSize = 4;
  if (VarDtype == "int64" || VarDtype == "uint64") {
    VarDtypeSize = 8;
  } else if (VarDtype == "float16") {
    VarDtypeSize = 2;
  } else if (VarDtype == "int8" || VarDtype == "uint8") {
    VarDtypeSize = 1;
  }
  if (addDataNum < varDataEachBlock) {
    runParams.coreNum = 1;
    runParams.indiceStep = maxIndice;
    int64_t VarBlockNum = 32 / VarDtypeSize;
    runParams.indiceStep = GetCeilTwoInt(runParams.indiceStep, VarBlockNum) * VarBlockNum;
    runParams.indiceStep = GetCeilTwoInt(runParams.indiceStep, addDataNum) * addDataNum;
  } else {
    runParams.indiceStep = GetCeilTwoInt(maxIndice, coreNum);
    int64_t VarBlockNum = 32 / VarDtypeSize;
    runParams.indiceStep = GetCeilTwoInt(runParams.indiceStep, VarBlockNum) * VarBlockNum;
    runParams.indiceStep = GetCeilTwoInt(runParams.indiceStep, addDataNum) * addDataNum;
    runParams.coreNum = GetCeilTwoInt(maxIndice, runParams.indiceStep);
  }
}

void SetRuningParams(const ScatterNonAliasingAddTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.indiceStep);
  ByteBufferPut(runInfo.tiling_data, params.coreNum);
  ByteBufferPut(runInfo.tiling_data, params.addsDataNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.addsNum);
  ByteBufferPut(runInfo.tiling_data, params.addsLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.addsLastNum);
  for (size_t i = 0; i < params.varOffset.size(); i++) {
    ByteBufferPut(runInfo.tiling_data, params.varOffset[i]);
  }
  ByteBufferPut(runInfo.tiling_data, params.indicesLastDim);
  ByteBufferPut(runInfo.tiling_data, params.indicesFrontDim);
  ByteBufferPut(runInfo.tiling_data, params.varNum);
}

void PrintTilingParams(const std::string& opType, const ScatterNonAliasingAddTilingParams& params) {
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : tilingMode=%ld.", params.tilingMode);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : indiceStep=%ld.", params.indiceStep);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : coreNum=%ld.", params.coreNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : addsDataNum=%ld.", params.addsDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : indicesLoopNum=%ld.", params.indicesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : indicesLastNum=%ld.", params.indicesLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : addsNum=%ld.", params.addsNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : addsLoopNum=%ld.", params.addsLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : addsLastNum=%ld.", params.addsLastNum);
  for (size_t i = 0; i < params.varOffset.size(); i++) {
    OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : varOffset[%ld]=%ld.", i, params.varOffset[i]);
  }
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : indicesLastDim=%ld.", params.indicesLastDim);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : indicesFrontDim=%ld.", params.indicesFrontDim);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : varNum=%ld.", params.varNum);
}

bool CheckScatterNonAliasingAddTensorShape(const std::string& opType, std::vector<int64_t> varShape,
                                           std::vector<int64_t> indicesShape, std::vector<int64_t> addsShape,
                                           std::vector<int64_t> outShape) {
  if (varShape != outShape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "the length of var must be same as the length of output.");
    return false;
  }

  if (indicesShape.size() == 1 && indicesShape[0] == 1 && varShape.size() - addsShape.size() == 1) {
    OP_LOGI(opType.c_str(), "Input indices is a scalar.");
    return true;
  }

  std::vector<int64_t> actualAddsShape = indicesShape;
  int64_t varSize = varShape.size();
  for (int64_t i = 1; i < varSize; i++) {
    actualAddsShape.push_back(varShape[i]);
  }
  return true;
}

bool GetScatteAddCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int64_t& coreNum,
                               int64_t& ubSize, int64_t& varSize, int64_t& indicesSize) {
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

  return true;
}
}  // namespace scatternonaliasingadd
bool ScatterNonAliasingAddTiling(const std::string& opType, const TeOpParas& opParas,
                                 const nlohmann::json& opCompileInfo, OpRunInfo& runInfo) {
  using namespace ge;
  using namespace scatternonaliasingadd;
  OP_LOGI(opType.c_str(), "ScatterNonAliasingAddTiling running.");
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
  const std::vector<int64_t>& addsShape = opParas.inputs[2].tensor[0].shape;
  const std::vector<int64_t>& outShape = opParas.outputs[0].tensor[0].shape;
  const std::string VarDtype = opParas.inputs[0].tensor[0].dtype;

  bool is_valid_shape = CheckScatterNonAliasingAddTensorShape(opType, varShape, indicesShape, addsShape, outShape);
  if (!is_valid_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckScatterNonAliasingAddTensorShape failed.");
    return false;
  }

  int64_t coreNum = 0;
  int64_t ubSize = 0;
  int64_t varSize = 0;
  int64_t indicesSize = 0;
  bool can_get_params = GetScatteAddCompileParams(opType, opCompileInfo, coreNum, ubSize, varSize, indicesSize);
  if (!can_get_params) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetScatteAddCompileParams error.");
    return false;
  }
  if (coreNum <= ZERO || ubSize <= ZERO || varSize <= ZERO || indicesSize <= ZERO) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "coreNum, ubSize, varSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum, ubSize,
        varSize, indicesSize);
    return false;
  }

  ScatterNonAliasingAddTilingParams runParams;
  InitRunningParams(runParams);
  int64_t indicesNum = std::accumulate(indicesShape.begin(), indicesShape.end(), 1, std::multiplies<int>());
  int64_t addsNum = std::accumulate(addsShape.begin(), addsShape.end(), 1, std::multiplies<int>());
  int64_t K = indicesShape.back();
  int64_t addDataNum =
      (varShape.size() > 1) ? (std::accumulate(varShape.begin() + K, varShape.end(), 1, std::multiplies<int>())) : 1;
  int64_t maxIndice = 1;

  maxIndice = std::accumulate(varShape.begin(), varShape.end(), 1, std::multiplies<int>());

  int64_t varDataEachBlock = BLOCK_SIZE / varSize;
  OP_LOGD(opType.c_str(), "BLOCK_SIZE=%ld.", BLOCK_SIZE);
  OP_LOGD(opType.c_str(), "varSize=%ld.", varSize);
  OP_LOGD(opType.c_str(), "addDataNum=%ld.", addDataNum);

  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : indicesNum=%ld.", indicesNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : addsNum=%ld.", addsNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : addDataNum=%ld.", addDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : maxIndice=%ld.", maxIndice);

  runParams.indicesLastDim = indicesShape.back();
  runParams.indicesFrontDim = std::accumulate(indicesShape.begin(), indicesShape.end() - 1, 1, std::multiplies<int>());
  CalRunningParams(runParams, indicesNum, addsNum, addDataNum, maxIndice, ubSize, coreNum, varSize, indicesSize,
                   varDataEachBlock, VarDtype);

  for (int64_t i = 0; i < indicesShape.back(); i++) {
    runParams.varOffset[i] = std::accumulate(varShape.begin() + i + 1, varShape.end(), 1, std::multiplies<int>());
  }

  SetRuningParams(runParams, runInfo);

  PrintTilingParams(opType, runParams);

  runInfo.block_dim = runParams.coreNum;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  OP_LOGI(opType.c_str(), "ScatterNonAliasingAddTiling run success.");

  return true;
}

// register tiling interface of the ScatterNonAliasingAdd op.
REGISTER_OP_TILING_FUNC_BUFFERED(ScatterNonAliasingAdd, ScatterNonAliasingAddTiling);
}  // namespace optiling
