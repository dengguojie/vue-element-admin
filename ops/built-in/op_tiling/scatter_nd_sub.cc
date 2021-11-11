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
 * \file scatter_nd_sub.cpp
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
namespace scatterndsub {
const int64_t BLOCK_SIZE = 32;
// subDataNum is 32b aligned, ub can store all subsNum
const int64_t TILING_MODE_1 = 1;
// subDataNum is 32b aligned, ub can't store all subsNum
const int64_t TILING_MODE_2 = 2;
// subDataNum isn't 32b aligned and less than 1 block
// ub can store all subsNum
const int64_t TILING_MODE_3 = 3;
// subDataNum isn't 32b aligned and less than 1 block
// ub can't store all subsNum
const int64_t TILING_MODE_4 = 4;
// subDataNum isn't 32b aligned and more than 1 block
const int64_t TILING_MODE_5 = 5;
// div 0 check
const int64_t ZERO = 0;

struct ScatterNdSubTilingParams {
  int64_t tilingMode;
  int64_t indiceStep;
  int64_t coreNum;
  int64_t subsDataNum;
  int64_t indicesLoopNum;
  int64_t indicesLastNum;
  int64_t subsNum;
  int64_t subsLoopNum;
  int64_t subsLastNum;
  std::vector<int64_t> varOffset = {0, 0, 0, 0, 0, 0, 0};
  int64_t indicesLastDim;
  int64_t indicesFrontDim;
};

void InitRunningParams(ScatterNdSubTilingParams& params) {
  params.tilingMode = TILING_MODE_1;
  params.indiceStep = 0;
  params.coreNum = 0;
  params.subsDataNum = 0;
  params.indicesLoopNum = 0;
  params.indicesLastNum = 0;
  params.subsNum = 0;
  params.subsLoopNum = 0;
  params.subsLastNum = 0;
}

void CalRunningParams(ScatterNdSubTilingParams& runParams, int64_t indicesNum, int64_t subsNum, int64_t subDataNum,
                      int64_t maxIndice, int64_t ubSize, int64_t coreNum, int64_t varSize, int64_t indicesSize,
                      int64_t varDataEachBlock, const std::string& VarDtype) {
  int64_t subSizeByte = varSize * subsNum;
  int64_t halfUbSize = ubSize / 2;
  OP_TILING_CHECK(halfUbSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd_sub", "halfUbSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(indicesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd_sub", "indicesSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(coreNum == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd_sub", "coreNum = 0 is not support"),
                  return );
  OP_TILING_CHECK(varSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd_sub", "varSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(runParams.indicesLastDim == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd_sub", "runParams.indicesLastDim = 0 is not support"),
                  return );
  OP_TILING_CHECK(runParams.indicesLastDim == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd_sub", "runParams.indicesLastDim = 0 is not support"),
                  return );
  OP_TILING_CHECK(varDataEachBlock == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd_sub", "varDataEachBlock = 0 is not support"),
                  return );
  int64_t halfUbIndicesNum = halfUbSize / indicesSize;
  OP_TILING_CHECK(halfUbIndicesNum == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd_sub", "halfUbIndicesNum = 0 is not support"),
                  return );
  runParams.subsLoopNum = subDataNum / (halfUbSize / varSize);
  runParams.subsLastNum = subDataNum % (halfUbSize / varSize);
  runParams.indicesLoopNum = (indicesNum / runParams.indicesLastDim) / (halfUbIndicesNum / runParams.indicesLastDim);
  runParams.indicesLastNum =
      indicesNum - runParams.indicesLoopNum * (halfUbIndicesNum / runParams.indicesLastDim * runParams.indicesLastDim);
  runParams.subsDataNum = subDataNum;
  runParams.subsNum = subsNum;

  if (subDataNum % varDataEachBlock == 0) {
    if (subSizeByte <= halfUbSize) {
      runParams.tilingMode = TILING_MODE_1;
    } else {
      runParams.tilingMode = TILING_MODE_2;
    }
  } else {
    if (subDataNum < varDataEachBlock) {
      if (subSizeByte <= halfUbSize) {
        runParams.tilingMode = TILING_MODE_3;
        runParams.subsLoopNum = subsNum / (halfUbSize / varSize);
        runParams.subsLastNum = subsNum % (halfUbSize / varSize);
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
  if (subDataNum < varDataEachBlock) {
    runParams.coreNum = 1;
    runParams.indiceStep = ceil(float(maxIndice) / coreNum);
    int64_t VarBlockNum = 32 / VarDtypeSize;
    runParams.indiceStep = ceil(float(runParams.indiceStep) / VarBlockNum) * VarBlockNum;
  } else {
    runParams.indiceStep = ceil(float(maxIndice) / coreNum);
    int64_t VarBlockNum = 32 / VarDtypeSize;
    runParams.indiceStep = ceil(float(runParams.indiceStep) / VarBlockNum) * VarBlockNum;
    runParams.coreNum = ceil(float(maxIndice) / runParams.indiceStep);
  }
}

void SetRuningParams(const ScatterNdSubTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.indiceStep);
  ByteBufferPut(runInfo.tiling_data, params.coreNum);
  ByteBufferPut(runInfo.tiling_data, params.subsDataNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.subsNum);
  ByteBufferPut(runInfo.tiling_data, params.subsLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.subsLastNum);
  for (size_t i = 0; i < params.varOffset.size(); i++) {
    ByteBufferPut(runInfo.tiling_data, params.varOffset[i]);
  }
  ByteBufferPut(runInfo.tiling_data, params.indicesLastDim);
  ByteBufferPut(runInfo.tiling_data, params.indicesFrontDim);
}

void PrintTilingParams(const std::string& opType, const ScatterNdSubTilingParams& params) {
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : tilingMode=%ld.", params.tilingMode);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : indiceStep=%ld.", params.indiceStep);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : coreNum=%ld.", params.coreNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : subsDataNum=%ld.", params.subsDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : indicesLoopNum=%ld.", params.indicesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : indicesLastNum=%ld.", params.indicesLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : subsNum=%ld.", params.subsNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : subsLoopNum=%ld.", params.subsLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : subsLastNum=%ld.", params.subsLastNum);
  for (size_t i = 0; i < params.varOffset.size(); i++) {
    OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : varOffset[%ld]=%ld.", i, params.varOffset[i]);
  }
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : indicesLastDim=%ld.", params.indicesLastDim);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : indicesFrontDim=%ld.", params.indicesFrontDim);
}

bool CheckScatterNdSubTensorShape(const std::string& opType, std::vector<int64_t> varShape,
                                  std::vector<int64_t> indicesShape, std::vector<int64_t> subsShape,
                                  std::vector<int64_t> outShape) {
  if (varShape != outShape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "the length of var must be same as the length of output.");
    return false;
  }

  if (indicesShape.size() == 1 && indicesShape[0] == 1 && varShape.size() - subsShape.size() == 1) {
    OP_LOGI(opType.c_str(), "Input indices is a scalar.");
    return true;
  }

  std::vector<int64_t> actualSubsShape = indicesShape;
  int64_t varSize = varShape.size();
  for (int64_t i = 1; i < varSize; i++) {
    actualSubsShape.push_back(varShape[i]);
  }
  return true;
}

bool GetScatteSubCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int64_t& coreNum,
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
}  // namespace scatterndsub
bool ScatterNdSubTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                        OpRunInfo& runInfo) {
  using namespace ge;
  using namespace scatterndsub;
  OP_LOGI(opType.c_str(), "ScatterNdSubTiling running.");
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
  const std::vector<int64_t>& subsShape = opParas.inputs[2].tensor[0].shape;
  const std::vector<int64_t>& outShape = opParas.outputs[0].tensor[0].shape;
  const std::string VarDtype = opParas.inputs[0].tensor[0].dtype;

  bool is_valid_shape = CheckScatterNdSubTensorShape(opType, varShape, indicesShape, subsShape, outShape);
  if (!is_valid_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckScatterNdSubTensorShape failed.");
    return false;
  }

  int64_t coreNum = 0;
  int64_t ubSize = 0;
  int64_t varSize = 0;
  int64_t indicesSize = 0;
  bool can_get_params = GetScatteSubCompileParams(opType, opCompileInfo, coreNum, ubSize, varSize, indicesSize);
  if (!can_get_params) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetScatteSubCompileParams error.");
    return false;
  }
  if (coreNum <= ZERO || ubSize <= ZERO || varSize <= ZERO || indicesSize <= ZERO) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "coreNum, ubSize, varSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum, ubSize,
        varSize, indicesSize);
    return false;
  }

  ScatterNdSubTilingParams runParams;
  InitRunningParams(runParams);
  int64_t indicesNum = std::accumulate(indicesShape.begin(), indicesShape.end(), 1, std::multiplies<int>());
  int64_t subsNum = std::accumulate(subsShape.begin(), subsShape.end(), 1, std::multiplies<int>());
  int64_t K = indicesShape.back();
  int64_t subDataNum =
      (varShape.size() > 1) ? (std::accumulate(varShape.begin() + K, varShape.end(), 1, std::multiplies<int>())) : 1;
  int64_t maxIndice = 1;

  maxIndice = std::accumulate(varShape.begin(), varShape.end(), 1, std::multiplies<int>());

  int64_t varDataEachBlock = BLOCK_SIZE / varSize;
  OP_LOGD(opType.c_str(), "BLOCK_SIZE=%ld.", BLOCK_SIZE);
  OP_LOGD(opType.c_str(), "varSize=%ld.", varSize);
  OP_LOGD(opType.c_str(), "subDataNum=%ld.", subDataNum);

  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : indicesNum=%ld.", indicesNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : subsNum=%ld.", subsNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : subDataNum=%ld.", subDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdSubTiling] : maxIndice=%ld.", maxIndice);

  runParams.indicesLastDim = indicesShape.back();
  runParams.indicesFrontDim = std::accumulate(indicesShape.begin(), indicesShape.end() - 1, 1, std::multiplies<int>());
  CalRunningParams(runParams, indicesNum, subsNum, subDataNum, maxIndice, ubSize, coreNum, varSize, indicesSize,
                   varDataEachBlock, VarDtype);

  for (int64_t i = 0; i < indicesShape.back(); i++) {
    runParams.varOffset[i] = std::accumulate(varShape.begin() + i + 1, varShape.end(), 1, std::multiplies<int>());
  }

  SetRuningParams(runParams, runInfo);

  PrintTilingParams(opType, runParams);

  runInfo.block_dim = runParams.coreNum;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  OP_LOGI(opType.c_str(), "ScatterNdSubTiling run success.");

  return true;
}

// register tiling interface of the ScatterNdSub op.
REGISTER_OP_TILING_FUNC_BUFFERED(ScatterNdSub, ScatterNdSubTiling);
}  // namespace optiling
