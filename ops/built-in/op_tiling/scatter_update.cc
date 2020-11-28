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
 * \file scatter_update.cpp
 * \brief
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"

// updatesDataNum is 32B aligned can store in ub
const int32_t TILING_MODE_1 = 1;
// data num in one block of float32 datatype
const int32_t VAR_DATA_EACH_BLOCK = 8;
// data num in one block of int32 datatype
const int32_t INDICES_DATA_EACH_BLOCK = 8;
// float32 and int32 size take up
const int32_t FP32_INT32_SIZE = 4;

namespace optiling {

struct ScatterUpdateTilingParams {
  int32_t tilingMode;
  int32_t indicesUbNumber;
  int32_t updatesUbNumber;
  int32_t indiceStep;
  int32_t coreNum;
  int32_t updatesDataNum;
  int32_t indicesLoopNum;
  int32_t indicesLastNum;
  int32_t indicesNum;
};

void InitRunningParams(ScatterUpdateTilingParams& params) {
  params.tilingMode = TILING_MODE_1;
  params.indicesUbNumber = 0;
  params.updatesUbNumber = 0;
  params.indiceStep = 0;
  params.coreNum = 0;
  params.updatesDataNum = 0;
  params.indicesLoopNum = 0;
  params.indicesLastNum = 0;
  params.indicesNum = 0;
}

void CalRunningParams(ScatterUpdateTilingParams& runParams, int32_t indicesNum, int32_t updateDataNum,
                      int32_t maxIndice, int32_t ubSize, int32_t coreNum) {
  runParams.updatesUbNumber = updateDataNum;
  runParams.indicesUbNumber = ubSize * 0.9 / FP32_INT32_SIZE - updateDataNum;
  runParams.indicesUbNumber =
      ceil(float(runParams.indicesUbNumber) / INDICES_DATA_EACH_BLOCK) * INDICES_DATA_EACH_BLOCK;
  runParams.indicesLoopNum = indicesNum / runParams.indicesUbNumber;
  runParams.indicesLastNum = indicesNum % runParams.indicesUbNumber;
  runParams.indicesNum = indicesNum;
  runParams.updatesDataNum = updateDataNum;
  if (updateDataNum < VAR_DATA_EACH_BLOCK) {
    runParams.coreNum = 1;
  } else {
    int32_t aiCoreNum = coreNum;
    runParams.indiceStep = ceil(float(maxIndice) / aiCoreNum);
    runParams.coreNum = ceil(float(maxIndice) / runParams.indiceStep);
  }
}

void SetRuningParams(const ScatterUpdateTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.indicesUbNumber);
  ByteBufferPut(runInfo.tiling_data, params.updatesUbNumber);
  ByteBufferPut(runInfo.tiling_data, params.indiceStep);
  ByteBufferPut(runInfo.tiling_data, params.coreNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesDataNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesNum);
}

void PrintTilingParams(const ScatterUpdateTilingParams& params) {
  GELOGD("op [ScatterUpdateTiling] : tilingMode=%d.", params.tilingMode);
  GELOGD("op [ScatterUpdateTiling] : indicesUbNumber=%d.", params.indicesUbNumber);
  GELOGD("op [ScatterUpdateTiling] : updatesUbNumber=%d.", params.updatesUbNumber);
  GELOGD("op [ScatterUpdateTiling] : indiceStep=%d.", params.indiceStep);
  GELOGD("op [ScatterUpdateTiling] : coreNum=%d.", params.coreNum);
  GELOGD("op [ScatterUpdateTiling] : updatesDataNum=%d.", params.updatesDataNum);
  GELOGD("op [ScatterUpdateTiling] : indicesLoopNum=%d.", params.indicesLoopNum);
  GELOGD("op [ScatterUpdateTiling] : indicesLastNum=%d.", params.indicesLastNum);
  GELOGD("op [ScatterUpdateTiling] : indicesNum=%d.", params.indicesNum);
}

bool CheckTensorShape(const std::string& opType, std::vector<int64_t> varShape, std::vector<int64_t> indicesShape,
                      std::vector<int64_t> updatesShape, std::vector<int64_t> outShape) {
  if (varShape != outShape) {
    ge::OpsOneInputShapeErrReport("ScatterUpdate", "var", "the length of var must be same as the length of output");
    OP_LOGE(opType.c_str(), "[ScatterUpdateTiling] : var_out's shape must be the same as var's shape.");
    return false;
  }
  std::vector<int64_t> actualUpdatesShape = indicesShape;
  int32_t varSize = varShape.size();
  for (int32_t i = 1; i < varSize; i++) {
    actualUpdatesShape.push_back(varShape[i]);
  }
  if (updatesShape != actualUpdatesShape) {
    ge::OpsOneInputShapeErrReport("ScatterUpdate", "updates",
                                  "updates does not satisfy the relation expression with actualUpdateShape");
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling] : updates's shape is illegal.");
    return false;
  }
  return true;
}

bool GetScatteUpdateCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int32_t& coreNum,
                                  int32_t& ubSize) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "core_num");
    OP_LOGE("op [ScatterUpdateTiling] : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int32_t>();
  if (allVars.count("ub_size") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "ub_size");
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling] : GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int32_t>();
  return true;
}

bool ScatterUpdateTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                         OpRunInfo& runInfo) {
  using namespace ge;
  GELOGI("op[%s] ScatterUpdateTiling running.", opType.c_str());
  const std::vector<int64_t>& varShape = opParas.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& indicesShape = opParas.inputs[1].tensor[0].shape;
  const std::vector<int64_t>& updatesShape = opParas.inputs[2].tensor[0].shape;
  const std::vector<int64_t>& outShape = opParas.outputs[0].tensor[0].shape;

  if (opCompileInfo == nullptr) {
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling] : opCompileInfo json error.");
    return false;
  }

  if (opParas.inputs.empty() || varShape.size() < 2 || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty() || opParas.inputs[2].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices or updates or var",
                                  "The length of var may be less than 2 or the input may be empty");
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling]: input shape error");
    return false;
  }
  if (opParas.outputs.empty() || outShape.size() < 2 || opParas.outputs[0].tensor.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "var_out",
                                   "The length of var_out may be less than 2 or the input may be empty");
    OP_LOGE(opType.c_str(), "ScatterUpdateTiling: output shape error");
    return false;
  }

  bool is_valid_shape = CheckTensorShape(opType, varShape, indicesShape, updatesShape, outShape);
  if (!is_valid_shape) {
    OP_LOGE(opType.c_str(), "ScatterUpdateTiling: CheckTensorShape failed.");
    return false;
  }

  int32_t coreNum = 0;
  int32_t ubSize = 0;
  bool can_get_params = GetScatteUpdateCompileParams(opType, opCompileInfo, coreNum, ubSize);
  if (!can_get_params) {
    OP_LOGE(opType.c_str(), "ScatterUpdateTiling: GetScatteUpdateCompileParams error.");
    return false;
  }

  ScatterUpdateTilingParams runParams;
  InitRunningParams(runParams);
  int32_t indicesNum = std::accumulate(indicesShape.begin(), indicesShape.end(), 1, std::multiplies<int>());
  int32_t updateDataNum =
      (varShape.size() > 1) ? (std::accumulate(varShape.begin() + 1, varShape.end(), 1, std::multiplies<int>())) : 1;
  int32_t maxIndice = varShape[0];
  GELOGD("op [ScatterUpdateTiling] : indicesNum=%d.", indicesNum);
  GELOGD("op [ScatterUpdateTiling] : updateDataNum=%d.", updateDataNum);
  GELOGD("op [ScatterUpdateTiling] : maxIndice=%d.", maxIndice);
  CalRunningParams(runParams, indicesNum, updateDataNum, maxIndice, ubSize, coreNum);

  SetRuningParams(runParams, runInfo);

  PrintTilingParams(runParams);

  runInfo.block_dim = runParams.coreNum;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  GELOGI("op[%s] tiling run success.", opType.c_str());

  return true;
}

// register tiling interface of the ScatterUpdate op.
REGISTER_OP_TILING_FUNC_BUFFERED(ScatterUpdate, ScatterUpdateTiling);
}  // namespace optiling
