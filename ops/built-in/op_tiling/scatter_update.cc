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

namespace optiling {

const int64_t BLOCK_SIZE = 32;
// updateDataNum is 32b aligned, ub can store all updatesNum
const int64_t TILING_MODE_1 = 1;
// updateDataNum is 32b aligned, ub can't store all updatesNum
const int64_t TILING_MODE_2 = 2;
// updateDataNum isn't 32b aligned and less than 1 block
// ub can store all updatesNum
const int64_t TILING_MODE_3 = 3;
// updateDataNum isn't 32b aligned and less than 1 block
// ub can't store all updatesNum
const int64_t TILING_MODE_4 = 4;
// updateDataNum isn't 32b aligned and more than 1 block
const int64_t TILING_MODE_5 = 5;

struct ScatterUpdateTilingParams {
  int64_t tilingMode;
  int64_t indiceStep;
  int64_t coreNum;
  int64_t updatesDataNum;
  int64_t indicesLoopNum;
  int64_t indicesLastNum;
  int64_t updatesNum;
  int64_t updatesLoopNum;
  int64_t updatesLastNum;
};

void InitRunningParams(ScatterUpdateTilingParams& params) {
  params.tilingMode = TILING_MODE_1;
  params.indiceStep = 0;
  params.coreNum = 0;
  params.updatesDataNum = 0;
  params.indicesLoopNum = 0;
  params.indicesLastNum = 0;
  params.updatesNum = 0;
  params.updatesLoopNum = 0;
  params.updatesLastNum = 0;
}

void CalRunningParams(ScatterUpdateTilingParams& runParams, int64_t indicesNum, int64_t updatesNum,
                      int64_t updateDataNum, int64_t maxIndice, int64_t ubSize, int64_t coreNum, int64_t varSize,
                      int64_t indicesSize, int64_t varDataEachBlock) {
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
          runParams.tilingMode = TILING_MODE_3;
          runParams.updatesLoopNum = updatesNum / (halfUbSize / varSize);
          runParams.updatesLastNum = updatesNum % (halfUbSize / varSize);
        } else {
            runParams.tilingMode = TILING_MODE_4;
        }
      } else {
          runParams.tilingMode = TILING_MODE_5;
      }
  }

  if (updateDataNum < varDataEachBlock) {
    runParams.coreNum = 1;
  } else {
    runParams.indiceStep = ceil(float(maxIndice) / coreNum);
    runParams.coreNum = ceil(float(maxIndice) / runParams.indiceStep);
  }
}

void SetRuningParams(const ScatterUpdateTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.indiceStep);
  ByteBufferPut(runInfo.tiling_data, params.coreNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesDataNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesLastNum);
}

void PrintTilingParams(const ScatterUpdateTilingParams& params) {
  GELOGD("op [ScatterUpdateTiling] : tilingMode=%ld.", params.tilingMode);
  GELOGD("op [ScatterUpdateTiling] : indiceStep=%ld.", params.indiceStep);
  GELOGD("op [ScatterUpdateTiling] : coreNum=%ld.", params.coreNum);
  GELOGD("op [ScatterUpdateTiling] : updatesDataNum=%ld.", params.updatesDataNum);
  GELOGD("op [ScatterUpdateTiling] : indicesLoopNum=%ld.", params.indicesLoopNum);
  GELOGD("op [ScatterUpdateTiling] : indicesLastNum=%ld.", params.indicesLastNum);
  GELOGD("op [ScatterUpdateTiling] : updatesNum=%ld.", params.updatesNum);
  GELOGD("op [ScatterUpdateTiling] : updatesLoopNum=%ld.", params.updatesLoopNum);
  GELOGD("op [ScatterUpdateTiling] : updatesLastNum=%ld.", params.updatesLastNum);
}

bool CheckScatterUpdateTensorShape(const std::string& opType, std::vector<int64_t> varShape,
                                   std::vector<int64_t> indicesShape, std::vector<int64_t> updatesShape,
                                   std::vector<int64_t> outShape) {
  if (varShape != outShape) {
    ge::OpsOneInputShapeErrReport("ScatterUpdate", "var", "the length of var must be same as the length of output");
    OP_LOGE(opType.c_str(), "[ScatterUpdateTiling] : var_out's shape must be the same as var's shape.");
    return false;
  }

  if (indicesShape.size() == 1 && indicesShape[0] == 1 && varShape.size() - updatesShape.size() == 1) {
    GELOGI("op[%s] Input indices is a scalar.", opType.c_str());
    return true;
  }

  std::vector<int64_t> actualUpdatesShape = indicesShape;
  int64_t varSize = varShape.size();
  for (int64_t i = 1; i < varSize; i++) {
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

bool GetScatteUpdateCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int64_t& coreNum,
                                  int64_t& ubSize, int64_t& varSize, int64_t& indicesSize) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];
  if (allVars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "core_num");
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling] : GetCompileParams, get core_num error");
    return false;
  }
  coreNum = allVars["core_num"].get<std::int64_t>();

  if (allVars.count("ub_size") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "ub_size");
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling] : GetCompileParams, get ub_size error");
    return false;
  }
  ubSize = allVars["ub_size"].get<std::int64_t>();

  if (allVars.count("var_size") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "var_size");
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling] : GetCompileParams, get var_size error");
    return false;
  }
  varSize = allVars["var_size"].get<std::int64_t>();

  if (allVars.count("indices_size") == 0) {
    ge::OpsGetCompileParamsErrReport(opType.c_str(), "indices_size");
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling] : GetCompileParams, get indices_size error");
    return false;
  }
  indicesSize = allVars["indices_size"].get<std::int64_t>();

  return true;  
}

bool ScatterUpdateTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                         OpRunInfo& runInfo) {
  using namespace ge;
  GELOGI("op[%s] ScatterUpdateTiling running.", opType.c_str());
  if (opCompileInfo == nullptr) {
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling] : opCompileInfo json error.");
    return false;
  }

  if (opParas.inputs.empty() || opParas.inputs[0].tensor.empty() ||
      opParas.inputs[1].tensor.empty() || opParas.inputs[2].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(opType.c_str(), "indices or updates or var",
                                  "The input may be empty");
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling] : input shape error");
    return false;
  }

  if (opParas.outputs.empty() || opParas.outputs[0].tensor.empty()) {
    ge::OpsOneOutputShapeErrReport(opType.c_str(), "var_out",
                                   "The output may be empty");
    OP_LOGE(opType.c_str(), "op [ScatterUpdateTiling] : output shape error");
    return false;
  }

  const std::vector<int64_t>& varShape = opParas.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& indicesShape = opParas.inputs[1].tensor[0].shape;
  const std::vector<int64_t>& updatesShape = opParas.inputs[2].tensor[0].shape;
  const std::vector<int64_t>& outShape = opParas.outputs[0].tensor[0].shape;

  bool is_valid_shape = CheckScatterUpdateTensorShape(opType, varShape, indicesShape, updatesShape, outShape);
  if (!is_valid_shape) {
    OP_LOGE(opType.c_str(), "ScatterUpdateTiling: CheckScatterUpdateTensorShape failed.");
    return false;
  }

  int64_t coreNum = 0;
  int64_t ubSize = 0;
  int64_t varSize = 0;
  int64_t indicesSize = 0;
  bool can_get_params = GetScatteUpdateCompileParams(opType, opCompileInfo, coreNum, ubSize, varSize, indicesSize);
  if (!can_get_params) {
    OP_LOGE(opType.c_str(), "ScatterUpdateTiling: GetScatteUpdateCompileParams error.");
    return false;
  }

  ScatterUpdateTilingParams runParams;
  InitRunningParams(runParams);
  int64_t indicesNum = std::accumulate(indicesShape.begin(), indicesShape.end(), 1, std::multiplies<int>());
  int64_t updatesNum = std::accumulate(updatesShape.begin(), updatesShape.end(), 1, std::multiplies<int>());
  int64_t updateDataNum = (varShape.size() > 1) ? (std::accumulate(varShape.begin() + 1, varShape.end(), 1,
                                                                   std::multiplies<int>())) : 1;
  int64_t maxIndice = varShape[0];
  int64_t varDataEachBlock = BLOCK_SIZE / varSize;

  GELOGD("op [ScatterUpdateTiling] : indicesNum=%ld.", indicesNum);
  GELOGD("op [ScatterUpdateTiling] : updatesNum=%ld.", updatesNum);
  GELOGD("op [ScatterUpdateTiling] : updateDataNum=%ld.", updateDataNum);
  GELOGD("op [ScatterUpdateTiling] : maxIndice=%ld.", maxIndice);

  CalRunningParams(runParams, indicesNum, updatesNum, updateDataNum, maxIndice, ubSize, coreNum,
                   varSize, indicesSize, varDataEachBlock);

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
