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
 * \file scatter_nd_update.cpp
 * \brief
 */
#include <math.h>

#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling_util.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"

namespace optiling {
namespace scatterndupdate {
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
// div 0 check
const int64_t ZERO = 0;
const int64_t INDICES_SIZE_INDEX = 3;
const int64_t VAR_SIZE_INDEX = 2;
const int64_t SHAPE_UPDATES_INDEX = 2;

struct ScatterNdUpdateTilingParams {
  int64_t tilingMode;
  int64_t indiceStep;
  int64_t coreNum;
  int64_t updatesDataNum;
  int64_t indicesLoopNum;
  int64_t indicesLastNum;
  int64_t updatesNum;
  int64_t updatesLoopNum;
  int64_t updatesLastNum;
  std::vector<int64_t> varOffset = {0, 0, 0, 0, 0, 0, 0};
  int64_t indicesLastDim;
  int64_t indicesFrontDim;
};

void InitRunningParams(ScatterNdUpdateTilingParams& params) {
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

void CalRunningParams(ScatterNdUpdateTilingParams& runParams, int64_t indicesNum, int64_t updatesNum,
                      int64_t updateDataNum, int64_t maxIndice, int64_t ubSize, int64_t coreNum, int64_t varSize,
                      int64_t indicesSize, int64_t varDataEachBlock) {
  int64_t updateSizeByte = varSize * updatesNum;
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
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd_sub", "varDataEachBlock = 0 is not support"), return );
  int64_t halfUbIndicesNum = halfUbSize / indicesSize;
  OP_TILING_CHECK(halfUbIndicesNum == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_nd_sub", "halfUbIndicesNum = 0 is not support"), return );
  runParams.updatesLoopNum = updateDataNum / (halfUbSize / varSize);
  runParams.updatesLastNum = updateDataNum % (halfUbSize / varSize);
  runParams.indicesLoopNum = (indicesNum / runParams.indicesLastDim) / (halfUbIndicesNum / runParams.indicesLastDim);
  runParams.indicesLastNum =
      indicesNum - runParams.indicesLoopNum * (halfUbIndicesNum / runParams.indicesLastDim * runParams.indicesLastDim);
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
    int64_t VarBlockNum = 32 / varSize;
    runParams.indiceStep = ceil(float(runParams.indiceStep) / VarBlockNum) * VarBlockNum;
    runParams.coreNum = ceil(float(maxIndice) / runParams.indiceStep);
  }
}

void SetRuningParams(const ScatterNdUpdateTilingParams& params, utils::OpRunInfo& runInfo) {
  runInfo.AddTilingData(params.tilingMode);
  runInfo.AddTilingData(params.indiceStep);
  runInfo.AddTilingData(params.coreNum);
  runInfo.AddTilingData(params.updatesDataNum);
  runInfo.AddTilingData(params.indicesLoopNum);
  runInfo.AddTilingData(params.indicesLastNum);
  runInfo.AddTilingData(params.updatesNum);
  runInfo.AddTilingData(params.updatesLoopNum);
  runInfo.AddTilingData(params.updatesLastNum);
  for (size_t i = 0; i < params.varOffset.size(); i++) {
    runInfo.AddTilingData(params.varOffset[i]);
  }
  runInfo.AddTilingData(params.indicesLastDim);
  runInfo.AddTilingData(params.indicesFrontDim);
}

void PrintTilingParams(const std::string& opType, const ScatterNdUpdateTilingParams& params) {
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : tilingMode=%ld.", params.tilingMode);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : indiceStep=%ld.", params.indiceStep);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : coreNum=%ld.", params.coreNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : updatesDataNum=%ld.", params.updatesDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : indicesLoopNum=%ld.", params.indicesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : indicesLastNum=%ld.", params.indicesLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : updatesNum=%ld.", params.updatesNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : updatesLoopNum=%ld.", params.updatesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : updatesLastNum=%ld.", params.updatesLastNum);
  for (size_t i = 0; i < params.varOffset.size(); i++) {
    OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : varOffset[%ld]=%ld.", i, params.varOffset[i]);
  }
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : indicesLastDim=%ld.", params.indicesLastDim);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : indicesFrontDim=%ld.", params.indicesFrontDim);
}

bool CheckScatterNdUpdateTensorShape(const std::string& opType, std::vector<int64_t> varShape,
                                     std::vector<int64_t> indicesShape, std::vector<int64_t> updatesShape,
                                     std::vector<int64_t> outShape) {
  if (varShape != outShape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "the length of var must be same as the length of output.");
    return false;
  }

  if (indicesShape.size() == 1 && indicesShape[0] == 1 && varShape.size() - updatesShape.size() == 1) {
    OP_LOGI(opType.c_str(), "Input indices is a scalar.");
  }

  return true;
}
}  // namespace scatterndupdate

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size", "var_size", "indices_size"};

bool ScatterNdUpdateTiling(const std::string& opType, const ge::Operator& opParas, const std::vector<int64_t>& op_info,
                           utils::OpRunInfo& runInfo) {
  using namespace ge;
  using namespace scatterndupdate;
  PROFILING_TILING_INIT(opType.c_str());
  OP_LOGI(opType.c_str(), "ScatterNdUpdateTiling running.");

  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  const std::vector<int64_t>& varShape = input_desc->MutableShape().GetDims();
  const ge::DataType VarDtype = input_desc->GetDataType();

  input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  const std::vector<int64_t>& indicesShape = input_desc->MutableShape().GetDims();

  input_desc = operator_info->MutableInputDesc(SHAPE_UPDATES_INDEX);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  const std::vector<int64_t>& updatesShape = input_desc->MutableShape().GetDims();

  auto output_desc = operator_info->MutableOutputDesc(0);
  OP_TILING_CHECK(output_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get output_desc failed."),
                  return false);
  const std::vector<int64_t>& outShape = output_desc->MutableShape().GetDims();

  bool is_valid_shape = CheckScatterNdUpdateTensorShape(opType, varShape, indicesShape, updatesShape, outShape);
  if (!is_valid_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckScatterNdUpdateTensorShape failed.");
    return false;
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  OP_TILING_CHECK(COMPILE_INFO_KEY.size() != op_info.size(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "parse op_info failed."), return false);
  int64_t coreNum = op_info[0];
  int64_t ubSize = op_info[1];
  int64_t varSize = op_info[VAR_SIZE_INDEX];
  int64_t indicesSize = op_info[INDICES_SIZE_INDEX];
  if (coreNum <= ZERO || ubSize <= ZERO || varSize <= ZERO || indicesSize <= ZERO) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "coreNum, ubSize, varSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum,
        ubSize, varSize, indicesSize);
    return false;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  ScatterNdUpdateTilingParams runParams;
  InitRunningParams(runParams);
  int64_t indicesNum = std::accumulate(indicesShape.begin(), indicesShape.end(), 1, std::multiplies<int>());
  int64_t updatesNum = std::accumulate(updatesShape.begin(), updatesShape.end(), 1, std::multiplies<int>());
  int64_t K = indicesShape.back();
  int64_t updateDataNum =
      (varShape.size() > 1) ? (std::accumulate(varShape.begin() + K, varShape.end(), 1, std::multiplies<int>())) : 1;
  int64_t maxIndice = 1;

  maxIndice = std::accumulate(varShape.begin(), varShape.end(), 1, std::multiplies<int>());

  int64_t varDataEachBlock = BLOCK_SIZE / varSize;
  OP_LOGD(opType.c_str(), "BLOCK_SIZE=%ld.", BLOCK_SIZE);
  OP_LOGD(opType.c_str(), "varSize=%ld.", varSize);
  OP_LOGD(opType.c_str(), "updateDataNum=%ld.", updateDataNum);

  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : indicesNum=%ld.", indicesNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : updatesNum=%ld.", updatesNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : updateDataNum=%ld.", updateDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterNdUpdateTiling] : maxIndice=%ld.", maxIndice);

  runParams.indicesLastDim = indicesShape.back();
  runParams.indicesFrontDim = std::accumulate(indicesShape.begin(), indicesShape.end() - 1, 1, std::multiplies<int>());
  CalRunningParams(runParams, indicesNum, updatesNum, updateDataNum, maxIndice, ubSize, coreNum, varSize, indicesSize,
                   varDataEachBlock);

  for (int64_t i = 0; i < indicesShape.back(); i++) {
    runParams.varOffset[i] = std::accumulate(varShape.begin() + i + 1, varShape.end(), 1, std::multiplies<int>());
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  SetRuningParams(runParams, runInfo);

  PrintTilingParams(opType, runParams);

  runInfo.SetBlockDim(runParams.coreNum);

  OP_LOGI(opType.c_str(), "ScatterNdUpdateTiling run success.");
  PROFILING_TILING_END();

  return true;
}

// register tiling interface of the ScatterNdUpdate op.
REGISTER_OP_TILING_V3_WITH_VECTOR(ScatterNdUpdate, ScatterNdUpdateTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
