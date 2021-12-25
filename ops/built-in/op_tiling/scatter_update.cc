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
 * \file scatter_update.cpp
 * \brief
 */
#include <string>
#include <math.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"

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
// div 0 check
const int64_t ZERO = 0;
constexpr int32_t BYTE_SIZE_2 = 2;
constexpr int32_t BYTE_SIZE_4 = 4;
constexpr int32_t VARSIZE_INDEX = 2;
constexpr int32_t UPDATES_DESC_INDEX = 2;
constexpr int32_t INDICE_DESC_INDEX = 1;
constexpr int32_t INDICE_INDEX = 3;
// define the compile key of json.vars
static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size", "var_size", "indices_size"};

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
  int64_t each_core_compute_num;
  int64_t each_core_loop_num;
  int64_t each_core_loop_compute_num;
  int64_t each_core_last_num;
  int64_t last_core_compute_num;
  int64_t last_core_loop_num;
  int64_t last_core_loop_compute_num;
  int64_t last_core_last_num;
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
  params.each_core_compute_num = 0;
  params.each_core_loop_num = 0;
  params.each_core_loop_compute_num = 0;
  params.each_core_last_num = 0;
  params.last_core_compute_num = 0;
  params.last_core_loop_num = 0;
  params.last_core_loop_compute_num = 0;
  params.last_core_last_num = 0;
}

void CalRunningParams(const std::string& opType, ScatterUpdateTilingParams& runParams, int64_t indicesNum,
                      int64_t updatesNum, int64_t updateDataNum, int64_t maxIndice, int64_t ubSize, int64_t coreNum,
                      int64_t varSize, int64_t indicesSize, int64_t varDataEachBlock, int64_t var_num,
                      ge::DataType var_dtype, int64_t var_first_shape) {
  int64_t updateSizeByte = varSize * updatesNum;
  int64_t halfUbSize = ubSize / 2;
  OP_TILING_CHECK(varSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_update", "varSize = 0 is not support"),
                  return);
  OP_TILING_CHECK(indicesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_update", "indicesSize = 0 is not support"),
                  return);
  OP_TILING_CHECK(varDataEachBlock == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING("scatter_update", "varDataEachBlock = 0 is not support"), return );
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
  if (opType == "InplaceUpdate") {
    int64_t dtype_bytes_size = 0;
    if (var_dtype == ge::DT_FLOAT16) {
      dtype_bytes_size = BYTE_SIZE_2;
    } else if (var_dtype == ge::DT_FLOAT) {
      dtype_bytes_size = BYTE_SIZE_4;
    } else if (var_dtype == ge::DT_INT32) {
      dtype_bytes_size = BYTE_SIZE_4;
    }
    runParams.each_core_compute_num = runParams.indiceStep * var_num;
    runParams.each_core_loop_num = (runParams.each_core_compute_num * dtype_bytes_size) / halfUbSize;
    if (runParams.each_core_loop_num != 0) {
      runParams.each_core_loop_compute_num = runParams.each_core_compute_num / runParams.each_core_loop_num;
    } else {
      runParams.each_core_loop_compute_num = 0;
    }
    runParams.each_core_last_num =
        runParams.each_core_compute_num - runParams.each_core_loop_num * runParams.each_core_loop_compute_num;
    runParams.last_core_compute_num = var_first_shape * var_num - runParams.each_core_compute_num * \
	                              (runParams.coreNum - 1);
    runParams.last_core_loop_num = (runParams.last_core_compute_num * dtype_bytes_size) / halfUbSize;
    if (runParams.last_core_loop_num != 0) {
      runParams.last_core_loop_compute_num = runParams.last_core_compute_num / runParams.last_core_loop_num;
    } else {
      runParams.last_core_loop_compute_num = 0;
    }
    runParams.last_core_last_num =
        runParams.last_core_compute_num - runParams.last_core_loop_num * runParams.last_core_loop_compute_num;
  }
}

void SetRuningParams(const std::string& opType, const ScatterUpdateTilingParams& params, utils::OpRunInfo& runInfo) {
  runInfo.AddTilingData(params.tilingMode);
  runInfo.AddTilingData(params.indiceStep);
  runInfo.AddTilingData(params.coreNum);
  runInfo.AddTilingData(params.updatesDataNum);
  runInfo.AddTilingData(params.indicesLoopNum);
  runInfo.AddTilingData(params.indicesLastNum);
  runInfo.AddTilingData(params.updatesNum);
  runInfo.AddTilingData(params.updatesLoopNum);
  runInfo.AddTilingData(params.updatesLastNum);
  if (opType == "InplaceUpdate") {
    runInfo.AddTilingData(params.each_core_compute_num);
    runInfo.AddTilingData(params.each_core_loop_num);
    runInfo.AddTilingData(params.each_core_loop_compute_num);
    runInfo.AddTilingData(params.each_core_last_num);
    runInfo.AddTilingData(params.last_core_compute_num);
    runInfo.AddTilingData(params.last_core_loop_num);
    runInfo.AddTilingData(params.last_core_loop_compute_num);
    runInfo.AddTilingData(params.last_core_last_num);
  }
}

void PrintTilingParams(const std::string& opType, const ScatterUpdateTilingParams& params) {
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : tilingMode=%ld.", params.tilingMode);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : indiceStep=%ld.", params.indiceStep);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : coreNum=%ld.", params.coreNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : updatesDataNum=%ld.", params.updatesDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : indicesLoopNum=%ld.", params.indicesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : indicesLastNum=%ld.", params.indicesLastNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : updatesNum=%ld.", params.updatesNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : updatesLoopNum=%ld.", params.updatesLoopNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : updatesLastNum=%ld.", params.updatesLastNum);
  if (opType == "InplaceUpdate") {
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : each_core_compute_num=%ld.", params.each_core_compute_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : each_core_loop_num=%ld.", params.each_core_loop_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : each_core_loop_compute_num=%ld.",
            params.each_core_loop_compute_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : each_core_last_num=%ld.", params.each_core_last_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : last_core_compute_num=%ld.", params.last_core_compute_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : last_core_loop_num=%ld.", params.last_core_loop_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : last_core_loop_compute_num=%ld.",
            params.last_core_loop_compute_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : last_core_last_num=%ld.", params.last_core_last_num);
  }
}

bool CheckScatterUpdateTensorShape(const std::string& opType, const GeShape& varShape, const GeShape& indicesShape,
                                   const GeShape& updatesShape, const GeShape& outShape) {
  if (!(varShape == outShape)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "the length of var must be same as the length of output.");
    return false;
  }

  if (indicesShape.GetDimNum() == 1 && indicesShape.GetDim(0) == 1 &&
      varShape.GetDimNum() - updatesShape.GetDimNum() == 1) {
    OP_LOGI(opType.c_str(), "Input indices is a scalar.");
    return true;
  }

  for (size_t i = 0; i < indicesShape.GetDimNum(); i++) {
    OP_TILING_CHECK(
      updatesShape.GetDim(i) != indicesShape.GetDim(i),
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "the updatesShape[%zu] is not equal to indicesShape[%zu].",
                                      i, i),
      return false);
  }

  for (size_t j = 1; j < varShape.GetDimNum(); j++) {
    int64_t index = j + indicesShape.GetDimNum() - 1;
    OP_TILING_CHECK(
      updatesShape.GetDim(index) != varShape.GetDim(j),
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "the updatesShape[%zu] is not equal to varShape[%zu].",
                                      index, j),
      return false);
  }
  return true;
}

bool GetScatteUpdateCompileParams(const std::string& opType, const std::vector<int64_t>& opCompileInfo,
                                  int64_t& coreNum, int64_t& ubSize, int64_t& varSize, int64_t& indicesSize) {
  OP_TILING_CHECK(
      opCompileInfo.size() != COMPILE_INFO_KEY.size(),
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "the compile info num is not equal expect compile_info(%zu), is %zu",
                                      COMPILE_INFO_KEY.size(), opCompileInfo.size()),
      return false);

  coreNum = opCompileInfo[0];
  ubSize = opCompileInfo[1];
  varSize = opCompileInfo[VARSIZE_INDEX];
  indicesSize = opCompileInfo[INDICE_INDEX];

  return true;
}

bool ScatterUpdateTiling(const std::string& opType, const ge::Operator& opParas,
                         const std::vector<int64_t>& opCompileInfo, utils::OpRunInfo& runInfo) {
  using namespace ge;
  PROFILING_TILING_INIT(opType.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(opParas);
  OP_LOGI(opType.c_str(), "ScatterUpdateTiling running.");
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get OpDesc failed."),
                  return false);

  auto var_desc = operator_info->MutableInputDesc(0);
  auto indices_desc = operator_info->MutableInputDesc(INDICE_DESC_INDEX);
  auto updates_desc = operator_info->MutableInputDesc(UPDATES_DESC_INDEX);
  OP_TILING_CHECK(var_desc == nullptr || indices_desc == nullptr || updates_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "get InputDesc failed."), return false);

  auto out_desc = operator_info->MutableOutputDesc(0);
  OP_TILING_CHECK(out_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get OutputDesc failed."), return false);

  const GeShape& varShape = var_desc->MutableShape();
  const GeShape& indicesShape = indices_desc->MutableShape();
  const GeShape& updatesShape = updates_desc->MutableShape();
  const GeShape& outShape = out_desc->MutableShape();
  ge::DataType var_dtype = indices_desc->GetDataType();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  int64_t var_first_shape = varShape.GetDim(0);
  int64_t var_num = 1;
  for (size_t i = 1; i < varShape.GetDimNum(); i++) {
    var_num = var_num * varShape.GetDim(i);
  }

  bool is_valid_shape = CheckScatterUpdateTensorShape(opType, varShape, indicesShape, updatesShape, outShape);
  if (!is_valid_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckScatterUpdateTensorShape failed.");
    return false;
  }

  int64_t coreNum = 0;
  int64_t ubSize = 0;
  int64_t varSize = 0;
  int64_t indicesSize = 0;
  bool can_get_params = GetScatteUpdateCompileParams(opType, opCompileInfo, coreNum, ubSize, varSize, indicesSize);
  if (!can_get_params) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetScatteUpdateCompileParams error.");
    return false;
  }
  if (coreNum <= ZERO || ubSize <= ZERO || varSize <= ZERO || indicesSize <= ZERO) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "coreNum, ubSize, varSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum,
        ubSize, varSize, indicesSize);
    return false;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  ScatterUpdateTilingParams runParams;
  InitRunningParams(runParams);
  int64_t indicesNum = GetTensorSize(indicesShape);
  int64_t updatesNum = updatesShape.GetShapeSize();
  int64_t updateDataNum = (varShape.GetDimNum() > 1) ? var_num : 1;
  int64_t maxIndice = varShape.GetDim(0);
  int64_t varDataEachBlock = BLOCK_SIZE / varSize;

  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : indicesNum=%ld.", indicesNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : updatesNum=%ld.", updatesNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : updateDataNum=%ld.", updateDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : maxIndice=%ld.", maxIndice);

  CalRunningParams(opType, runParams, indicesNum, updatesNum, updateDataNum, maxIndice, ubSize, coreNum, varSize,
                   indicesSize, varDataEachBlock, var_num, var_dtype, var_first_shape);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  SetRuningParams(opType, runParams, runInfo);

  PrintTilingParams(opType, runParams);

  runInfo.SetBlockDim(runParams.coreNum);
  PROFILING_TILING_END();
  OP_LOGI(opType.c_str(), "ScatterUpdateTiling run success.");

  return true;
}

// register tiling interface of the ScatterUpdate op.
REGISTER_OP_TILING_V3_WITH_VECTOR(ScatterUpdate, ScatterUpdateTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
REGISTER_OP_TILING_V3_WITH_VECTOR(InplaceUpdate, ScatterUpdateTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
