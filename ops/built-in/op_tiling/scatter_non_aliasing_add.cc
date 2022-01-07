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
 * \file scatter_non_aliasing_add.cpp
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
// varsize intput compile params
const int64_t VARSIZE_COMPILE_INDEX = 2;
// indicessize intput compile params
const int64_t INDICESSIZE_COMPILE_INDEX = 3;

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
                      int64_t indicesSize, int64_t varDataEachBlock, const ge::DataType VarDtype) {
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
  if (VarDtype == ge::DT_INT64 || VarDtype == ge::DT_UINT64) {
    VarDtypeSize = 8;
  } else if (VarDtype == ge::DT_FLOAT16) {
    VarDtypeSize = 2;
  } else if (VarDtype == ge::DT_INT8 || VarDtype == ge::DT_UINT8) {
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

void SetRuningParams(const ScatterNonAliasingAddTilingParams& params, utils::OpRunInfo& runInfo) {
  runInfo.AddTilingData(params.tilingMode);
  runInfo.AddTilingData(params.indiceStep);
  runInfo.AddTilingData(params.coreNum);
  runInfo.AddTilingData(params.addsDataNum);
  runInfo.AddTilingData(params.indicesLoopNum);
  runInfo.AddTilingData(params.indicesLastNum);
  runInfo.AddTilingData(params.addsNum);
  runInfo.AddTilingData(params.addsLoopNum);
  runInfo.AddTilingData(params.addsLastNum);
  for (size_t i = 0; i < params.varOffset.size(); i++) {
    runInfo.AddTilingData(params.varOffset[i]);
  }
  runInfo.AddTilingData(params.indicesLastDim);
  runInfo.AddTilingData(params.indicesFrontDim);
  runInfo.AddTilingData(params.varNum);
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

bool CheckScatterNonAliasingAddTensorShape(const std::string& opType, const GeShape& varShape,
                                           const GeShape& indicesShape, const GeShape& addsShape,
                                           const GeShape& outShape) {
  if (!(varShape == outShape)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "the length of var must be same as the length of output.");
    return false;
  }

  if (indicesShape.GetDimNum() == 1 && indicesShape.GetDim(0) == 1 &&
      varShape.GetDimNum() - addsShape.GetDimNum() == 1) {
    OP_LOGI(opType.c_str(), "Input indices is a scalar.");
  }

  return true;
}

int64_t forMul(const GeShape& inputShape, const int64_t& begin, const int64_t& end) {
  int64_t addIndices = 1;
  for (int64_t i = begin; i < end; i++) {
    addIndices *= inputShape.GetDim(i);
  }
  return addIndices;
}
}  // namespace scatternonaliasingadd
static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size", "var_size", "indices_size"};

bool ScatterNonAliasingAddTiling(const std::string& opType, const ge::Operator& opParas,
                                 const std::vector<int64_t>& op_info, utils::OpRunInfo& runInfo) {
  using namespace scatternonaliasingadd;
  OP_LOGI(opType.c_str(), "ScatterNonAliasingAddTiling running.");
  PROFILING_TILING_INIT(opType.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(opParas);

  OP_TILING_CHECK(COMPILE_INFO_KEY.size() != op_info.size(),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType, "parse op_info failed."), return false);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info failed."),
                  return false);

  auto var_desc = operator_info->MutableInputDesc(0);
  auto indices_desc = operator_info->MutableInputDesc(1);
  auto adds_desc = operator_info->MutableInputDesc(VARSIZE_COMPILE_INDEX);
  auto out_desc = operator_info->MutableOutputDesc(0);

  OP_TILING_CHECK(var_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get var_desc shape error."),
                  return false);
  OP_TILING_CHECK(indices_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get indices_desc shape error."),
                  return false);
  OP_TILING_CHECK(adds_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get adds_desc shape error."),
                  return false);
  OP_TILING_CHECK(out_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get output_desc shape error."),
                  return false);

  const GeShape& varShape = var_desc->MutableShape();
  const GeShape& indicesShape = indices_desc->MutableShape();
  const GeShape& addsShape = adds_desc->MutableShape();
  const GeShape& outShape = out_desc->MutableShape();
  const ge::DataType VarDtype = var_desc->GetDataType();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  bool is_valid_shape = CheckScatterNonAliasingAddTensorShape(opType, varShape, indicesShape, addsShape, outShape);
  if (!is_valid_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckScatterNonAliasingAddTensorShape failed.");
    return false;
  }

  int64_t coreNum = op_info[0];
  int64_t ubSize = op_info[1];
  int64_t varSize = op_info[VARSIZE_COMPILE_INDEX];
  int64_t indicesSize = op_info[INDICESSIZE_COMPILE_INDEX];
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  if (coreNum <= ZERO || ubSize <= ZERO || varSize <= ZERO || indicesSize <= ZERO) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "coreNum, ubSize, varSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum,
        ubSize, varSize, indicesSize);
    return false;
  }

  ScatterNonAliasingAddTilingParams runParams;
  InitRunningParams(runParams);
  int64_t indicesNum = GetTensorSize(indicesShape);
  int64_t addsNum = GetTensorSize(addsShape);
  int64_t K = indicesShape.GetDim(indicesShape.GetDimNum() - 1);
  int64_t addDataNum = forMul(varShape, K, varShape.GetDimNum());
  int64_t maxIndice = GetTensorSize(varShape);
  int64_t varDataEachBlock = BLOCK_SIZE / varSize;
  OP_LOGD(opType.c_str(), "BLOCK_SIZE=%ld.", BLOCK_SIZE);
  OP_LOGD(opType.c_str(), "varSize=%ld.", varSize);
  OP_LOGD(opType.c_str(), "addDataNum=%ld.", addDataNum);

  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : indicesNum=%ld.", indicesNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : addsNum=%ld.", addsNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : addDataNum=%ld.", addDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterNonAliasingAddTiling] : maxIndice=%ld.", maxIndice);

  runParams.indicesLastDim = K;
  int64_t addIndices = forMul(indicesShape, 0, indicesShape.GetDimNum() - 1);
  runParams.indicesFrontDim = addIndices;
  CalRunningParams(runParams, indicesNum, addsNum, addDataNum, maxIndice, ubSize, coreNum, varSize, indicesSize,
                   varDataEachBlock, VarDtype);

  for (int64_t i = 0; i < K; i++) {
    runParams.varOffset[i] = forMul(varShape, i + 1, varShape.GetDimNum());
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  SetRuningParams(runParams, runInfo);
  PrintTilingParams(opType, runParams);
  runInfo.SetBlockDim(runParams.coreNum);
  OP_LOGI(opType.c_str(), "ScatterNonAliasingAddTiling run success.");
  PROFILING_TILING_END();
  return true;
}

// register tiling interface of the ScatterNonAliasingAdd op.
REGISTER_OP_TILING_V3_WITH_VECTOR(ScatterNonAliasingAdd, ScatterNonAliasingAddTiling, COMPILE_INFO_KEY,
                                  NO_OPTIONAL_VALUE);
}  // namespace optiling
