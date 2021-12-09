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
 * \file scatter_nd_add.cpp
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
namespace scatterndadd {
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

struct ScatterNdAddTilingParams {
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
};

void InitRunningParams(ScatterNdAddTilingParams& params) {
  params.tilingMode = TILING_MODE_1;
  params.indiceStep = 0;
  params.coreNum = 0;
  params.addsDataNum = 0;
  params.indicesLoopNum = 0;
  params.indicesLastNum = 0;
  params.addsNum = 0;
  params.addsLoopNum = 0;
  params.addsLastNum = 0;
}

void CalRunningParams(ScatterNdAddTilingParams& runParams, int64_t indicesNum, int64_t addsNum, int64_t addDataNum,
                      int64_t maxIndice, int64_t ubSize, int64_t coreNum, int64_t varSize, int64_t indicesSize,
                      int64_t varDataEachBlock, const ge::DataType& VarDtype, const std::string& opType) {
  int64_t addSizeByte = varSize * addsNum;
  int64_t halfUbSize = ubSize / 2;
  OP_TILING_CHECK(halfUbSize == 0, VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "halfUbSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(indicesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "indicesSize = 0 is not support"),
                  return );
  OP_TILING_CHECK(coreNum == 0, VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "coreNum = 0 is not support"), return );
  OP_TILING_CHECK(varSize == 0, VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "varSize = 0 is not support"), return );
  OP_TILING_CHECK(runParams.indicesLastDim == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "runParams.indicesLastDim = 0 is not support"),
                  return );
  OP_TILING_CHECK(runParams.indicesLastDim == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "runParams.indicesLastDim = 0 is not support"),
                  return );
  OP_TILING_CHECK(varDataEachBlock == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "varDataEachBlock = 0 is not support"), return );
  int64_t halfUbIndicesNum = halfUbSize / indicesSize;
  OP_TILING_CHECK(halfUbIndicesNum == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "halfUbIndicesNum = 0 is not support"), return );
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

void SetRuningParams(const ScatterNdAddTilingParams& params, utils::OpRunInfo& runInfo) {
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
}

void PrintTilingParams(const std::string& opType, const ScatterNdAddTilingParams& params) {
  OP_LOGD(opType.c_str(), "tilingMode=%ld.", params.tilingMode);
  OP_LOGD(opType.c_str(), "indiceStep=%ld.", params.indiceStep);
  OP_LOGD(opType.c_str(), "coreNum=%ld.", params.coreNum);
  OP_LOGD(opType.c_str(), "addsDataNum=%ld.", params.addsDataNum);
  OP_LOGD(opType.c_str(), "indicesLoopNum=%ld.", params.indicesLoopNum);
  OP_LOGD(opType.c_str(), "indicesLastNum=%ld.", params.indicesLastNum);
  OP_LOGD(opType.c_str(), "addsNum=%ld.", params.addsNum);
  OP_LOGD(opType.c_str(), "addsLoopNum=%ld.", params.addsLoopNum);
  OP_LOGD(opType.c_str(), "addsLastNum=%ld.", params.addsLastNum);
  for (size_t i = 0; i < params.varOffset.size(); i++) {
    OP_LOGD(opType.c_str(), "varOffset[%ld]=%ld.", i, params.varOffset[i]);
  }
  OP_LOGD(opType.c_str(), "indicesLastDim=%ld.", params.indicesLastDim);
  OP_LOGD(opType.c_str(), "indicesFrontDim=%ld.", params.indicesFrontDim);
}

bool CheckScatterNdAddTensorShape(const std::string& opType, const GeShape& varShape, const GeShape& indicesShape,
                                  const GeShape& addsShape, const GeShape& outShape) {
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

}  // namespace scatterndadd

static const std::vector<std::string> COMPILE_INFO_KEY = {"core_num", "ub_size", "var_size", "indices_size"};

bool ScatterNdAddTiling(const std::string& opType, const ge::Operator& opParas, const std::vector<int64_t>& op_info,
                        utils::OpRunInfo& runInfo) {
  using namespace ge;
  using namespace scatterndadd;
  PROFILING_TILING_INIT(opType.c_str());
  OP_LOGI(opType.c_str(), "ScatterNdAddTiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  const GeShape& varShape = input_desc->MutableShape();
  const ge::DataType VarDtype = input_desc->GetDataType();

  input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  const GeShape& indicesShape = input_desc->MutableShape();

  input_desc = operator_info->MutableInputDesc(2);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get input_desc failed."),
                  return false);
  const GeShape& addsShape = input_desc->MutableShape();

  auto output_desc = operator_info->MutableOutputDesc(0);
  OP_TILING_CHECK(output_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opType, "get output_desc failed."),
                  return false);
  const GeShape& outShape = output_desc->MutableShape();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  bool is_valid_shape = CheckScatterNdAddTensorShape(opType, varShape, indicesShape, addsShape, outShape);
  if (!is_valid_shape) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "CheckScatterNdAddTensorShape failed.");
    return false;
  }

  int64_t coreNum = op_info[0];
  int64_t ubSize = op_info[1];
  int64_t varSize = op_info[2];
  int64_t indicesSize = op_info[3];
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  if (coreNum <= ZERO || ubSize <= ZERO || varSize <= ZERO || indicesSize <= ZERO) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        opType, "coreNum, ubSize, varSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum,
        ubSize, varSize, indicesSize);
    return false;
  }

  ScatterNdAddTilingParams runParams;
  InitRunningParams(runParams);
  int64_t indicesNum = indicesShape.GetShapeSize();
  int64_t addsNum = addsShape.GetShapeSize();
  int64_t indicesLastDim = (indicesShape.GetDimNum() > 0) ? indicesShape.GetDim(indicesShape.GetDimNum() - 1) : 0;
  int64_t maxIndice = varShape.GetShapeSize();
  int64_t addDataNum = 1;
  int64_t varDimNum = varShape.GetDimNum();
  if (varDimNum > 1) {
    for (int64_t i = indicesLastDim; i < varDimNum; i++) {
      addDataNum *= varShape.GetDim(i);
    }
  }

  int64_t varDataEachBlock = BLOCK_SIZE / varSize;
  OP_LOGD(opType.c_str(), "BLOCK_SIZE=%ld.", BLOCK_SIZE);
  OP_LOGD(opType.c_str(), "varSize=%ld.", varSize);

  OP_LOGD(opType.c_str(), "indicesNum=%ld.", indicesNum);
  OP_LOGD(opType.c_str(), "addsNum=%ld.", addsNum);
  OP_LOGD(opType.c_str(), "addDataNum=%ld.", addDataNum);
  OP_LOGD(opType.c_str(), "maxIndice=%ld.", maxIndice);

  runParams.indicesLastDim = indicesLastDim;
  if (indicesLastDim > 0) {
    runParams.indicesFrontDim = indicesNum / indicesLastDim;
  } else {
    runParams.indicesFrontDim = 1;
    for (int64_t i = indicesShape.GetDimNum() - 1; i >= 0; i--) {
      runParams.indicesFrontDim *= indicesShape.GetDim(i);
    }
  }
  CalRunningParams(runParams, indicesNum, addsNum, addDataNum, maxIndice, ubSize, coreNum, varSize, indicesSize,
                   varDataEachBlock, VarDtype, opType);

  for (int64_t i = 0; i < indicesLastDim; i++) {
    runParams.varOffset[i] = 1;
    for (int64_t j = i + 1; j < varDimNum; j++)
      runParams.varOffset[i] *= varShape.GetDim(j);
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  SetRuningParams(runParams, runInfo);

  PrintTilingParams(opType, runParams);

  runInfo.SetBlockDim(runParams.coreNum);

  OP_LOGI(opType.c_str(), "Tiling run success.");
  PROFILING_TILING_END();

  return true;
}

// register tiling interface of the ScatterNdAdd op.
REGISTER_OP_TILING_V3_WITH_VECTOR(ScatterNdAdd, ScatterNdAddTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
REGISTER_OP_TILING_V3_WITH_VECTOR(ScatterNdSub, ScatterNdAddTiling, COMPILE_INFO_KEY, NO_OPTIONAL_VALUE);
}  // namespace optiling
