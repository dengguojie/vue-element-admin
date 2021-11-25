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
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

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
		      int64_t updatesNum,int64_t updateDataNum, int64_t maxIndice, int64_t ubSize, int64_t coreNum, 
		      int64_t varSize,int64_t indicesSize, int64_t varDataEachBlock, int64_t var_num, 
		      std::string var_dtype, int64_t var_first_shape) {
  int64_t updateSizeByte = varSize * updatesNum;
  int64_t halfUbSize = ubSize / 2;
  OP_TILING_CHECK(varSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_update", "varSize = 0 is not support"),
                  return);
  OP_TILING_CHECK(indicesSize == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_update", "indicesSize = 0 is not support"),
                  return);
  OP_TILING_CHECK(varDataEachBlock == 0, VECTOR_INNER_ERR_REPORT_TILIING("scatter_update", 
		  "varDataEachBlock = 0 is not support"),
                  return);
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
    if (var_dtype == "float16") {
      dtype_bytes_size = BYTE_SIZE_2;
    }
    if (var_dtype == "float32") {
      dtype_bytes_size = BYTE_SIZE_4;
    }
    if (var_dtype == "int32") {
      dtype_bytes_size = BYTE_SIZE_4;
    }
    runParams.each_core_compute_num = runParams.indiceStep * var_num;
    runParams.each_core_loop_num = (runParams.each_core_compute_num * dtype_bytes_size) / halfUbSize;
    if (runParams.each_core_loop_num != 0) {
      runParams.each_core_loop_compute_num = runParams.each_core_compute_num / runParams.each_core_loop_num;
    } else {
      runParams.each_core_loop_compute_num = 0;
    }
    runParams.each_core_last_num = runParams.each_core_compute_num - runParams.each_core_loop_num * \
				      runParams.each_core_loop_compute_num;
    runParams.last_core_compute_num = var_first_shape * var_num - runParams.each_core_compute_num * \ 
	                              (runParams.coreNum - 1);
    runParams.last_core_loop_num = (runParams.last_core_compute_num * dtype_bytes_size) / halfUbSize;
    if (runParams.last_core_loop_num != 0) {
      runParams.last_core_loop_compute_num = runParams.last_core_compute_num / runParams.last_core_loop_num;
    } else {
      runParams.last_core_loop_compute_num = 0;
    }
    runParams.last_core_last_num = runParams.last_core_compute_num - runParams.last_core_loop_num * runParams.last_core_loop_compute_num;
  }
}

void SetRuningParams(const std::string& opType, const ScatterUpdateTilingParams& params, OpRunInfo& runInfo) {
  ByteBufferPut(runInfo.tiling_data, params.tilingMode);
  ByteBufferPut(runInfo.tiling_data, params.indiceStep);
  ByteBufferPut(runInfo.tiling_data, params.coreNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesDataNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.indicesLastNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesLoopNum);
  ByteBufferPut(runInfo.tiling_data, params.updatesLastNum);
  if (opType == "InplaceUpdate") {
    ByteBufferPut(runInfo.tiling_data, params.each_core_compute_num);
    ByteBufferPut(runInfo.tiling_data, params.each_core_loop_num);
    ByteBufferPut(runInfo.tiling_data, params.each_core_loop_compute_num);
    ByteBufferPut(runInfo.tiling_data, params.each_core_last_num);
    ByteBufferPut(runInfo.tiling_data, params.last_core_compute_num);
    ByteBufferPut(runInfo.tiling_data, params.last_core_loop_num);
    ByteBufferPut(runInfo.tiling_data, params.last_core_loop_compute_num);
    ByteBufferPut(runInfo.tiling_data, params.last_core_last_num);
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
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : each_core_compute_num=%ld.", 
            params.each_core_compute_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : each_core_loop_num=%ld.", params.each_core_loop_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : each_core_loop_compute_num=%ld.", params.each_core_loop_compute_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : each_core_last_num=%ld.", params.each_core_last_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : last_core_compute_num=%ld.", params.last_core_compute_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : last_core_loop_num=%ld.", params.last_core_loop_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : last_core_loop_compute_num=%ld.", 
	    params.last_core_loop_compute_num);
    OP_LOGD(opType.c_str(), "op [InplaceUpdateTiling] : last_core_last_num=%ld.", params.last_core_last_num);
  }
}

bool CheckScatterUpdateTensorShape(const std::string& opType, std::vector<int64_t> varShape,
                                   std::vector<int64_t> indicesShape, std::vector<int64_t> updatesShape,
                                   std::vector<int64_t> outShape) {
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
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "updates does not satisfy the relation expression with actualUpdateShape.");
    return false;
  }
  return true;
}

bool GetScatteUpdateCompileParams(const std::string& opType, const nlohmann::json& opCompileInfo, int64_t& coreNum,
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

bool ScatterUpdateTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                         OpRunInfo& runInfo) {
  using namespace ge;
  std::cout << opType << endl;
  OP_LOGI(opType.c_str(), "ScatterUpdateTiling running.");
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
  std::string var_dtype = opParas.inputs[1].tensor[0].dtype;

  int64_t var_num = std::accumulate(varShape.begin() + 1, varShape.end(), 1, std::multiplies<int64_t>());
  int64_t var_first_shape = varShape[0];
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
        opType, "coreNum, ubSize, varSize, indicesSize must be greater to 0, but got %ld, %ld, %ld, %ld", coreNum, ubSize,
        varSize, indicesSize);
    return false;
  }

  ScatterUpdateTilingParams runParams;
  InitRunningParams(runParams);
  int64_t indicesNum = std::accumulate(indicesShape.begin(), indicesShape.end(), 1, std::multiplies<int>());
  int64_t updatesNum = std::accumulate(updatesShape.begin(), updatesShape.end(), 1, std::multiplies<int>());
  int64_t updateDataNum =
      (varShape.size() > 1) ? (std::accumulate(varShape.begin() + 1, varShape.end(), 1, std::multiplies<int>())) : 1;
  int64_t maxIndice = varShape[0];
  int64_t varDataEachBlock = BLOCK_SIZE / varSize;

  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : indicesNum=%ld.", indicesNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : updatesNum=%ld.", updatesNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : updateDataNum=%ld.", updateDataNum);
  OP_LOGD(opType.c_str(), "op [ScatterUpdateTiling] : maxIndice=%ld.", maxIndice);

  CalRunningParams(opType, runParams, indicesNum, updatesNum, updateDataNum, maxIndice, ubSize, coreNum, varSize, 
		  indicesSize, varDataEachBlock, var_num, var_dtype, var_first_shape);

  SetRuningParams(opType, runParams, runInfo);

  PrintTilingParams(opType, runParams);

  runInfo.block_dim = runParams.coreNum;
  std::vector<int64_t> workspace;
  runInfo.workspaces = workspace;

  OP_LOGI(opType.c_str(), "ScatterUpdateTiling run success.");

  return true;
}

// register tiling interface of the ScatterUpdate op.
REGISTER_OP_TILING_FUNC_BUFFERED(ScatterUpdate, ScatterUpdateTiling);
REGISTER_OP_TILING_FUNC_BUFFERED(InplaceUpdate, ScatterUpdateTiling);
}  // namespace optiling
