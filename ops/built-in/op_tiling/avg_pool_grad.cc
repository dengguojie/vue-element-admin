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
 * \file avg_pool_grad.cc
 * \brief tiling function of avg_pool_grad
 */
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "cube_tiling.h"
#include "op_log.h"

namespace optiling {
/*
 * @brief: tiling function of avg_pool_grad
 * @param [in] op_type: op_type of avg_pool_grad
 * @param [in] op_paras: inputs/outputs/atts of avg_pool_grad
 * @param [in] op_compile_info: compile time generated info of avg_pool_grad
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool AvgPoolGradTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                         OpRunInfo& runInfo) {
  int32_t nDim = 0;
  int32_t hDim = 2;
  int32_t wDim = 3;

  if (opParas.inputs.empty() || opParas.outputs.empty() || opParas.inputs.size() < 3 ||
      opParas.inputs[1].tensor.empty() || opParas.outputs[0].tensor.empty() ||
      opParas.inputs[1].tensor[0].shape.empty() || opParas.inputs[1].tensor[0].shape.size() < 4 ||
      opParas.outputs[0].tensor[0].shape.empty() || opParas.outputs[0].tensor[0].shape.size() < 4) {
    return false;
  }
  std::vector<std::string>varMap = opCompileInfo.at("_vars")["10000"];
  std::vector<int64_t> var_value;
  if (std::find(varMap.begin(), varMap.end(), "batch_n") != varMap.end()) {
    var_value.insert(var_value.end(), opParas.outputs[0].tensor[0].shape[nDim]);
  }
  if (std::find(varMap.begin(), varMap.end(), "dx_h") != varMap.end()) {
    var_value.insert(var_value.end(), opParas.inputs[1].tensor[0].shape[hDim]);
    var_value.insert(var_value.end(), opParas.outputs[0].tensor[0].shape[hDim]);
  }
  if (std::find(varMap.begin(), varMap.end(), "dx_w") != varMap.end()) {
    var_value.insert(var_value.end(), opParas.inputs[1].tensor[0].shape[wDim]);
    var_value.insert(var_value.end(), opParas.outputs[0].tensor[0].shape[wDim]);
  }

  return cube_tiling(opType, opParas.outputs[0].tensor[0].shape, var_value, opCompileInfo, runInfo);
}

// register tiling interface of the avg_pool_grad
REGISTER_OP_TILING_FUNC_BUFFERED(AvgPoolGrad, AvgPoolGradTiling);
}  // namespace optiling