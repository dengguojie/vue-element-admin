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
 * \file avg_pool.cc
 * \brief tiling function of avg_pool
 */
#include <string>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "cube_tiling.h"
#include "op_log.h"

using namespace std;
namespace optiling {
/*
 * @brief: tiling function of avg_pool
 * @param [in] op_type: op_type of the avg_pool
 * @param [in] op_paras: inputs/outputs/atts of the avg_pool
 * @param [in] op_compile_info: compile time generated info of the avg_pool
 * @param [out] run_info: result data
 * @return bool: success or not
 */

bool AvgPoolTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                  OpRunInfo& runInfo) {
  int32_t nDim = 0;
  int32_t hDim = 2;
  int32_t wDim = 3;

  if (opParas.inputs.empty() || opParas.outputs.empty() || opParas.inputs[0].tensor.empty() ||
      opParas.outputs[0].tensor.empty() || opParas.inputs[0].tensor[0].shape.empty() ||
      opParas.outputs[0].tensor[0].shape.empty()) {
    return false;
  }

  if(opCompileInfo.empty()) {
    GELOGD("op compile info is empty");
    return false;
  }
  // accurate build has only one item
  // fuzzy build has multiple items
  std::vector<std::string> varMap;
  GELOGD("original compile info is: %s", opCompileInfo.dump().c_str());

  varMap = opCompileInfo.at("_vars").begin().value().get<std::vector<std::string>>();

  std::vector<int64_t> var_value;
  if (std::find(varMap.begin(), varMap.end(), "batch_n") != varMap.end()) {
    var_value.insert(var_value.end(), opParas.inputs[0].tensor[0].shape[nDim]);
  }
  if (std::find(varMap.begin(), varMap.end(), "fmap_h") != varMap.end()) {
    var_value.insert(var_value.end(), opParas.inputs[0].tensor[0].shape[hDim]);
    var_value.insert(var_value.end(), opParas.outputs[0].tensor[0].shape[hDim]);
  }
  if (std::find(varMap.begin(), varMap.end(), "fmap_w") != varMap.end()) {
    var_value.insert(var_value.end(), opParas.inputs[0].tensor[0].shape[wDim]);
    var_value.insert(var_value.end(), opParas.outputs[0].tensor[0].shape[wDim]);
  }

  return cube_tiling(opType, opParas.inputs[0].tensor[0].shape, var_value, opCompileInfo, runInfo);
}
// register tiling interface of the avgpool
REGISTER_OP_TILING_FUNC_BUFFERED(AvgPool, AvgPoolTiling);
}  // namespace optiling