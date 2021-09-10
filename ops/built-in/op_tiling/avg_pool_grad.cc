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
 * \file avg_pool_grad.cc
 * \brief tiling function of avg_pool_grad
 */
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "cube_tiling_new.h"
#include "graph/debug/ge_log.h"
#include "external/graph/operator.h"
#include "op_tiling.h"
#include "op_log.h"

namespace optiling {
/*
 * @brief: tiling function of cube operators
 * @param [in] compile_info: compile time generated info of operator
 * @param [out] opInfo: merge compile time generated info of operator
 * @return : void
 */
void merge_compile_info(const nlohmann::json& compile_info, nlohmann::json &opInfo)
{
  if (compile_info.is_object()) {
    opInfo = compile_info;
  } else if (compile_info.is_array()) {
    nlohmann::json item;
    opInfo = compile_info[0];
    for (std::size_t i = 1; i < compile_info.size(); i++) {
      item = compile_info[i];
      std::vector<std::string> key_list = {"repo_seeds", "repo_range", "cost_range"};
      for (auto key : key_list) {
        if (item[key].is_object() && !item[key].empty()) {
          std::vector<int32_t> list_value = item[key].begin().value().get<std::vector<int32_t>>();
          opInfo[key][item[key].begin().key()] = list_value;
        }
      }
      std::vector<std::string> key_int = {"block_dim"};
      for (auto key: key_int) {
        if (item[key].is_object() && !item[key].empty()) {
          int32_t int_value = item[key].begin().value().get<int32_t>();
          opInfo[key][item[key].begin().key()] = int_value;
        }
      }
    }
  }
}

/*
 * @brief: tiling function of avg_pool_grad
 * @param [in] op_type: op_type of avg_pool_grad
 * @param [in] op_paras: inputs/outputs/atts of avg_pool_grad
 * @param [in] op_compile_info: compile time generated info of avg_pool_grad
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool AvgPoolGradTiling(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                         utils::OpRunInfo& runInfo) {
  int32_t nDim = 0;
  int32_t hDim = 2;
  int32_t wDim = 3;
  if (opParas.GetInputsSize() < 2 || opParas.GetOutputsSize() == 0 ||
      opParas.GetInputDesc(1).GetShape().GetDimNum() < 4 || opParas.GetOutputDesc(0).GetShape().GetDimNum() < 4){
    return false;
  }

  try {
    if (opCompileInfo.empty()) {
      GELOGD("op compile info is empty");
      return false;
    }

    nlohmann::json opInfo;
    merge_compile_info(opCompileInfo, opInfo);

    std::vector<std::string>varMap = opInfo.at("_vars").begin().value().get<std::vector<std::string>>();

    std::vector<int64_t> var_value;
    if (std::find(varMap.begin(), varMap.end(), "batch_n") != varMap.end()) {
        var_value.insert(var_value.end(), opParas.GetOutputDesc(0).GetShape().GetDim(nDim));
    }
    if (std::find(varMap.begin(), varMap.end(), "dx_h") != varMap.end()) {
      var_value.insert(var_value.end(), opParas.GetInputDesc(1).GetShape().GetDim(hDim));
      var_value.insert(var_value.end(), opParas.GetOutputDesc(0).GetShape().GetDim(hDim));
    }
    if (std::find(varMap.begin(), varMap.end(), "dx_w") != varMap.end()) {
      var_value.insert(var_value.end(), opParas.GetInputDesc(1).GetShape().GetDim(wDim));
      var_value.insert(var_value.end(), opParas.GetOutputDesc(0).GetShape().GetDim(wDim));
    }

    return cube_tiling(opType,
                       opParas.GetOutputDesc(0).GetShape().GetDims(),
                       var_value,
                       opInfo,
                       runInfo);
  } catch (...) {
    GELOGD("get unknown exception, please check compile info json.");
    return false;
  }
}

// register tiling interface of the avg_pool_grad
REGISTER_OP_TILING_FUNC_BUFFERED_V2(AvgPoolGrad, AvgPoolGradTiling);
}  // namespace optiling
