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
 * \file conv2d.cpp
 * \brief tiling function of conv2d
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
 * @brief: tiling function of conv2d
 * @param [in] op_type: op_type of the conv2d
 * @param [in] op_paras: inputs/outputs/atts of the conv2d
 * @param [in] op_compile_info: compile time generated info of the conv2d
 * @param [out] run_info: result data
 * @return bool: success or not
 */

bool Conv2DTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                  OpRunInfo& runInfo) {
  int32_t nDim = 0;
  int32_t hDim = 2;
  int32_t wDim = 3;

  if (opParas.inputs.empty() || opParas.outputs.empty() || opParas.inputs[0].tensor.empty() ||
      opParas.outputs[0].tensor.empty() || opParas.inputs[0].tensor[0].shape.empty() ||
      opParas.outputs[0].tensor[0].shape.empty()) {
    return false;
  }

  if (opCompileInfo.contains("fmap_c1") && opParas.inputs[0].tensor[0].shape[1] != opCompileInfo["fmap_c1"]) {
    OP_LOGE(opType.c_str(), "Not support, input x channel should be equal to filter channel*groups");
    return false;
  }

  if(opCompileInfo.empty()) {
    GELOGD("op compile info is empty");
    return false;
  }

  int32_t batch = opParas.inputs[0].tensor[0].shape[nDim];
  int32_t hi = opParas.inputs[0].tensor[0].shape[hDim];
  int32_t wi = opParas.inputs[0].tensor[0].shape[wDim];
  int32_t ho = opParas.outputs[0].tensor[0].shape[hDim];
  int32_t wo = opParas.outputs[0].tensor[0].shape[wDim];
  // accurate build has only one item
  // fuzzy build has multiple items
  std::vector<std::string> varMap;
  nlohmann::json opInfo;
  GELOGD("original compile info is: %s", opCompileInfo.dump().c_str());
  if (opCompileInfo.is_array()) {
    // >>> start: splice compile info
    opInfo = opCompileInfo[0];
    varMap = opInfo.at("_vars").begin().value().get<std::vector<std::string>>();
    nlohmann::json item;
    for (size_t i = 1; i < opCompileInfo.size(); ++i) {
      item = opCompileInfo[i];
      std::vector<std::string> key_list = {"repo_seeds", "repo_range", "cost_range"};
      for (auto key: key_list) {
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
    // <<< end: put together compile info
    GELOGD("compile info after splice is: %s", opInfo.dump().c_str());
  } else if (opCompileInfo.is_object()) {
    varMap = opCompileInfo.at("_vars")["10000"].get<std::vector<std::string>>();
    opInfo = opCompileInfo;
  }

  std::vector<int64_t> var_value;
  if (std::find(varMap.begin(), varMap.end(), "batch_n") != varMap.end()) {
    var_value.insert(var_value.end(), batch);
  }
  if (std::find(varMap.begin(), varMap.end(), "fmap_h") != varMap.end()) {
    var_value.insert(var_value.end(), hi);
    var_value.insert(var_value.end(), ho);
  }
  if (std::find(varMap.begin(), varMap.end(), "fmap_w") != varMap.end()) {
    var_value.insert(var_value.end(), wi);
    var_value.insert(var_value.end(), wo);
  }

  bool res = cube_tiling(opType, opParas.inputs[0].tensor[0].shape, var_value, opInfo, runInfo);
  GELOGD("conv2d tiling_data is %d, %d, %d, %d, %d, %d", runInfo.tiling_key, batch, hi, ho, wi, wo);

  return res;
}

// register tiling interface of the conv2d
REGISTER_OP_TILING_FUNC_BUFFERED(Conv2D, Conv2DTiling);
}  // namespace optiling
