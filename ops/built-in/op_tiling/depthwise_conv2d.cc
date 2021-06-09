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
 * \file depthwise_conv2d.cc
 * \brief tiling function of depthwiseconv2d
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
 * @brief: tiling function of depthwiseconv2d
 * @param [in] op_type: op_type of the depthwiseconv2d
 * @param [in] op_paras: inputs/outputs/atts of the depthwiseconv2d
 * @param [in] op_compile_info: compile time generated info of the depthwiseconv2d
 * @param [out] run_info: result data
 * @return bool: success or not
 */

bool DepthwiseConv2DTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                           OpRunInfo& runInfo) {
  if (opParas.inputs.empty() || opParas.outputs.empty() || opParas.inputs[0].tensor.empty() ||
      opParas.outputs[0].tensor.empty() || opParas.inputs[0].tensor[0].shape.empty() ||
      opParas.outputs[0].tensor[0].shape.empty()) {
    return false;
  }

  GELOGD("Current format is %s, Ori format is %s",
         opParas.inputs[0].tensor[0].format.c_str(),
         opParas.inputs[0].tensor[0].ori_format.c_str());

  int32_t nDim = 0;
  int32_t hDim = 2;
  int32_t wDim = 3;

  int32_t batch = opParas.inputs[0].tensor[0].shape[nDim];
  int32_t hi = opParas.inputs[0].tensor[0].shape[hDim];
  int32_t wi = opParas.inputs[0].tensor[0].shape[wDim];
  int32_t ho = opParas.outputs[0].tensor[0].shape[hDim];
  int32_t wo = opParas.outputs[0].tensor[0].shape[wDim];

  std::vector<std::string> varMap = opCompileInfo.at("_vars")["10000"];
  std::vector<int64_t> var_value;
  for (auto var:varMap) {
    if (var == "batch_n") {
      var_value.insert(var_value.end(), batch);
    } else if (var == "fmap_h") {
      var_value.insert(var_value.end(), hi);
    } else if (var == "fmap_w") {
      var_value.insert(var_value.end(), wi);
    } else if (var == "ho") {
      var_value.insert(var_value.end(), ho);
    } else if (var == "wo") {
      var_value.insert(var_value.end(), wo);
    }
  }

  bool res = cube_tiling(opType, opParas.inputs[0].tensor[0].shape, var_value, opCompileInfo, runInfo);
  GELOGD("depthwiseconv2d tiling_data is %d, %d, %d, %d, %d, %d", runInfo.tiling_key, batch, hi, ho, wi, wo);
  return res;
}

// register tiling interface of the conv2d
REGISTER_OP_TILING_FUNC_BUFFERED(DepthwiseConv2D, DepthwiseConv2DTiling);
}  // namespace optiling
