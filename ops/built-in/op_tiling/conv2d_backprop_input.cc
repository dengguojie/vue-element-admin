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
 * \file conv2d_backprop_input.cpp
 * \brief tiling function of conv2d_backprop_input
 */
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "register/op_tiling.h"
#include "graph/debug/ge_log.h"
#include "conv_tiling.h"
#include "op_log.h"

namespace optiling {
/*
 * @brief: tiling function of conv2d_backprop_input
 * @param [in] op_type: op_type of conv2d_backprop_input
 * @param [in] op_paras: inputs/outputs/atts of conv2d_backprop_input
 * @param [in] op_compile_info: compile time generated info of conv2d_backprop_input
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool Conv2DBpInputTiling(const std::string& opType, const TeOpParas& opParas, const nlohmann::json& opCompileInfo,
                         OpRunInfo& runInfo) {
  if (opParas.inputs.empty() || opParas.outputs.empty() || opParas.inputs[2].tensor.empty() ||
      opParas.outputs[0].tensor.empty() || opParas.inputs[2].tensor[0].shape.empty() ||
      opParas.outputs[0].tensor[0].shape.empty()) {
    return false;
  }
  std::string mode = opCompileInfo["dynamic_mode"].get<std::string>();
  GELOGD("dynamic_mode is [%s]", mode.c_str());
  GELOGD("Current format is %s, Ori format is %s", opParas.outputs[0].tensor[0].format.c_str(),
         opParas.outputs[0].tensor[0].ori_format.c_str());

  int32_t tilingId = 0;
  if (mode == "dynamic_hw") {
    int32_t dedyH = opParas.inputs[2].tensor[0].shape[2];
    int32_t dedyW = opParas.inputs[2].tensor[0].shape[3];
    int32_t dxH = opParas.outputs[0].tensor[0].shape[2];
    int32_t dxW = opParas.outputs[0].tensor[0].shape[3];
    tilingId = ConvTiling({dxH, dxW}, mode, opCompileInfo, runInfo);
    ByteBufferPut(runInfo.tiling_data, tilingId);
    ByteBufferPut(runInfo.tiling_data, dedyH);
    ByteBufferPut(runInfo.tiling_data, dedyW);
    ByteBufferPut(runInfo.tiling_data, dxH);
    ByteBufferPut(runInfo.tiling_data, dxW);

    GELOGD("tiling_data is %d, %d, %d, %d, %d", tilingId, dedyH, dedyW, dxH, dxW);
  } else if (mode == "dynamic_batch") {
    int32_t batch = opParas.outputs[0].tensor[0].shape[0];
    tilingId = ConvTiling({batch}, mode, opCompileInfo, runInfo);
    ByteBufferPut(runInfo.tiling_data, tilingId);
    ByteBufferPut(runInfo.tiling_data, batch);

    GELOGD("Input info is %d, %d", tilingId, batch);
  } else {
    OP_LOGE("mode: %s is not supported", mode.c_str());
    return false;
  }

  if (tilingId == 0) {
    OP_LOGE(
        "This shape is not covered by any tiling, "
        "please modify range and recompile");
    return false;
  }
  return true;
}

// register tiling interface of the conv2d_backprop_input
REGISTER_OP_TILING_FUNC(Conv2DBackpropInput, Conv2DBpInputTiling);
}  // namespace optiling
