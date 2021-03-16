/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file conv3d_backprop_input.cc
 * \brief tiling function of conv3d_backprop_input and conv3d_transpose
 */
#include <string>
#include <nlohmann/json.hpp>
#include <limits>
#include "op_tiling.h"
#include "cube_tiling.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"

namespace {
  constexpr int32_t kConv3dBpInputDimSizeLimit = 6;
  constexpr int32_t kConv3dBpInputDedyInputIndex = 2;
}

namespace optiling {
/*
 * @brief: tiling function of conv3d_backprop_input
 * @param [in] op_type: op_type of the conv3d_backprop_input
 * @param [in] op_paras: inputs/outputs/atts of the conv3d_backprop_input
 * @param [in] compile_info: compile time generated info of the conv3d_backprop_input
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool Conv3DBackpropInputTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& compile_info,
                               OpRunInfo& run_info) {
  if ((op_paras.inputs.size() <= kConv3dBpInputDedyInputIndex) || op_paras.outputs.empty() ||
      op_paras.inputs[kConv3dBpInputDedyInputIndex].tensor.empty() || op_paras.outputs[0].tensor.empty() ||
      (op_paras.inputs[kConv3dBpInputDedyInputIndex].tensor[0].shape.size() != kConv3dBpInputDimSizeLimit) ||
      (op_paras.outputs[0].tensor[0].shape.size() != kConv3dBpInputDimSizeLimit)) {
    OP_LOGE(op_type.c_str(), "param check failed");
    return false;
  }

  return Conv3DCommonTiling("Conv3DBackpropInput", op_paras.outputs[0].tensor[0].shape,
                            op_paras.inputs[kConv3dBpInputDedyInputIndex].tensor[0].shape, 
                            compile_info, run_info);
}

REGISTER_OP_TILING_FUNC_BUFFERED(Conv3DBackpropInput, Conv3DBackpropInputTiling);
}  // namespace optiling
