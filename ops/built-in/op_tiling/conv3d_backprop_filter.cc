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
 * \file conv3d_backprop_filter.cc
 * \brief tiling function of conv3d_backprop_filter
 */
#include <string>
#include <nlohmann/json.hpp>
#include <limits>
#include "cube_tiling.h"
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"

namespace {
  constexpr int32_t kConv3dDimSizeLimit = 6;
  constexpr int32_t kConv3dBpInputsNum = 3;
}

namespace optiling {
/*
 * @brief: tiling function of conv3d_backprop_input and conv3d_transpose
 * @param [in] op_type: op_type of the conv3d_backprop_input or conv3d_transpose
 * @param [in] op_paras: inputs/outputs/atts of the conv3d_backprop_input or conv3d_transpose
 * @param [in] compile_info: compile time generated info of the conv3d_backprop_input or conv3d_transpose
 * @param [out] run_info: result data
 * @return bool: success or not
 */
bool Conv3DBpFilterTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& compile_info,
                        OpRunInfo& run_info) {
  bool invalid_input = (op_paras.inputs.size() < kConv3dBpInputsNum || op_paras.outputs.empty() ||
      op_paras.inputs[0].tensor.empty() || op_paras.inputs[2].tensor.empty() ||
      (op_paras.inputs[0].tensor[0].shape.size() != kConv3dDimSizeLimit) ||
      (op_paras.inputs[2].tensor[0].shape.size() != kConv3dDimSizeLimit));
  if (invalid_input) {
    OP_LOGE(op_type.c_str(), "Input paramters check failed");
    return false;
  }

  if (compile_info.contains("fmap_c1") && op_paras.inputs[0].tensor[0].shape[2] != compile_info["fmap_c1"]) {
    OP_LOGE(op_type.c_str(), "not support, input x channel should be equal to filter * groups");
    return false;
  }

  if (compile_info.contains("dedy_c1") && op_paras.inputs[2].tensor[0].shape[2] != compile_info["dedy_c1"]) {
    OP_LOGE(op_type.c_str(), "not support, input dedy channel should be equal to filter");
    return false;
  }

  return Conv3DCommonTiling("Conv3DBackpropFilter", op_paras.inputs[0].tensor[0].shape,
                            op_paras.inputs[2].tensor[0].shape,
                            compile_info, run_info);
}

REGISTER_OP_TILING_FUNC_BUFFERED(Conv3DBackpropFilter, Conv3DBpFilterTiling);
}  // namespace optiling
