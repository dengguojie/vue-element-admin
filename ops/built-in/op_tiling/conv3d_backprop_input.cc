/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
#include "error_util.h"

namespace {
  constexpr int32_t kConv3dBpInputDimSizeLimit = 6;
  constexpr int32_t kConv3dBpInputDedyInputIndex = 2;
  static const int kDimIndex = 2;
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
bool Conv3DBackpropInputTiling(const std::string& op_type,
                               const ge::Operator& op_paras,
                               const nlohmann::json& compile_info,
                               utils::OpRunInfo& run_info) {
  if ((op_paras.GetInputsSize() <= kConv3dBpInputDedyInputIndex) ||
      (op_paras.GetInputDesc(kConv3dBpInputDedyInputIndex).GetShape().GetDimNum() != kConv3dBpInputDimSizeLimit) ||
      (op_paras.GetOutputDesc(0).GetShape().GetDimNum() != kConv3dBpInputDimSizeLimit)) {
    CUBE_INNER_ERR_REPORT(op_type.c_str(), "param check failed");
    return false;
  }

  if (compile_info.empty()) {
    CUBE_INNER_ERR_REPORT(op_type.c_str(), "op compile info is empty");
    return false;
  }

  // the dim index of input dedy channel is 2
  if (compile_info.contains("dedy_c1") &&
      op_paras.GetInputDesc(kConv3dBpInputDedyInputIndex).GetShape().GetDim(kDimIndex) != compile_info["dedy_c1"]) {
    CUBE_INNER_ERR_REPORT(op_type.c_str(), "not support, input dedy channel should be equal to filter");
    return false;
  }

  nlohmann::json opInfo;
  deal_with_compile_info(compile_info, opInfo);

  return Conv3DCommonTiling("Conv3DBackpropInput",
                            op_paras.GetOutputDesc(0).GetShape().GetDims(),
                            op_paras.GetInputDesc(kConv3dBpInputDedyInputIndex).GetShape().GetDims(),
                            opInfo,
                            run_info);
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(Conv3DBackpropInput, Conv3DBackpropInputTiling);
}  // namespace optiling
