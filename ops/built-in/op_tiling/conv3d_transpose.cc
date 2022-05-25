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
  constexpr int32_t kConv3dTransposeDimSizeLimit = 6;
  constexpr int32_t kConv3dTransposeDedyInputIndex = 1;
  static const int kDimIndex = 2;
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
bool Conv3DTransposeTiling(const std::string& op_type,
                           const ge::Operator& op_paras,
                           const nlohmann::json& compile_info,
                           utils::OpRunInfo& run_info) {
  GELOGD("Start Conv3DTransposeTiling process");
  int32_t input_size = op_paras.GetInputsSize();
  int32_t dedy_dimnum = op_paras.GetInputDesc(kConv3dTransposeDedyInputIndex).GetShape().GetDimNum();
  int32_t output_dimnum = op_paras.GetOutputDesc(0).GetShape().GetDimNum();
  OP_LOGE_IF(input_size <= kConv3dTransposeDedyInputIndex, false, op_type,
             "The number of inputs to the operator should not be less than 2, but it is actually %d.",
             input_size);
  OP_LOGE_IF(dedy_dimnum != kConv3dTransposeDimSizeLimit, false, op_type,
             "The dimension of the dedy should be 6, but it is actually %d.", dedy_dimnum);
  OP_LOGE_IF(output_dimnum != kConv3dTransposeDimSizeLimit, false, op_type,
             "The dimension of the output should be 6, but it is actually %d.", output_dimnum);
  // the dim index of input dedy channel is 2
  int32_t dedy_channel_dim = op_paras.GetInputDesc(kConv3dTransposeDedyInputIndex).GetShape().GetDim(kDimIndex);
  if (compile_info.contains("dedy_c1")) {
    int32_t dedy_c1 = compile_info["dedy_c1"];
    OP_LOGE_IF(dedy_channel_dim != dedy_c1, false, op_type,
               "unsupported input, input dedy channel [%d] should be equal to filter [%d]", dedy_channel_dim,
               dedy_c1);
  } else {
    OP_LOGE(op_type, "compile_info does not contain the key value of the dedy_c1");
  }

  return Conv3DCommonTiling("Conv3DTranspose",
                            op_paras.GetOutputDesc(0).GetShape().GetDims(),
                            op_paras.GetInputDesc(kConv3dTransposeDedyInputIndex).GetShape().GetDims(),
                            compile_info,
                            run_info);
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(Conv3DTranspose, Conv3DTransposeTiling);
}  // namespace optiling
