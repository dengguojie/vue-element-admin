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
  constexpr int32_t kConv3DBpFilterDedyInputIdx = 2;
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
bool Conv3DBpFilterTiling(const std::string& op_type,
                          const ge::Operator& op_paras,
                          const nlohmann::json& compile_info,
                          utils::OpRunInfo& run_info) {
  bool invalid_input = (op_paras.GetInputsSize() < kConv3dBpInputsNum ||
      (op_paras.GetInputDesc(0).GetShape().GetDimNum() != kConv3dDimSizeLimit) ||
      (op_paras.GetInputDesc(kConv3DBpFilterDedyInputIdx).GetShape().GetDimNum() != kConv3dDimSizeLimit));
  if (invalid_input) {
    OP_LOGE(op_type.c_str(), "Input paramters check failed");
    return false;
  }
  // the dim index of input x channel is 2
  if (compile_info.contains("fmap_c1") && op_paras.GetInputDesc(0).GetShape().GetDim(2) != compile_info["fmap_c1"]) {
    OP_LOGE(op_type.c_str(), "not support, input x channel should be equal to filter * groups");
    return false;
  }
  // the dim index of input dedy channel is 2
  if (compile_info.contains("dedy_c1") &&
      op_paras.GetInputDesc(kConv3DBpFilterDedyInputIdx).GetShape().GetDim(kDimIndex) != compile_info["dedy_c1"]) {
    OP_LOGE(op_type.c_str(), "not support, input dedy channel should be equal to filter");
    return false;
  }

  return Conv3DCommonTiling("Conv3DBackpropFilter",
                            op_paras.GetInputDesc(0).GetShape().GetDims(),
                            op_paras.GetInputDesc(kConv3DBpFilterDedyInputIdx).GetShape().GetDims(),
                            compile_info,
                            run_info);
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(Conv3DBackpropFilter, Conv3DBpFilterTiling);
}  // namespace optiling
