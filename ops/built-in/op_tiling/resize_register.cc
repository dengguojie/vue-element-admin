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
 * \file resize_register.cc
 * \brief
 */

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
 * \file resize_register.cc
 * \brief
 */
#include "resize_common.h"

namespace optiling {
static bool ResizeCommonTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                               utils::OpRunInfo& run_info) {
  using namespace ge;
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed.");
    return false;
  }

  OP_LOGI(op_type, "tiling run begin.");
  PROFILING_TILING_INIT(op_type.c_str());

  // get input_shape and output_shape
  auto input_x_desc = operator_info->MutableInputDesc(0);
  if (input_x_desc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed.");
    return false;
  }

  auto out_desc = operator_info->MutableOutputDesc(0);
  if (out_desc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get out_desc failed.");
    return false;
  }

  const std::vector<int64_t>& input_shape = input_x_desc->MutableShape().GetDims();
  const std::vector<int64_t>& output_shape = out_desc->MutableShape().GetDims();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  OP_TILING_CHECK(
      input_shape.size() != 5,
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the input shape size must be 5(NC1HWC0) but %lu.", input_shape.size()),
      return false);
  OP_TILING_CHECK(output_shape.size() != 5,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the output shape size must be 5(NC1HWC0) but %lu.",
                                                  output_shape.size()),
                  return false);

  // get compile data begin
  ResizeClassCompileParams compile_params;
  // init compile data
  compile_params.core_num = 0;
  compile_params.max_w_len = 0;
  compile_params.align_corners = 0;
  compile_params.half_pixel_centers = 0;
  compile_params.op_type = op_type;
  // get compile data
  if (!GetResizeClassCompileParams(op_info, compile_params)) {
    return false;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  // get compile data end

  // get auto turn params
  if (GetResizeClassTuneParams(op_info, compile_params)) {
    OP_LOGI(compile_params.op_type, "Get auto tune params success.");
  }
  // get auto turn params end

  // get tiling data begin
  ResizeClassTilingParams tiling_params;
  // init tiling data
  tiling_params.tiling_key = DEFAULT_TILING_MODE;
  tiling_params.input_batch = input_shape[0];
  tiling_params.input_c1 = input_shape[1];
  tiling_params.input_height = input_shape[2];
  tiling_params.output_height = output_shape[2];
  tiling_params.input_width = input_shape[3];
  tiling_params.output_width = output_shape[3];
  tiling_params.cut_batch_c1_num = 1;
  tiling_params.cut_height_num = 1;
  tiling_params.cut_width_num = 1;

  // calcu tiling
  bool get_tiling_result = false;
  if (op_type == "ResizeNearestNeighborV2") {
    get_tiling_result = GetResizeNearestNeighborV2Tiling(compile_params, tiling_params);
  } else if (op_type == "ResizeBilinearV2") {
    get_tiling_result = GetResizeBilinearV2Tiling(compile_params, tiling_params);
  } else if (op_type == "SyncResizeBilinearV2") {
    get_tiling_result = GetResizeBilinearV2Tiling(compile_params, tiling_params);
  } else {
    get_tiling_result = GetResizeNearestNeighborV2GradTiling(compile_params, tiling_params);
  }
  if (!get_tiling_result) {
    PrintTilingParams(op_type, tiling_params, compile_params);
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get tiling data failed.");
    return false;
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  // get tiling data end
  PrintTilingParams(op_type, tiling_params, compile_params);
  SetTilingParams(tiling_params, run_info);
  run_info.SetBlockDim(compile_params.core_num);

  PROFILING_TILING_END();
  OP_LOGI(op_type, "tiling run success.");

  return true;
}

// register tiling interface of the ResizeNearestNeighborV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(ResizeNearestNeighborV2, ResizeCommonTiling);
// register tiling interface of the ResizeBilinearV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(ResizeBilinearV2, ResizeCommonTiling);
// register tiling interface of the ResizeNearestNeighborV2Grad op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(ResizeNearestNeighborV2Grad, ResizeCommonTiling);
// register tiling interface of the SyncResizeBilinearV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(SyncResizeBilinearV2, ResizeCommonTiling);
}  // namespace optiling
