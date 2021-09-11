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

static bool ResizeCommonTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                               OpRunInfo& run_info) {
  using namespace ge;
  OP_LOGI(op_type, "tiling run begin.");

  OP_TILING_CHECK(op_paras.inputs.empty(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs is empty."),
                  return false);

  OP_TILING_CHECK(op_paras.outputs.empty(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.outputs is empty."),
                  return false);

  // get input_shape and output_shape
  const std::vector<int64_t>& input_shape = op_paras.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& output_shape = op_paras.outputs[0].tensor[0].shape;
  OP_TILING_CHECK(
      input_shape.size() != 5,
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the input shape size must be 5(NC1HWC0) but %lu.", input_shape.size()),
      return false);
  OP_TILING_CHECK(
      output_shape.size() != 5,
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the output shape size must be 5(NC1HWC0) but %lu.", output_shape.size()),
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
  } else {
    get_tiling_result = GetResizeNearestNeighborV2GradTiling(compile_params, tiling_params);
  }
  if (!get_tiling_result) {
    PrintTilingParams(op_type, tiling_params, compile_params);
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get tiling data failed.");
    return false;
  }

  // get tiling data end
  PrintTilingParams(op_type, tiling_params, compile_params);
  SetTilingParams(tiling_params, run_info);
  run_info.block_dim = compile_params.core_num;
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;
  OP_LOGI(op_type, "tiling run success.");

  return true;
}

// register tiling interface of the ResizeNearestNeighborV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED(ResizeNearestNeighborV2, ResizeCommonTiling);
// register tiling interface of the ResizeBilinearV2 op.
REGISTER_OP_TILING_FUNC_BUFFERED(ResizeBilinearV2, ResizeCommonTiling);
// register tiling interface of the ResizeNearestNeighborV2Grad op.
REGISTER_OP_TILING_FUNC_BUFFERED(ResizeNearestNeighborV2Grad, ResizeCommonTiling);
}  // namespace optiling
