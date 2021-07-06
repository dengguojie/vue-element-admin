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
 * \file resize_common.cc
 * \brief
 */
#include "resize_common.h"

namespace optiling {

bool GetResizeClassCompileParams(const nlohmann::json& compile_info, ResizeClassCompileParams& compile_params) {
  using namespace nlohmann;
  auto allVars = compile_info["vars"];
  OP_TILING_CHECK(allVars.count("core_num") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "get compile core_num error, num = 0"),
                  return false);
  compile_params.core_num = allVars["core_num"].get<std::int64_t>();

  OP_TILING_CHECK(allVars.count("max_w_len") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "get compile max_w_len error, num = 0"),
                  return false);
  compile_params.max_w_len = allVars["max_w_len"].get<std::int64_t>();

  OP_TILING_CHECK(allVars.count("align_corners") == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "get compile align_corners error, num = 0"),
                  return false);
  compile_params.align_corners = allVars["align_corners"].get<std::int64_t>();

  OP_TILING_CHECK(
      allVars.count("half_pixel_centers") == 0,
      VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "get compile half_pixel_centers error, num = 0"),
      return false);
  compile_params.half_pixel_centers = allVars["half_pixel_centers"].get<std::int64_t>();

  return true;
}

void SetTilingParams(const ResizeClassTilingParams& tiling_params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, tiling_params.tiling_key);
  ByteBufferPut(run_info.tiling_data, tiling_params.input_batch);
  ByteBufferPut(run_info.tiling_data, tiling_params.input_c1);
  ByteBufferPut(run_info.tiling_data, tiling_params.input_height);
  ByteBufferPut(run_info.tiling_data, tiling_params.input_width);
  ByteBufferPut(run_info.tiling_data, tiling_params.output_height);
  ByteBufferPut(run_info.tiling_data, tiling_params.output_width);
  ByteBufferPut(run_info.tiling_data, tiling_params.cut_batch_c1_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.cut_height_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.cut_width_num);
}

void PrintTilingParams(const std::string& op_type, const ResizeClassTilingParams& tiling_params,
                       const ResizeClassCompileParams& compile_params) {
  // print tiling_params
  OP_LOGD(op_type, "tiling_data, tiling_key = %d.", tiling_params.tiling_key);
  OP_LOGD(op_type, "tiling_data, input_batch_c1 = %d.", tiling_params.input_batch * tiling_params.input_c1);
  OP_LOGD(op_type, "tiling_data, input_height = %d.", tiling_params.input_height);
  OP_LOGD(op_type, "tiling_data, input_width = %d.", tiling_params.input_width);
  OP_LOGD(op_type, "tiling_data, output_height = %d.", tiling_params.output_height);
  OP_LOGD(op_type, "tiling_data, output_width = %d.", tiling_params.output_width);
  OP_LOGD(op_type, "tiling_data, cut_batch_c1_num = %d.", tiling_params.cut_batch_c1_num);
  OP_LOGD(op_type, "tiling_data, cut_height_num = %d.", tiling_params.cut_height_num);
  OP_LOGD(op_type, "tiling_data, cut_width_num = %d.", tiling_params.cut_width_num);

  // print compile_params
  OP_LOGD(op_type, "compile_data, core_num = %d.", compile_params.core_num);
  OP_LOGD(op_type, "compile_data, max_w_len = %d.", compile_params.max_w_len);
  OP_LOGD(op_type, "compile_data, align_corners = %d.", compile_params.align_corners);
  OP_LOGD(op_type, "compile_data, half_pixel_centers = %d.", compile_params.half_pixel_centers);
}

void GetTilingForHW2MHNW(const ResizeClassCompileParams& compile_params, ResizeClassTilingParams& tiling_params) {
  auto image_batch_c1 = tiling_params.input_batch * tiling_params.input_c1;
  auto cut_batch_c1_sigment = (image_batch_c1 + compile_params.core_num - 1) / compile_params.core_num;
  tiling_params.cut_batch_c1_num = (image_batch_c1 + cut_batch_c1_sigment - 1) / cut_batch_c1_sigment;
  auto left_core_num = compile_params.core_num - tiling_params.cut_batch_c1_num;
  if (left_core_num != 0) {
    if (tiling_params.input_width > left_core_num * 4) {
      // charge whether continue cut width
      left_core_num = (compile_params.core_num + tiling_params.cut_batch_c1_num - 1) / tiling_params.cut_batch_c1_num;
      tiling_params.cut_batch_c1_num = compile_params.core_num / left_core_num;
      auto cut_w_sigment = (tiling_params.input_width + left_core_num - 1) / left_core_num;
      int64_t min_w_sigment = 16;
      cut_w_sigment = max(min_w_sigment, cut_w_sigment);
      tiling_params.cut_width_num = (tiling_params.input_width + cut_w_sigment - 1) / cut_w_sigment;
    }
  }
  left_core_num = compile_params.core_num / (tiling_params.cut_batch_c1_num * tiling_params.cut_width_num);
  // continue cut height
  if (left_core_num > 1) {
    auto cut_h_sigment = (tiling_params.input_height + left_core_num - 1) / left_core_num;
    auto h_max = (tiling_params.input_height + cut_h_sigment - 1) / cut_h_sigment;
    tiling_params.cut_height_num = min(left_core_num, h_max);
  }
}

}  // namespace optiling
