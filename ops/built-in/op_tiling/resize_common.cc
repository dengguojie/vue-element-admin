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
 * \file resize_common.cc
 * \brief
 */
#include "resize_common.h"

namespace {
constexpr int32_t TILING_KEY_100110 = 100110;
constexpr int32_t TILING_KEY_100000 = 100000;
}  // namespace

namespace optiling {

bool ResizeCommonTuneParamsFunc(const std::string& op_type, const nlohmann::json& compile_info,
                                ResizeCommonInputCompile& compile_value) {
  if (compile_info.count(INNERTUNEPARAM) == 0) {
    OP_LOGD(op_type, "GetResizeClassTuneParams do not contain %s in json.", INNERTUNEPARAM);
    return false;
  }
  const nlohmann::json& tuneParamOut = compile_info[INNERTUNEPARAM];

  if (tuneParamOut.count(TUNEPARAM) == 0) {
    OP_LOGD(op_type, "%s is not in %s.", TUNEPARAM, INNERTUNEPARAM);
    return false;
  }
  const nlohmann::json& tuneParam = tuneParamOut[TUNEPARAM];

  OP_TILING_CHECK(!GetCompileValue(tuneParam, "tiling_key", compile_value.tune_params[INDEX_0]),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "tiling_key is not in %s.", TUNEPARAM),
                  return false);
  OP_LOGD(op_type, "tiling_key of tune param: %lld", compile_value.tune_params[INDEX_0]);

  if (compile_value.tune_params[INDEX_0] == TILING_KEY_100110 ||
      compile_value.tune_params[INDEX_0] == TILING_KEY_100000) {
    OP_TILING_CHECK(!GetCompileValue(tuneParam, "cut_batch_c1_num", compile_value.tune_params[INDEX_1]),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "cut_batch_c1_num is not in %s.", TUNEPARAM),
                    return false);
    OP_TILING_CHECK(!GetCompileValue(tuneParam, "cut_height_num", compile_value.tune_params[INDEX_2]),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "cut_height_num is not in %s.", TUNEPARAM),
                    return false);
    OP_TILING_CHECK(!GetCompileValue(tuneParam, "cut_width_num", compile_value.tune_params[INDEX_3]),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "cut_width_num is not in %s.", TUNEPARAM),
                    return false);
  }

  return true;
}

bool ResizeCommonParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                           ResizeCommonInputCompile& compile_value) {
  if (compile_info.count("vars") == 0) {
    return false;
  }
  const nlohmann::json& all_vars = compile_info["vars"];

  OP_TILING_CHECK(!GetCompileValue(all_vars, "core_num", compile_value.core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "ResizeCommonParseFunc, get core_num error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(all_vars, "max_w_len", compile_value.max_w_len),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "ResizeCommonParseFunc, get max_w_len error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(all_vars, "align_corners", compile_value.align_corners),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "ResizeCommonParseFunc, get align_corners error"),
                  return false);
  OP_TILING_CHECK(
      !GetCompileValue(all_vars, "half_pixel_centers", compile_value.half_pixel_centers),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "ResizeCommonParseFunc, get half_pixel_centers error"),
      return false);

  if (ResizeCommonTuneParamsFunc(op_type, compile_info, compile_value)) {
    OP_LOGI(op_type, "Get auto tune params success.");
  }

  return true;
}

bool GetResizeClassCompileParams(const ResizeCommonInputCompile& compile_info,
                                 ResizeClassCompileParams& compile_params) {
  compile_params.core_num = compile_info.core_num;
  compile_params.max_w_len = compile_info.max_w_len;
  compile_params.align_corners = compile_info.align_corners;
  compile_params.half_pixel_centers = compile_info.half_pixel_centers;

  return true;
}

void GetResizeClassTuneParams(const ResizeCommonInputCompile& compile_info, ResizeClassCompileParams& compile_params) {
  compile_params.tuneParams.tiling_key = compile_info.tune_params[INDEX_0];
  compile_params.tuneParams.cut_batch_c1_num = compile_info.tune_params[INDEX_1];
  compile_params.tuneParams.cut_height_num = compile_info.tune_params[INDEX_2];
  compile_params.tuneParams.cut_width_num = compile_info.tune_params[INDEX_3];
}

void SetTilingParams(const ResizeClassTilingParams& tiling_params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(tiling_params.tiling_key);
  run_info.AddTilingData(tiling_params.input_batch);
  run_info.AddTilingData(tiling_params.input_c1);
  run_info.AddTilingData(tiling_params.input_height);
  run_info.AddTilingData(tiling_params.input_width);
  run_info.AddTilingData(tiling_params.output_height);
  run_info.AddTilingData(tiling_params.output_width);
  run_info.AddTilingData(tiling_params.cut_batch_c1_num);
  run_info.AddTilingData(tiling_params.cut_height_num);
  run_info.AddTilingData(tiling_params.cut_width_num);
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
