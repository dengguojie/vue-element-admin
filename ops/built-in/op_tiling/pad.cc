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
 * \file pad.cpp
 * \brief
 */
#include <string>
#include <math.h>

#include "op_tiling_util.h"
#include "../op_proto/strided_slice_infer_shape.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace {
constexpr int64_t CUT_LEN = 3;
constexpr int64_t DST_LEN = 2;
constexpr int64_t PADDING_FACTOR = 2;
constexpr int64_t DTYPE_RATE_INDEX = 2;
constexpr int64_t OUTPUT_SIZE_INDEX = 3;
constexpr int64_t BEGIN_MASK_INDEX = 4;
constexpr int64_t END_MASK_INDEX = 5;
constexpr int64_t ELLIPSIS_MASK_INDEX = 6;
constexpr int64_t AXIS_MASK_INDEX = 7;
constexpr int64_t SHRINK_AXIS_INDEX = 8;
constexpr int64_t SHAPE_DIM_INDEX_2 = 2;
constexpr int64_t SHAPE_DIM_INDEX_3 = 3;
constexpr int64_t SHAPE_DIM_INDEX_4 = 4;
constexpr int64_t SHAPE_DIM_INDEX_5 = 5;
constexpr int64_t SHAPE_DIM_INDEX_6 = 6;
constexpr int64_t SHAPE_DIM_INDEX_7 = 7;
constexpr int64_t TILING_PADDING_INDEX_2 = 2;
constexpr int64_t TILING_PADDING_INDEX_3 = 3;
constexpr int64_t TILING_PADDING_INDEX_4 = 4;
constexpr int64_t TILING_PADDING_INDEX_5 = 5;
constexpr int64_t TILING_PADDING_INDEX_6 = 6;
constexpr int64_t TILING_PADDING_INDEX_7 = 7;
constexpr int64_t TILING_PADDING_INDEX_8 = 8;
constexpr int64_t TILING_PADDING_INDEX_9 = 9;
constexpr int64_t TILING_PADDING_INDEX_10 = 10;
constexpr int64_t TILING_PADDING_INDEX_11 = 11;
constexpr int64_t TILING_PADDING_INDEX_12 = 12;
constexpr int64_t TILING_PADDING_INDEX_13 = 13;
constexpr int64_t TILING_PADDING_INDEX_14 = 14;
constexpr int64_t TILING_PADDING_INDEX_15 = 15;
}  // namespace

namespace optiling {
const int64_t ONE_BLOCK_BYTES = 32;

const int64_t TILING_MODE_0 = 0;

const int64_t TILING_MODE_1 = 1;

const int64_t TILING_MODE_2 = 2;

const int64_t TILING_MODE_3 = 3;

const std::string OP_STRIDED_SLICE_GRAD = "StridedSliceGrad";

const int64_t SHAPE_C0 = 16;

const int64_t MAX_SHAPE_LEN = 8;

const int64_t COMPILE_INFO_SIZE_4 = 4;

const int64_t COMPILE_INFO_SIZE_9 = 9;

// define the TILING_MODE_0 sigment when the shape len = 2
const int64_t TILING_0_SIGMENT_1 = 128;

// define the TILING_MODE_0 sigment when the shape len = 1; 16 mean fp16 num in one block
const int64_t TILING_0_SIGMENT_2 = 16;

// define a map format with N(D)CHW dim idx vector
const map<Format, vector<int32_t>>& FORMAT_FOR_NCHW_IDX = {
    {FORMAT_NCHW, {0, 1, 2, 3}},     {FORMAT_NHWC, {0, 3, 1, 2}},     {FORMAT_HWCN, {3, 2, 0, 1}},
    {FORMAT_NCDHW, {0, 2, 1, 3, 4}}, {FORMAT_DHWCN, {4, 0, 3, 1, 2}}, {FORMAT_NDHWC, {0, 1, 4, 2, 3}}};

// define a map for special format
const std::vector<Format> SPECIAL_FORMAT = {FORMAT_NC1HWC0, FORMAT_NDC1HWC0};

struct SliceParameters {
  std::vector<int64_t> input;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> end_list;
  std::vector<int64_t> stride_list;
};

struct SliceMasks {
  uint64_t beginmask = 0;
  uint64_t endmask = 0;
  uint64_t ellipsismask = 0;
  uint64_t newaxismask = 0;
  uint64_t shrinkaxismask = 0;
};

struct PadCompileParams {
  int64_t core_num;
  int64_t ub_size;
  int64_t dtype_rate;
  std::string op_type;
  // Add For StridedSliceGrad
  uint64_t begin_mask;
  uint64_t end_mask;
  uint64_t ellipsis_mask;
  uint64_t new_axis_mask;
  uint64_t shrink_axis_mask;
  uint64_t tiling_two_max_output_size;
};

struct PadTilingParams {
  int64_t tiling_key;
  int64_t tiling_input_dim_0;
  int64_t tiling_input_dim_1;
  int64_t tiling_input_dim_2;
  int64_t tiling_input_dim_3;
  int64_t tiling_input_dim_4;
  int64_t tiling_input_dim_5;
  int64_t tiling_input_dim_6;
  int64_t tiling_input_dim_7;
  int64_t tiling_pading_00;
  int64_t tiling_pading_01;
  int64_t tiling_pading_10;
  int64_t tiling_pading_11;
  int64_t tiling_pading_20;
  int64_t tiling_pading_21;
  int64_t tiling_pading_30;
  int64_t tiling_pading_31;
  int64_t tiling_pading_40;
  int64_t tiling_pading_41;
  int64_t tiling_pading_50;
  int64_t tiling_pading_51;
  int64_t tiling_pading_60;
  int64_t tiling_pading_61;
  int64_t tiling_pading_70;
  int64_t tiling_pading_71;
  int64_t tiling_input_dim_cut_axis;
};

void InitRunningParams(PadTilingParams& params) {
  params.tiling_key = TILING_MODE_1;
  params.tiling_input_dim_0 = 1;
  params.tiling_input_dim_1 = 1;
  params.tiling_input_dim_2 = 1;
  params.tiling_input_dim_3 = 1;
  params.tiling_input_dim_4 = 1;
  params.tiling_input_dim_5 = 1;
  params.tiling_input_dim_6 = 1;
  params.tiling_input_dim_7 = 1;
  params.tiling_pading_00 = 0;
  params.tiling_pading_01 = 0;
  params.tiling_pading_10 = 0;
  params.tiling_pading_11 = 0;
  params.tiling_pading_20 = 0;
  params.tiling_pading_21 = 0;
  params.tiling_pading_30 = 0;
  params.tiling_pading_31 = 0;
  params.tiling_pading_40 = 0;
  params.tiling_pading_41 = 0;
  params.tiling_pading_50 = 0;
  params.tiling_pading_51 = 0;
  params.tiling_pading_60 = 0;
  params.tiling_pading_61 = 0;
  params.tiling_pading_70 = 0;
  params.tiling_pading_71 = 0;
  params.tiling_input_dim_cut_axis = 1;
}

static bool GetPadCompileParams(const std::vector<int64_t>& compile_info, PadCompileParams& compile_params) {
  OP_TILING_CHECK(compile_info.size() < COMPILE_INFO_SIZE_4,
                  VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "the compile info num must >= 4, is %zu",
                                                  compile_info.size()),
                  return false);
  compile_params.core_num = compile_info[0];
  compile_params.ub_size = compile_info[1];
  compile_params.dtype_rate = compile_info[DTYPE_RATE_INDEX];
  compile_params.tiling_two_max_output_size = compile_info[OUTPUT_SIZE_INDEX];
  if (compile_params.op_type == OP_STRIDED_SLICE_GRAD) {
    OP_TILING_CHECK(compile_info.size() != COMPILE_INFO_SIZE_9,
                    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "the compile info num must == 9, is %zu",
                                                    compile_info.size()),
                    return false);
    compile_params.begin_mask = static_cast<std::uint64_t>(compile_info[BEGIN_MASK_INDEX]);
    compile_params.end_mask = static_cast<std::uint64_t>(compile_info[END_MASK_INDEX]);
    compile_params.ellipsis_mask = static_cast<std::uint64_t>(compile_info[ELLIPSIS_MASK_INDEX]);
    compile_params.new_axis_mask = static_cast<std::uint64_t>(compile_info[AXIS_MASK_INDEX]);
    compile_params.shrink_axis_mask = static_cast<std::uint64_t>(compile_info[SHRINK_AXIS_INDEX]);
  }
  return true;
}

void SetRuningParams(const PadTilingParams& params, utils::OpRunInfo& run_info) {
  run_info.AddTilingData(params.tiling_key);
  run_info.AddTilingData(params.tiling_input_dim_0);
  run_info.AddTilingData(params.tiling_input_dim_1);
  run_info.AddTilingData(params.tiling_input_dim_2);
  run_info.AddTilingData(params.tiling_input_dim_3);
  run_info.AddTilingData(params.tiling_input_dim_4);
  run_info.AddTilingData(params.tiling_input_dim_5);
  run_info.AddTilingData(params.tiling_input_dim_6);
  run_info.AddTilingData(params.tiling_input_dim_7);
  run_info.AddTilingData(params.tiling_pading_00);
  run_info.AddTilingData(params.tiling_pading_01);
  run_info.AddTilingData(params.tiling_pading_10);
  run_info.AddTilingData(params.tiling_pading_11);
  run_info.AddTilingData(params.tiling_pading_20);
  run_info.AddTilingData(params.tiling_pading_21);
  run_info.AddTilingData(params.tiling_pading_30);
  run_info.AddTilingData(params.tiling_pading_31);
  run_info.AddTilingData(params.tiling_pading_40);
  run_info.AddTilingData(params.tiling_pading_41);
  run_info.AddTilingData(params.tiling_pading_50);
  run_info.AddTilingData(params.tiling_pading_51);
  run_info.AddTilingData(params.tiling_pading_60);
  run_info.AddTilingData(params.tiling_pading_61);
  run_info.AddTilingData(params.tiling_pading_70);
  run_info.AddTilingData(params.tiling_pading_71);
  run_info.AddTilingData(params.tiling_input_dim_cut_axis);
}

void PrintTilingParams(const PadTilingParams& params, const std::string& op_type) {
  OP_LOGD(op_type, "tiling_key=%ld. ", params.tiling_key);
  OP_LOGD(op_type, "tiling_input_dim_0=%ld.", params.tiling_input_dim_0);
  OP_LOGD(op_type, "tiling_input_dim_1=%ld.", params.tiling_input_dim_1);
  OP_LOGD(op_type, "tiling_input_dim_2=%ld.", params.tiling_input_dim_2);
  OP_LOGD(op_type, "tiling_input_dim_3=%ld.", params.tiling_input_dim_3);
  OP_LOGD(op_type, "tiling_input_dim_4=%ld.", params.tiling_input_dim_4);
  OP_LOGD(op_type, "tiling_input_dim_5=%ld.", params.tiling_input_dim_5);
  OP_LOGD(op_type, "tiling_input_dim_6=%ld.", params.tiling_input_dim_6);
  OP_LOGD(op_type, "tiling_input_dim_7=%ld.", params.tiling_input_dim_7);
  OP_LOGD(op_type, "tiling_pading_00=%ld.", params.tiling_pading_00);
  OP_LOGD(op_type, "tiling_pading_01=%ld.", params.tiling_pading_01);
  OP_LOGD(op_type, "tiling_pading_10=%ld.", params.tiling_pading_10);
  OP_LOGD(op_type, "tiling_pading_11=%ld.", params.tiling_pading_11);
  OP_LOGD(op_type, "tiling_pading_20=%ld.", params.tiling_pading_20);
  OP_LOGD(op_type, "tiling_pading_21=%ld.", params.tiling_pading_21);
  OP_LOGD(op_type, "tiling_pading_30=%ld.", params.tiling_pading_30);
  OP_LOGD(op_type, "tiling_pading_31=%ld.", params.tiling_pading_31);
  OP_LOGD(op_type, "tiling_pading_40=%ld.", params.tiling_pading_40);
  OP_LOGD(op_type, "tiling_pading_41=%ld.", params.tiling_pading_41);
  OP_LOGD(op_type, "tiling_pading_50=%ld.", params.tiling_pading_50);
  OP_LOGD(op_type, "tiling_pading_51=%ld.", params.tiling_pading_51);
  OP_LOGD(op_type, "tiling_pading_60=%ld.", params.tiling_pading_60);
  OP_LOGD(op_type, "tiling_pading_61=%ld.", params.tiling_pading_61);
  OP_LOGD(op_type, "tiling_pading_70=%ld.", params.tiling_pading_70);
  OP_LOGD(op_type, "tiling_pading_71=%ld.", params.tiling_pading_71);
  OP_LOGD(op_type, "tiling_input_dim_cut_axis=%ld.", params.tiling_input_dim_cut_axis);
}

void PrintTensorValue(const PadCompileParams& compile_params, const std::vector<int64_t>& in, const std::string& name) {
  using namespace std;
  string vec_str;
  for (size_t i = 0; i < in.size(); ++i) {
    OP_LOGD(compile_params.op_type, "Func[PrintTensorValue] [%s][%ld]: [%ld].", name.c_str(), i, in[i]);
  }
}

static bool CalcuPaddingForStridedSliceGrad(const ge::Operator& op_paras, const PadCompileParams& compile_params,
                                            std::vector<int64_t>& paddings_const_values,
                                            std::vector<int64_t>& paddings_input_shape) {
  struct SliceParameters slice_params_output = {};
  struct SliceMasks slicemasks_output;
  map<string, std::pair<int64_t, vector<int64_t>&>> const_params = {
      {"shape", {0, slice_params_output.input}},
      {"begin", {1, slice_params_output.begin_list}},
      {"end", {2, slice_params_output.end_list}},
      {"strides", {3, slice_params_output.stride_list}},
  };

  slicemasks_output.beginmask = compile_params.begin_mask;
  slicemasks_output.endmask = compile_params.end_mask;
  slicemasks_output.ellipsismask = compile_params.ellipsis_mask;
  slicemasks_output.newaxismask = compile_params.new_axis_mask;
  slicemasks_output.shrinkaxismask = compile_params.shrink_axis_mask;

  // Get Const Input
  for (auto& item : const_params) {
    auto& name = item.first;
    int64_t index = item.second.first;
    auto& values = item.second.second;
    if (!ops::GetConstIntData(op_paras, index, values)) {
      VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "Get %s values failed", name.c_str());
      return false;
    }
  }

  OP_LOGI(compile_params.op_type, "original information shape:%s begin:%s end:%s stride:%s",
          ops::to_string(slice_params_output.input).c_str(), ops::to_string(slice_params_output.begin_list).c_str(),
          ops::to_string(slice_params_output.end_list).c_str(),
          ops::to_string(slice_params_output.stride_list).c_str());

  // input_shape is not change, it will be used in calc padding.
  std::vector<int64_t> input_shape = slice_params_output.input;
  vector<int64_t> ori_begin = slice_params_output.begin_list;

  StridedSliceParams input_params = {
      slice_params_output.input,
      slice_params_output.begin_list,
      slice_params_output.end_list,
      slice_params_output.stride_list,
      vector<pair<int64_t, int64_t>>(),
      slicemasks_output.beginmask,
      slicemasks_output.endmask,
      slicemasks_output.ellipsismask,
      slicemasks_output.newaxismask,
      slicemasks_output.shrinkaxismask,
      true,
      true,
      true,
  };

  vector<int64_t> output_shape;
  vector<pair<int64_t, int64_t>> output_ranges;
  if (!StridedSliceCommonInferShape(compile_params.op_type, input_params, output_shape, output_ranges)) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "StridedSliceCommonInferShape failed.");
    return false;
  }

  OP_LOGI(compile_params.op_type, "after infershape, begin end and stride is %s, %s, and %s.",
          ops::to_string(input_params.begin).c_str(), ops::to_string(input_params.end).c_str(),
          ops::to_string(input_params.strides).c_str());

  slice_params_output.begin_list = input_params.begin;
  slice_params_output.end_list = input_params.end;
  slice_params_output.stride_list = input_params.strides;
  slice_params_output.input = input_params.input_shape;

  OP_LOGI(compile_params.op_type, "after infershape shape:%s begin:%s end:%s stride:%s",
          ops::to_string(slice_params_output.input).c_str(), ops::to_string(slice_params_output.begin_list).c_str(),
          ops::to_string(slice_params_output.end_list).c_str(),
          ops::to_string(slice_params_output.stride_list).c_str());

  vector<int64_t> grad_input_dy_shape;
  uint64_t length = static_cast<uint64_t>(slice_params_output.input.size());
  for (uint64_t i = 0; i < length; i++) {
    // StridedSliceCommonInferShape make sure that input, begin_list and end_list are same length
    int64_t begin_i = slice_params_output.begin_list[i];
    int64_t end_i = slice_params_output.end_list[i];
    int64_t shape_i = slice_params_output.input[i];

    begin_i = (begin_i < 0) ? begin_i + shape_i : begin_i;
    end_i = (end_i < 0) ? end_i + shape_i : end_i;

    paddings_const_values.push_back(begin_i);
    paddings_const_values.push_back(shape_i - end_i);
    grad_input_dy_shape.push_back(end_i - begin_i);
  }
  paddings_input_shape = grad_input_dy_shape;
  return true;
}

static bool GetTilingParam(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& paddings_const_values,
                           const PadCompileParams& compile_params, PadTilingParams& tiling_params) {
  auto shape_len = input_shape.size();
  std::vector<int64_t> merge_input_shape_dims;
  std::vector<int64_t> merge_paddings_values;
  merge_input_shape_dims.reserve(MAX_SHAPE_LEN);
  merge_paddings_values.reserve(MAX_SHAPE_LEN * PADDING_FACTOR);
  int64_t merge_input_shape_dim = 1;
  bool pre_dim_pidding_flag = true;
  for (size_t i = 0; i < shape_len; i++) {
    auto cu_idx = shape_len - i - 1;
    auto cu_input_dim = input_shape[cu_idx];
    auto cu_paddings_value_left = paddings_const_values[cu_idx * PADDING_FACTOR];
    auto cu_paddings_value_right = paddings_const_values[cu_idx * PADDING_FACTOR + 1];
    auto cu_output_dim = cu_input_dim + cu_paddings_value_left + cu_paddings_value_right;
    if (i == shape_len - 1) {
      merge_input_shape_dims.insert(merge_input_shape_dims.begin(), cu_input_dim * merge_input_shape_dim);
      merge_paddings_values.insert(merge_paddings_values.begin(), cu_paddings_value_right * merge_input_shape_dim);
      merge_paddings_values.insert(merge_paddings_values.begin(), cu_paddings_value_left * merge_input_shape_dim);
    } else if (cu_output_dim == cu_input_dim) {
      pre_dim_pidding_flag = false;
      merge_input_shape_dim = merge_input_shape_dim * cu_input_dim;
    } else {
      if (pre_dim_pidding_flag) {
        merge_input_shape_dims.insert(merge_input_shape_dims.begin(), cu_input_dim);
        merge_paddings_values.insert(merge_paddings_values.begin(), cu_paddings_value_right);
        merge_paddings_values.insert(merge_paddings_values.begin(), cu_paddings_value_left);
      } else {
        merge_input_shape_dims.insert(merge_input_shape_dims.begin(), cu_input_dim * merge_input_shape_dim);
        merge_paddings_values.insert(merge_paddings_values.begin(), cu_paddings_value_right * merge_input_shape_dim);
        merge_paddings_values.insert(merge_paddings_values.begin(), cu_paddings_value_left * merge_input_shape_dim);
        merge_input_shape_dim = 1;
      }
      pre_dim_pidding_flag = true;
    }
  }

  shape_len = merge_input_shape_dims.size();
  for (size_t i = 0; i < MAX_SHAPE_LEN - shape_len; i++) {
    merge_input_shape_dims.insert(merge_input_shape_dims.begin(), 1);
    merge_paddings_values.insert(merge_paddings_values.begin(), 0);
    merge_paddings_values.insert(merge_paddings_values.begin(), 0);
  }

  tiling_params.tiling_input_dim_0 = merge_input_shape_dims[0];
  tiling_params.tiling_input_dim_1 = merge_input_shape_dims[1];
  tiling_params.tiling_input_dim_2 = merge_input_shape_dims[SHAPE_DIM_INDEX_2];
  tiling_params.tiling_input_dim_3 = merge_input_shape_dims[SHAPE_DIM_INDEX_3];
  tiling_params.tiling_input_dim_4 = merge_input_shape_dims[SHAPE_DIM_INDEX_4];
  tiling_params.tiling_input_dim_5 = merge_input_shape_dims[SHAPE_DIM_INDEX_5];
  tiling_params.tiling_input_dim_6 = merge_input_shape_dims[SHAPE_DIM_INDEX_6];
  tiling_params.tiling_input_dim_7 = merge_input_shape_dims[SHAPE_DIM_INDEX_7] * compile_params.dtype_rate;
  tiling_params.tiling_pading_00 = merge_paddings_values[0];
  tiling_params.tiling_pading_01 = merge_paddings_values[1];
  tiling_params.tiling_pading_10 = merge_paddings_values[TILING_PADDING_INDEX_2];
  tiling_params.tiling_pading_11 = merge_paddings_values[TILING_PADDING_INDEX_3];
  tiling_params.tiling_pading_20 = merge_paddings_values[TILING_PADDING_INDEX_4];
  tiling_params.tiling_pading_21 = merge_paddings_values[TILING_PADDING_INDEX_5];
  tiling_params.tiling_pading_30 = merge_paddings_values[TILING_PADDING_INDEX_6];
  tiling_params.tiling_pading_31 = merge_paddings_values[TILING_PADDING_INDEX_7];
  tiling_params.tiling_pading_40 = merge_paddings_values[TILING_PADDING_INDEX_8];
  tiling_params.tiling_pading_41 = merge_paddings_values[TILING_PADDING_INDEX_9];
  tiling_params.tiling_pading_50 = merge_paddings_values[TILING_PADDING_INDEX_10];
  tiling_params.tiling_pading_51 = merge_paddings_values[TILING_PADDING_INDEX_11];
  tiling_params.tiling_pading_60 = merge_paddings_values[TILING_PADDING_INDEX_12];
  tiling_params.tiling_pading_61 = merge_paddings_values[TILING_PADDING_INDEX_13];
  tiling_params.tiling_pading_70 = merge_paddings_values[TILING_PADDING_INDEX_14] * compile_params.dtype_rate;
  tiling_params.tiling_pading_71 = merge_paddings_values[TILING_PADDING_INDEX_15] * compile_params.dtype_rate;

  auto last_dim_output =
      tiling_params.tiling_input_dim_7 + tiling_params.tiling_pading_70 + tiling_params.tiling_pading_71;
  OP_LOGD(compile_params.op_type, "tiling_two_max_output_size=%ld. ", compile_params.tiling_two_max_output_size);
  if (static_cast<uint64_t>(last_dim_output) < compile_params.tiling_two_max_output_size) {
    tiling_params.tiling_key = TILING_MODE_2;
    tiling_params.tiling_input_dim_cut_axis = 2;
    if (shape_len == DST_LEN && (tiling_params.tiling_pading_60 + tiling_params.tiling_pading_61 == 0)) {
      // when origin shape len is 2, will split the core_dim to dim 3, base the core_num
      int64_t split_dim = 1;
      for (int64_t i = 0; i < compile_params.core_num; i++) {
        int64_t cu_split_dim = compile_params.core_num - i;
        if (tiling_params.tiling_input_dim_6 % cu_split_dim == 0) {
          split_dim = cu_split_dim;
          break;
        }
      }
      tiling_params.tiling_input_dim_6 = tiling_params.tiling_input_dim_6 / split_dim;
      tiling_params.tiling_input_dim_5 = split_dim;
    }
  }
  if ((shape_len == DST_LEN && tiling_params.tiling_input_dim_7 > TILING_0_SIGMENT_1 * compile_params.core_num) ||
      (shape_len == 1 && tiling_params.tiling_input_dim_7 >= TILING_0_SIGMENT_2 * compile_params.core_num)) {
    tiling_params.tiling_key = TILING_MODE_0;
    tiling_params.tiling_input_dim_cut_axis = 0;
  }
  return true;
}

bool PadTiling(const std::string& op_type, const ge::Operator& op_paras, const std::vector<int64_t>& op_compile_info,
               utils::OpRunInfo& run_info) {
  using namespace ge;
  OP_LOGI(op_type, "begin to run tiling.");
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get OpDesc failed."),
                  return false);

  // begin to get compile data
  PadCompileParams compile_params;
  compile_params.op_type = op_type;
  OP_TILING_CHECK(!GetPadCompileParams(op_compile_info, compile_params),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info from nlohmann json failed."),
                  return false);
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  int64_t pad_input_idx = op_type == OP_STRIDED_SLICE_GRAD ? 4 : 0;
  auto pad_input_desc = operator_info->MutableInputDesc(pad_input_idx);
  std::vector<int64_t> input_shape = pad_input_desc->MutableShape().GetDims();
  std::vector<int64_t> paddings_const_values;
  if (op_type == OP_STRIDED_SLICE_GRAD) {
    OP_TILING_CHECK(!CalcuPaddingForStridedSliceGrad(op_paras, compile_params, paddings_const_values, input_shape),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "calcu paddings const value failed."), return false);
  } else {
    // the paddings input index is 1
    OP_TILING_CHECK(!ops::GetConstIntData(op_paras, 1, paddings_const_values),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get paddings const value failed."), return false);
  }
  PrintTensorValue(compile_params, input_shape, "input_shape");
  PrintTensorValue(compile_params, paddings_const_values, "paddings");
  auto pad_format = pad_input_desc->GetFormat();
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  // end to get compile data
  PadTilingParams run_params;
  InitRunningParams(run_params);
  auto special_format_it = find(SPECIAL_FORMAT.begin(), SPECIAL_FORMAT.end(), pad_format);
  if (special_format_it != SPECIAL_FORMAT.end()) {
    std::vector<int64_t> paddings_new_values(input_shape.size() * 2, 0);
    auto pad_ori_format = pad_input_desc->GetOriginFormat();
    OP_LOGD(op_type, "special pad case, format = %s, ori format = %s.", to_string(pad_format).c_str(),
            to_string(pad_ori_format).c_str());
    vector<int32_t> format_nchw_idx;
    auto ori_foramt_it = FORMAT_FOR_NCHW_IDX.find(pad_ori_format);
    OP_TILING_CHECK(ori_foramt_it == FORMAT_FOR_NCHW_IDX.end(),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "do not support ori format(%s) when format is (%s)",
                                                    to_string(pad_ori_format).c_str(), to_string(pad_format).c_str()),
                    return false);
    format_nchw_idx = ori_foramt_it->second;
    size_t shape_expect = format_nchw_idx.size() + 1;
    OP_TILING_CHECK(
        shape_expect != input_shape.size(),
        VECTOR_INNER_ERR_REPORT_TILIING(
            op_type, "the len of input_shape(%ld) must be equal with len(%ld) of format. when format is special format",
            input_shape.size(), shape_expect),
        return false);
    for (size_t i = 0; i < format_nchw_idx.size(); i++) {
      auto dim_idx = format_nchw_idx[i];
      paddings_new_values[i * PADDING_FACTOR] = paddings_const_values[dim_idx * PADDING_FACTOR];
      paddings_new_values[i * PADDING_FACTOR + 1] = paddings_const_values[dim_idx * PADDING_FACTOR + 1];
    }

    // modify C1 padding = C padding / SHAPE_C0
    auto padding_c1_idx = (format_nchw_idx.size() - CUT_LEN) * PADDING_FACTOR;
    paddings_new_values[padding_c1_idx] = paddings_new_values[padding_c1_idx] / SHAPE_C0;
    paddings_new_values[padding_c1_idx + 1] = paddings_new_values[padding_c1_idx + 1] / SHAPE_C0;

    PrintTensorValue(compile_params, paddings_new_values, "hd_paddings");
    GetTilingParam(input_shape, paddings_new_values, compile_params, run_params);
  } else {
    GetTilingParam(input_shape, paddings_const_values, compile_params, run_params);
  }
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  // get tiling end
  SetRuningParams(run_params, run_info);

  PrintTilingParams(run_params, op_type);

  run_info.SetBlockDim(compile_params.core_num);
  // set workspace for pad_v2
  if (op_type == "PadV2") {
    std::vector<int64_t> workspace = {compile_params.core_num * ONE_BLOCK_BYTES};
    for (auto ws : workspace) {
      run_info.AddWorkspace(ws);
    }
  }

  OP_LOGI(op_type, "end to run tiling, succ!");
  PROFILING_TILING_END();
  return true;
}

static const std::vector<std::string> PAD_COMPILE_INFO_KEY = {"core_num", "ub_size", "dtype_rate",
                                                              "tiling_two_max_output_size"};
static const std::map<std::string, std::int64_t> OPTIONAL_VALUE = {{"tiling_two_max_output_size", 960}};
REGISTER_OP_TILING_V4_WITH_VECTOR(Pad, PadTiling, PAD_COMPILE_INFO_KEY, OPTIONAL_VALUE);
REGISTER_OP_TILING_V4_WITH_VECTOR(PadV2, PadTiling, PAD_COMPILE_INFO_KEY, OPTIONAL_VALUE);

// register tiling interface of the StridedSliceGrad op.
static const std::vector<std::string> STRIDED_COMPILE_INFO_KEY = {
    "core_num",      "ub_size",       "dtype_rate",      "tiling_two_max_output_size", "begin_mask", "end_mask",
    "ellipsis_mask", "new_axis_mask", "shrink_axis_mask"};
REGISTER_OP_TILING_V4_WITH_VECTOR(StridedSliceGrad, PadTiling, STRIDED_COMPILE_INFO_KEY, OPTIONAL_VALUE);
}  // namespace optiling
