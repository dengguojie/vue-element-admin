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
 * \file pad.cpp
 * \brief
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/strided_slice_infer_shape.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

const int64_t TILING_MODE_0 = 0;

const int64_t TILING_MODE_1 = 1;

const int64_t TILING_MODE_2 = 2;

const int64_t TILING_MODE_3 = 3;

const std::string OP_STRIDED_SLICE_GRAD = "StridedSliceGrad";

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
};

struct PadTilingParams {
  int64_t tiling_key;
  int64_t tiling_input_dim_0;
  int64_t tiling_input_dim_1;
  int64_t tiling_input_dim_2;
  int64_t tiling_input_dim_3;
  int64_t tiling_input_dim_4;
  int64_t tiling_input_dim_5;
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
  params.tiling_input_dim_cut_axis = 1;
}

static bool GetPaddingsConstValue(const TeOpParas& paras, const string& name,
                                  const string& dtype, vector<int64_t>& values) {
  values.clear();
  if (paras.const_inputs.count(name) == 0 || std::get<0>(paras.const_inputs.at(name)) == nullptr) {
    return false;
  }

  auto size = std::get<1>(paras.const_inputs.at(name));
  if (dtype == "int64") {
    int count = size / sizeof(int64_t);
    const int64_t *data_addr = reinterpret_cast<const int64_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i=0; i < count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  } else if (dtype == "int32") {
    int count = size / sizeof(int32_t);
    const int32_t *data_addr = reinterpret_cast<const int32_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i=0; i < count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  }

  return true;
}

static bool GetPadCompileParams(const nlohmann::json& compile_info,
                                PadCompileParams& compile_params) {
  using namespace nlohmann;
  auto allVars = compile_info["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get core_num error");
    return false;
  }
  compile_params.core_num = allVars["core_num"].get<std::int64_t>();
  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get ub_size error");
    return false;
  }
  compile_params.ub_size = allVars["ub_size"].get<std::int64_t>();
  if (allVars.count("dtype_rate") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get dtype_rate error");
    return false;
  }
  compile_params.dtype_rate = allVars["dtype_rate"].get<std::int64_t>();

  // add for StridedSliceGrad
  if (compile_params.op_type == OP_STRIDED_SLICE_GRAD) {
    if (allVars.count("begin_mask") == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get begin_mask error");
      return false;
    }
    compile_params.begin_mask = allVars["begin_mask"].get<std::uint64_t>();

    if (allVars.count("end_mask") == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get end_mask error");
      return false;
    }
    compile_params.end_mask = allVars["end_mask"].get<std::uint64_t>();

    if (allVars.count("ellipsis_mask") == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get ellipsis_mask error");
      return false;
    }
    compile_params.ellipsis_mask = allVars["ellipsis_mask"].get<std::uint64_t>();

    if (allVars.count("new_axis_mask") == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get new_axis_mask error");
      return false;
    }
    compile_params.new_axis_mask = allVars["new_axis_mask"].get<std::uint64_t>();

    if (allVars.count("shrink_axis_mask") == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get shrink_axis_mask error");
      return false;
    }
    compile_params.shrink_axis_mask = allVars["shrink_axis_mask"].get<std::uint64_t>();
  }
  return true;
}

void SetRuningParams(const PadTilingParams& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.tiling_key);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_0);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_1);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_2);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_3);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_4);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_5);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_00);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_01);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_10);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_11);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_20);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_21);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_30);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_31);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_40);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_41);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_50);
  ByteBufferPut(run_info.tiling_data, params.tiling_pading_51);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_cut_axis);
}

void PrintTilingParams(const PadTilingParams& params, const std::string& op_type) {
  OP_LOGD(op_type, "tiling_key=%ld. ", params.tiling_key);
  OP_LOGD(op_type, "tiling_input_dim_0=%ld.", params.tiling_input_dim_0);
  OP_LOGD(op_type, "tiling_input_dim_1=%ld.", params.tiling_input_dim_1);
  OP_LOGD(op_type, "tiling_input_dim_2=%ld.", params.tiling_input_dim_2);
  OP_LOGD(op_type, "tiling_input_dim_3=%ld.", params.tiling_input_dim_3);
  OP_LOGD(op_type, "tiling_input_dim_4=%ld.", params.tiling_input_dim_4);
  OP_LOGD(op_type, "tiling_input_dim_5=%ld.", params.tiling_input_dim_5);
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
  OP_LOGD(op_type, "tiling_input_dim_cut_axis=%ld.", params.tiling_input_dim_cut_axis);
}

void _printTensorValue(const PadCompileParams& compile_params,
                       const std::vector<int64_t>& in,
                       const std::string& name) {
  using namespace std;
  string vec_str;
  for (auto item : in) {
    vec_str += to_string(item);
    vec_str += ",";
  }
  OP_LOGD(compile_params.op_type, "Func[_printTensorValue] [%s]: [%s].", name.c_str(), vec_str.c_str());
}

static bool CalcuPaddingForStridedSliceGrad(const TeOpParas& op_paras,
                                            const PadCompileParams& compile_params,
                                            std::vector<int64_t>& paddings_const_values,
                                            std::vector<int64_t>& paddings_input_shape) {
  struct SliceParameters slice_params_output = {};
  struct SliceMasks slicemasks_output;
  map<string, std::pair<int, vector<int64_t>&>> const_params = {
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
    int index = item.second.first;
    auto& values = item.second.second;
    if (!GetPaddingsConstValue(op_paras, name, op_paras.inputs[index].tensor[0].dtype, values)) {
      VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "Get %s values failed", name.c_str());
      return false;
    }
  }

  // Print Org_Tensor'values
  _printTensorValue(compile_params, slice_params_output.input, "shape");
  _printTensorValue(compile_params, slice_params_output.begin_list, "begin");
  _printTensorValue(compile_params, slice_params_output.end_list, "end");
  _printTensorValue(compile_params, slice_params_output.stride_list, "strides");

  // Calc Paddings
  bool cond0 = slice_params_output.input.size() > slice_params_output.begin_list.size();
  bool cond1 = slice_params_output.begin_list.size() == 2;
  bool cond2 = slicemasks_output.ellipsismask == 1;
  bool cond3 = slicemasks_output.shrinkaxismask == 2;
  uint64_t length = slice_params_output.input.size();
  // input_shape is not change, it will be used in calc padding.
  std::vector<int64_t> input_shape = slice_params_output.input;
  vector<int64_t> ori_begin = slice_params_output.begin_list;

  StridedSliceParams input_params = {
      input_shape,
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
  OP_LOGI(compile_params.op_type, "original begin end and stride is %s, %s, and %s.",
          ops::to_string(input_params.begin).c_str(),
          ops::to_string(input_params.end).c_str(),
          ops::to_string(input_params.strides).c_str());

  vector<int64_t> output_shape;
  vector<pair<int64_t, int64_t>> output_ranges;

  if (!StridedSliceCommonInferShape(compile_params.op_type, input_params, output_shape, output_ranges)) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "StridedSliceCommonInferShape failed.");
    return false;
  }

  OP_LOGI(compile_params.op_type, "after infershape, begin end and stride is %s, %s, and %s.",
          ops::to_string(input_params.begin).c_str(),
          ops::to_string(input_params.end).c_str(),
          ops::to_string(input_params.strides).c_str());

  slice_params_output.begin_list = input_params.begin;
  slice_params_output.end_list = input_params.end;
  slice_params_output.stride_list = input_params.strides;

  _printTensorValue(compile_params, slice_params_output.input, "shape");
  _printTensorValue(compile_params, slice_params_output.begin_list, "begin");
  _printTensorValue(compile_params, slice_params_output.end_list, "end");
  _printTensorValue(compile_params, slice_params_output.stride_list, "strides");

  vector<int64_t> grad_input_dy_shape;
  if (cond0 && cond1 && cond2 && cond3) {
    for (uint64_t i = 0; i < length; i++) {
      paddings_const_values.push_back(0);
      paddings_const_values.push_back(0);
      grad_input_dy_shape.push_back(input_shape[i]);
    }
    paddings_const_values[(length - 1) * 2] = ori_begin[1];
    paddings_const_values[(length - 1) * 2 + 1] = input_shape[length - 1] - ori_begin[1] - 1;
    grad_input_dy_shape[length - 1] = 1;
  } else {
    for (uint64_t i = 0; i < length; i++) {
      int64_t begin_i = slice_params_output.begin_list[i];
      int64_t end_i = slice_params_output.end_list[i];
      int64_t shape_i = input_shape[i];

      begin_i = (begin_i < 0) ? begin_i + shape_i : begin_i;
      end_i = (end_i < 0) ? end_i + shape_i : end_i;

      paddings_const_values.push_back(begin_i);
      paddings_const_values.push_back(shape_i - end_i);
      grad_input_dy_shape.push_back(end_i - begin_i);
    }
  }
  paddings_input_shape = grad_input_dy_shape;
  return true;
}

static bool GetTilingParam(const std::vector<int64_t>& input_shape,
                           const std::vector<int64_t>& paddings_const_values,
                           const PadCompileParams& compile_params,
                           PadTilingParams& tiling_params) {
  auto shape_len = input_shape.size();
  std::vector<int64_t> merge_input_shape_dims;
  std::vector<int64_t> merge_paddings_values;
  int64_t merge_input_shape_dim = 1;
  bool pre_dim_pidding_flag = true;
  for (size_t i = 0; i < shape_len; i++) {
    auto cu_idx = shape_len - i - 1;
    auto cu_input_dim = input_shape[cu_idx];
    auto cu_paddings_value_left = paddings_const_values[cu_idx * 2];
    auto cu_paddings_value_right = paddings_const_values[cu_idx * 2 + 1];
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
  for (size_t i = 0; i < 6 - shape_len; i++) {
    merge_input_shape_dims.insert(merge_input_shape_dims.begin(), 1);
    merge_paddings_values.insert(merge_paddings_values.begin(), 0);
    merge_paddings_values.insert(merge_paddings_values.begin(), 0);
  }

  tiling_params.tiling_input_dim_0 = merge_input_shape_dims[0];
  tiling_params.tiling_input_dim_1 = merge_input_shape_dims[1];
  tiling_params.tiling_input_dim_2 = merge_input_shape_dims[2];
  tiling_params.tiling_input_dim_3 = merge_input_shape_dims[3];
  tiling_params.tiling_input_dim_4 = merge_input_shape_dims[4];
  tiling_params.tiling_input_dim_5 = merge_input_shape_dims[5] * compile_params.dtype_rate;
  tiling_params.tiling_pading_00 = merge_paddings_values[0];
  tiling_params.tiling_pading_01 = merge_paddings_values[1];
  tiling_params.tiling_pading_10 = merge_paddings_values[2];
  tiling_params.tiling_pading_11 = merge_paddings_values[3];
  tiling_params.tiling_pading_20 = merge_paddings_values[4];
  tiling_params.tiling_pading_21 = merge_paddings_values[5];
  tiling_params.tiling_pading_30 = merge_paddings_values[6];
  tiling_params.tiling_pading_31 = merge_paddings_values[7];
  tiling_params.tiling_pading_40 = merge_paddings_values[8];
  tiling_params.tiling_pading_41 = merge_paddings_values[9];
  tiling_params.tiling_pading_50 = merge_paddings_values[10] * compile_params.dtype_rate;
  tiling_params.tiling_pading_51 = merge_paddings_values[11] * compile_params.dtype_rate;

  auto last_dim_output = tiling_params.tiling_input_dim_5
                         + tiling_params.tiling_pading_50 + tiling_params.tiling_pading_51;
  if (last_dim_output < 960) {
    tiling_params.tiling_key = TILING_MODE_2;
    tiling_params.tiling_input_dim_cut_axis = 2;
    if (shape_len == 2 && (tiling_params.tiling_pading_41 + tiling_params.tiling_pading_40 == 0)) {
      // when origin shape len is 2, will split the core_dim to dim 3, base the core_num
      int64_t split_dim = 1;
      for (int64_t i = 0; i < compile_params.core_num; i++) {
        int64_t cu_split_dim = compile_params.core_num - i;
        if (tiling_params.tiling_input_dim_4 % cu_split_dim == 0) {
          split_dim = cu_split_dim;
          break;
        }
      }
      tiling_params.tiling_input_dim_4 = tiling_params.tiling_input_dim_4 / split_dim;
      tiling_params.tiling_input_dim_3 = split_dim;
    }
  }
  if ((shape_len == 2 && last_dim_output > 128 * compile_params.core_num) || shape_len == 1) {
    tiling_params.tiling_key = TILING_MODE_0;
    tiling_params.tiling_input_dim_cut_axis = 0;
  }
  return true;
}

bool PadTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_compile_info,
               OpRunInfo& run_info) {
  using namespace ge;

  OP_LOGI(op_type, "begin to run tiling.");
  if (op_compile_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op [PadTiling] : op_compile_info json error.");
    return false;
  }

  if (op_paras.inputs.empty() || op_paras.inputs[0].tensor.empty() ||
      op_paras.inputs[1].tensor.empty()) {

    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op [PadTiling] : input shape error");
    return false;
  }
  // begin to get compile data
  PadCompileParams compile_params;
  compile_params.op_type = op_type;
  if (!GetPadCompileParams(op_compile_info, compile_params)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info from nlohmann json failed.");
    return false;
  }
  int64_t pad_input_idx = op_type == OP_STRIDED_SLICE_GRAD ? 4 : 0;
  const std::vector<int64_t>& input_shape_const = op_paras.inputs[pad_input_idx].tensor[0].shape;
  std::vector<int64_t> input_shape = input_shape_const;
  std::vector<int64_t> paddings_const_values;
  if (op_type == OP_STRIDED_SLICE_GRAD) {
    CalcuPaddingForStridedSliceGrad(op_paras, compile_params, paddings_const_values, input_shape);
  } else {
    GetPaddingsConstValue(op_paras, "paddings", op_paras.inputs[1].tensor[0].dtype, paddings_const_values);
  }
  _printTensorValue(compile_params, input_shape, "input_shape");
  _printTensorValue(compile_params, paddings_const_values, "paddings");

  // end to get compile data
  PadTilingParams run_params;
  InitRunningParams(run_params);
  GetTilingParam(input_shape, paddings_const_values, compile_params, run_params);
  SetRuningParams(run_params, run_info);

  PrintTilingParams(run_params, op_type);

  run_info.block_dim = compile_params.core_num;
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;

  OP_LOGI(op_type, "end to run tiling, succ!");

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(Pad, PadTiling);
// register tiling interface of the StridedSliceGrad op.
REGISTER_OP_TILING_FUNC_BUFFERED(StridedSliceGrad, PadTiling);
}  // namespace optiling

