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
 * \file pad_v3.cc
 * \brief
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "reflection_pad_v3.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

static const int64_t TILING_MODE_0 = 0;

static const int64_t TILING_MODE_1 = 1;

static const int64_t TILING_MODE_2 = 2;

static const int64_t TILING_MODE_4 = 4;

static const int64_t TILING_MODE_5 = 5;
struct PadV3CompileParams {
  int64_t core_num;
  int64_t ub_size;
  int64_t dtype_rate;
  std::string op_type;
};

struct PadV3TilingParams {
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

static void InitRunningParams(PadV3TilingParams& params) {
  OP_LOGD("begin to InitRunningParams.");
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
  OP_LOGD("begin to GetPaddingsConstValue.");
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

static bool GetPadV3CompileParams(const nlohmann::json& compile_info,
                                  PadV3CompileParams& compile_params) {
  using namespace nlohmann;
  OP_LOGD("begin to GetPadV3CompileParams.");
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
  return true;
}

static void SetRuningParams(const PadV3TilingParams& params, OpRunInfo& run_info) {
  OP_LOGD("begin to SetRuningParams.");
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

static void PrintTilingParams(const PadV3TilingParams& params, const std::string& op_type) {
  OP_LOGD("begin to PrintTilingParams.");
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

static void _printTensorValue(const PadV3CompileParams& compile_params,
                              const std::vector<int64_t>& in,
                              const std::string& name) {
  using namespace std;
  OP_LOGD("begin to _printTensorValue.");
  string vec_str;
  for (auto item : in) {
    vec_str += to_string(item);
    vec_str += ",";
  }
  OP_LOGD(compile_params.op_type, "Func[_printTensorValue] [%s]: [%s].", name.c_str(), vec_str.c_str());
}

static bool GetTilingParam(const std::vector<int64_t>& input_shape,
                           const std::vector<int64_t>& paddings_const_values,
                           const PadV3CompileParams& compile_params,
                           PadV3TilingParams& tiling_params) {
  OP_LOGD("begin to GetTilingParam.");
  auto shape_len = input_shape.size();
  std::vector<int64_t> merge_input_shape_dims;
  std::vector<int64_t> merge_paddings_values;
  if (compile_params.core_num == 1) {
    auto paddings_len = shape_len + shape_len;
    for (size_t i = 0; i < shape_len; i++) {
      merge_input_shape_dims.insert(merge_input_shape_dims.end(), input_shape[i]);
    }
    for (size_t i = 0; i < paddings_len; i++) {
      merge_paddings_values.insert(merge_paddings_values.end(), paddings_const_values[i]);
    }
  }
  else {
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
  
  bool all_zero = true;
  for (auto ele: merge_paddings_values) {
    if (ele != 0) {
      all_zero = false;
    }
  }

  if (all_zero) {
    tiling_params.tiling_key = TILING_MODE_4;
    tiling_params.tiling_input_dim_cut_axis = -1;
  }
  else if (compile_params.core_num == 1) {
    tiling_params.tiling_key = TILING_MODE_5;
    tiling_params.tiling_input_dim_cut_axis = -1;
  }
  else if (last_dim_output < 960) {
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
  else if ((shape_len == 2 && last_dim_output > 128 * compile_params.core_num) || shape_len == 1) {
    tiling_params.tiling_key = TILING_MODE_0;
    tiling_params.tiling_input_dim_cut_axis = 0;
  }
  return true;
}

bool PadV3Tiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_compile_info,
                 OpRunInfo& run_info) {
  auto allVars = op_compile_info["vars"];
  auto mode = allVars["mode"];
  auto padding_contiguous = allVars["padding_contiguous"];
  if (mode == "constant") {
    using namespace ge;
    OP_LOGD("begin to run tiling.");
    if (op_compile_info == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op [PadV3Tiling] : op_compile_info json error.");
        return false;
    }

    if (op_paras.inputs.empty() || op_paras.inputs[0].tensor.empty() ||
        op_paras.inputs[1].tensor.empty()) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op [PadV3Tiling] : input shape error");
        return false;
    }
    // begin to get compile data
    PadV3CompileParams compile_params;
    compile_params.op_type = op_type;
    if (!GetPadV3CompileParams(op_compile_info, compile_params)) {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info from nlohmann json failed.");
        return false;
    }
    int64_t pad_input_idx = 0;
    const std::vector<int64_t>& input_shape_const = op_paras.inputs[pad_input_idx].tensor[0].shape;
    std::vector<int64_t> input_shape = input_shape_const;
    std::vector<int64_t> paddings_const_values;
    std::vector<int64_t> paddings_const_values_origin;
    GetPaddingsConstValue(op_paras, "paddings", op_paras.inputs[1].tensor[0].dtype, paddings_const_values_origin);
    // change paddings format for onnx 9 version
    if (!padding_contiguous) {
        auto rank = input_shape_const.size();
        for (size_t i = 0; i < rank;  i++) {
            paddings_const_values.push_back(paddings_const_values_origin[i]);
            paddings_const_values.push_back(paddings_const_values_origin[i + rank]);
        }
    } 
    else {
        paddings_const_values = paddings_const_values_origin;
    }
    _printTensorValue(compile_params, input_shape, "input_shape");
    _printTensorValue(compile_params, paddings_const_values, "paddings");

    // end to get compile data
    PadV3TilingParams run_params;
    InitRunningParams(run_params);
    GetTilingParam(input_shape, paddings_const_values, compile_params, run_params);
    SetRuningParams(run_params, run_info);

    PrintTilingParams(run_params, op_type);

    run_info.block_dim = compile_params.core_num;
    std::vector<int64_t> workspace;
    if (compile_params.core_num > 1) {
        workspace.push_back(compile_params.core_num * 32);
    }
    run_info.workspaces = workspace;

    OP_LOGI(op_type, "end to run tiling, succ!");
  }
  if (mode == "reflect" || mode == "edge") {
    ReflectionPadV3Tiling(op_type, op_paras, op_compile_info, run_info);
  }
  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(PadV3, PadV3Tiling);
}  // namespace optiling

