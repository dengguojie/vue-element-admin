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
 * \file mirror_pad.cc
 * \brief
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/util.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {
// Mode0 support all case divide [0,1,2,3] and [4]
const int64_t TILING_MODE_0 = 0;
// Mode1 divide [0,1,2] and [3,4]
const int64_t TILING_MODE_1 = 1;
// Mode2 support net case better perf
const int64_t TILING_MODE_2 = 2;

const int64_t MAX_AXIS = 5;

const int64_t BLOCK_NUM = 16;
constexpr int32_t DIM_VAL = 16;

struct MirrorPadCompileParams {
  int64_t core_num;
  int64_t output_dim4_max_cnt;
  int64_t dtype_rate;
  std::string op_type;
};

struct MirrorPadTilingParams { 
  int64_t tiling_key;
  // input dim num
  int64_t tiling_input_dim_0;
  int64_t tiling_input_dim_1;
  int64_t tiling_input_dim_2;
  int64_t tiling_input_dim_3;
  int64_t tiling_input_dim_4;
  // padding number
  int64_t tiling_padding_00;
  int64_t tiling_padding_01;
  int64_t tiling_padding_10;
  int64_t tiling_padding_11;
  int64_t tiling_padding_20;
  int64_t tiling_padding_21;
  int64_t tiling_padding_30;
  int64_t tiling_padding_31;
  int64_t tiling_padding_40;
  int64_t tiling_padding_41;
  // calculate core
  int64_t core_used_num;
  int64_t num_per_core;
  int64_t num_tail_core;
};

void InitRunningParams(MirrorPadTilingParams& params) {
  params.tiling_key = TILING_MODE_1;
  params.tiling_input_dim_0 = 1;
  params.tiling_input_dim_1 = 1;
  params.tiling_input_dim_2 = 1;
  params.tiling_input_dim_3 = 1;
  params.tiling_input_dim_4 = 1;
  params.tiling_padding_00 = 0;
  params.tiling_padding_01 = 0;
  params.tiling_padding_10 = 0;
  params.tiling_padding_11 = 0;
  params.tiling_padding_20 = 0;
  params.tiling_padding_21 = 0;
  params.tiling_padding_30 = 0;
  params.tiling_padding_31 = 0;
  params.tiling_padding_40 = 0;
  params.tiling_padding_41 = 0;
  params.core_used_num = 0;
  params.num_per_core = 0;
  params.num_tail_core = 0;
}

static bool GetMirrorPaddingsConstValue(const TeOpParas& paras, const string& name, const string& dtype,
                                        vector<int64_t>& values) {
  values.clear();
  if (paras.const_inputs.count(name) == 0 || std::get<0>(paras.const_inputs.at(name)) == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING("MirrorPadTiling", "GetMirrorPaddingsConstValue, get paddings const error");
    return false;
  }

  auto size = std::get<1>(paras.const_inputs.at(name));
  if (dtype == "int64") {
    int count = size / sizeof(int64_t);
    const int64_t* data_addr = reinterpret_cast<const int64_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i = 0; i < count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  } else if (dtype == "int32") {
    int count = size / sizeof(int32_t);
    const int32_t* data_addr = reinterpret_cast<const int32_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i = 0; i < count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  }

  return true;
}

static bool GetMirrorPadCompileParams(const nlohmann::json& compile_info, MirrorPadCompileParams& compile_params) {
  using namespace nlohmann;
  auto allVars = compile_info["vars"];
  if (allVars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get core_num error");
    return false;
  }
  compile_params.core_num = allVars["core_num"].get<std::int64_t>();
  if (allVars.count("output_dim4_max_cnt") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get output_dim4_max_cnt error");
    return false;
  }
  compile_params.output_dim4_max_cnt = allVars["output_dim4_max_cnt"].get<std::int64_t>();
  if (allVars.count("dtype_rate") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(compile_params.op_type, "GetCompileParams, get dtype_rate error");
    return false;
  }
  compile_params.dtype_rate = allVars["dtype_rate"].get<std::int64_t>();
  return true;
}

void SetRuningParams(const MirrorPadTilingParams& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.tiling_key);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_0);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_1);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_2);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_3);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_4);
  ByteBufferPut(run_info.tiling_data, params.tiling_padding_00);
  ByteBufferPut(run_info.tiling_data, params.tiling_padding_01);
  ByteBufferPut(run_info.tiling_data, params.tiling_padding_10);
  ByteBufferPut(run_info.tiling_data, params.tiling_padding_11);
  ByteBufferPut(run_info.tiling_data, params.tiling_padding_20);
  ByteBufferPut(run_info.tiling_data, params.tiling_padding_21);
  ByteBufferPut(run_info.tiling_data, params.tiling_padding_30);
  ByteBufferPut(run_info.tiling_data, params.tiling_padding_31);
  ByteBufferPut(run_info.tiling_data, params.tiling_padding_40);
  ByteBufferPut(run_info.tiling_data, params.tiling_padding_41);
  ByteBufferPut(run_info.tiling_data, params.core_used_num);
  ByteBufferPut(run_info.tiling_data, params.num_per_core);
  ByteBufferPut(run_info.tiling_data, params.num_tail_core);
}

void PrintTilingParams(const MirrorPadTilingParams& params, const std::string& op_type) {
  OP_LOGD(op_type, "tiling_key=%ld. ", params.tiling_key);
  OP_LOGD(op_type, "tiling_input_dim_0=%ld.", params.tiling_input_dim_0);
  OP_LOGD(op_type, "tiling_input_dim_1=%ld.", params.tiling_input_dim_1);
  OP_LOGD(op_type, "tiling_input_dim_2=%ld.", params.tiling_input_dim_2);
  OP_LOGD(op_type, "tiling_input_dim_3=%ld.", params.tiling_input_dim_3);
  OP_LOGD(op_type, "tiling_input_dim_4=%ld.", params.tiling_input_dim_4);
  OP_LOGD(op_type, "tiling_padding_00=%ld.", params.tiling_padding_00);
  OP_LOGD(op_type, "tiling_padding_01=%ld.", params.tiling_padding_01);
  OP_LOGD(op_type, "tiling_padding_10=%ld.", params.tiling_padding_10);
  OP_LOGD(op_type, "tiling_padding_11=%ld.", params.tiling_padding_11);
  OP_LOGD(op_type, "tiling_padding_20=%ld.", params.tiling_padding_20);
  OP_LOGD(op_type, "tiling_padding_21=%ld.", params.tiling_padding_21);
  OP_LOGD(op_type, "tiling_padding_30=%ld.", params.tiling_padding_30);
  OP_LOGD(op_type, "tiling_padding_31=%ld.", params.tiling_padding_31);
  OP_LOGD(op_type, "tiling_padding_40=%ld.", params.tiling_padding_40);
  OP_LOGD(op_type, "tiling_padding_41=%ld.", params.tiling_padding_41);
  OP_LOGD(op_type, "core_used_num=%ld.", params.core_used_num);
  OP_LOGD(op_type, "num_per_core=%ld.", params.num_per_core);
  OP_LOGD(op_type, "num_tail_core=%ld.", params.num_tail_core);
}

void _printTensorValue(const MirrorPadCompileParams& compile_params, const std::vector<int64_t>& in,
                       const std::string& name) {
  using namespace std;
  string vec_str;
  for (auto item : in) {
    vec_str += to_string(item);
    vec_str += ",";
  }
  OP_LOGD(compile_params.op_type, "Func[_printTensorValue] [%s]: [%s].", name.c_str(), vec_str.c_str());
}

static bool GetTilingParam(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& paddings_const_values,
                           const MirrorPadCompileParams& compile_params, MirrorPadTilingParams& tiling_params) {
  std::vector<int64_t> merge_input_shape_dims = input_shape;
  std::vector<int64_t> merge_paddings_values = paddings_const_values;
  auto shape_len = input_shape.size();
  for (size_t i = 0; i < MAX_AXIS - shape_len; i++) {
    merge_input_shape_dims.insert(merge_input_shape_dims.begin(), 1);
    merge_paddings_values.insert(merge_paddings_values.begin(), 0);
    merge_paddings_values.insert(merge_paddings_values.begin(), 0);
  }
  tiling_params.tiling_input_dim_0 = merge_input_shape_dims[0];
  tiling_params.tiling_input_dim_1 = merge_input_shape_dims[1];
  tiling_params.tiling_input_dim_2 = merge_input_shape_dims[2];
  tiling_params.tiling_input_dim_3 = merge_input_shape_dims[3];
  tiling_params.tiling_input_dim_4 = merge_input_shape_dims[4] * compile_params.dtype_rate;
  tiling_params.tiling_padding_00 = merge_paddings_values[0];
  tiling_params.tiling_padding_01 = merge_paddings_values[1];
  tiling_params.tiling_padding_10 = merge_paddings_values[2];
  tiling_params.tiling_padding_11 = merge_paddings_values[3];
  tiling_params.tiling_padding_20 = merge_paddings_values[4];
  tiling_params.tiling_padding_21 = merge_paddings_values[5];
  tiling_params.tiling_padding_30 = merge_paddings_values[6];
  tiling_params.tiling_padding_31 = merge_paddings_values[7];
  tiling_params.tiling_padding_40 = merge_paddings_values[8] * compile_params.dtype_rate;
  tiling_params.tiling_padding_41 = merge_paddings_values[9] * compile_params.dtype_rate;

  auto output_dim_4 =
      tiling_params.tiling_input_dim_4 + tiling_params.tiling_padding_40 + tiling_params.tiling_padding_41;
  auto output_dim_3 =
      tiling_params.tiling_input_dim_3 + tiling_params.tiling_padding_30 + tiling_params.tiling_padding_31;
  auto inner_num = output_dim_4 * output_dim_3;
  auto outer_num = (merge_input_shape_dims[0] + merge_paddings_values[0] + merge_paddings_values[1]) *
                   (merge_input_shape_dims[1] + merge_paddings_values[2] + merge_paddings_values[3]) *
                   (merge_input_shape_dims[2] + merge_paddings_values[4] + merge_paddings_values[5]);
  auto outer_num_mode_0 = outer_num * (merge_input_shape_dims[3] + merge_paddings_values[6] + merge_paddings_values[7]);
  bool input_dim3_addressrollback =
      (tiling_params.tiling_input_dim_3 * output_dim_4 < DIM_VAL) && tiling_params.tiling_padding_31 == 0;
  bool padding31_addressrollback =
      (tiling_params.tiling_padding_31 * output_dim_4 < DIM_VAL) && tiling_params.tiling_padding_31 != 0;

  // network case tiling mode
  if (output_dim_4 % BLOCK_NUM == 0 && output_dim_4 == tiling_params.tiling_input_dim_4) {
    tiling_params.tiling_key = TILING_MODE_2;
    tiling_params.num_per_core = (outer_num_mode_0 + compile_params.core_num - 1) / compile_params.core_num;
    tiling_params.core_used_num = (outer_num_mode_0 + tiling_params.num_per_core - 1) / tiling_params.num_per_core;
    tiling_params.num_tail_core = outer_num_mode_0 - (tiling_params.num_per_core * (tiling_params.core_used_num - 1));
    return true;
  }
  if (inner_num < BLOCK_NUM || input_dim3_addressrollback || padding31_addressrollback) {
    tiling_params.tiling_key = TILING_MODE_1;
    tiling_params.core_used_num = 1;
    tiling_params.num_tail_core = outer_num;
  } else {
    if (output_dim_4 <= compile_params.output_dim4_max_cnt) {
      tiling_params.num_per_core = (outer_num + compile_params.core_num - 1) / compile_params.core_num;
      tiling_params.core_used_num = (outer_num + tiling_params.num_per_core - 1) / tiling_params.num_per_core;
      tiling_params.num_tail_core = outer_num - (tiling_params.num_per_core * (tiling_params.core_used_num - 1));
      tiling_params.tiling_key = TILING_MODE_1;
    } else {
      tiling_params.num_per_core = (outer_num_mode_0 + compile_params.core_num - 1) / compile_params.core_num;
      tiling_params.core_used_num = (outer_num_mode_0 + tiling_params.num_per_core - 1) / tiling_params.num_per_core;
      tiling_params.num_tail_core = outer_num_mode_0 - (tiling_params.num_per_core * (tiling_params.core_used_num - 1));
      tiling_params.tiling_key = TILING_MODE_0;
    }
  }
  return true;
}

bool MirrorPadTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_compile_info,
                     OpRunInfo& run_info) {
  using namespace ge;

  OP_LOGI(op_type, "begin to run tiling.");
  if (op_compile_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info json error.");
    return false;
  }

  if (op_paras.inputs.empty() || op_paras.inputs[0].tensor.empty() || op_paras.inputs[1].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape error");
    return false;
  }
  // begin to get compile data
  MirrorPadCompileParams compile_params;
  compile_params.op_type = op_type;
  if (!GetMirrorPadCompileParams(op_compile_info, compile_params)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile info from nlohmann json failed.");
    return false;
  }
  const std::vector<int64_t>& input_shape_const = op_paras.inputs[0].tensor[0].shape;
  std::vector<int64_t> input_shape = input_shape_const;
  std::vector<int64_t> paddings_const_values;
  GetMirrorPaddingsConstValue(op_paras, "paddings", op_paras.inputs[1].tensor[0].dtype, paddings_const_values);
  _printTensorValue(compile_params, input_shape, "input_shape");
  _printTensorValue(compile_params, paddings_const_values, "paddings");

  // end to get compile data
  MirrorPadTilingParams run_params;
  InitRunningParams(run_params);
  GetTilingParam(input_shape, paddings_const_values, compile_params, run_params);
  SetRuningParams(run_params, run_info);
  PrintTilingParams(run_params, op_type);

  run_info.block_dim = compile_params.core_num;
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;

  OP_LOGI(op_type, "end to run tiling, success!");

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(MirrorPad, MirrorPadTiling);
}  // namespace optiling
