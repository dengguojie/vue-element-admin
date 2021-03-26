/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "graph/debug/ge_log.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"

namespace optiling {

const int64_t BLOCK_SIZE = 32;

const int64_t TILING_MODE_1 = 1;

const int64_t TILING_MODE_2 = 2;

const int64_t TILING_MODE_3 = 3;

struct PadCompileParams {
  int64_t core_num;
  int64_t ub_size;
  int64_t dtype_rate;
  std::string op_type;
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

static bool GetPadCompileParams(const nlohmann::json& compile_info,
                                PadCompileParams& compile_params) {
  using namespace nlohmann;
  auto allVars = compile_info["vars"];
  if (allVars.count("core_num") == 0) {
    OP_LOGE(compile_params.op_type, "GetCompileParams, get core_num error");
    return false;
  }
  compile_params.core_num = allVars["core_num"].get<std::int64_t>();
  if (allVars.count("ub_size") == 0) {
    OP_LOGE(compile_params.op_type, "GetCompileParams, get ub_size error");
    return false;
  }
  compile_params.ub_size = allVars["ub_size"].get<std::int64_t>();
  if (allVars.count("dtype_rate") == 0) {
    OP_LOGE(compile_params.op_type, "GetCompileParams, get dtype_rate error");
    return false;
  }
  compile_params.dtype_rate = allVars["dtype_rate"].get<std::int64_t>();

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

void PrintTilingParams(const PadTilingParams& params) {
  GELOGD("op [PadTiling] : tiling_key=%ld. ", params.tiling_key);
  GELOGD("op [PadTiling] : tiling_input_dim_0=%ld.", params.tiling_input_dim_0);
  GELOGD("op [PadTiling] : tiling_input_dim_1=%ld.", params.tiling_input_dim_1);
  GELOGD("op [PadTiling] : tiling_input_dim_2=%ld.", params.tiling_input_dim_2);
  GELOGD("op [PadTiling] : tiling_input_dim_3=%ld.", params.tiling_input_dim_3);
  GELOGD("op [PadTiling] : tiling_input_dim_4=%ld.", params.tiling_input_dim_4);
  GELOGD("op [PadTiling] : tiling_input_dim_5=%ld.", params.tiling_input_dim_5);
  GELOGD("op [PadTiling] : tiling_pading_00=%ld.", params.tiling_pading_00);
  GELOGD("op [PadTiling] : tiling_pading_01=%ld.", params.tiling_pading_01);
  GELOGD("op [PadTiling] : tiling_pading_10=%ld.", params.tiling_pading_10);
  GELOGD("op [PadTiling] : tiling_pading_11=%ld.", params.tiling_pading_11);
  GELOGD("op [PadTiling] : tiling_pading_20=%ld.", params.tiling_pading_20);
  GELOGD("op [PadTiling] : tiling_pading_21=%ld.", params.tiling_pading_21);
  GELOGD("op [PadTiling] : tiling_pading_30=%ld.", params.tiling_pading_30);
  GELOGD("op [PadTiling] : tiling_pading_31=%ld.", params.tiling_pading_31);
  GELOGD("op [PadTiling] : tiling_pading_40=%ld.", params.tiling_pading_40);
  GELOGD("op [PadTiling] : tiling_pading_41=%ld.", params.tiling_pading_41);
  GELOGD("op [PadTiling] : tiling_pading_50=%ld.", params.tiling_pading_50);
  GELOGD("op [PadTiling] : tiling_pading_51=%ld.", params.tiling_pading_51);
  GELOGD("op [PadTiling] : tiling_input_dim_cut_axis=%ld.", params.tiling_input_dim_cut_axis);
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
  for (int64_t i = 0; i < shape_len; i++) {
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
  for (int64_t i = 0; i < 6 - shape_len; i++) {
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
    tiling_params.tiling_key = 2;
    tiling_params.tiling_input_dim_cut_axis = 2;
  }
  if ((shape_len == 2 && last_dim_output > 128 * compile_params.core_num) || shape_len == 1) {
    tiling_params.tiling_key = 0;
    tiling_params.tiling_input_dim_cut_axis = 0;
  }
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

bool PadTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_compile_info,
               OpRunInfo& run_info) {
  using namespace ge;

  GELOGI("op[%s] PadTiling running.", op_type.c_str());
  if (op_compile_info == nullptr) {
    OP_LOGE(op_type.c_str(), "op [PadTiling] : op_compile_info json error.");
    return false;
  }

  if (op_paras.inputs.empty() || op_paras.inputs[0].tensor.empty() ||
      op_paras.inputs[1].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(op_type.c_str(), "input_x or paddings",
                                  "The input may be empty");
    OP_LOGE(op_type.c_str(), "op [PadTiling] : input shape error");
    return false;
  }

  const std::vector<int64_t>& input_shape = op_paras.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& paddings_shape = op_paras.inputs[1].tensor[0].shape;

  std::vector<int64_t> paddings_const_values;
  GetPaddingsConstValue(op_paras, "paddings", op_paras.inputs[1].tensor[0].dtype, paddings_const_values);

  // begin to get compile data
  PadCompileParams compile_params;
  compile_params.op_type = op_type;
  if (!GetPadCompileParams(op_compile_info, compile_params)) {
    OP_LOGE(op_type, "get compile info from nlohmann json failed.");
    return false;
  }
  // end to get compile data
  PadTilingParams run_params;
  InitRunningParams(run_params);
  GetTilingParam(input_shape, paddings_const_values, compile_params, run_params);
  SetRuningParams(run_params, run_info);

  PrintTilingParams(run_params);

  run_info.block_dim = compile_params.core_num;
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;

  GELOGI("op[%s] tiling run success.", op_type.c_str());

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(Pad, PadTiling);
}  // namespace optiling

