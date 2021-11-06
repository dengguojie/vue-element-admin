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
 **/

/*!
 * \file reflection_pad_v3.cc
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

static const int64_t TILING_MODE_0 = 0;
static const int64_t TILING_MODE_1 = 1;
static const int64_t TILING_MODE_2 = 2;
static const int64_t PADDINGS_DIM_INDEX_2 = 4;
static const int64_t PADDINGS_DIM_INDEX_3 = 6;
struct ReflectionPadV3CompileParams {
  int64_t core_num;
  std::string op_type;
};

struct ReflectionPadV3TilingParams {
  int64_t tiling_key;
  int64_t tiling_input_dim_0;
  int64_t tiling_input_dim_1;
  int64_t tiling_input_dim_2;
  int64_t tiling_input_dim_3;
  int64_t tiling_output_dim_0;
  int64_t tiling_output_dim_1;
  int64_t tiling_output_dim_2;
  int64_t tiling_output_dim_3;
  int64_t core_uesd_num;
  int64_t padding_index_0;
  int64_t padding_index_1;
  int64_t padding_index_2;
  int64_t padding_index_3;
  int64_t not_last_core_num;
  int64_t last_core_num;
};

static void InitRunningParams(ReflectionPadV3TilingParams& params) {
  OP_LOGD("begin to InitRunningParams.");
  params.tiling_key = TILING_MODE_0;
  params.tiling_input_dim_0 = 1;
  params.tiling_input_dim_1 = 1;
  params.tiling_input_dim_2 = 1;
  params.tiling_input_dim_3 = 1;
  params.tiling_output_dim_0 = 1;
  params.tiling_output_dim_1 = 1;
  params.tiling_output_dim_2 = 1;
  params.tiling_output_dim_3 = 1;
  params.core_uesd_num = 1;
  params.padding_index_0 = 1;
  params.padding_index_1 = 1;
  params.padding_index_2 = 1;
  params.padding_index_3 = 1;
  params.not_last_core_num = 1;
  params.last_core_num = 1;
}

static bool GetPaddingsConstValue(const TeOpParas& paras, const std::string& name,
                                  const std::string& dtype, std::vector<int64_t>& values) {
  OP_LOGD("begin to GetPaddingsConstValue.");
  values.clear();
  if (paras.const_inputs.count(name) == 0 || std::get<0>(paras.const_inputs.at(name)) == nullptr) {
    return false;
  }

  auto size = std::get<1>(paras.const_inputs.at(name));
  if (dtype == "int64") {
    int count = size / sizeof(int64_t);
    const int64_t *data_addr = reinterpret_cast<const int64_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i = 0; i < count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  } else if (dtype == "int32") {
    int count = size / sizeof(int32_t);
    const int32_t *data_addr = reinterpret_cast<const int32_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i = 0; i < count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  }

  return true;
}

static bool GetReflectionPadV3CompileParams(const nlohmann::json& compile_info,
                                            ReflectionPadV3CompileParams& compile_params) {
  using namespace nlohmann;
  OP_LOGD("begin to GetReflectionPadV3CompileParams.");
  auto allVars = compile_info["vars"];
  if (allVars.count("core_num") == 0) {
    OP_LOGE(compile_params.op_type, "GetCompileParams, get core_num error");
    return false;
  }
  compile_params.core_num = allVars["core_num"].get<std::int64_t>();
  return true;
}

static void SetRuningParams(const ReflectionPadV3TilingParams& params, OpRunInfo& run_info) {
  OP_LOGD("begin to SetRuningParams.");
  ByteBufferPut(run_info.tiling_data, params.tiling_key);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_0);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_1);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_2);
  ByteBufferPut(run_info.tiling_data, params.tiling_input_dim_3);
  ByteBufferPut(run_info.tiling_data, params.tiling_output_dim_0);
  ByteBufferPut(run_info.tiling_data, params.tiling_output_dim_1);
  ByteBufferPut(run_info.tiling_data, params.tiling_output_dim_2);
  ByteBufferPut(run_info.tiling_data, params.tiling_output_dim_3);
  ByteBufferPut(run_info.tiling_data, params.core_uesd_num);
  ByteBufferPut(run_info.tiling_data, params.padding_index_0);
  ByteBufferPut(run_info.tiling_data, params.padding_index_1);
  ByteBufferPut(run_info.tiling_data, params.padding_index_2);
  ByteBufferPut(run_info.tiling_data, params.padding_index_3);
  ByteBufferPut(run_info.tiling_data, params.not_last_core_num);
  ByteBufferPut(run_info.tiling_data, params.last_core_num);
}

static void PrintTilingParams(const ReflectionPadV3TilingParams& params, const std::string& op_type) {
  OP_LOGD("begin to PrintTilingParams.");
  OP_LOGD(op_type, "tiling_key=%ld. ", params.tiling_key);
  OP_LOGD(op_type, "tiling_input_dim_0=%ld.", params.tiling_input_dim_0);
  OP_LOGD(op_type, "tiling_input_dim_1=%ld.", params.tiling_input_dim_1);
  OP_LOGD(op_type, "tiling_input_dim_2=%ld.", params.tiling_input_dim_2);
  OP_LOGD(op_type, "tiling_input_dim_3=%ld.", params.tiling_input_dim_3);
  OP_LOGD(op_type, "tiling_output_dim_0=%ld.", params.tiling_output_dim_0);
  OP_LOGD(op_type, "tiling_output_dim_1=%ld.", params.tiling_output_dim_1);
  OP_LOGD(op_type, "tiling_output_dim_2=%ld.", params.tiling_output_dim_2);
  OP_LOGD(op_type, "tiling_output_dim_3=%ld.", params.tiling_output_dim_3);
  OP_LOGD(op_type, "core_uesd_num=%ld.", params.core_uesd_num);
  OP_LOGD(op_type, "padding_index_0=%ld.", params.padding_index_0);
  OP_LOGD(op_type, "padding_index_1=%ld.", params.padding_index_1);
  OP_LOGD(op_type, "padding_index_2=%ld.", params.padding_index_2);
  OP_LOGD(op_type, "padding_index_3=%ld.", params.padding_index_3);
  OP_LOGD(op_type, "not_last_core_num=%ld.", params.not_last_core_num);
  OP_LOGD(op_type, "last_core_num=%ld.", params.last_core_num);
}

static void _printTensorValue(const ReflectionPadV3CompileParams& compile_params,
                              const std::vector<int64_t>& in,
                              const std::string& name) {
  using namespace std;
  OP_LOGD("begin to _printTensorValue.");
  std::string vec_str;
  for (auto item : in) {
    vec_str += to_string(item);
    vec_str += ",";
  }
  OP_LOGD(compile_params.op_type, "Func[_printTensorValue] [%s]: [%s].", name.c_str(), vec_str.c_str());
}

static int64_t get_core_num(std::vector<int64_t> x_shape, int64_t core_num) {
  OP_TILING_CHECK(core_num == 0, VECTOR_INNER_ERR_REPORT_TILIING("reflection_pad_v3",
    "core_num = 0 is not support"), return -1);
  auto nc_total = x_shape[0] * x_shape[1];
  auto hc_total = x_shape[2] * x_shape[3];
  auto block = 16;
  auto diff = 1;
  auto ele_per_core = (nc_total - diff) / core_num + diff;
  auto core_used = (nc_total - diff) / ele_per_core + diff;
  if (hc_total < (block * block)) {
    while ((hc_total * ele_per_core < block) && (core_used > diff)) {
      core_used -= diff;
      ele_per_core = (nc_total - diff) / core_used + diff;
    }
  }
  core_used = (nc_total - diff) / ele_per_core + diff;
  return core_used;
}

static bool GetTilingParam(const std::vector<int64_t>& input_shape,
                           const std::vector<int64_t>& paddings_const_values,
                           const ReflectionPadV3CompileParams& compile_params,
                           ReflectionPadV3TilingParams& tiling_params) {
  OP_LOGD("begin to GetTilingParam.");
  auto tiling_key = 0;
  auto output_third = paddings_const_values[PADDINGS_DIM_INDEX_2] + input_shape[2]
                      + paddings_const_values[PADDINGS_DIM_INDEX_2 + 1];
  auto output_fourth = paddings_const_values[PADDINGS_DIM_INDEX_3] + input_shape[3]
                      + paddings_const_values[PADDINGS_DIM_INDEX_3 + 1];
  int64_t core_used = 1;
  int64_t not_last_core_numel = 0;
  int64_t last_core_numel = 0;
  auto diff = 1;
  auto block = 16;
  auto nc_total = input_shape[0] * input_shape[1];
  core_used = get_core_num(input_shape, compile_params.core_num);
  OP_TILING_CHECK(core_used == 0, VECTOR_INNER_ERR_REPORT_TILIING("reflection_pad_v3",
    "core_used = 0 is not support"), return false);
  not_last_core_numel = (nc_total - diff) / core_used + diff;
  last_core_numel = nc_total - (core_used - diff) * not_last_core_numel;
  if (input_shape[2] < block && input_shape[3] < block) {
    tiling_key = TILING_MODE_0;
  }
  else if (input_shape[2] >= block && input_shape[3] >= block) {
    tiling_key = TILING_MODE_1;
  }
  else {
    tiling_key = TILING_MODE_2;
  }
  tiling_params.tiling_key = tiling_key;
  tiling_params.tiling_input_dim_0 = input_shape[0];
  tiling_params.tiling_input_dim_1 = input_shape[1];
  tiling_params.tiling_input_dim_2 = input_shape[2];
  tiling_params.tiling_input_dim_3 = input_shape[3];
  tiling_params.tiling_output_dim_0 = input_shape[0];
  tiling_params.tiling_output_dim_1 = input_shape[1];
  tiling_params.tiling_output_dim_2 = output_third;
  tiling_params.tiling_output_dim_3 = output_fourth;
  tiling_params.core_uesd_num = core_used;
  // paddings_const_values is a vector containing 8 elements
  tiling_params.padding_index_0 = paddings_const_values[PADDINGS_DIM_INDEX_3];
  tiling_params.padding_index_1 = paddings_const_values[PADDINGS_DIM_INDEX_3 + 1];
  tiling_params.padding_index_2 = paddings_const_values[PADDINGS_DIM_INDEX_2];
  tiling_params.padding_index_3 = paddings_const_values[PADDINGS_DIM_INDEX_2 + 1];
  tiling_params.not_last_core_num = not_last_core_numel;
  tiling_params.last_core_num = last_core_numel;
  return true;
  }

bool ReflectionPadV3Tiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_compile_info,
                           OpRunInfo& run_info) {
  using namespace ge;
  auto allVars = op_compile_info["vars"];
  auto padding_contiguous = allVars["padding_contiguous"];
  OP_LOGD("begin to run tiling.");
  if (op_compile_info == nullptr) {
    OP_LOGE(op_type, "op [ReflectionPadV3Tiling] : op_compile_info json error.");
    return false;
  }

  if (op_paras.inputs.empty() || op_paras.inputs[0].tensor.empty() ||
      op_paras.inputs[1].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(op_type.c_str(), "input_x or paddings",
                                  "The input may be empty");
    OP_LOGE(op_type, "op [ReflectionPadV3Tiling] : input shape error");
    return false;
  }
  // begin to get compile data
  ReflectionPadV3CompileParams compile_params;
  compile_params.op_type = op_type;
  if (!GetReflectionPadV3CompileParams(op_compile_info, compile_params)) {
    OP_LOGE(op_type, "get compile info from nlohmann json failed.");
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
  ReflectionPadV3TilingParams run_params;
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

REGISTER_OP_TILING_FUNC_BUFFERED(ReflectionPadV3, ReflectionPadV3Tiling);
}  // namespace optiling

