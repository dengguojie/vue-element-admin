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
 * \file pad_v3_grad.cc
 * \brief
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "../op_proto/util/util.h"
#include "../op_proto/util/error_util.h"
#include "op_log.h"

namespace optiling {

static const int64_t TILING_MODE_0 = 0;
static const int64_t TILING_MODE_1 = 1;
static const int64_t PADDINGS_DIM_INDEX_2 = 4;
static const int64_t PADDINGS_DIM_INDEX_3 = 6;

struct PadV3GradCompileParams {
  int64_t core_num;
  std::string op_type;
};

struct PadV3GradTilingParams {
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

static void InitRunningParams(PadV3GradTilingParams& params) {
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

static bool GetPadV3GradCompileParams(const nlohmann::json& compile_info,
                                PadV3GradCompileParams& compile_params) {
  using namespace nlohmann;
  OP_LOGD("begin to GetPadV3GradCompileParams.");
  auto allVars = compile_info["vars"];
  if (allVars.count("core_num") == 0) {
    OP_LOGE(compile_params.op_type, "GetCompileParams, get core_num error");
    return false;
  }
  compile_params.core_num = allVars["core_num"].get<std::int64_t>();
  return true;
}

static void SetRuningParams(const PadV3GradTilingParams& params, OpRunInfo& run_info) {
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

static void PrintTilingParams(const PadV3GradTilingParams& params, const std::string& op_type) {
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

static void _printTensorValue(const PadV3GradCompileParams& compile_params,
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

static int64_t get_core_num(std::vector<int64_t> x_shape, int64_t core_num, int64_t wh_out_total) {
  auto nc_total = x_shape[0] * x_shape[1];
  auto block = 16;
  auto threshold = block * block;
  auto ele_per_core = (nc_total - 1) / core_num + 1;
  auto core_used = (nc_total - 1) / ele_per_core + 1;
  if (wh_out_total < threshold) {
    while ((wh_out_total * ele_per_core < block) && (core_used > 1)) {
      core_used -= 1;
      ele_per_core = (nc_total - 1) / core_used + 1;
    }
  }
  core_used = (nc_total - 1) / ele_per_core + 1; 
  return core_used;
}

static bool GetTilingParam(const std::vector<int64_t>& input_shape,
                           const std::vector<int64_t>& paddings_const_values,
                           const PadV3GradCompileParams& compile_params,
                           PadV3GradTilingParams& tiling_params) {
  OP_LOGD("begin to GetTilingParam.");
  auto tiling_key = 0;
  auto output_third = input_shape[2] - paddings_const_values[PADDINGS_DIM_INDEX_2] 
                      - paddings_const_values[PADDINGS_DIM_INDEX_2 + 1];
  auto output_fourth = input_shape[3] - paddings_const_values[PADDINGS_DIM_INDEX_3] 
                       - paddings_const_values[PADDINGS_DIM_INDEX_3 + 1];
  int64_t core_used = 1;
  int64_t not_last_core_numel = 0;
  int64_t last_core_numel = 0;
  int64_t split = 160;
  int64_t low_split = 48;
  int64_t high_split = 480;
  auto nc_total = input_shape[0] * input_shape[1];
  auto wh_output_total = output_third * output_fourth;
  core_used = get_core_num(input_shape, compile_params.core_num, wh_output_total);
  not_last_core_numel = (nc_total - 1) / core_used + 1;
  last_core_numel = nc_total - (core_used - 1) * not_last_core_numel;
  if (output_third > split && output_fourth > split) {
    tiling_key = 1;
  }
  else if (output_third > high_split && output_fourth > low_split) {
	tiling_key = 1;
  }
  else if (output_third > low_split && output_fourth > high_split) {
	tiling_key = 1;
  }
  else {
    tiling_key = 0;
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
  tiling_params.padding_index_0 = paddings_const_values[PADDINGS_DIM_INDEX_3];
  tiling_params.padding_index_1 = paddings_const_values[PADDINGS_DIM_INDEX_3 + 1];
  tiling_params.padding_index_2 = paddings_const_values[PADDINGS_DIM_INDEX_2];
  tiling_params.padding_index_3 = paddings_const_values[PADDINGS_DIM_INDEX_2 + 1];
  tiling_params.not_last_core_num = not_last_core_numel;
  tiling_params.last_core_num = last_core_numel;
  return true;
}

bool PadV3GradTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_compile_info,
               OpRunInfo& run_info) {
  using namespace ge;

  auto allVars = op_compile_info["vars"];
  auto padding_contiguous = allVars["padding_contiguous"];
  OP_LOGD("begin to run tiling.");
  if (op_compile_info == nullptr) {
    OP_LOGE(op_type, "op [PadV3GradTiling] : op_compile_info json error.");
    return false;
  }

  if (op_paras.inputs.empty() || op_paras.inputs[0].tensor.empty() ||
      op_paras.inputs[1].tensor.empty()) {
    ge::OpsOneInputShapeErrReport(op_type.c_str(), "input_x or paddings",
                                  "The input may be empty");
    OP_LOGE(op_type, "op [PadV3GradTiling] : input shape error");
    return false;
  }
  // begin to get compile data
  PadV3GradCompileParams compile_params;
  compile_params.op_type = op_type;
  if (!GetPadV3GradCompileParams(op_compile_info, compile_params)) {
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
  PadV3GradTilingParams run_params;
  InitRunningParams(run_params);
  GetTilingParam(input_shape, paddings_const_values, compile_params, run_params);
  SetRuningParams(run_params, run_info);

  PrintTilingParams(run_params, op_type);

  run_info.block_dim = compile_params.core_num;
  std::vector<int64_t> workspace;
  int64_t work_space_size = 9600000;
  workspace.push_back(work_space_size);
  run_info.workspaces = workspace;

  OP_LOGI(op_type, "end to run tiling, succ!");

  return true;
}

REGISTER_OP_TILING_FUNC_BUFFERED(PadV3Grad, PadV3GradTiling);
}  // namespace optiling
  