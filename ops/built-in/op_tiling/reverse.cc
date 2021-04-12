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
 * \file reverse.cc
 * \brief
 */
#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "op_log.h"
#include "../op_proto/util/error_util.h"

namespace optiling {
// elements num in one block for int16
const int64_t block_num = 16;
// vnhwc process the min numbers
const int64_t vnhwc_block_num = 256;

struct ResizeV2TilingParams {
  int64_t tiling_key;
  int64_t inner_shape_0;
  int64_t inner_shape_1;
  int64_t inner_shape_2;
  int64_t inner_shape_3;
  int64_t inner_shape_4;
  int64_t inner_shape_5;
  int64_t inner_shape_6;
  int64_t inner_axis_0;
  int64_t inner_axis_1;
  int64_t inner_axis_2;
  int64_t inner_axis_3;
  int64_t inner_axis_4;
  int64_t inner_axis_5;
  int64_t inner_axis_6;
  int64_t outer_shape_0;
  int64_t outer_shape_1;
  int64_t outer_shape_2;
  int64_t outer_shape_3;
  int64_t outer_shape_4;
  int64_t outer_shape_5;
  int64_t outer_shape_6;
  int64_t outer_axis_0;
  int64_t outer_axis_1;
  int64_t outer_axis_2;
  int64_t outer_axis_3;
  int64_t outer_axis_4;
  int64_t outer_axis_5;
  int64_t outer_axis_6;
  int64_t is_split_axi_reverse;
  int64_t split_part_num;
  int64_t split_dim;
};

struct ReverseV2CompileParams {
  int64_t core_num;
  int64_t max_elements;
  int64_t max_elements_last_large_size;
  int64_t dtype_rate;
  std::string op_type;
};

static bool GetReverseV2CompileParams(const nlohmann::json& compile_info, ReverseV2CompileParams& compile_params) {
  using namespace nlohmann;
  auto allVars = compile_info["vars"];
  if (allVars.count("core_num") == 0) {
    OP_LOGE(compile_params.op_type, "GetCompileParams, get core_num error");
    return false;
  }
  compile_params.core_num = allVars["core_num"].get<std::int64_t>();
  // get max_elements
  if (allVars.count("max_elements") == 0) {
    OP_LOGE(compile_params.op_type, "GetCompileParams, get max_elements error");
    return false;
  }
  compile_params.max_elements = allVars["max_elements"].get<std::int64_t>();

  // get max_elements_last_large_size
  if (allVars.count("max_elements_last_large_size") == 0) {
    OP_LOGE(compile_params.op_type, "GetCompileParams, get max_elements_last_large_size error");
    return false;
  }
  compile_params.max_elements_last_large_size = allVars["max_elements_last_large_size"].get<std::int64_t>();

  // get dtype_rate
  if (allVars.count("dtype_rate") == 0) {
    OP_LOGE(compile_params.op_type, "GetCompileParams, get dtype_rate error");
    return false;
  }
  compile_params.dtype_rate = allVars["dtype_rate"].get<std::int64_t>();
  return true;
}

static bool GetAxesConstValue(const TeOpParas& paras, const string& name, const string& dtype,
                              vector<int64_t>& values) {
  values.clear();
  if (paras.const_inputs.count(name) == 0 || std::get<0>(paras.const_inputs.at(name)) == nullptr) {
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

static void PrintVectorValues(const std::string& op_type, const std::string& print_key,
                              const std::vector<int64_t>& print_vec) {
  // print tiling_params
  for (size_t i = 0; i < print_vec.size(); ++i) {
    OP_LOGD(op_type, "the index %d of %s = %d.", i, print_key.c_str(), print_vec[i]);
  }
}

void SetRuningParams(const ResizeV2TilingParams& tiling_params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_shape_0);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_shape_1);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_shape_2);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_shape_3);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_shape_4);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_shape_5);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_shape_6);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_axis_0);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_axis_1);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_axis_2);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_axis_3);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_axis_4);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_axis_5);
  ByteBufferPut(run_info.tiling_data, tiling_params.inner_axis_6);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_shape_0);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_shape_1);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_shape_2);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_shape_3);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_shape_4);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_shape_5);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_shape_6);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_axis_0);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_axis_1);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_axis_2);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_axis_3);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_axis_4);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_axis_5);
  ByteBufferPut(run_info.tiling_data, tiling_params.outer_axis_6);
  ByteBufferPut(run_info.tiling_data, tiling_params.is_split_axi_reverse);
  ByteBufferPut(run_info.tiling_data, tiling_params.split_part_num);
  ByteBufferPut(run_info.tiling_data, tiling_params.split_dim);
  ByteBufferPut(run_info.tiling_data, tiling_params.tiling_key);
}

void PrintTilingParams(const ResizeV2TilingParams& tiling_params, const std::string& op_type) {
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_shape_0 = %d.", tiling_params.inner_shape_0);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_shape_1 = %d.", tiling_params.inner_shape_1);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_shape_2 = %d.", tiling_params.inner_shape_2);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_shape_3 = %d.", tiling_params.inner_shape_3);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_shape_4 = %d.", tiling_params.inner_shape_4);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_shape_5 = %d.", tiling_params.inner_shape_5);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_shape_6 = %d.", tiling_params.inner_shape_6);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_axis_0 = %d.", tiling_params.inner_axis_0);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_axis_1 = %d.", tiling_params.inner_axis_1);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_axis_2 = %d.", tiling_params.inner_axis_2);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_axis_3 = %d.", tiling_params.inner_axis_3);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_axis_4 = %d.", tiling_params.inner_axis_4);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_axis_5 = %d.", tiling_params.inner_axis_5);
  OP_LOGD(op_type, "tiling_data, tiling_params.inner_axis_6 = %d.", tiling_params.inner_axis_6);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_shape_0 = %d.", tiling_params.outer_shape_0);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_shape_1 = %d.", tiling_params.outer_shape_1);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_shape_2 = %d.", tiling_params.outer_shape_2);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_shape_3 = %d.", tiling_params.outer_shape_3);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_shape_4 = %d.", tiling_params.outer_shape_4);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_shape_5 = %d.", tiling_params.outer_shape_5);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_shape_6 = %d.", tiling_params.outer_shape_6);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_axis_0 = %d.", tiling_params.outer_axis_0);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_axis_1 = %d.", tiling_params.outer_axis_1);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_axis_2 = %d.", tiling_params.outer_axis_2);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_axis_3 = %d.", tiling_params.outer_axis_3);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_axis_4 = %d.", tiling_params.outer_axis_4);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_axis_5 = %d.", tiling_params.outer_axis_5);
  OP_LOGD(op_type, "tiling_data, tiling_params.outer_axis_6 = %d.", tiling_params.outer_axis_6);
  OP_LOGD(op_type, "tiling_data, tiling_params.is_split_axi_reverse = %d.", tiling_params.is_split_axi_reverse);
  OP_LOGD(op_type, "tiling_data, tiling_params.split_part_num = %d.", tiling_params.split_part_num);
  OP_LOGD(op_type, "tiling_data, tiling_params.split_dim = %d.", tiling_params.split_dim);
  OP_LOGD(op_type, "tiling_data, tiling_key = %d.", tiling_params.tiling_key);
}

bool ReverseV2Tiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                     OpRunInfo& run_info) {
  using namespace ge;
  // get compile data begin
  ReverseV2CompileParams compile_params;
  // init compile data
  compile_params.core_num = 0;
  compile_params.op_type = op_type;
  // get compile data
  if (!GetReverseV2CompileParams(op_info, compile_params)) {
    OP_LOGE(op_type, "get compile info from nlohmann json failed.");
    return false;
  }

  OP_LOGI(op_type, "tiling run begin.");

  if (op_paras.inputs.size() != 2) {
    OP_LOGE(op_type, "the num of inputs must be 2. but is %d", op_paras.inputs.size());
    return false;
  }
  if (op_paras.outputs.empty()) {
    OP_LOGE(op_type, "the num of outputs is 0. return false");
    return false;
  }
  const std::vector<int64_t>& input_shape_const = op_paras.inputs[0].tensor[0].shape;
  std::vector<int64_t> input_shape = input_shape_const;
  std::vector<int64_t> axis_vec;
  if (!GetAxesConstValue(op_paras, "axis", op_paras.inputs[1].tensor[0].dtype, axis_vec)) {
    OP_LOGE(compile_params.op_type, "Get axis values failed");
    return false;
  }

  for (size_t i = 0; i < axis_vec.size(); ++i) {
    if (axis_vec[i] < 0) {
      axis_vec[i] = axis_vec[i] + static_cast<int64_t>(input_shape.size());
    }
  }
  // append the dim for other dtype
  input_shape.push_back(compile_params.dtype_rate);

  // print input
  PrintVectorValues(op_type, "input_shape", input_shape);
  PrintVectorValues(op_type, "axis_value", axis_vec);
  std::vector<int64_t> status_vec;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    std::vector<int64_t>::iterator it;
    it = find(axis_vec.begin(), axis_vec.end(), i);

    if (it != axis_vec.end()) {
      status_vec.push_back(1);
    } else {
      status_vec.push_back(0);
    }
  }

  std::vector<int64_t> modified_input;
  std::vector<int64_t> modified_axis;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (input_shape[i] == 1) {
      continue;
    } else {
      modified_input.push_back(input_shape[i]);
      modified_axis.push_back(status_vec[i]);
    }
  }

  std::vector<int64_t> merged_shape;
  std::vector<int64_t> merged_axis;

  int64_t axis_num = 0;
  int64_t previous_status = modified_axis[0];
  for (size_t i = 0; i < modified_axis.size(); ++i) {
    if (axis_num == 0) {
      axis_num = modified_input[i];
      continue;
    }
    if (previous_status == modified_axis[i]) {
      axis_num = modified_input[i] * axis_num;
      if (i == modified_axis.size() - 1) {
        merged_shape.push_back(axis_num);
        merged_axis.push_back(modified_axis[i]);
      }
      continue;
    } else {
      merged_shape.push_back(axis_num);
      axis_num = modified_input[i];
      merged_axis.push_back(previous_status);
      previous_status = modified_axis[i];
      if (i == modified_axis.size() - 1) {
        merged_shape.push_back(axis_num);
        merged_axis.push_back(modified_axis[i]);
      }
    }
  }
  // split dim base on aicore num
  if (merged_shape.size() < 7 && merged_shape.size() > 0) {
    int64_t split_dim = 1;
    for (int64_t i = 0; i < compile_params.core_num; i++) {
      int64_t cu_split_core_dim = compile_params.core_num - i;
      if (merged_shape[0] % cu_split_core_dim == 0) {
        split_dim = cu_split_core_dim;
        break;
      }
    }
    if (split_dim != 1 && merged_shape[0] / split_dim != 1) {
      merged_shape.insert(merged_shape.begin(), split_dim);
      merged_shape[1] = merged_shape[1] / split_dim;
      auto axis_status = merged_axis[0];
      merged_axis.insert(merged_axis.begin(), axis_status);
    }
  }

  int64_t max_len = compile_params.max_elements;
  int64_t max_len_for_last_large_size = compile_params.max_elements_last_large_size;
  ResizeV2TilingParams tiling_params;
  PrintVectorValues(op_type, "merged_shape", merged_shape);
  PrintVectorValues(op_type, "merged_axis", merged_axis);
  // charge tiling_key base on last dim size
  if (merged_axis[merged_axis.size() - 1] == 1) {
    if (merged_shape[merged_shape.size() - 1] > max_len_for_last_large_size) {
      tiling_params.tiling_key = 6;
      max_len = max_len_for_last_large_size;
    } else if (merged_shape[merged_shape.size() - 1] <= max_len_for_last_large_size &&
               merged_shape[merged_shape.size() - 1] > 128) {
      tiling_params.tiling_key = 5;
    } else {
      tiling_params.tiling_key = 4;
    }
  } else {
    if (merged_shape[merged_shape.size() - 1] > max_len_for_last_large_size) {
      tiling_params.tiling_key = 3;
      max_len = max_len_for_last_large_size;
    } else if (merged_shape[merged_shape.size() - 1] <= max_len_for_last_large_size &&
               merged_shape[merged_shape.size() - 1] > block_num &&
               merged_shape[merged_shape.size() - 1] % block_num != 0) {
      tiling_params.tiling_key = 2;
    } else if (merged_shape[merged_shape.size() - 1] <= max_len_for_last_large_size &&
               merged_shape[merged_shape.size() - 1] > block_num &&
               merged_shape[merged_shape.size() - 1] % block_num == 0) {
      tiling_params.tiling_key = 1;
    } else {
      tiling_params.tiling_key = 0;
    }
  }

  int64_t count = 0;
  int64_t inner_real_count = 0;
  int64_t sub_tensor_num = 0;
  int64_t inner_first_dim = 0;
  for (int64_t i = static_cast<int64_t>(merged_shape.size()) - 1; i >= 0; --i) {
    if (count == 0) {
      inner_real_count = merged_shape[i];
      count = (merged_shape[i] + block_num - 1) / block_num * block_num;
      if (merged_axis[merged_axis.size() - 1] == 1) {
        if (merged_shape[merged_shape.size() - 1] > 128) {
          count = (merged_shape[i] + vnhwc_block_num - 1) / vnhwc_block_num * vnhwc_block_num;
        }
      }
      continue;
    }
    if (count > max_len) {
      inner_first_dim = i + 1;
      break;
    }
    if (i == 0 && inner_real_count > 64) {
      inner_first_dim = 1;
      break;
    }
    count = count * merged_shape[i];
    inner_real_count = inner_real_count * merged_shape[i];
  }

  std::vector<int64_t> inner_shape;
  std::vector<int64_t> inner_axis;
  std::vector<int64_t> outer_shape;
  std::vector<int64_t> outer_axis;
  // max process dim numbers is 7
  int64_t inner_fill_num = 7 - (static_cast<int64_t>(merged_shape.size()) - inner_first_dim);
  int64_t outer_fill_num = 7 - inner_first_dim;

  for (int64_t i = 0; i < 7; ++i) {
    if (i < inner_fill_num) {
      inner_shape.push_back(1);
      inner_axis.push_back(0);
    } else {
      inner_shape.push_back(merged_shape[i - inner_fill_num + inner_first_dim]);
      inner_axis.push_back(merged_axis[i - inner_fill_num + inner_first_dim]);
    }
  }

  for (int64_t i = 0; i < 7; ++i) {
    if (i < outer_fill_num) {
      outer_shape.push_back(1);
      outer_axis.push_back(0);
    } else {
      outer_shape.push_back(merged_shape[i - outer_fill_num]);
      outer_axis.push_back(merged_axis[i - outer_fill_num]);
    }
  }
  PrintVectorValues(op_type, "merged_shape", merged_shape);
  PrintVectorValues(op_type, "merged_axis", merged_axis);

  tiling_params.inner_shape_0 = inner_shape[0];
  tiling_params.inner_shape_1 = inner_shape[1];
  tiling_params.inner_shape_2 = inner_shape[2];
  tiling_params.inner_shape_3 = inner_shape[3];
  tiling_params.inner_shape_4 = inner_shape[4];
  tiling_params.inner_shape_5 = inner_shape[5];
  tiling_params.inner_shape_6 = inner_shape[6];
  tiling_params.inner_axis_0 = inner_axis[0];
  tiling_params.inner_axis_1 = inner_axis[1];
  tiling_params.inner_axis_2 = inner_axis[2];
  tiling_params.inner_axis_3 = inner_axis[3];
  tiling_params.inner_axis_4 = inner_axis[4];
  tiling_params.inner_axis_5 = inner_axis[5];
  tiling_params.inner_axis_6 = inner_axis[6];
  tiling_params.outer_shape_0 = outer_shape[0];
  tiling_params.outer_shape_1 = outer_shape[1];
  tiling_params.outer_shape_2 = outer_shape[2];
  tiling_params.outer_shape_3 = outer_shape[3];
  tiling_params.outer_shape_4 = outer_shape[4];
  tiling_params.outer_shape_5 = outer_shape[5];
  tiling_params.outer_shape_6 = outer_shape[6];
  tiling_params.outer_axis_0 = outer_axis[0];
  tiling_params.outer_axis_1 = outer_axis[1];
  tiling_params.outer_axis_2 = outer_axis[2];
  tiling_params.outer_axis_3 = outer_axis[3];
  tiling_params.outer_axis_4 = outer_axis[4];
  tiling_params.outer_axis_5 = outer_axis[5];
  tiling_params.outer_axis_6 = outer_axis[6];
  tiling_params.is_split_axi_reverse = inner_axis[inner_fill_num];
  tiling_params.split_part_num = max_len / (count / inner_shape[inner_fill_num]);
  tiling_params.split_dim = inner_shape[inner_fill_num];
  if (tiling_params.split_part_num > tiling_params.split_dim) {
    tiling_params.split_part_num = tiling_params.split_dim;
  }

  SetRuningParams(tiling_params, run_info);

  PrintTilingParams(tiling_params, op_type);

  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;
  run_info.block_dim = compile_params.core_num;
  OP_LOGI(op_type, "tiling run success.");
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(ReverseV2, ReverseV2Tiling);
}  // namespace optiling
