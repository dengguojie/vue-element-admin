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
 * \file eletwise.cpp
 * \brief
 */
#include "eletwise.h"
#include "broadcast.h"

#include <algorithm>
#include <unordered_map>

#include "vector_tiling.h"
#include "error_log.h"

namespace optiling {

namespace {
  static const std::unordered_map<int64_t, int64_t> SPLIT_FACTORS{
      {1, 32767},
      {2, 32767},
      {4, 16383},
      {8, 8191},
  };
}

const int64_t GetElementByType(const std::string& dtype) {
  // element nums in one block, default, fp16, int16, uin16
  int64_t element_in_block = 16;
  if (dtype == "float32" || dtype == "int32" || dtype == "uint32") {
    // element nums in one block by b32
    element_in_block = 8;
  } else if (dtype == "int8" || dtype == "uint8" || dtype == "bool") {
    // element nums in one block by b8
    element_in_block = 32;
  } else if (dtype == "int64" || dtype == "uint64") {
    // element nums in one block by b64
    element_in_block = 4;
  } else if (dtype == "uint1") {
    // element nums in one block by uint1
    element_in_block = 256;
  }
  return element_in_block;
}

bool Eletwise::Init() {
  V_OP_TILING_CHECK((!op_paras.inputs.empty() && !op_paras.inputs[0].tensor.empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape cannot be empty"),
                    return false);
  in_type = op_paras.inputs[0].tensor[0].dtype;
  V_OP_TILING_CHECK((!op_paras.outputs.empty() && !op_paras.outputs[0].tensor.empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output shape cannot be empty"),
                    return false);
  out_type = op_paras.outputs[0].tensor[0].dtype;
  int64_t type_size = GetElementByType(out_type);
  for (size_t i = 1; i < op_paras.outputs.size(); i++) {
    V_OP_TILING_CHECK(!op_paras.outputs[i].tensor.empty(),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output shape cannot be empty"),
                      return false);
    int64_t cur_type_size = GetElementByType(op_paras.outputs[i].tensor[0].dtype);
    if (cur_type_size > type_size) {
      out_type = op_paras.outputs[i].tensor[0].dtype;
      type_size = cur_type_size;
    }
  }
  V_CHECK_GE(flag_info.size(), 1,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "flag info error"),
             return false);
  only_const_tiling = flag_info[0];
  if (!only_const_tiling) {
    const size_t special_pattern_index = 3;
    V_CHECK_GT(flag_info.size(), special_pattern_index,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type,"flag info has no _use_special_pattern"),
               return false);
    use_special_pattern = flag_info[3];
  }
  return true;
}

bool Eletwise::GenerateOutputShape() {
  const std::vector<int64_t>& shapes = op_paras.outputs[0].tensor[0].shape;
  int64_t fused_output = std::accumulate(shapes.begin(), shapes.end(), 1LL, std::multiplies<int64_t>());
  V_CHECK_LE(fused_output, INT32_MAX,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The output shape is too large"),
             return false);
  V_CHECK_GT(fused_output, 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The output shape must be greater than 0"),
             return false);
  output_shape = {fused_output};
  return true;
}

bool Eletwise::CalcTiling() {
  // "_base_info": ["_core_num", "_max_dtype", "_max_available_ub", "_max_available_ub_db"]
  std::string pattern_key = "100";
  if (only_const_tiling || !use_special_pattern) {
    pattern_key = "000";
  }
  try {
    const auto& base_info = op_info.at("_base_info").at(pattern_key);
    const size_t base_info_size = 4;
    V_CHECK_EQ(base_info.size(), base_info_size,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "base info must be _ub_size, _max_dtype, _coexisting_quantity"),
               return false);
    core_num = base_info[0];
    max_dtype = base_info[1];
    max_available_ub = base_info[2];
    max_available_ub_db = base_info[3];
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_base_info] error. Error message: %s", e.what());
    return false;
  }
  V_CHECK_GT(output_shape.size(), 0,
           VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output_shape index out of range"),
           return false);
  const int64_t multi_core_threshold = GetElementByType(out_type) * core_num * DOUBLE_BUFFER_SIZE;
  if (output_shape[0] < multi_core_threshold) {
    need_multi_core = false;
  }
  return true;
}

bool Eletwise::DoBlockTiling() {
  int64_t cur_core = core_num;
  bool outs_uint1 = op_info.at("_outs_uint1");
  int64_t ele_in_block = outs_uint1 ? ELEWISE_UINT1_REPEATE_NUMS : ELEWISE_REPEATE_NUMS;
  block_axis = 0;
  V_OP_TILING_CHECK((!output_shape.empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output shape cannot be empty"),
                    return false);
  V_CHECK_GT(core_num, 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "baseInfo core_num error, it is [%ld]", core_num),
             return false);
  block_factor = std::ceil(output_shape[0] * 1.0 / cur_core);
  block_factor = std::ceil(block_factor * 1.0 / ele_in_block) * ele_in_block;
  block_dims = std::ceil(output_shape[0] * 1.0 / block_factor);
  return true;
}


bool Eletwise::DoUbTiling() {
  ub_axis = 0;
  ub_factor = block_factor;
  int64_t limit = std::min(max_available_ub, SPLIT_FACTORS.at(max_dtype));
  if (limit < ub_factor) {
    bool outs_uint1 = op_info.at("_outs_uint1");
    int64_t ele_in_block = outs_uint1 ? ELEWISE_UINT1_REPEATE_NUMS : ELEWISE_REPEATE_NUMS;
    V_CHECK_GT(limit, 0,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub limit error, it is [%ld]",limit),
               return false);
    int64_t ub_for_num = std::ceil(ub_factor * 1.0 / limit);
    int64_t adjust_factor = std::ceil(ub_factor * 1.0 / ub_for_num);
    int64_t align_factor = std::ceil(adjust_factor * 1.0 / ele_in_block);
    ub_factor = align_factor * ele_in_block;
    if (ub_factor > limit) {
      ub_factor = std::floor(adjust_factor * 1.0 / ele_in_block) * ele_in_block;
    }
  }
  return true;
}

void Eletwise::CalcKey() {
  // special pattern: COMMON
  key = 210000000;
  if (!use_special_pattern) {
    key = 0;
  }
  int64_t doubleBufferKey = 10000;
  if (need_double_buffer) {
    key += doubleBufferKey;
  }
}

bool Eletwise::WriteTilingData(OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "tiling key:%lld", key);
  OP_LOGD(op_type.c_str(), "tiling block_dims:%lld", block_dims);
  OP_LOGD(op_type.c_str(), "tiling block_factor:%lld", block_factor);
  OP_LOGD(op_type.c_str(), "tiling ub_factor:%lld", ub_factor);
  OP_LOGD(op_type.c_str(), "tiling block_axis:%lld", block_axis);
  OP_LOGD(op_type.c_str(), "tiling ub_axis:%lld", ub_axis);

  run_info.block_dim = static_cast<uint32_t>(block_dims);
  if (only_const_tiling) {
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(need_multi_core));
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(block_axis));
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(block_factor));
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(ub_axis));
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(ub_factor));
    if (need_double_buffer) {
      ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(1));
    } else {
      ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(0));
    }
    return true;
  }

  run_info.tiling_key = static_cast<int32_t>(key);
  std::string str_key = "210000000";
  if (!use_special_pattern) {
    str_key = "0";
  }
  if (need_double_buffer) {
    str_key = "210010000";
    if (!use_special_pattern) {
      str_key = "10000";
    }
  }
  try {
    const auto& all_vars = op_info.at("_elewise_vars").at(str_key);
    for (const auto& var : all_vars) {
      if (var >= 30000) {
        V_CHECK_GE(ub_axis, 0,
                   VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Not cut ub"),
                   return false);
        ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(ub_factor));
      } else if (var >= 20000) {
        V_CHECK_GE(block_axis, 0,
                   VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Not cut block"),
                   return false);
        ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(block_factor));
      } else {
        int64_t var_value = var;
        var_value /= 100;
        size_t dim_index = var_value % 100;
        V_CHECK_LT(dim_index, output_shape.size(),
                   VECTOR_INNER_ERR_REPORT_TILIING(op_type, "dim_index out of range output_shape index"),
                   return false);
        ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(output_shape[dim_index]));
      }
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_elewise_vars] error. Error message: %s", e.what());
    return false;
  }
  if (op_info.contains("_attr_vars")) {
    try {
      const auto& all_vars = op_info.at("_attr_vars").at(str_key);
      for (const auto& var : all_vars) {
        size_t attr_size = 0;
        const uint8_t *attr = op_paras.var_attrs.GetData(var.at("name"), var.at("type"), attr_size);
        ByteBufferPut(run_info.tiling_data, attr, attr_size);
      }
    } catch (const std::exception &e) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_attr_vars] error. Error message: %s", e.what());
      return false;
    }
  }
  return true;
}

bool Eletwise::DoTiling() {
  OP_LOGI(op_type.c_str(), "eletwise tiling running");
  bool ret = Init();
  ret = ret && GenerateOutputShape();
  ret = ret && CalcTiling();
  if (need_multi_core) {
    // cut block
    ret = ret && DoBlockTiling();
    V_OP_TILING_CHECK((SPLIT_FACTORS.find(max_dtype) != SPLIT_FACTORS.end()),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "baseInfo max_dtype not in SPLIT_FACTORS"),
                      return false);
    if (ret && block_factor > std::min(max_available_ub, SPLIT_FACTORS.at(max_dtype))) {
      need_double_buffer = true;
      max_available_ub = max_available_ub_db;
    }
    ret = ret && DoUbTiling();
  } else {
    ub_axis = 0;
    ub_factor = output_shape[0];
    block_axis = 0;
    block_factor = output_shape[0];
  }
  if (ret && !only_const_tiling) {
    CalcKey();
  }
  return ret;
}

bool CompletedShapes(std::array<std::array<int64_t, B_MAX_DIM_LEN>, B_MAX_INPUT_NUMS>& input_shapes,
                     size_t& dim_len, bool& is_pure_elementwise,
                     const std::string& op_type, const TeOpParas& op_paras) {
  size_t input_num = op_paras.inputs.size();
  V_CHECK_LE(input_num, B_MAX_INPUT_NUMS,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "more than 70 input are not supported"),
             return false);
  for (size_t i = 0; i < input_num; i++) {
    V_OP_TILING_CHECK(!op_paras.inputs[i].tensor.empty(),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor cannot be empty"),
                      return false);
    input_shapes[i].fill(1LL);
    if (op_paras.inputs[i].tensor[0].shape.size() > dim_len) {
      dim_len = op_paras.inputs[i].tensor[0].shape.size();
    }
  }
  V_CHECK_LE(dim_len, B_MAX_DIM_LEN,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "more than 16 dims are not supported"),
             return false);
  for (size_t i = 0; i < input_num; i++) {
    size_t cur_dim_len = op_paras.inputs[i].tensor[0].shape.size();
    size_t start_index = dim_len - cur_dim_len;
    for (size_t j = 0; j < cur_dim_len; j++) {
      input_shapes[i][start_index++] = op_paras.inputs[i].tensor[0].shape[j];
    }
  }
  for (size_t i = 0; i < dim_len; i++) {
    int64_t max_output = input_shapes[0][i];
    for (size_t j = 1; j < input_num; j++) {
      bool verify_broadcast = input_shapes[j][i] != 1 &&
          (input_shapes[j][i] != max_output && max_output != 1);
      V_OP_TILING_CHECK((!verify_broadcast),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shapes [%s] cannot broadcast to shape [%s]",
                                std::to_string(input_shapes[j][i]).c_str(), std::to_string(max_output).c_str()),
                        return false);
      if (input_shapes[j][i] != max_output) {
        is_pure_elementwise = false;
      }
      if (input_shapes[j][i] > max_output) {
        max_output = input_shapes[j][i];
      }
    }
  }
  return true;
}

bool MatchConstShape(const std::string& op_type,
                     const nlohmann::json& op_info,
                     const std::vector<int64_t>& const_shapes,
                     size_t& key_index) {
  try {
    const auto& compile_const_shapes = op_info.at("_const_shapes");
    for (size_t i = 0; i < compile_const_shapes.size(); i++) {
      bool shape_equal = true;
      V_CHECK_EQ(const_shapes.size(), compile_const_shapes[i].size(),
                 VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape and const shape not match"),
                 return false);
      for (size_t j = 0; j < compile_const_shapes[i].size(); j++) {
        if (const_shapes[j] != compile_const_shapes[i][j].get<int64_t>()) {
          shape_equal = false;
        }
      }
      if (shape_equal) {
        key_index = i;
        break;
      }
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_const_shapes] error. Error message: %s", e.what());
    return false;
  }
  return true;
}

bool CalcConstKey(const std::string& op_type, const TeOpParas& op_paras,
                  const nlohmann::json& op_info, const bool is_support_broadcast,
                  int64_t& key, int64_t& block_dims) {
  OP_LOGI(op_type.c_str(), "tiling running");
  size_t key_index = 0;
  bool ret = true;
  if (is_support_broadcast) {
    bool verify_input =
            op_paras.inputs.size() == 2 && !op_paras.inputs[0].tensor.empty() && !op_paras.inputs[1].tensor.empty();
    V_OP_TILING_CHECK(verify_input,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape cannot be empty"),
                      return false);
    std::vector<int64_t> input_shape_x = op_paras.inputs[0].tensor[0].shape;
    std::vector<int64_t> input_shape_y = op_paras.inputs[1].tensor[0].shape;
    size_t shape_len_x = input_shape_x.size();
    size_t shape_len_y = input_shape_y.size();
    size_t max_len = shape_len_x > shape_len_y ? shape_len_x : shape_len_y;
    if (shape_len_x < max_len) {
      input_shape_x.insert(input_shape_x.begin(), max_len - shape_len_x, 1);
    } else if (shape_len_y < max_len) {
      input_shape_y.insert(input_shape_y.begin(), max_len - shape_len_y, 1);
    }
    V_CHECK_EQ(input_shape_x.size(), input_shape_y.size(),
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape must be same"),
               return false);
    std::vector<int64_t> const_shapes(input_shape_x.size(), 0);
    for (size_t i = 0; i < input_shape_x.size(); i++) {
      const_shapes[i] = static_cast<uint64_t>(input_shape_x[i]) & static_cast<uint64_t>(input_shape_y[i]);
    }
    ret = MatchConstShape(op_type, op_info, const_shapes, key_index);
  }
  if (ret) {
    try {
      const auto& const_block_dims = op_info["_const_block_dims"];
      V_CHECK_GT(const_block_dims.size(), key_index,
                 VECTOR_INNER_ERR_REPORT_TILIING(op_type, "const_block_dims index out of range"),
                 return false);
      block_dims = const_block_dims[key_index].get<int64_t>();
      key = 100000000 + key_index;
    } catch (const std::exception &e) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_const_block_dims] error. Error message: %s", e.what());
      return false;
    }
  }
  return ret;
}

bool IsEmptyTensor(const std::string& op_type, const TeOpParas& op_paras) {
  bool has_zero = false;
  for (size_t i = 0; i < op_paras.outputs.size(); i++) {
    int64_t output_size = std::accumulate(op_paras.outputs[i].tensor[0].shape.begin(),
        op_paras.outputs[i].tensor[0].shape.end(), 1LL, std::multiplies<int64_t>());
    if (output_size == 0) {
      has_zero = true;
    } else {
      V_OP_TILING_CHECK(!has_zero,
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "multi-output only supports all 0 output"),
                        return false);
    }
  }
  return has_zero;
}

bool WriteConstTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                      OpRunInfo& run_info, const int64_t& key, const int64_t& block_dims) {
  OP_LOGD(op_type.c_str(), "tiling key:%lld", key);
  OP_LOGD(op_type.c_str(), "tiling block_dims:%lld", block_dims);
  run_info.block_dim = static_cast<uint32_t>(block_dims);
  run_info.tiling_key = static_cast<uint32_t>(key);
  if (op_info.contains("_attr_vars") && op_info.at("_attr_vars").contains(std::to_string(key))) {
    try {
      const auto& all_vars = op_info.at("_attr_vars").at(std::to_string(key));
      for (const auto& var : all_vars) {
        size_t attr_size = 0;
        const uint8_t* attr = op_paras.var_attrs.GetData(var.at("name"), var.at("type"), attr_size);
        ByteBufferPut(run_info.tiling_data, attr, attr_size);
      }
    } catch (const std::exception& e) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_attr_vars] error. Error message: %s", e.what());
      return false;
    }
  }
  return true;
}

bool EletwiseTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                    OpRunInfo& run_info) {
  std::array<std::array<int64_t, B_MAX_DIM_LEN>, B_MAX_INPUT_NUMS> input_shapes{};
  std::vector<bool> flag_info;
  try {
    flag_info = op_info.at("_flag_info").get<std::vector<bool>>();
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_flag_info] error. Error message: %s", e.what());
    return false;
  }
  bool is_const = false;
  bool is_support_broadcast = true;
  bool use_special_pattern = true;
  if (flag_info.size() > 2) {
    is_const = flag_info[1];
    is_support_broadcast = flag_info[2];
    use_special_pattern = flag_info[3];
  }
  size_t dim_len = 0;
  bool is_pure_elementwise = true;
  bool ret = CompletedShapes(input_shapes, dim_len, is_pure_elementwise, op_type, op_paras);
  if (is_const) {
    int64_t key{0};
    int64_t block_dims{1};
    ret = ret && CalcConstKey(op_type, op_paras, op_info, is_support_broadcast, key, block_dims);
    ret = ret && WriteConstTiling(op_type, op_paras, op_info, run_info, key, block_dims);
  } else if (IsEmptyTensor(op_type, op_paras)) {
    ret = ret && WriteConstTiling(op_type, op_paras, op_info, run_info, INT32_MIN, 1);
  } else if ((is_pure_elementwise && !(is_support_broadcast && !use_special_pattern)) || !is_support_broadcast) {
    Eletwise eletwise(op_type, op_paras, op_info, flag_info);
    ret = ret && eletwise.DoTiling();
    ret = ret && eletwise.WriteTilingData(run_info);
  } else {
    Broadcast broadcast(op_type, op_paras, op_info, flag_info, dim_len, input_shapes);
    ret = ret && broadcast.DoTiling();
    ret = ret && broadcast.WriteTilingData(run_info);
  }
  return ret;
}

}  // namespace optiling
