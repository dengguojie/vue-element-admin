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

#include "graph/debug/ge_log.h"
#include "error_log.h"
#include "vector_tiling.h"

namespace optiling {

static const std::unordered_map<int64_t, int64_t> SPLIT_FACTORS{
    {1, 32767},
    {2, 32767},
    {4, 16383},
    {8, 8191},
};

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
  CHECK((!op_paras.inputs.empty() && !op_paras.inputs[0].tensor.empty()), "op [%s] : input shape cannot be empty",
        op_type.c_str());
  in_type = op_paras.inputs[0].tensor[0].dtype;
  CHECK((!op_paras.outputs.empty() && !op_paras.outputs[0].tensor.empty()), "op [%s] : output shape cannot be empty",
        op_type.c_str());
  out_type = op_paras.outputs[0].tensor[0].dtype;
  int64_t type_size = GetElementByType(out_type);
  for (size_t i = 1; i < op_paras.outputs.size(); i++) {
    int64_t cur_type_size = GetElementByType(op_paras.outputs[i].tensor[0].dtype);
    if (cur_type_size > type_size) {
      out_type = op_paras.outputs[i].tensor[0].dtype;
      type_size = cur_type_size;
    }
  }
  CHECK_GE(flag_info.size(), 1, "op [%s] : flag info error", op_type.c_str());
  only_const_tiling = flag_info[0];
  if (!only_const_tiling) {
    const size_t special_pattern_index = 3;
    CHECK_GT(flag_info.size(), special_pattern_index, "op [%s] : flag info has no _use_special_pattern",
             op_type.c_str());
    use_special_pattern = flag_info[3];
  }
  return true;
}

bool Eletwise::GenerateOutputShape() {
  const std::vector<int64_t>& shapes = op_paras.outputs[0].tensor[0].shape;
  int64_t fused_output = std::accumulate(shapes.begin(), shapes.end(), 1ll, std::multiplies<int64_t>());
  CHECK_LE(fused_output, INT32_MAX, "op [%s] : The output shape is too large", op_type.c_str());
  output_shape = {fused_output};
  return true;
}

bool Eletwise::CalcTiling() {
  CHECK((op_info.find("_base_info") != op_info.end()), "op [%s] : compile info not contain [_base_info]",
        op_type.c_str());
  // "_base_info": ["_ub_size", "_max_dtype", "_coexisting_quantity", "_core_num"]
  std::string pattern_key = "100";
  if (only_const_tiling || !use_special_pattern) {
    pattern_key = "000";
  }
  CHECK((op_info["_base_info"].find(pattern_key) != op_info["_base_info"].end()), "op [%s] : _base_info not contain [%s]",
        op_type.c_str(), pattern_key.c_str());
  const auto& base_info = op_info["_base_info"][pattern_key];
  const size_t base_info_size = 4;
  CHECK_EQ(base_info.size(), base_info_size, "op [%s] : base info must be _ub_size, _max_dtype, _coexisting_quantity"
           " and _core_num", op_type.c_str());
  baseInfo.ub_size = base_info[0];
  baseInfo.max_dtype = base_info[1];
  baseInfo.coexisting_quantity = base_info[2];
  baseInfo.core_num = base_info[3];
  CHECK_GT(baseInfo.coexisting_quantity, 0, "op [%s] : baseInfo coexisting_quantity error, it is [%d]",
           op_type.c_str(), baseInfo.coexisting_quantity);
  CHECK_GT(baseInfo.max_dtype, 0, "op [%s] : baseInfo max_dtype error, it is [%d]",
           op_type.c_str(), baseInfo.max_dtype);
  max_available_ub =
          (((baseInfo.ub_size / baseInfo.coexisting_quantity) / BLOCK_SIZE) * BLOCK_SIZE) / baseInfo.max_dtype;
  CHECK_GT(output_shape.size(), 0, "op [%s] : output_shape index out of range", op_type.c_str());
  const int64_t multi_core_threshold = GetElementByType(out_type) * baseInfo.core_num * DOUBLE_BUFFER_SIZE;
  if (output_shape[0] < multi_core_threshold) {
    need_multi_core = false;
  }
  return true;
}

bool Eletwise::DoBlockTiling() {
  int64_t cur_core = baseInfo.core_num;
  int64_t ele_in_block = (out_type == "uint1") ? ELEWISE_UINT1_REPEATE_NUMS : ELEWISE_REPEATE_NUMS;
  block_axis = 0;
  CHECK((!output_shape.empty()), "op [%s] : output shape cannot be empty", op_type.c_str())
  CHECK_GT(baseInfo.core_num, 0, "op [%s] : baseInfo core_num error, it is [%d]",
           op_type.c_str(), baseInfo.core_num)
  block_factor = std::ceil(output_shape[0] * 1.0 / cur_core);
  block_factor = std::ceil(block_factor * 1.0 / ele_in_block) * ele_in_block;
  block_dims = std::ceil(output_shape[0] * 1.0 / block_factor);
  return true;
}

bool Eletwise::DoUbTiling() {
  ub_axis = 0;
  ub_factor = block_factor;
  int64_t limit = std::min(max_available_ub, SPLIT_FACTORS.at(baseInfo.max_dtype));
  if (limit < ub_factor) {
    int64_t ele_in_block = (out_type == "uint1") ? ELEWISE_UINT1_REPEATE_NUMS : ELEWISE_REPEATE_NUMS;
    CHECK_GT(limit, 0, "op [%s] : ub limit error, it is [%d]", op_type.c_str(), limit)
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
  GELOGD("op [%s] tiling key:%lld", op_type.c_str(), key);
  GELOGD("op [%s] tiling block_dims:%lld", op_type.c_str(), block_dims);
  GELOGD("op [%s] tiling block_factor:%lld", op_type.c_str(), block_factor);
  GELOGD("op [%s] tiling ub_factor:%lld", op_type.c_str(), ub_factor);
  GELOGD("op [%s] tiling block_axis:%lld", op_type.c_str(), block_axis);
  GELOGD("op [%s] tiling ub_axis:%lld", op_type.c_str(), ub_axis);

  run_info.block_dim = static_cast<uint32_t>(block_dims);
  if (only_const_tiling) {
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(need_multi_core));
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(block_axis));
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(block_factor));
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(ub_axis));
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(ub_factor));
    return true;
  }
  ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(key));
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
  CHECK((op_info.find("_elewise_vars") != op_info.end()), "op [%s] : compile info not contain [_elewise_vars]", op_type.c_str());
  CHECK((op_info["_elewise_vars"].find(str_key) != op_info["_elewise_vars"].end()), "op [%s] : _base_info not contain [%s]",
        op_type.c_str(), str_key.c_str());
  const auto& all_vars = op_info["_elewise_vars"][str_key];
  for (const auto& var : all_vars) {
    if (var >= 30000) {
      CHECK((ub_axis >= 0), "op [%s] : Not cut ub", op_type.c_str())
      ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(ub_factor));
    } else if (var >= 20000) {
      CHECK((ub_axis >= 0), "op [%s] : Not cut block", op_type.c_str())
      ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(block_factor));
    } else {
      int64_t var_value = var;
      var_value /= 100;
      size_t dim_index = var_value % 100;
      CHECK_LT(dim_index, output_shape.size(), "op [%s] : more than 16 dims are not supported", op_type.c_str())
      ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(output_shape[dim_index]));
    }
  }
  return true;
}

bool Eletwise::DoTiling() {
  GELOGI("op [%s]: tiling running", op_type.c_str());
  bool ret = Init();
  ret = ret && GenerateOutputShape();
  ret = ret && CalcTiling();
  if (need_multi_core) {
    // cut block
    ret = ret && DoBlockTiling();
    CHECK((SPLIT_FACTORS.find(baseInfo.max_dtype) != SPLIT_FACTORS.end()),
        "op [%s] : baseInfo max_dtype not in SPLIT_FACTORS", op_type.c_str())
    if (block_factor > std::min(max_available_ub, SPLIT_FACTORS.at(baseInfo.max_dtype))) {
      need_double_buffer = true;
      max_available_ub =
              (((baseInfo.ub_size / DOUBLE_BUFFER_SIZE / baseInfo.coexisting_quantity) / BLOCK_SIZE)
                * BLOCK_SIZE) / baseInfo.max_dtype;
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

bool CompletedShapes(std::array<std::array<int64_t, MAX_DIM_LEN>, MAX_INPUT_NUMS>& input_shapes,
                     const size_t input_num, size_t& dim_len, bool& is_pure_elementwise,
                     const std::string& op_type, const TeOpParas& op_paras) {
  CHECK_LE(input_num, MAX_INPUT_NUMS, "op [%s] : more than 70 input are not supported", op_type.c_str())
  for (size_t i = 0; i < input_num; i++) {
    CHECK(!op_paras.inputs[i].tensor.empty(), "op [%s] : input tensor cannot be empty", op_type.c_str());
    input_shapes[i].fill(1ll);
    if (op_paras.inputs[i].tensor[0].shape.size() > dim_len) {
      dim_len = op_paras.inputs[i].tensor[0].shape.size();
    }
  }
  CHECK_LE(dim_len, MAX_DIM_LEN, "op [%s] : more than 16 dims are not supported", op_type.c_str())
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
      if (input_shapes[j][i] != max_output) {
        is_pure_elementwise = false;
      }
      if (input_shapes[j][i] > max_output) {
        max_output = input_shapes[j][i];
      }
      CHECK(!verify_broadcast, "op [%s] : input shapes [%s] cannot broadcast to shape [%s]", op_type.c_str(),
            std::to_string(input_shapes[j][i]).c_str(), std::to_string(max_output).c_str());
    }
  }
  return true;
}

bool MatchConstShape(const std::string& op_type,
                     const nlohmann::json& op_info,
                     const std::vector<int64_t>& const_shapes,
                     const std::vector<int64_t>& const_block_dims,
                     size_t& key_index) {
  CHECK((op_info.find("_const_shapes") != op_info.end()),
      "op [%s] : compile info not contain [_const_shapes]", op_type.c_str());
  const std::vector<std::vector<int64_t>>& compile_const_shapes =
          op_info["_const_shapes"].get<std::vector<std::vector<int64_t>>>();
  for (size_t i = 0; i < compile_const_shapes.size(); i++) {
    bool shape_equal = true;
    CHECK_EQ(const_shapes.size(), compile_const_shapes[i].size(),
        "op [%s] : input shape and const shape not match", op_type.c_str());
    CHECK_EQ(const_block_dims.size(), compile_const_shapes.size(),
        "op [%s] : block dims and const shape not match", op_type.c_str());
    for (size_t j = 0; j < compile_const_shapes[i].size(); j++) {
      if (const_shapes[j] != compile_const_shapes[i][j]) {
        shape_equal = false;
      }
    }
    if (shape_equal) {
      key_index = i;
      break;
    }
  }
  return true;
}

bool CalcConstKey(const std::string& op_type, const TeOpParas& op_paras,
                  const nlohmann::json& op_info, const bool is_support_broadcast,
                  int64_t& key, int64_t& block_dims) {
  GELOGI("op [%s]: tiling running", op_type.c_str());
  size_t key_index = 0;
  CHECK((op_info.find("_const_block_dims") != op_info.end()), "op [%s] : compile info not contain [_const_block_dims]",
        op_type.c_str());
  const std::vector<int64_t>& const_block_dims = op_info["_const_block_dims"].get<std::vector<int64_t>>();
  bool ret = true;
  if (is_support_broadcast) {
    bool verify_input =
            op_paras.inputs.size() == 2 && !op_paras.inputs[0].tensor.empty() && !op_paras.inputs[1].tensor.empty();
    CHECK(verify_input, "op [%s] : input shape cannot be empty", op_type.c_str());
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
    CHECK_EQ(input_shape_x.size(), input_shape_y.size(), "op [%s] : input shape must be same", op_type.c_str());
    std::vector<int64_t> const_shapes(input_shape_x.size(), 0);
    for (size_t i = 0; i < input_shape_x.size(); i++) {
      const_shapes[i] = static_cast<uint64_t>(input_shape_x[i]) & static_cast<uint64_t>(input_shape_y[i]);
    }
    ret = MatchConstShape(op_type, op_info, const_shapes, const_block_dims, key_index);
  } else {
    CHECK(!const_block_dims.empty(), "op [%s] : block dims and const shape not match", op_type.c_str());
  }
  if (ret) {
    CHECK_GT(const_block_dims.size(), key_index, "op [%s] : const_block_dims index out of range", op_type.c_str());
    block_dims = const_block_dims[key_index];
    key = 100000000 + key_index;
  }
  return ret;
}

void WriteConstTiling(const std::string& op_type, OpRunInfo& run_info, const int64_t& key, const int64_t& block_dims) {
  GELOGD("op [%s] tiling key:%lld", op_type.c_str(), key);
  GELOGD("op [%s] tiling block_dims:%lld", op_type.c_str(), block_dims);
  run_info.block_dim = static_cast<uint32_t>(block_dims);
  ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(key));
}

bool EletwiseTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                    OpRunInfo& run_info) {
  std::array<std::array<int64_t, MAX_DIM_LEN>, MAX_INPUT_NUMS> input_shapes{};
  CHECK((op_info.find("_flag_info") != op_info.end()), "op [%s] : compile info not contain [_flag_info]",
        op_type.c_str());
  const std::vector<bool>& flag_info = op_info["_flag_info"];
  bool is_const = false;
  bool is_support_broadcast = false;
  if (flag_info.size() > 2) {
    is_const = flag_info[1];
    is_support_broadcast = flag_info[2];
  }
  size_t input_num = op_paras.inputs.size();
  size_t dim_len = 0;
  bool is_pure_elementwise = true;
  bool ret = CompletedShapes(input_shapes, input_num, dim_len, is_pure_elementwise, op_type, op_paras);
  if (is_const) {
    int64_t key{0};
    int64_t block_dims{1};
    ret = ret && CalcConstKey(op_type, op_paras, op_info, is_support_broadcast, key, block_dims);
    if (ret) {
      WriteConstTiling(op_type, run_info, key, block_dims);
    }
  } else if (is_pure_elementwise) {
    Eletwise eletwise(op_type, op_paras, op_info, flag_info);
    ret = eletwise.DoTiling();
    ret = ret && eletwise.WriteTilingData(run_info);
  } else {
    Broadcast broadcast(op_type, op_paras, op_info, flag_info, input_num, dim_len, input_shapes);
    ret = broadcast.DoTiling();
    ret = ret && broadcast.WriteTilingData(run_info);
  }
  return ret;
}

}  // namespace optiling
