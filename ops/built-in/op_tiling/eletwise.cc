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

#include <algorithm>
#include <unordered_map>

#include "error_log.h"
#include "vector_tiling.h"

namespace optiling {

static std::unordered_map<int32_t, int32_t> split_factors{
    {1, 32767},
    {2, 32767},
    {4, 16383},
    {8, 8191},

};

static std::unordered_map<int32_t, Pattern> special_pattern{
    {100, Pattern::COMMON},    {120, Pattern::COMMON_BROADCAST}, {121, Pattern::COMMON_BROADCAST_COMMON},
    {200, Pattern::BROADCAST}, {210, Pattern::BROADCAST_COMMON},
};

int32_t GetTypeSize(const std::string& dtype) {
  int32_t type_size = 2;
  if (dtype == "int8" || dtype == "uint8") {
    type_size = 1;
  } else if (dtype == "float32" || dtype == "int32" || dtype == "uint32") {
    type_size = 4;
  } else if (dtype == "int64" || dtype == "uint64") {
    type_size = 8;
  } else if (dtype == "bool") {
    type_size = 1;
  }
  return type_size;
}

bool Eletwise::Init() {
  need_multi_core = true;
  block_axis = -1;
  ub_axis = -1;
  block_dims = 1;
  s_pattern = Pattern::ORIGINAL;
  CHECK((!op_paras.inputs.empty() && !op_paras.inputs[0].tensor.empty()), "op [%s] : input shape cannot be empty",
        op_type.c_str());
  in_type = op_paras.inputs[0].tensor[0].dtype;
  CHECK((!op_paras.outputs.empty() && !op_paras.outputs[0].tensor.empty()), "op [%s] : output shape cannot be empty",
        op_type.c_str());
  out_type = op_paras.outputs[0].tensor[0].dtype;
  CHECK((op_info.find("_only_const_tiling") != op_info.end()), "op [%s] : compile info not contain [_only_const_tiling]",
        op_type.c_str());
  only_const_tiling = op_info["_only_const_tiling"];
  if (only_const_tiling) {
    return true;
  }
  CHECK((op_info.find("_flag_info") != op_info.end()), "op [%s] : compile info not contain [_flag_info]",
        op_type.c_str());
  // "_flag_info": ["_is_support_broadcast", "_use_special_pattern",
  // "_is_support_absorbable_broadcast", "_is_const_shapes", "_fusion"]
  const int32_t flag_info_size = 5;
  const std::vector<int32_t>& flag_info = op_info["_flag_info"];
  CHECK_EQ(flag_info.size(), flag_info_size, "op [%s] : base info must be _is_support_broadcast, _use_special_pattern, "
                                "_is_support_absorbable_broadcast, _is_const_shapes and _fusion", op_type.c_str());
  compileInfo.is_support_broadcast = flag_info[0];
  compileInfo.use_special_pattern = flag_info[1];
  compileInfo.is_support_absorbable_broadcast = flag_info[2];
  is_const = flag_info[3];
  compileInfo.fusion_flag = flag_info[4];
  return true;
}

void FusionContinuousAxis(const std::array<int64_t, MAX_DIM_LEN>& input_shape_x,
                          const std::array<int64_t, MAX_DIM_LEN>& input_shape_y,
                          std::vector<int64_t>& fused_shape_x,
                          std::vector<int64_t>& fused_shape_y,
                          size_t dim_len) {
  bool state = (input_shape_x[0] == input_shape_y[0]);
  size_t last = 0;
  for (size_t i = 1; i < dim_len; i++) {
    if (input_shape_x[i] == 1 && input_shape_y[i] == 1) {
      continue;
    }
    if (state && (input_shape_x[i] == input_shape_y[i])) {
      fused_shape_x[fused_shape_x.size() - 1] *= input_shape_x[i];
      fused_shape_y[fused_shape_y.size() - 1] *= input_shape_y[i];
    } else if (((input_shape_x[i] == input_shape_x[last]) && input_shape_x[i] == 1) ||
               ((input_shape_y[i] == input_shape_y[last]) && input_shape_y[i] == 1)) {
      fused_shape_x[fused_shape_x.size() - 1] *= input_shape_x[i];
      fused_shape_y[fused_shape_y.size() - 1] *= input_shape_y[i];
      state = (input_shape_x[i] == input_shape_y[i]);
    } else {
      fused_shape_x.push_back(input_shape_x[i]);
      fused_shape_y.push_back(input_shape_y[i]);
      state = (input_shape_x[i] == input_shape_y[i]);
      if (fused_shape_x.size() > MAX_PATTERN_DIM) {
        break;
      }
    }
    last = i;
  }
}

bool Eletwise::TrySwitchToPerfPattern() {
  if (compileInfo.fusion_flag == 0 || !compileInfo.use_special_pattern) {
    return true;
  }
  std::vector<int64_t> fused_shape_x{input_shapes[0][0]};
  std::vector<int64_t> fused_shape_y{input_shapes[1][0]};
  FusionContinuousAxis(input_shapes[0], input_shapes[1], fused_shape_x, fused_shape_y, dim_len);
  if (fused_shape_x.size() > MAX_PATTERN_DIM) {
    return true;
  }
  int32_t pattern_key = 0;
  int32_t base = 100;
  for (size_t i = 0; i < fused_shape_x.size(); i++) {
    if (fused_shape_x[i] == fused_shape_y[i]) {
      pattern_key += base;
    } else {
      pattern_key += (base * BROADCAST_BASE_KEY);
      broadcast_aixs = i;
    }
    base /= 10;
  }
  if (special_pattern.find(pattern_key) != special_pattern.end()) {
    s_pattern = special_pattern[pattern_key];
    if (s_pattern == BROADCAST && compileInfo.is_support_absorbable_broadcast) {
      s_pattern = fused_shape_x[0] == 1 ? SCALAR_BROADCAST : BROADCAST_SCALAR;
    }
    dim_len = fused_shape_x.size();
    for (size_t i = 0; i < dim_len; i++) {
      input_shapes[0][i] = fused_shape_x[i];
      input_shapes[1][i] = fused_shape_y[i];
      bool verify_broadcast = fused_shape_x[i] != fused_shape_y[i] && fused_shape_x[i] != 1 && fused_shape_y[i] != 1;
      CHECK(!verify_broadcast, "op [%s] : input shapes [%s] cannot broadcast to shape [%s]", op_type.c_str(),
          std::to_string(fused_shape_x[i]).c_str(), std::to_string(fused_shape_y[i]).c_str());
      output_shape.push_back(std::max(input_shapes[0][i], input_shapes[1][i]));
    }
  }
  return true;
}

void Eletwise::MulFusionContinuousAxis(std::vector<std::vector<int64_t>>& fusion_shapes, size_t& fusion_length) {
  int32_t last_index = 0;
  bool input_all_one = true;
  for (size_t i = 0; i < dim_len; i++) {
    bool all_one = true;
    bool state_same = true;
    for (size_t j = 0; j < input_num; j++) {
      if (input_shapes[j][i] != 1) {
        all_one = false;
      }
      if (state_same && input_shapes[j][i] != input_shapes[j][last_index] &&
          (input_shapes[j][i] == 1 || input_shapes[j][last_index] == 1)) {
        state_same = false;
      }
    }
    if (all_one) {
      continue;
    }
    if (input_all_one || state_same) {
      input_all_one = false;
      for (size_t j = 0; j < input_num; j++) {
        fusion_shapes[j][fusion_length] *= input_shapes[j][i];
      }
    } else {
      for (size_t j = 0; j < input_num; j++) {
        fusion_shapes[j].push_back(input_shapes[j][i]);
      }
      fusion_length++;
      if (fusion_length > (MAX_PATTERN_DIM - 1)) {
        break;
      }
    }
    last_index = i;
  }
}

bool Eletwise::MulTrySwitchToPerfPattern() {
  if (compileInfo.fusion_flag == 0 || !compileInfo.use_special_pattern) {
    return true;
  }
  std::vector<std::vector<int64_t>> fusion_shapes(input_num, std::vector<int64_t>{1});
  size_t fusion_length = 0;
  MulFusionContinuousAxis(fusion_shapes, fusion_length);
  if (fusion_length <= (MAX_PATTERN_DIM - 1)) {
    int32_t pattern_key = 0;
    int32_t base = 100;
    for (size_t i = 0; i <= fusion_length; i++) {
      bool is_broadcast = false;
      int64_t shape = fusion_shapes[0][i];
      for (size_t j = 1; j < input_num; j++) {
        if (shape != fusion_shapes[j][i]) {
          is_broadcast = true;
          break;
        }
      }
      if (is_broadcast) {
        pattern_key += (base * BROADCAST_BASE_KEY);
        broadcast_aixs = i;
      } else {
        pattern_key += base;
      }
      base /= 10;
    }
    if (special_pattern.find(pattern_key) != special_pattern.end()) {
      s_pattern = special_pattern[pattern_key];
      dim_len = fusion_shapes[0].size();
      for (size_t i = 0; i < dim_len; i++) {
        int32_t max_output = 1;
        for (size_t j = 0; j < input_num; j++) {
          input_shapes[j][i] = fusion_shapes[j][i];
          if (input_shapes[j][i] > max_output) {
            max_output = input_shapes[j][i];
          }
          bool verify_broadcast = input_shapes[j][i] != 1 && input_shapes[j][i] != max_output;
          CHECK(!verify_broadcast, "op [%s] : input shapes [%s] cannot broadcast to shape [%s]", op_type.c_str(),
                std::to_string(input_shapes[j][i]).c_str(), std::to_string(max_output).c_str());
        }
        output_shape.push_back(max_output);
      }
    }
  }
  return true;
}

bool Eletwise::GetCompletedShapes() {
  input_num = op_paras.inputs.size();
  dim_len = 0;
  for (size_t i = 0; i < input_num; i++) {
    CHECK(!op_paras.inputs[i].tensor.empty(), "op [%s] : input tensor cannot be empty", op_type.c_str());
    input_shapes[i].fill(1);
    if (op_paras.inputs[i].tensor[0].shape.size() > dim_len) {
      dim_len = op_paras.inputs[i].tensor[0].shape.size();
    }
  }
  for (size_t i = 0; i < input_num; i++) {
    size_t cur_dim_len = op_paras.inputs[i].tensor[0].shape.size();
    size_t start_index = dim_len - cur_dim_len;
    for (size_t j = 0; j < cur_dim_len; j++) {
      input_shapes[i][start_index++] = op_paras.inputs[i].tensor[0].shape[j];
    }
  }
  return true;
}

bool Eletwise::GenerateOutputShape() {
  bool ret = true;
  if (only_const_tiling) {
    output_shape = op_paras.outputs[0].tensor[0].shape;
  } else if (compileInfo.is_support_broadcast) {
    ret = GetCompletedShapes();
    if (input_num == SPECIAL_BROADCAST_INPUT_NUMS) {
      ret = ret && TrySwitchToPerfPattern();
    } else {
      ret = ret && MulTrySwitchToPerfPattern();
    }
    if (compileInfo.fusion_flag == 2 && s_pattern == Pattern::ORIGINAL) {
      ret = ret && RefineShapesForBroadcast();
    } else if (s_pattern == Pattern::ORIGINAL) {
      ret = ret && BroadcastShapes();
    }
  } else {
    const std::vector<int64_t>& shapes = op_paras.outputs[0].tensor[0].shape;
    if (compileInfo.fusion_flag > 0 || shapes.size() == 1) {
      int64_t fused_output = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<int64_t>());
      output_shape.push_back(fused_output);
      CHECK(!output_shape.empty(), "op [%s] : output shape cannot be empty", op_type.c_str());
      if (compileInfo.use_special_pattern) {
        s_pattern = Pattern::COMMON;
      }
    } else {
      output_shape = shapes;
    }
    for (size_t i = 0; i < output_shape.size(); i++) {
      input_shapes[0][i] = output_shape[i];
    }
  }
  return ret;
}

bool Eletwise::BroadcastShapes() {
  output_shape.reserve(dim_len);
  for (size_t i = 0; i < dim_len; i++) {
    int64_t max_output = 1;
    int64_t min_output = 2;
    for (size_t j = 0; j < input_num; j++) {
      if (input_shapes[j][i] > max_output) {
        max_output = input_shapes[j][i];
      }
      if (input_shapes[j][i] < min_output) {
        min_output = input_shapes[j][i];
      }
      bool verify_broadcast = input_shapes[j][i] != 1 && input_shapes[j][i] != max_output;
      CHECK(!verify_broadcast, "op [%s] : input shapes [%s] cannot broadcast to shape [%s]", op_type.c_str(),
            std::to_string(input_shapes[j][i]).c_str(), std::to_string(max_output).c_str());
    }
    output_shape.push_back(max_output);
    if (min_output == 1 && max_output != 1) {
      broadcast_aixs = i;
    }
  }
  return true;
}

bool Eletwise::RefineShapesForBroadcast() {
  CHECK((op_info.find("_fusion_index") != op_info.end()), "op [%s] : compile info not contain [_fusion_index]",
        op_type.c_str());
  const auto& fusion_index = op_info["_fusion_index"];
  size_t fusion_len = fusion_index.size();
  output_shape.reserve(fusion_len);
  for (size_t i = 0; i < fusion_len; i++) {
    std::vector<int64_t> fused(input_num, 1);
    int64_t max_output = 1;
    int64_t min_output = 2;
    for (size_t j = 0; j < input_num; j++) {
      for (const auto& k : fusion_index[i]) {
        fused[j] *= input_shapes[j][k];
      }
      if (fused[j] > max_output) {
        max_output = fused[j];
      }
      if (fused[j] < min_output) {
        min_output = fused[j];
      }
      bool verify_broadcast = fused[j] != 1 && fused[j] != max_output;
      CHECK(!verify_broadcast, "op [%s] : input shapes [%s] cannot broadcast to shape [%s]", op_type.c_str(),
            std::to_string(fused[j]).c_str(), std::to_string(max_output).c_str());
    }
    output_shape.push_back(max_output);
    if (min_output == 1 && max_output != 1) {
      broadcast_aixs = i;
    }
  }
  return true;
}

bool Eletwise::CalcTiling() {
  int32_t pattern = s_pattern;
  int32_t key_len = 2;
  char keys[4] = {'0', '0', '0', '\0'};
  while (pattern) {
    keys[key_len] = '0' + pattern % 10;
    pattern /= 10;
    key_len--;
  }
  std::string pattern_key = keys;
  CHECK((op_info.find("_base_info") != op_info.end()), "op [%s] : compile info not contain [_base_info]",
        op_type.c_str());
  // "_base_info": ["_ub_size", "_max_dtype", "_coexisting_quantity", "_core_num"]
  const int32_t base_info_size = 4;
  const std::vector<int32_t>& base_info = op_info["_base_info"][pattern_key];
  CHECK_EQ(base_info.size(), base_info_size, "op [%s] : base info must be _ub_size, _max_dtype, _coexisting_quantity"
           " and _core_num", op_type.c_str());
  compileInfo.ub_size = base_info[0];
  compileInfo.max_dtype = base_info[1];
  compileInfo.coexisting_quantity = base_info[2];
  compileInfo.core_num = base_info[3];
  max_available_ub =
          (((compileInfo.ub_size / compileInfo.coexisting_quantity) / BLOCK_SIZE) * BLOCK_SIZE) / compileInfo.max_dtype;
  int64_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  CHECK_LE(output_size, INT32_MAX, "op [%s] : The output shape is too large", op_type.c_str());
  const int64_t multi_core_threshold = 1024;
  if (output_size < multi_core_threshold) {
    need_multi_core = false;
  }
  return true;
}

bool Eletwise::DoBlockTiling() {
  int32_t cur_core = compileInfo.core_num;
  size_t len = output_shape.size();
  int32_t ele_in_block = BLOCK_SIZE / GetTypeSize(out_type);
  if (len == 1) {
    block_axis = 0;
    if (output_shape.empty()) {
      return -1;
    }
    CHECK((!output_shape.empty()), "op [%s] : output shape cannot be empty", op_type.c_str())
    block_factor = std::ceil(output_shape[0] * 1.0 / cur_core);
    block_factor = std::ceil(block_factor * 1.0 / ele_in_block) * ele_in_block;
    block_dims = std::ceil(output_shape[0] * 1.0 / block_factor);
    output_shape[0] = block_factor;
  } else {
    for (size_t i = 0; i < output_shape.size(); i++) {
      if (output_shape[i] > cur_core) {
        block_axis = i;
        block_factor = std::ceil(output_shape[i] * 1.0 / cur_core);
        block_dims *= std::ceil(output_shape[i] * 1.0 / block_factor);
        output_shape[i] = block_factor;
        break;
      } else {
        cur_core /= output_shape[i];
        block_dims *= output_shape[i];
      }
    }
  }
  return true;
}

bool Eletwise::DoUbTiling() {
  int32_t limit = max_available_ub;
  if (output_shape.size() == 1) {
    ub_axis = 0;
    ub_factor = static_cast<int32_t>(output_shape[0]);
    limit = std::min(limit, split_factors[compileInfo.max_dtype]);
    if (limit < ub_factor) {
      int32_t ele_in_block = BLOCK_SIZE / GetTypeSize(out_type);
      int32_t ub_for_num = std::ceil(ub_factor * 1.0 / limit);
      int32_t adjust_factor = std::ceil(ub_factor * 1.0 / ub_for_num);
      int32_t align_factor = std::ceil(adjust_factor * 1.0 / ele_in_block);
      ub_factor = align_factor * ele_in_block;
      if (ub_factor > limit) {
        ub_factor = std::floor(adjust_factor * 1.0 / ele_in_block) * ele_in_block;
      }
    }
  } else {
    int32_t shape_len = static_cast<int32_t>(output_shape.size()) - 1;
    int64_t under_ub_shape = 1;
    int32_t ele_in_block = BLOCK_SIZE / GetTypeSize(in_type);
    for (int32_t i = shape_len; i >= block_axis; i--) {
      if (broadcast_aixs == i && broadcast_aixs != shape_len) {
        bool is_cut_under_b = (under_ub_shape > N_LAST_BROADCAST_THRESHOLD ||
                               (under_ub_shape > (ele_in_block * 2) && output_shape[i] < ele_in_block)) &&
                              under_ub_shape % ele_in_block != 0 &&
			      under_ub_shape >= BLOCK_SIZE / GetTypeSize(out_type);
        if (is_cut_under_b) {
          ub_axis = i + 1;
          ub_factor = output_shape[i + 1];
          break;
        } else if (output_shape[i] > (ele_in_block * 3) && under_ub_shape % ele_in_block != 0) {
          ub_axis = i;
          ub_factor = std::min(static_cast<int32_t>(output_shape[i]), limit);
          break;
        }
      }
      if (output_shape[i] >= limit) {
        ub_axis = i;
        ub_factor = limit;
        break;
      } else {
        limit /= output_shape[i];
        under_ub_shape *= output_shape[i];
      }
    }
    if (ub_axis < 0) {
      ub_axis = block_axis;
      ub_factor = output_shape[block_axis];
    } else {
      if (block_axis == ub_axis) {
        int32_t ub_for_num = std::ceil(output_shape[ub_axis] * 1.0 / ub_factor);
        ub_factor = std::ceil(output_shape[ub_axis] * 1.0 / ub_for_num);
      }
      // Adjust the UB factor to avoid tail block less than 32 bytes
      int32_t ele_in_block = BLOCK_SIZE / GetTypeSize(out_type);
      int32_t ub_tail = output_shape[ub_axis] % ub_factor;
      if (ub_tail != 0 && (under_ub_shape * ub_tail < ele_in_block)) {
        int32_t need_tail = std::ceil(ele_in_block * 1.0 / under_ub_shape);
        int32_t ub_gap = std::ceil((need_tail - ub_tail) * 1.0 / (output_shape[ub_axis] / ub_factor));
        ub_factor -= ub_gap;
      }
    }
  }
  return true;
}

void Eletwise::CalcKey() {
  int32_t base_key = 0;
  int32_t doubleBufferKey = 10000;
  if (s_pattern != Pattern::ORIGINAL) {
    base_key = 200000000 + s_pattern * 100000;
  }
  if (need_double_buffer) {
    base_key += doubleBufferKey;
  }
  if (output_shape.size() == 1) {
    key = base_key;
  } else {
    if (ub_axis == -1 && block_axis == -1) {
      key = base_key;
    } else {
      key = base_key + block_axis * output_shape.size() + ub_axis + 1;
    }
  }
}

bool Eletwise::CalcConstKey() {
  int32_t key_index = 0;
  CHECK((op_info.find("_const_block_dims") != op_info.end()), "op [%s] : compile info not contain [_const_block_dims]",
        op_type.c_str());
  const std::vector<int32_t>& const_block_dims = op_info["_const_block_dims"].get<std::vector<int32_t>>();
  if (compileInfo.is_support_broadcast) {
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
    CHECK((op_info.find("_const_shapes") != op_info.end()), "op [%s] : compile info not contain [_const_shapes]",
          op_type.c_str());
    const std::vector<std::vector<int64_t>>& compile_const_shapes =
            op_info["_const_shapes"].get<std::vector<std::vector<int64_t>>>();
    for (size_t i = 0; i < compile_const_shapes.size(); i++) {
      bool shape_equal = true;
      CHECK_EQ(const_shapes.size(), compile_const_shapes[i].size(), "op [%s] : input shape and const shape not match",
               op_type.c_str());
      CHECK_EQ(const_block_dims.size(), compile_const_shapes.size(), "op [%s] : block dims and const shape not match",
               op_type.c_str());
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
  } else {
    CHECK(!const_block_dims.empty(), "op [%s] : block dims and const shape not match", op_type.c_str());
  }
  block_dims = const_block_dims[key_index];
  key = 100000000 + key_index;
  return true;
}

bool Eletwise::WriteTilingData(OpRunInfo& run_info) {
  GELOGD("op [%s] tiling key:%d", op_type.c_str(), key);
  GELOGD("op [%s] tiling block_dims:%d", op_type.c_str(), block_dims);
  GELOGD("op [%s] tiling block_factor:%d", op_type.c_str(), block_factor);
  GELOGD("op [%s] tiling ub_factor:%d", op_type.c_str(), ub_factor);
  GELOGD("op [%s] tiling block_axis:%d", op_type.c_str(), block_axis);
  GELOGD("op [%s] tiling ub_axis:%d", op_type.c_str(), ub_axis);

  run_info.block_dim = block_dims;
  if (only_const_tiling) {
    ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(need_multi_core));
    ByteBufferPut(run_info.tiling_data, block_axis);
    ByteBufferPut(run_info.tiling_data, block_factor);
    ByteBufferPut(run_info.tiling_data, ub_axis);
    ByteBufferPut(run_info.tiling_data, ub_factor);
    return true;
  }
  ByteBufferPut(run_info.tiling_data, key);
  if (!is_const) {
    CHECK((op_info.find("_vars") != op_info.end()), "op [%s] : compile info not contain [_vars]", op_type.c_str());
    int32_t key_len = key == 0 ? 7 : 8;
    char keys[10] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '\0'};
    while(key && key_len >= 0) {
      keys[key_len] = '0' + key % 10;
      key /= 10;
      key_len--;
    }
    std::string str_key = keys + key_len + 1;
    const auto& all_vars = op_info["_elewise_vars"][str_key];
    for (const auto& var : all_vars) {
      if (var >= 30000) {
        CHECK((ub_axis >= 0), "op [%s] : Not cut ub", op_type.c_str())
        ByteBufferPut(run_info.tiling_data, ub_factor);
      } else if (var >= 20000) {
        CHECK((ub_axis >= 0), "op [%s] : Not cut block", op_type.c_str())
        ByteBufferPut(run_info.tiling_data, block_factor);
      } else {
        int32_t var_value = var;
        int32_t operator_index = var_value % 100;
        var_value /= 100;
        size_t dim_index = var_value % 100;
        ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(input_shapes[operator_index][dim_index]));
      }
    }
  }
  return true;
}

bool Eletwise::IsNeedDoubleBuffer() {
  return ((s_pattern == Pattern::COMMON) &&
   (block_factor > std::min(max_available_ub, split_factors[compileInfo.max_dtype]))) ||
   ((s_pattern == Pattern::COMMON_BROADCAST) && (output_shape[1] >= max_available_ub));
}

bool Eletwise::DoTiling() {
  GELOGI("op [%s]: tiling running", op_type.c_str());
  bool ret = true;
  ret = ret && Init();
  if (is_const) {
    ret = ret && CalcConstKey();
  } else {
    ret = ret && GenerateOutputShape();
    ret = ret && CalcTiling();
    if (need_multi_core) {
      // cut block
      ret = ret && DoBlockTiling();
      if (IsNeedDoubleBuffer()) {
        need_double_buffer = true;
        max_available_ub =
                (((compileInfo.ub_size / DOUBLE_BUFFER_SIZE / compileInfo.coexisting_quantity) / BLOCK_SIZE)
                 * BLOCK_SIZE) / compileInfo.max_dtype;
      }
      ret = ret && DoUbTiling();
    } else {
      if (output_shape.size() == 1) {
        ub_axis = 0;
        ub_factor = output_shape[0];
        block_axis = 0;
        block_factor = output_shape[0];
      }
    }
    if (ret && !only_const_tiling) {
      CalcKey();
    }
  }
  return ret;
}

bool EletwiseTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                    OpRunInfo& run_info) {
  Eletwise eletwise(op_type, op_paras, op_info);
  bool ret = eletwise.DoTiling();
  ret = ret && eletwise.WriteTilingData(run_info);
  return ret;
}

}  // namespace optiling
