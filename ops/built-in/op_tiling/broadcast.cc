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
 * \file broadcast.cpp
 * \brief
 */
#include "broadcast.h"

#include <algorithm>
#include <unordered_map>

#include "error_log.h"
#include "vector_tiling.h"

namespace optiling {

namespace {
  static const std::unordered_map<int64_t, int64_t> SPLIT_FACTORS{
      {1, 32767},
      {2, 32767},
      {4, 16383},
      {8, 8191},
  };

  static const std::unordered_map<int64_t, Pattern> SPECIAL_PATTERN{
      {100, Pattern::COMMON},    {120, Pattern::COMMON_BROADCAST}, {121, Pattern::COMMON_BROADCAST_COMMON},
      {200, Pattern::BROADCAST}, {210, Pattern::BROADCAST_COMMON},
  };
}

const int64_t BGetElementByType(const std::string& dtype) {
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

bool Broadcast::Init() {
  CHECK((!op_paras.inputs.empty() && !op_paras.inputs[0].tensor.empty()), "op [%s] : input shape cannot be empty",
        op_type.c_str());
  in_type = op_paras.inputs[0].tensor[0].dtype;
  input_num = op_paras.inputs.size();
  CHECK((!op_paras.outputs.empty() && !op_paras.outputs[0].tensor.empty()), "op [%s] : output shape cannot be empty",
        op_type.c_str());
  out_type = op_paras.outputs[0].tensor[0].dtype;
  int64_t type_size = BGetElementByType(out_type);
  max_output_shape_size = op_paras.outputs[0].tensor[0].shape.size();
  for (size_t i = 1; i < op_paras.outputs.size(); i++) {
    CHECK(!op_paras.outputs[i].tensor.empty(), "op [%s] : output shape cannot be empty",
        op_type.c_str());
    int64_t cur_type_size = BGetElementByType(op_paras.outputs[i].tensor[0].dtype);
    if (cur_type_size > type_size) {
      out_type = op_paras.outputs[i].tensor[0].dtype;
      type_size = cur_type_size;
    }
    if (op_paras.outputs[i].tensor[0].shape.size() > max_output_shape_size) {
      max_output_shape_size = op_paras.outputs[i].tensor[0].shape.size();
    }
  }
  is_multi_output = op_paras.outputs.size() > 1;
  // "_flag_info": ["_only_const_tiling", "_is_const_shapes", "_is_support_broadcast", "_use_special_pattern",
  // "_is_support_absorbable_broadcast", , "_unknown_rank"]
  CHECK_GE(flag_info.size(), 1, "op [%s] : flag info error", op_type.c_str());
  only_const_tiling = flag_info[0];
  if (!only_const_tiling) {
    const size_t flag_info_size = 6;
    CHECK_EQ(flag_info.size(), flag_info_size, "op [%s] : flag info must be _only_const_tiling, _is_const_shapes, "
             "_is_support_broadcast, _use_special_pattern, _is_support_absorbable_broadcast", op_type.c_str());
    compileInfo.is_support_broadcast = flag_info[2];
    compileInfo.use_special_pattern = flag_info[3];
    compileInfo.is_support_absorbable_broadcast = flag_info[4];
    compileInfo.is_unknown_rank = flag_info[5];
  }
  return true;
}

void Broadcast::FusionContinuousAxis(std::vector<int64_t>& fused_shape_x, std::vector<int64_t>& fused_shape_y) {
  const std::array<int64_t, B_MAX_DIM_LEN>& input_shape_x = input_shapes[0];
  const std::array<int64_t, B_MAX_DIM_LEN>& input_shape_y = input_shapes[1];
  std::vector<size_t> current_index = {0};
  bool state = (input_shape_x[0] == input_shape_y[0]);
  size_t last = 0;
  for (size_t i = 1; i < dim_len; i++) {
    if (input_shape_x[i] == 1 && input_shape_y[i] == 1) {
      continue;
    }
    if (state && (input_shape_x[i] == input_shape_y[i])) {
      fused_shape_x[fused_shape_x.size() - 1] *= input_shape_x[i];
      fused_shape_y[fused_shape_y.size() - 1] *= input_shape_y[i];
      current_index.push_back(i);
    } else if (((input_shape_x[i] == input_shape_x[last]) && input_shape_x[i] == 1) ||
               ((input_shape_y[i] == input_shape_y[last]) && input_shape_y[i] == 1)) {
      fused_shape_x[fused_shape_x.size() - 1] *= input_shape_x[i];
      fused_shape_y[fused_shape_y.size() - 1] *= input_shape_y[i];
      current_index.push_back(i);
      state = (input_shape_x[i] == input_shape_y[i]);
    } else {
      fused_shape_x.push_back(input_shape_x[i]);
      fused_shape_y.push_back(input_shape_y[i]);
      state = (input_shape_x[i] == input_shape_y[i]);
      fusion_index.push_back(current_index);
      current_index = {i};
      if (fused_shape_x.size() > MAX_PATTERN_DIM) {
        break;
      }
    }
    last = i;
  }
  fusion_index.push_back(current_index);
}

bool Broadcast::TrySwitchToPerfPattern() {
  if (!compileInfo.use_special_pattern) {
    return true;
  }
  std::vector<int64_t> fused_shape_x{input_shapes[0][0]};
  std::vector<int64_t> fused_shape_y{input_shapes[1][0]};
  FusionContinuousAxis(fused_shape_x, fused_shape_y);
  if (fused_shape_x.size() > MAX_PATTERN_DIM) {
    return true;
  }
  int64_t pattern_key = 0;
  int64_t base = 100;
  size_t b_axis = 0;
  for (size_t i = 0; i < fused_shape_x.size(); i++) {
    if (fused_shape_x[i] == fused_shape_y[i]) {
      pattern_key += base;
    } else {
      pattern_key += (base * BROADCAST_BASE_KEY);
      b_axis = i;
    }
    base /= 10;
  }
  if (SPECIAL_PATTERN.find(pattern_key) != SPECIAL_PATTERN.end()) {
    s_pattern = SPECIAL_PATTERN.at(pattern_key);
    if (s_pattern == BROADCAST && compileInfo.is_support_absorbable_broadcast) {
      s_pattern = fused_shape_x[0] == 1 ? SCALAR_BROADCAST : BROADCAST_SCALAR;
    }
    dim_len = fused_shape_x.size();
    for (size_t i = 0; i < dim_len; i++) {
      input_shapes[0][i] = fused_shape_x[i];
      input_shapes[1][i] = fused_shape_y[i];
      output_shape.push_back(std::max(input_shapes[0][i], input_shapes[1][i]));
    }
    broadcast_axis[b_axis] = true;
  }
  return true;
}

void Broadcast::MulFusionContinuousAxis(std::vector<std::vector<int64_t>>& fusion_shapes, size_t& fusion_length) {
  int64_t last_index = 0;
  bool input_all_one = true;
  std::vector<size_t> current_index = {};
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
      current_index.push_back(i);
    } else {
      for (size_t j = 0; j < input_num; j++) {
        fusion_shapes[j].push_back(input_shapes[j][i]);
      }
      fusion_length++;
      fusion_index.push_back(current_index);
      current_index = {i};
      if (fusion_length > (MAX_PATTERN_DIM - 1)) {
        break;
      }
    }
    last_index = i;
  }
  fusion_index.push_back(current_index);
}

bool Broadcast::MulTrySwitchToPerfPattern() {
  if (!compileInfo.use_special_pattern) {
    return true;
  }
  std::vector<std::vector<int64_t>> fusion_shapes(input_num, std::vector<int64_t>{1});
  size_t fusion_length = 0;
  MulFusionContinuousAxis(fusion_shapes, fusion_length);
  if (fusion_length <= (MAX_PATTERN_DIM - 1)) {
    int64_t pattern_key = 0;
    int64_t base = 100;
    size_t b_axis = 0;
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
        b_axis = i;
      } else {
        pattern_key += base;
      }
      base /= 10;
    }
    if (SPECIAL_PATTERN.find(pattern_key) != SPECIAL_PATTERN.end()) {
      s_pattern = SPECIAL_PATTERN.at(pattern_key);
      dim_len = fusion_shapes[0].size();
      for (size_t i = 0; i < dim_len; i++) {
        int64_t max_output = 1;
        for (size_t j = 0; j < input_num; j++) {
          input_shapes[j][i] = fusion_shapes[j][i];
          if (input_shapes[j][i] > max_output) {
            max_output = input_shapes[j][i];
          }
        }
        output_shape.push_back(max_output);
      }
      broadcast_axis[b_axis] = true;
    }
  }
  return true;
}

bool Broadcast::GenerateOutputShape() {
  bool ret = true;
  broadcast_axis.fill(false);
  if (only_const_tiling) {
    output_shape = op_paras.outputs[0].tensor[0].shape;
    try {
      const auto& b_axis = op_info.at("_broadcast_axis");
      for (size_t i = 0; i < b_axis.size(); i++) {
        broadcast_axis[i] = b_axis[i];
        fusion_index.push_back({i});
      }
    } catch (const std::exception &e) {
      GE_LOGE("op [%s] : get compile_info[_broadcast_axis] error. Error message: %s", op_type.c_str(), e.what());
      return false;
    }
  } else {
    CHECK(compileInfo.is_support_broadcast, "op [%s] : compile shape and runtime shape not same", op_type.c_str());
    if (input_num == SPECIAL_BROADCAST_INPUT_NUMS) {
      ret = ret && TrySwitchToPerfPattern();
    } else {
      ret = ret && MulTrySwitchToPerfPattern();
    }
    if (s_pattern == Pattern::ORIGINAL) {
      ret = ret && RefineShapesForBroadcast();
    }
  }
  return ret;
}

bool Broadcast::RefineShapesForBroadcast() {
  size_t fusion_len = 0;
  if (!op_info.contains("_fusion_index")) {
    fusion_index = {};
    fusion_len = compileInfo.is_unknown_rank ? MAX_UNKNOWN_RANK : dim_len;
    for (size_t i = 0; i < fusion_len; i++) {
      fusion_index.push_back({i});
    }
  } else {
    try {
      fusion_index = op_info.at("_fusion_index").get<std::vector<std::vector<size_t>>>();
    } catch (const std::exception &e) {
      GE_LOGE("op [%s] : get compile_info[_fusion_index] error. Error message: %s", op_type.c_str(), e.what());
      return false;
    }
  }
  if (compileInfo.is_unknown_rank) {
    for (size_t i = 0; i < input_num; i++) {
      input_shapes[i].fill(1LL);
      size_t cur_dim_len = op_paras.inputs[i].tensor[0].shape.size();
      size_t start_index = MAX_UNKNOWN_RANK - cur_dim_len;
      for (size_t j = 0; j < cur_dim_len; j++) {
        input_shapes[i][start_index++] = op_paras.inputs[i].tensor[0].shape[j];
      }
    }
  }
  fusion_len = fusion_index.size();
  output_shape.reserve(fusion_len);
  for (size_t i = 0; i < fusion_len; i++) {
    int64_t max_output = 1;
    int64_t min_output = 2;
    for (size_t j = 0; j < input_num; j++) {
      int64_t fused = 1;
      for (const auto& k : fusion_index[i]) {
        fused *= input_shapes[j][k];
      }
      input_shapes[j][i] = fused;
      max_output = std::max(max_output, fused);
      min_output = std::min(min_output, fused);
    }
    output_shape.push_back(max_output);
    if (min_output == 1 && max_output != 1) {
      broadcast_axis[i] = true;
    }
  }
  return true;
}

bool Broadcast::CalcTiling() {
  int64_t pattern = s_pattern;
  int64_t key_len = 2;
  char keys[4] = {'0', '0', '0', '\0'};
  while (pattern) {
    keys[key_len] = '0' + pattern % 10;
    pattern /= 10;
    key_len--;
  }
  std::string pattern_key = keys;
  try {
    const auto& base_info = op_info.at("_base_info").at(pattern_key);
    // "_base_info": ["_ub_size", "_max_dtype", "_coexisting_quantity", "_core_num"]
    const size_t base_info_size = 4;
    CHECK_EQ(base_info.size(), base_info_size, "op [%s] : base info must be _ub_size, _max_dtype, _coexisting_quantity"
            " and _core_num", op_type.c_str());
    compileInfo.ub_size = base_info[0];
    compileInfo.max_dtype = base_info[1];
    compileInfo.coexisting_quantity = base_info[2];
    compileInfo.core_num = base_info[3];
  } catch (const std::exception &e) {
    GE_LOGE("op [%s] : get compile_info[_base_info] error. Error message: %s", op_type.c_str(), e.what());
    return false;
  }
  CHECK_GT(compileInfo.coexisting_quantity, 0, "op [%s] : compileInfo coexisting_quantity error, it is [%d]",
           op_type.c_str(), compileInfo.coexisting_quantity);
  CHECK_GT(compileInfo.max_dtype, 0, "op [%s] : compileInfo max_dtype error, it is [%d]",
           op_type.c_str(), compileInfo.max_dtype);
  max_available_ub =
          (((compileInfo.ub_size / compileInfo.coexisting_quantity) / BLOCK_SIZE) * BLOCK_SIZE) / compileInfo.max_dtype;
  output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1LL, std::multiplies<int64_t>());
  CHECK_LE(output_size, INT32_MAX, "op [%s] : The output shape is too large", op_type.c_str());
  CHECK_GT(output_size, 0, "op [%s] : The output shape must be greater than 0", op_type.c_str());
  const int64_t multi_core_threshold = BGetElementByType(out_type) * compileInfo.core_num * DOUBLE_BUFFER_SIZE;
  if (output_size < multi_core_threshold) {
    need_multi_core = false;
  }
  return true;
}

bool Broadcast::DoBlockTiling() {
  int64_t cur_core = compileInfo.core_num;
  CHECK_GT(compileInfo.core_num, 0, "op [%s] : compileInfo core_num error, it is [%d]",
           op_type.c_str(), compileInfo.core_num);
  for (size_t i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] > cur_core) {
      multi_core_output = output_shape[i];
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
  if (output_shape.size() == 1) {
    int64_t ele_in_block = (out_type == "uint1") ? ELEWISE_UINT1_REPEATE_NUMS : ELEWISE_REPEATE_NUMS;
    block_factor = std::ceil(block_factor * 1.0 / ele_in_block) * ele_in_block;
    output_shape[0] = block_factor;
    block_dims = std::ceil(multi_core_output * 1.0 / block_factor);
  } else {
    CheckUpdateBlockTiling();
  }
  return true;
}

void Broadcast::CheckUpdateBlockTiling() {
  bool need_single_core = false;
  if (is_multi_output) {
    // multi output check
    for (const auto& output: op_paras.outputs) {
      int64_t ele_in_block = BGetElementByType(output.tensor[0].dtype);
      const auto& out_shape = output.tensor[0].shape;
      int64_t start = fusion_index[block_axis][0] - max_output_shape_size + out_shape.size();
      int64_t end = fusion_index[block_axis].back() - max_output_shape_size + out_shape.size();
      int64_t cut_output = 1;
      int64_t under_block = 1;
      if (start >= 0) {
        cut_output = std::accumulate(out_shape.begin() + start, out_shape.begin() + end + 1,
            1LL, std::multiplies<int64_t>());
        under_block = std::accumulate(out_shape.begin() + end + 1, out_shape.end(),
            1LL, std::multiplies<int64_t>());
      } else {
        under_block = std::accumulate(out_shape.begin(), out_shape.end(), 1LL, std::multiplies<int64_t>());
      }
      int64_t cur_block_factor = block_factor;
      if (cut_output % block_factor != 0 && (cut_output % block_factor) * under_block < ele_in_block) {
        block_factor = multi_core_output;
        output_shape[block_axis] = multi_core_output;
        cur_block_factor = std::min(multi_core_output, cut_output);
      }
      need_single_core = cut_output % cur_block_factor == 0 && cur_block_factor * under_block < ele_in_block;
      if (need_single_core) {
        break;
      }
    }
  } else {
    // single output check
    int64_t ele_in_block = BGetElementByType(out_type);
    int64_t under_block = std::accumulate(output_shape.begin() + block_axis + 1,
        output_shape.end(), 1LL, std::multiplies<int64_t>());
    if (multi_core_output % block_factor != 0 && (multi_core_output % block_factor) * under_block < ele_in_block) {
      block_factor = multi_core_output;
      output_shape[block_axis] = multi_core_output;
    }
    need_single_core = block_factor * under_block < ele_in_block;
  }
  if (need_single_core) {
    output_shape[block_axis] = multi_core_output;
    block_axis = 0;
    block_factor = output_shape[block_axis];
    multi_core_output = block_factor;
    block_dims = 1;
  }
}

bool Broadcast::DoUbTiling() {
  int64_t limit = max_available_ub;
  CHECK((SPLIT_FACTORS.find(compileInfo.max_dtype) != SPLIT_FACTORS.end()),
        "op [%s] : compileInfo max_dtype not in SPLIT_FACTORS", op_type.c_str())
  if (output_shape.size() == 1 &&  max_available_ub > SPLIT_FACTORS.at(compileInfo.max_dtype)) {
    limit = SPLIT_FACTORS.at(compileInfo.max_dtype);
  }
  int64_t shape_len = static_cast<int64_t>(output_shape.size()) - 1;
  int64_t under_ub_shape = 1;
  int64_t ele_in_block = BGetElementByType(in_type);
  for (int64_t i = shape_len; i >= block_axis; i--) {
    if (broadcast_axis[i] && i != shape_len) {
      bool is_cut_under_b = (under_ub_shape > N_LAST_BROADCAST_THRESHOLD ||
                              (under_ub_shape > (ele_in_block * LAST_AND_N_LAST_FACTOR) &&
                              output_shape[i] < (ele_in_block * LAST_AND_N_LAST_BASE))) &&
                            under_ub_shape % ele_in_block != 0 &&
                            under_ub_shape >= BGetElementByType(out_type);
      if (is_cut_under_b) {
        ub_axis = i + 1;
        ub_factor = output_shape[i + 1];
        break;
      } else if (output_shape[i] > (ele_in_block * 3) && under_ub_shape % ele_in_block != 0) {
        ub_axis = i;
        ub_factor = std::min(output_shape[i], limit);
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
  AdjustUbTiling(under_ub_shape, limit);
  if (output_shape.size() != 1) {
    CheckUpdateUbTiling();
  }

  return true;
}

void Broadcast::AdjustUbTiling(const int64_t under_ub_shape, const int64_t limit) {
  if (ub_axis < 0) {
    ub_axis = block_axis;
    ub_factor = output_shape[block_axis];
  } else {
    if (block_axis == ub_axis) {
      int64_t ub_for_num = std::ceil(output_shape[ub_axis] * 1.0 / ub_factor);
      ub_factor = std::ceil(output_shape[ub_axis] * 1.0 / ub_for_num);
    }
    int64_t shape_len = static_cast<int64_t>(output_shape.size()) - 1;
    if (ub_axis == shape_len && ub_factor != output_shape[shape_len]) {
      int64_t ele_in_block = BGetElementByType(out_type);
      if (output_shape.size() == 1) {
        ele_in_block = (out_type == "uint1") ? ELEWISE_UINT1_REPEATE_NUMS : ELEWISE_REPEATE_NUMS;
      }
      int64_t last_factor = ub_factor;
      int64_t align_factor = std::ceil(ub_factor * 1.0 / ele_in_block);
      ub_factor = align_factor * ele_in_block;
      if (ub_factor > limit) {
        ub_factor = std::floor(last_factor * 1.0 / ele_in_block) * ele_in_block;
      }
    }
    // Adjust the UB factor to avoid tail block less than 32 bytes
    int64_t ele_in_block = BGetElementByType(out_type);
    int64_t ub_tail = output_shape[ub_axis] % ub_factor;
    if (ub_tail != 0 && (under_ub_shape * ub_tail < ele_in_block)) {
      int64_t need_tail = std::ceil(ele_in_block * 1.0 / under_ub_shape);
      int64_t ub_gap = std::ceil((need_tail - ub_tail) * 1.0 / (output_shape[ub_axis] / ub_factor));
      ub_factor -= ub_gap;
    }
  }
}

void Broadcast::CheckUpdateUbTiling() {
  bool need_single_core = false;
  if (is_multi_output) {
    // multi output check
    for (const auto& output: op_paras.outputs) {
      int64_t ele_in_block = BGetElementByType(output.tensor[0].dtype);
      const auto& out_shape = output.tensor[0].shape;
      int64_t start = fusion_index[ub_axis][0] - max_output_shape_size + out_shape.size();
      int64_t end = fusion_index[ub_axis].back() - max_output_shape_size + out_shape.size();
      int64_t cut_output = 1;
      int64_t under_ub = 1;
      if (start >= 0) {
        cut_output = std::accumulate(out_shape.begin() + start, out_shape.begin() + end + 1,
            1LL, std::multiplies<int64_t>());
        under_ub = std::accumulate(out_shape.begin() + end + 1, out_shape.end(),
            1LL, std::multiplies<int64_t>());
      } else {
        under_ub = std::accumulate(out_shape.begin(), out_shape.end(), 1LL, std::multiplies<int64_t>());
      }
      need_single_core = (cut_output % ub_factor != 0 &&
          (cut_output % ub_factor) * under_ub < ele_in_block) ||
              (cut_output % ub_factor == 0 && ub_factor * under_ub < ele_in_block);
      if (block_axis == ub_axis && cut_output != 1) {
        int64_t tail = cut_output % block_factor % ub_factor;
        need_single_core = need_single_core || (tail != 0 && tail * under_ub < ele_in_block);
      }
      if (need_single_core) {
        break;
      }
    }
  } else {
    // single output check
    int64_t ele_in_block = BGetElementByType(out_type);
    int64_t cut_output = output_shape[ub_axis];
    int64_t under_ub = std::accumulate(output_shape.begin() + ub_axis + 1, output_shape.end(),
        1LL, std::multiplies<int64_t>());
    need_single_core = (cut_output % ub_factor != 0 &&
        (cut_output % ub_factor) * under_ub < ele_in_block) ||
            (cut_output % ub_factor == 0 && ub_factor * under_ub < ele_in_block);
    if (block_axis == ub_axis) {
      int64_t tail = multi_core_output % block_factor % ub_factor;
      need_single_core = need_single_core || (tail != 0 && tail * under_ub < ele_in_block);
    }
  }
  if (need_single_core) {
    output_shape[block_axis] = multi_core_output;
    block_axis = 0;
    block_factor = output_shape[block_axis];
    block_dims = 1;
  }
}

void Broadcast::CalcKey() {
  int64_t base_key = 0;
  int64_t doubleBufferKey = 10000;
  if (s_pattern != Pattern::ORIGINAL) {
    base_key = 200000000 + static_cast<int64_t>(s_pattern) * 100000;
  }
  if (need_double_buffer) {
    base_key += doubleBufferKey;
  }
  key = base_key;
  if (output_shape.size() != 1 && need_multi_core) {
    key = base_key + block_axis * output_shape.size() + ub_axis + 1;
  }
}

bool Broadcast::WriteTilingData(OpRunInfo& run_info) const {
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
  try {
    run_info.tiling_key = static_cast<int32_t>(key);
    int status = op_info.at("push_status");
    if (status == 0) {
      ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(key));
    }
  } catch (const std::exception &e) {
    GE_LOGE("op [%s]: get push_status error. Error message: %s", op_type.c_str(), e.what());
    return false;
  }

  CHECK_GE(key, 0, "op [%s] : Tiling key error, it is [%d], please check it", op_type.c_str(), key);
  int64_t cur_key = key;
  int64_t key_len = cur_key == 0 ? 7 : 8;
  char keys[10] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '\0'};
  while(cur_key && key_len >= 0) {
    keys[key_len] = '0' + cur_key % 10;
    cur_key /= 10;
    key_len--;
  }
  std::string str_key = keys + key_len + 1;
  try {
    const auto& all_vars = op_info.at("_elewise_vars").at(str_key);
    for (const auto& var : all_vars) {
      if (var >= 30000) {
        CHECK((ub_axis >= 0), "op [%s] : Not cut ub", op_type.c_str());
        ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(ub_factor));
      } else if (var >= 20000) {
        CHECK((block_axis >= 0), "op [%s] : Not cut block", op_type.c_str());
        ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(block_factor));
      } else {
        int64_t var_value = var;
        size_t operator_index = var_value % 100;
        var_value /= 100;
        size_t dim_index = var_value % 100;
        CHECK_LT(operator_index, B_MAX_INPUT_NUMS, "op [%s] : more than 70 input are not supported", op_type.c_str());
        CHECK_LT(dim_index, B_MAX_DIM_LEN, "op [%s] : more than 16 dims are not supported", op_type.c_str());
        ByteBufferPut(run_info.tiling_data, static_cast<int32_t>(input_shapes[operator_index][dim_index]));
      }
    }
  } catch (const std::exception &e) {
    GE_LOGE("op [%s] : get compile_info[_elewise_vars] error. Error message: %s", op_type.c_str(), e.what());
    return false;
  }
  return true;
}

bool Broadcast::IsNeedDoubleBuffer() const {
  return ((s_pattern == Pattern::COMMON_BROADCAST) && (output_shape[1] >= max_available_ub));
}

bool Broadcast::DoTiling() {
  GELOGI("op [%s]: tiling running", op_type.c_str());
  bool ret = Init();
  ret = ret && GenerateOutputShape();
  ret = ret && CalcTiling();
  if (need_multi_core) {
    // cut block
    ret = ret && DoBlockTiling();
    if (ret && IsNeedDoubleBuffer()) {
      need_double_buffer = true;
      max_available_ub =
              (((compileInfo.ub_size / DOUBLE_BUFFER_SIZE / compileInfo.coexisting_quantity) / BLOCK_SIZE)
                * BLOCK_SIZE) / compileInfo.max_dtype;
    }
    // cut ub
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

}  // namespace optiling
