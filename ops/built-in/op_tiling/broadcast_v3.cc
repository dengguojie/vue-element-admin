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
 * \file broadcast.cpp
 * \brief
 */
#include "broadcast_v3.h"
#include <algorithm>
#include <unordered_map>
#include <tuple>

#include "graph/utils/op_desc_utils.h"
#include "vector_tiling.h"
#include "error_log.h"
#include "tiling_handler.h"

namespace optiling {
namespace v3 {
namespace {
const std::unordered_map<int64_t, int64_t> SPLIT_FACTORS{
    {1, 32767},
    {2, 32767},
    {4, 16383},
    {8, 8191},
};

const std::unordered_map<int64_t, Pattern> SPECIAL_PATTERN{
    {100, Pattern::COMMON},    {120, Pattern::COMMON_BROADCAST}, {121, Pattern::COMMON_BROADCAST_COMMON},
    {200, Pattern::BROADCAST}, {210, Pattern::BROADCAST_COMMON},
};

const std::string ALL_UNKNOWN_PATTERN = "999";
const std::string MILAN = "Ascend920";

constexpr std::int32_t DTYPE_UINT1 = 100;
constexpr std::int32_t ELEMENT_IN_BLOCK_DEFAULT = 16;
constexpr std::int32_t ELEMENT_IN_BLOCK_B32 = 8;
constexpr std::int32_t ELEMENT_IN_BLOCK_B8 = 32;
constexpr std::int32_t ELEMENT_IN_BLOCK_B64 = 4;
constexpr std::int32_t ELEMENT_IN_BLOCK_UINT1 = 256;

constexpr std::int32_t ONLY_CONST_TILING_INDEX = 0;
constexpr std::int32_t IS_CONST_INDEX = 1;
constexpr std::int32_t IS_SUPPORT_BROADCAST_INDEX = 2;
constexpr std::int32_t USE_SPECIAL_PATTERN_INDEX = 3;
constexpr std::int32_t IS_SUPPORT_ABSORBABLE_BROADCAST_INDEX = 4;
constexpr std::int32_t IS_UNKNOWN_RANK_INDEX = 5;
constexpr std::int32_t HAS_ALL_UNKNOWN_INDEX = 6;

constexpr std::int32_t MAX_UB_INDEX = 2;
constexpr std::int32_t CUR_CORE_INDEX = 0;

constexpr std::int32_t MAX_AVAILABLE_UB_INDEX = 2;
constexpr std::int32_t MAX_AVAILABLE_UB_DB_INDEX = 3;

constexpr std::int32_t NUM_TEN = 10;
constexpr std::int32_t NUM_TWO = 2;
constexpr std::int32_t NUM_ONE_HUNDRED = 100;

constexpr std::int32_t MIN_SPLIT_FACTOR = 2;
constexpr std::int32_t SPLIT_FACTOR_STEP = 2;

constexpr std::int32_t BASE_KEY_NUM = 200000000;
constexpr std::int32_t ORIGINAL_NO_DB_TILING_LEN = 7;
constexpr std::int32_t TILING_LEN = 8;

constexpr std::int32_t MIN_BLOCK_CUT_INDEX = 20000;
constexpr std::int32_t MIN_UB_CUT_INDEX = 30000;

constexpr int64_t BLOCK_SIZE = 32;
constexpr size_t MAX_UNKNOWN_RANK = 8;
constexpr int64_t DOUBLE_BUFFER_SIZE = 2;
constexpr int64_t BLOCK_NUM = 8;
constexpr int64_t MAX_REPEAT_TIMES = 8;
constexpr int64_t N_LAST_BROADCAST_THRESHOLD = 1024;
constexpr int64_t LAST_AND_N_LAST_FACTOR = 7;
constexpr int64_t MAX_PATTERN_DIM = 3;
constexpr int64_t SPECIAL_BROADCAST_INPUT_NUMS = 2;
constexpr int64_t BROADCAST_BASE_KEY = 2;
constexpr int64_t ELEWISE_REPEATE_NUMS = 128;
constexpr int64_t ELEWISE_UINT1_REPEATE_NUMS = 256;
constexpr int64_t NONE_BRC_AXIS_OPTIMIZE_BLOCK_NUMS = 3;
constexpr float MIDDLE_AXIS_OPTIMIZE_BLOCK_NUMS = 1.5;
}

const int64_t BGetElementByType(const ge::DataType& dtype) {
  // element nums in one block, default, fp16, int16, uin16
  int64_t element_in_block = ELEMENT_IN_BLOCK_DEFAULT;
  if (dtype == ge::DataType::DT_FLOAT || dtype == ge::DataType::DT_INT32 || dtype == ge::DataType::DT_UINT32) {
    // element nums in one block by b32
    element_in_block = ELEMENT_IN_BLOCK_B32;
  } else if (dtype == ge::DataType::DT_INT8 || dtype == ge::DataType::DT_UINT8 || dtype == ge::DataType::DT_BOOL) {
    // element nums in one block by b8
    element_in_block = ELEMENT_IN_BLOCK_B8;
  } else if (dtype == ge::DataType::DT_INT64 || dtype == ge::DataType::DT_UINT64) {
    // element nums in one block by b64
    element_in_block = ELEMENT_IN_BLOCK_B64;
  }else if (dtype == DTYPE_UINT1) {
    // element nums in one block by uint1
    element_in_block = ELEMENT_IN_BLOCK_UINT1;
  }
  return element_in_block;
}

bool Broadcast::Init() {
  // "_flag_info": ["_only_const_tiling", "_is_const_shapes", "_is_support_broadcast", "_use_special_pattern",
  // "_is_support_absorbable_broadcast", , "_unknown_rank"]
  V_CHECK_GE(broadcast_compile_info.flag_info_compile.size(), 1,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "flag info size error"),
             return false);
  only_const_tiling = broadcast_compile_info.flag_info_compile[ONLY_CONST_TILING_INDEX];
  if (!only_const_tiling) {
    const size_t flag_info_size = 7;
    V_CHECK_EQ(broadcast_compile_info.flag_info_compile.size(), flag_info_size,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "flag info must be _only_const_tiling, _is_const_shapes, "
                                                        "_is_support_broadcast, _use_special_pattern,"
                                                        " _is_support_absorbable_broadcast"),
               return false);
    is_support_broadcast_compile = broadcast_compile_info.flag_info_compile[IS_SUPPORT_BROADCAST_INDEX];
    use_special_pattern_compile = broadcast_compile_info.flag_info_compile[USE_SPECIAL_PATTERN_INDEX];
    is_support_absorbable_broadcast_compile =
      broadcast_compile_info.flag_info_compile[IS_SUPPORT_ABSORBABLE_BROADCAST_INDEX];
    is_unknown_rank_compile = broadcast_compile_info.flag_info_compile[IS_UNKNOWN_RANK_INDEX];
    has_all_unknown_compile = broadcast_compile_info.flag_info_compile[HAS_ALL_UNKNOWN_INDEX];
  }
  if (broadcast_compile_info.soc_version.first) {
    std::string soc_version = broadcast_compile_info.soc_version.second;
    is_milan_soc = soc_version == MILAN;
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
      if (fused_shape_x.size() > MAX_PATTERN_DIM && !has_all_unknown_compile) {
        break;
      }
    }
    last = i;
  }
  fusion_index.push_back(current_index);
}

void Broadcast::TrySwitchToPerfPattern() {
  fusion_shapes.push_back({input_shapes[0][0]});
  fusion_shapes.push_back({input_shapes[1][0]});
  FusionContinuousAxis(fusion_shapes[0], fusion_shapes[1]);
  if ((fusion_shapes[0].size() > MAX_PATTERN_DIM && !is_milan_soc) || !use_special_pattern_compile) {
    return ;
  }
  int64_t pattern_key = 0;
  int64_t base = 100;
  size_t b_axis = 0;
  for (size_t i = 0; i < fusion_shapes[0].size(); i++) {
    if (fusion_shapes[0][i] == fusion_shapes[1][i]) {
      pattern_key += base;
    } else {
      pattern_key += (base * BROADCAST_BASE_KEY);
      b_axis = i;
    }
    base /= NUM_TEN;
  }
  if (SPECIAL_PATTERN.find(pattern_key) != SPECIAL_PATTERN.end()) {
    s_pattern = SPECIAL_PATTERN.at(pattern_key);
    if (s_pattern == BROADCAST && is_support_absorbable_broadcast_compile) {
      s_pattern = fusion_shapes[0][0] == 1 ? SCALAR_BROADCAST : BROADCAST_SCALAR;
    }
    dim_len = fusion_shapes[0].size();
    for (size_t i = 0; i < dim_len; i++) {
      input_shapes[0][i] = fusion_shapes[0][i];
      input_shapes[1][i] = fusion_shapes[1][i];
      output_shape.push_back(std::max(input_shapes[0][i], input_shapes[1][i]));
    }
    broadcast_axis[b_axis] = true;
  } else if (is_milan_soc && original_dim_len > fusion_shapes[0].size()) {
    TrySwitchToPerfPatternMilan();
  }
}

void Broadcast::TrySwitchToPerfPatternMilan() {
  s_pattern = Pattern::UNKNWON_UNKNOWN;
  dim_len = fusion_shapes[0].size();
  size_t start = original_dim_len - dim_len - 1;
  for (size_t i = 0; i < start; i++) {
    input_shapes[0][i] = 1;
    input_shapes[1][i] = 1;
    output_shape.push_back(1);
  }
  for (size_t i = 0; i < dim_len; i++) {
    input_shapes[0][i + start] = fusion_shapes[0][i];
    input_shapes[1][i + start] = fusion_shapes[1][i];
    output_shape.push_back(std::max(input_shapes[0][i], input_shapes[1][i]));
  }
}

void Broadcast::MulFusionContinuousAxis(std::vector<std::vector<int64_t>>& fusion_shapes, size_t& fusion_length) {
  int64_t last_index = 0;
  bool input_all_one = true;
  std::vector<size_t> current_index = {};
  for (size_t i = 0; i < dim_len; i++) {
    bool all_one = true;
    bool state_same = true;
    for (size_t j = 0; j < input_num; j++) {
      all_one = all_one && input_shapes[j][i] == 1;
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
      if (fusion_length > (MAX_PATTERN_DIM - 1) && !has_all_unknown_compile) {
        break;
      }
    }
    last_index = i;
  }
  fusion_index.push_back(current_index);
}

void Broadcast::MulTrySwitchToPerfPattern() {
  std::vector<std::vector<int64_t>> shapes(input_num, std::vector<int64_t>{1});
  fusion_shapes = std::move(shapes);
  size_t fusion_length = 0;
  MulFusionContinuousAxis(fusion_shapes, fusion_length);
  if (!use_special_pattern_compile && !is_milan_soc) {
    return ;
  }
  if (is_milan_soc) {
    return MulTrySwitchToPerfPatternMilan();
  }
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
      base /= NUM_TEN;
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
}

void Broadcast::MulTrySwitchToPerfPatternMilan() {
  if (original_dim_len == fusion_shapes[0].size()) {
    return ;
  }
  int64_t pattern_key = 0;
  int64_t base = 100;
  for (size_t i = 0; i < fusion_shapes[0].size(); i++) {
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
      broadcast_axis[i] = true;
    } else {
      pattern_key += base;
    }
    base /= NUM_TEN;
  }
  if (SPECIAL_PATTERN.find(pattern_key) != SPECIAL_PATTERN.end()) {
    s_pattern = SPECIAL_PATTERN.at(pattern_key);
  } else {
    s_pattern = Pattern::UNKNWON_UNKNOWN;
  }
  dim_len = fusion_shapes[0].size();
  size_t start = original_dim_len - dim_len - 1;
  for (size_t i = 0; i < start; i++) {
    for (size_t j = 0; j < input_num; j++) {
      input_shapes[j][i] = 1;
      output_shape.push_back(1);
    }
  }
  for (size_t i = 0; i < dim_len; i++) {
    int64_t max_output = 1;
    for (size_t j = 0; j < input_num; j++) {
      input_shapes[j][i + start] = fusion_shapes[j][i];
      if (input_shapes[j][i + start] > max_output) {
        max_output = input_shapes[j][i + start];
      }
    }
    output_shape.push_back(max_output);
  }
}

bool Broadcast::GenerateOutputShape() {
  bool ret = true;
  broadcast_axis.fill(false);
  if (only_const_tiling) {
    output_shape = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableOutputDesc(0)->MutableShape().GetDims();
    if (!broadcast_compile_info.broadcast_axis_compile.first) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_broadcast_axis] error.");
      return false;
    }
    const auto& b_axis = broadcast_compile_info.broadcast_axis_compile.second;
    for (size_t i = 0; i < b_axis.size(); i++) {
        broadcast_axis[i] = b_axis[i];
        fusion_index.push_back({i});
      }
  } else {
    V_OP_TILING_CHECK(is_support_broadcast_compile,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile shape and runtime shape not same"),
                      return false);
    if (is_milan_soc) {
      if (!broadcast_compile_info.fusion_index_compile.first) {
        original_dim_len = dim_len;
      } else {
        original_dim_len = broadcast_compile_info.fusion_index_compile.second.size();
      }
    }
    if (input_num == SPECIAL_BROADCAST_INPUT_NUMS) {
      TrySwitchToPerfPattern();
    } else {
      MulTrySwitchToPerfPattern();
    }
    if (s_pattern == Pattern::ORIGINAL) {
      ret = ret && RefineShapesForBroadcast();
    }
  }
  return ret;
}

void GenOutputAndBrcAxis(std::vector<int64_t>& out_shape, std::vector<bool>& brc_axis,
                         const std::vector<std::vector<int64_t>>& fusion_shapes, const size_t shape_len) {
  for (size_t i = 0; i < shape_len; i++) {
    int64_t max_output = 1;
    int64_t min_output = 2;
    for (size_t j = 0; j < fusion_shapes.size(); j++) {
      max_output = std::max(max_output, fusion_shapes[j][i]);
      min_output = std::min(min_output, fusion_shapes[j][i]);
    }
    out_shape[i] = max_output;
    if (min_output == 1 && max_output != 1) {
      brc_axis[i] = true;
    }
  }
}

int64_t FindAlignFactor(const int64_t max_ub_shape, const int64_t ele_in_block) {
  int64_t split_factor = -1;
  for (int64_t f = MIN_SPLIT_FACTOR; f <= ele_in_block; f += SPLIT_FACTOR_STEP) {
    if ((max_ub_shape * f) % ele_in_block == 0) {
      split_factor = f;
      break;
    }
  }
  return split_factor;
}

bool Broadcast::CalcSplitFactor(std::vector<int64_t>& out_shape, const std::vector<bool>& brc_axis,
                                const int64_t ele_in_block, int64_t& split_axis, int64_t& split_factor) {
  int64_t cur_core;
  int64_t max_ub;
  try {
    const auto& base_info = broadcast_compile_info.base_info_compile.second.at(ALL_UNKNOWN_PATTERN);
    const size_t base_info_size = 4;
    V_CHECK_EQ(base_info.size(), base_info_size,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type,
               "base info must be _ub_size, _max_dtype, _coexisting_quantity and _core_num"),
               return false);
    cur_core = base_info[CUR_CORE_INDEX];
    max_ub = base_info[MAX_UB_INDEX];
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                    "get all unknown compile_info[_base_info] error. Error message: %s", e.what());
    return false;
  }
  int64_t b_axis = 0;
  int64_t block_output = -1;
  const int64_t multi_core_threshold = BGetElementByType(out_type) * cur_core * DOUBLE_BUFFER_SIZE;
  if (output_size > multi_core_threshold) {
    for (size_t i = 0; i < out_shape.size(); i++) {
      if (out_shape[i] > cur_core) {
        b_axis = i;
        int64_t factor = std::ceil(out_shape[i] * 1.0 / cur_core);
        block_output = out_shape[i];
        out_shape[i] = factor;
        break;
      } else {
        cur_core /= out_shape[i];
      }
    }
  }
  int64_t max_ub_shape = 1;
  int64_t last_index = static_cast<int64_t>(out_shape.size()) - 1;
  for (int64_t i = last_index; i >= b_axis; i--) {
    if (out_shape[i] < max_ub) {
      max_ub /= out_shape[i];
      if (brc_axis[i] && i != last_index && i != b_axis &&
          ((max_ub_shape * out_shape[i]) % ele_in_block == 0) && max_ub_shape % ele_in_block != 0) {
        split_axis = i;
        split_factor = FindAlignFactor(max_ub_shape, ele_in_block);
        break;
      }
      max_ub_shape *= out_shape[i];
    }
  }
  out_shape[b_axis] = output_size > multi_core_threshold ? block_output : out_shape[b_axis];
  return true;
}

std::tuple<bool, int64_t> LastFuseOutput(const std::string& op_type,
                                         const std::vector<std::vector<int64_t>>& fusion_shapes) {
  int64_t fuse_last = 0;
  for (size_t i = 0; i < fusion_shapes.size(); i++) {
    V_CHECK_GT(fusion_shapes[i].size(), 0,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The input shape must be greater than 0"),
               return std::make_tuple(false, 0));
    fuse_last = fusion_shapes[i].back();
    if (fuse_last != 1) {
      break;
    }
  }
  return std::make_tuple(true, fuse_last);
}

void Broadcast::GenerateAllUnknown(const std::vector<int64_t>& out_shape, const std::vector<bool>& brc_axis,
                                   const int64_t split_axis, const int64_t split_factor) {
  V_OP_TILING_CHECK((split_factor != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "split_factor cannot be zero."),
                    return);
  int64_t shape_len = fusion_shapes[0].size();
  int64_t fusion_len = output_shape.size();
  output_shape.clear();
  size_t start = split_axis == -1 ? fusion_len - 1 - shape_len : fusion_len - shape_len - NUM_TWO;
  for (size_t i = 0; i < start; i++) {
    for (size_t j = 0; j < input_num; j++) {
      input_shapes[j][i] = 1;
    }
    output_shape.push_back(1);
    broadcast_axis[i] = false;
  }
  size_t dim_index = start;
  for (int64_t i = 0; i < shape_len; i++) {
    for (int64_t j = 0; j < static_cast<int64_t>(input_num); j++) {
      if (i == split_axis) {
        input_shapes[j][dim_index] = fusion_shapes[j][i] == 1 ? 1 : fusion_shapes[j][i] / split_factor;
        input_shapes[j][dim_index + 1] = fusion_shapes[j][i] == 1 ? 1 : split_factor;
      } else {
        input_shapes[j][dim_index] = fusion_shapes[j][i];
      }
    }
    broadcast_axis[dim_index] = brc_axis[i];
    dim_index++;
    if (i == split_axis) {
      broadcast_axis[dim_index] = brc_axis[i];
      output_shape.push_back(out_shape[i] / split_factor);
      output_shape.push_back(split_factor);
      dim_index++;
    } else {
      output_shape.push_back(out_shape[i]);
    }
  }
  s_pattern = Pattern::UNKNWON_UNKNOWN;
}

bool Broadcast::TryMatchAllUnknown() {
  V_CHECK_GT(fusion_shapes.size(), 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The input number must be greater than 0"),
             return false);
  size_t shape_len = fusion_shapes[0].size();
  int64_t split_axis = -1;
  int64_t split_factor = -1;
  std::vector<int64_t> out_shape(shape_len, 1);
  std::vector<bool> brc_axis(shape_len, false);
  GenOutputAndBrcAxis(out_shape, brc_axis, fusion_shapes, shape_len);
  output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1LL, std::multiplies<int64_t>());
  int64_t ele_in_block = BGetElementByType(in_type);
  V_OP_TILING_CHECK((ele_in_block != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ele_in_block cannot be zero."),
                    return false);
  bool ret = true;
  if ((output_shape.size() - 1) > shape_len && shape_len > NUM_TWO && output_size % ele_in_block == 0) {
    ret = CalcSplitFactor(out_shape, brc_axis, ele_in_block, split_axis, split_factor);
  }
  int64_t fuse_last = 0;
  bool check_res = true;
  std::tie(check_res, fuse_last) = LastFuseOutput(op_type, fusion_shapes);
  bool need_fuse_axis = output_shape.back() != fuse_last;
  if (ret && check_res && (need_fuse_axis || split_axis != -1)) {
    GenerateAllUnknown(out_shape, brc_axis, split_axis, split_factor);
  }
  return ret;
}

bool Broadcast::RefineShapesForBroadcast() {
  size_t fusion_len = 0;
  if (!broadcast_compile_info.fusion_index_compile.first) {
    fusion_index = {};
    fusion_len = is_unknown_rank_compile ? MAX_UNKNOWN_RANK : dim_len;
    for (size_t i = 0; i < fusion_len; i++) {
      fusion_index.push_back({i});
    }
  } else {
    fusion_index = broadcast_compile_info.fusion_index_compile.second;
  }
  if (is_unknown_rank_compile) {
    for (size_t i = 0; i < input_num; i++) {
      std::array<int64_t, B_MAX_DIM_LEN> ori_input_shape = input_shapes[i];
      input_shapes[i].fill(1LL);
      size_t start_index = MAX_UNKNOWN_RANK - dim_len;
      for (size_t j = 0; j < dim_len; j++) {
        input_shapes[i][start_index++] = ori_input_shape[j];
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
  bool maybe_all_unknown = !is_milan_soc && has_all_unknown_compile;
  if (maybe_all_unknown) {
    return TryMatchAllUnknown();
  }
  return true;
}

bool Broadcast::CalcTiling() {
  int64_t pattern = s_pattern;
  int64_t key_len = 2;
  char keys[4] = {'0', '0', '0', '\0'};
  while (pattern) {
    keys[key_len] = '0' + pattern % NUM_TEN;
    pattern /= NUM_TEN;
    key_len--;
  }
  std::string pattern_key = keys;
  try {
    const auto& base_info = broadcast_compile_info.base_info_compile.second.at(pattern_key);
    // "_base_info": ["_core_num", "_max_dtype", "_max_available_ub", "_max_available_ub_db"]
    const size_t base_info_size = 4;
    V_CHECK_EQ(base_info.size(), base_info_size,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type,
               "base info must be _ub_size, _max_dtype, _coexisting_quantity and _core_num"),
               return false);
    core_num_compile = base_info[0];
    max_dtype_compile = base_info[1];
    max_available_ub = base_info[MAX_AVAILABLE_UB_INDEX];
    max_available_ub_db = base_info[MAX_AVAILABLE_UB_DB_INDEX];
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_base_info] error. Error message: %s", e.what());
    return false;
  }
  output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1LL, std::multiplies<int64_t>());
  V_CHECK_LE(output_size, INT32_MAX,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The output shape is too large"),
             return false);
  V_CHECK_GT(output_size, 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The output shape must be greater than 0"),
             return false);
  const int64_t multi_core_threshold = BGetElementByType(out_type) * core_num_compile * DOUBLE_BUFFER_SIZE;
  // block factor whole cut when the shape size is less than the cores size
  const int64_t block_align_threshold = BGetElementByType(out_type) * BLOCK_NUM *
                                        MAX_REPEAT_TIMES * core_num_compile;
  if (output_size <= multi_core_threshold) {
    need_multi_core = false;
  } else if (output_size <= block_align_threshold) {
    need_block_align = true;
  }
  return true;
}

int64_t CalcAlignCore(const int64_t& shape, const int64_t& core,
                      const int64_t& block_dims, const int64_t& half_core) {
  int64_t align_core = core;
  for (; align_core > 0; align_core--) {
    if (shape % align_core == 0) {
      break;
    }
  }
  return (block_dims * align_core) > half_core ? align_core : core;
}

bool Broadcast::DoBlockTiling() {
  int64_t cur_core = core_num_compile;
  V_CHECK_GT(core_num_compile, 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compileInfo core_num error, it is [%ld]", core_num_compile),
             return false);
  // multi core need more than half of cores
  int64_t half_core = core_num_compile / NUM_TWO;
  bool is_one_dim = output_shape.size() == 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    if (output_shape[i] > cur_core) {
      int64_t align_core =
          need_block_align ? CalcAlignCore(output_shape[i], cur_core, block_dims, half_core) : cur_core;
      multi_core_output = output_shape[i];
      block_axis = i;
      block_factor = std::ceil(output_shape[i] * 1.0 / align_core);
      block_dims *= std::ceil(output_shape[i] * 1.0 / block_factor);
      output_shape[i] = block_factor;
      break;
    } else {
      if (need_block_align && cur_core % output_shape[i] != 0 && block_dims * output_shape[i] > half_core) {
        multi_core_output = output_shape[i];
        block_axis = i;
        block_factor = 1;
        block_dims *= output_shape[i];
        output_shape[i] = block_factor;
        if (!is_one_dim) {
          block_axis = i + 1;
          block_factor = output_shape[i + 1];
          output_shape[i] = multi_core_output;
          multi_core_output = output_shape[i + 1];
        }
        break;
      } else {
        cur_core /= output_shape[i];
        block_dims *= output_shape[i];
      }
    }
  }
  if (output_shape.size() == 1) {
    bool outs_uint1 = broadcast_compile_info.outs_uint1_compile;
    int64_t ele_in_block = outs_uint1 ? ELEWISE_UINT1_REPEATE_NUMS : ELEWISE_REPEATE_NUMS;
    block_factor = std::ceil(block_factor * 1.0 / ele_in_block) * ele_in_block;
    output_shape[0] = block_factor;
    block_dims = std::ceil(multi_core_output * 1.0 / block_factor);
  } else if (!is_milan_soc) {
    CheckUpdateBlockTiling();
  }
  return true;
}

void Broadcast::CheckUpdateBlockTiling() {
  bool need_single_core = false;
  if (is_multi_output) {
    // multi output check
    for (size_t i = 0; i < op_paras.GetOutputsSize(); i++) {
      auto output = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableOutputDesc(i);
      int64_t ele_in_block = BGetElementByType(output->GetDataType());
      const auto& out_shape = output->MutableShape().GetDims();
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

int64_t Broadcast::FindLowestMiddle() {
  int64_t shape_len = static_cast<int64_t>(output_shape.size()) - 1;
  int64_t lowest_middle_index = shape_len - 1;
  if (!broadcast_axis[shape_len] && input_num == SPECIAL_BROADCAST_INPUT_NUMS) {
    if (input_shapes[0][shape_len] == 1 && input_shapes[1][shape_len] == 1) {
      lowest_middle_index--;
      for (int64_t i = shape_len - 1; i >= ub_axis; i--) {
        if (input_shapes[0][i] == 1 && input_shapes[1][i] == 1) {
          lowest_middle_index--;
        } else {
          break;
        }
      }
    }
  }
  bool maby_continuous_brc = broadcast_axis[lowest_middle_index] && broadcast_axis[lowest_middle_index + 1] &&
                             input_num == SPECIAL_BROADCAST_INPUT_NUMS;
  if (maby_continuous_brc) {
    for (int64_t i = lowest_middle_index; i >= ub_axis; i--) {
      if (!broadcast_axis[i]) {
        break;
      }
      bool same_broadcast_direct = (input_shapes[0][i] == input_shapes[0][i + 1] && input_shapes[0][i] == 1) ||
                                   (input_shapes[0][i] != 1 && input_shapes[0][i + 1] != 1);
      if (same_broadcast_direct) {
        lowest_middle_index--;
      } else {
        break;
      }
    }
  }
  return lowest_middle_index;
}

int64_t Broadcast::SplitUb(const int64_t& max_ub_shape, const int64_t& ele_in_block) {
  int64_t last_ub_axis = ub_axis;
  int64_t ub_output = output_shape[last_ub_axis];
  output_shape[last_ub_axis] = ub_factor;
  int64_t shape_len = static_cast<int64_t>(output_shape.size()) - 1;
  int64_t last_broadcast_size = 1;
  bool is_middle_optimize = false;
  int64_t last_under_ub_shape = 1;
  int64_t under_ub_shape = 1;
  int64_t lowest_middle_index = FindLowestMiddle();
  for (int64_t i = shape_len; i >= last_ub_axis; i--) {
    if (broadcast_axis[i] && i != shape_len) {
      if (under_ub_shape > N_LAST_BROADCAST_THRESHOLD && !is_middle_optimize) {
        ub_axis = i + 1;
        ub_factor = output_shape[i + 1];
        break;
      } else if (i <= lowest_middle_index && output_shape[i] >= (ele_in_block * MIDDLE_AXIS_OPTIMIZE_BLOCK_NUMS) &&
                 output_shape[i] > last_broadcast_size) {
        ub_axis = i;
        ub_factor = output_shape[i];
        last_under_ub_shape = under_ub_shape;
        is_middle_optimize = true;
        last_broadcast_size = output_shape[i];
      } else if (!broadcast_axis[i + 1] &&
                 under_ub_shape > (max_ub_shape / under_ub_shape * NONE_BRC_AXIS_OPTIMIZE_BLOCK_NUMS) &&
                 under_ub_shape > (BLOCK_NUM * ele_in_block) && !is_middle_optimize) {
        ub_axis = i + 1;
        ub_factor = output_shape[i + 1];
        break;
      }
    }
    if (i != last_ub_axis) {
      under_ub_shape *= output_shape[i];
    }
  }
  output_shape[last_ub_axis] = ub_output;
  return is_middle_optimize ? last_under_ub_shape : under_ub_shape;
}

bool Broadcast::MilanUbTiling() {
  int64_t limit = max_available_ub;
  V_OP_TILING_CHECK((SPLIT_FACTORS.find(max_dtype_compile) != SPLIT_FACTORS.end()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compileInfo max_dtype not in SPLIT_FACTORS"),
                    return false);
  int64_t shape_len = static_cast<int64_t>(output_shape.size()) - 1;
  int64_t under_ub_shape = 1;
  int64_t ele_in_block = BGetElementByType(out_type);
  V_OP_TILING_CHECK((ele_in_block != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ele_in_block cannot be zero."),
                    return false);
  for (int64_t i = shape_len; i >= block_axis; i--) {
    int64_t cur_shape = output_shape[i];
    if (i == shape_len) {
      cur_shape = std::ceil(cur_shape * 1.0 / ele_in_block) * ele_in_block;
    }
    if (cur_shape >= limit) {
      ub_axis = i;
      ub_factor = std::min(output_shape[i], limit);
      break;
    } else {
      limit /= cur_shape;
      under_ub_shape *= cur_shape;
      ub_axis = i;
      ub_factor = output_shape[i];
    }
  }
  OptimizeUbTiling();
  return true;
}

bool Broadcast::DefaultUbTiling() {
  int64_t limit = max_available_ub;
  V_OP_TILING_CHECK((SPLIT_FACTORS.find(max_dtype_compile) != SPLIT_FACTORS.end()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compileInfo max_dtype not in SPLIT_FACTORS"),
                    return false);
  if (output_shape.size() == 1 &&  max_available_ub > SPLIT_FACTORS.at(max_dtype_compile)) {
    limit = SPLIT_FACTORS.at(max_dtype_compile);
  }
  int64_t shape_len = static_cast<int64_t>(output_shape.size()) - 1;
  int64_t max_ub_shape = 1;
  int64_t ele_in_block = BGetElementByType(in_type);
  V_OP_TILING_CHECK((ele_in_block != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ele_in_block cannot be zero."),
                    return false);
  bool has_ub_align = false;
  for (int64_t i = shape_len; i >= block_axis; i--) {
    if (output_shape[i] >= limit) {
      ub_axis = i;
      ub_factor = limit;
      has_ub_align = has_ub_align || (broadcast_axis[i] && (max_ub_shape % ele_in_block == 0));
      max_ub_shape *= ub_factor;
      break;
    } else {
      limit /= output_shape[i];
      has_ub_align = has_ub_align || (broadcast_axis[i] && (max_ub_shape % ele_in_block == 0));
      max_ub_shape *= output_shape[i];
      ub_axis = i;
      ub_factor = output_shape[i];
    }
  }
  int64_t under_ub_shape = 1;
  if (!has_ub_align) {
    under_ub_shape = SplitUb(max_ub_shape, ele_in_block);
  } else {
    under_ub_shape = max_ub_shape / ub_factor;
  }
  AdjustUbTiling(under_ub_shape, limit);
  if (output_shape.size() != 1) {
    CheckUpdateUbTiling();
  }
  OptimizeUbTiling();
  return true;
}

bool Broadcast::DoUbTiling() {
  if (is_milan_soc){
    return MilanUbTiling();
  }
  return DefaultUbTiling();
}

void Broadcast::OptimizeUbTiling() {
  // tiling optimize for ub factor
  // if BROADCAST axis greater than a half elem_in_block, split ub form split COMMON axis to BROADCAST axis
  if (!only_const_tiling && block_axis < ub_axis &&
      output_shape[ub_axis - 1] >= (BGetElementByType(in_type) / NUM_TWO) &&
      broadcast_axis[ub_axis - 1] && !broadcast_axis[ub_axis] && ub_factor == output_shape[ub_axis]) {
    ub_axis--;
    ub_factor = 1;
  }
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
        bool outs_uint1 = broadcast_compile_info.outs_uint1_compile;
        ele_in_block = outs_uint1 ? ELEWISE_UINT1_REPEATE_NUMS : ELEWISE_REPEATE_NUMS;
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
    for (size_t i = 0; i < op_paras.GetOutputsSize(); i++) {
      auto output = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableOutputDesc(i);
      int64_t ele_in_block = BGetElementByType(output->GetDataType());
      const auto& out_shape = output->MutableShape().GetDims();
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
  if (s_pattern != Pattern::ORIGINAL) {
    constexpr int64_t s_pattern_key_num = 100000;
    base_key = BASE_KEY_NUM + static_cast<int64_t>(s_pattern) * s_pattern_key_num;
  }
  if (need_double_buffer) {
    constexpr int64_t doubleBufferKey = 10000;
    base_key += doubleBufferKey;
  }
  key = base_key;
  if (output_shape.size() != 1 && need_multi_core) {
    key = base_key + block_axis * output_shape.size() + ub_axis + 1;
  }
}

bool Broadcast::WriteTilingData() const {
  OP_LOGD(op_type.c_str(), "tiling key:%lld", key);
  OP_LOGD(op_type.c_str(), "tiling block_dims:%lld", block_dims);
  OP_LOGD(op_type.c_str(), "tiling block_factor:%lld", block_factor);
  OP_LOGD(op_type.c_str(), "tiling ub_factor:%lld", ub_factor);
  OP_LOGD(op_type.c_str(), "tiling block_axis:%lld", block_axis);
  OP_LOGD(op_type.c_str(), "tiling ub_axis:%lld", ub_axis);

  run_info.SetBlockDim(static_cast<uint32_t>(block_dims));
  if (only_const_tiling) {
    run_info.AddTilingData(static_cast<int32_t>(need_multi_core));
    run_info.AddTilingData(static_cast<int32_t>(block_axis));
    run_info.AddTilingData(static_cast<int32_t>(block_factor));
    run_info.AddTilingData(static_cast<int32_t>(ub_axis));
    run_info.AddTilingData(static_cast<int32_t>(ub_factor));
    run_info.AddTilingData(0);
    return true;
  }
  run_info.SetTilingKey(static_cast<uint32_t>(key));

  V_CHECK_GE(key, 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Tiling key error, it is [%lu], please check it", key),
             return false);
  int64_t cur_key = key;
  int64_t key_len = cur_key == 0 ? ORIGINAL_NO_DB_TILING_LEN : TILING_LEN;
  char keys[10] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '\0'};
  while (cur_key && key_len >= 0) {
    keys[key_len] = '0' + cur_key % NUM_TEN;
    cur_key /= NUM_TEN;
    key_len--;
  }
  std::string str_key = keys + key_len + 1;
  try {
    const auto& all_vars = broadcast_compile_info.elewise_vars_compile.second.at(str_key);
    for (const auto& var : all_vars) {
      if (var >= MIN_UB_CUT_INDEX) {
        V_CHECK_GE(ub_axis, 0,
                   VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Not cut ub"),
                   return false);
        run_info.AddTilingData(static_cast<int32_t>(ub_factor));
      } else if (var >= MIN_BLOCK_CUT_INDEX) {
        V_CHECK_GE(block_axis, 0,
                   VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Not cut block"),
                   return false);
        run_info.AddTilingData(static_cast<int32_t>(block_factor));
      } else {
        int64_t var_value = var;
        size_t operator_index = var_value % NUM_ONE_HUNDRED;
        var_value /= NUM_ONE_HUNDRED;
        size_t dim_index = var_value % NUM_ONE_HUNDRED;
        V_CHECK_LT(operator_index, B_MAX_INPUT_NUMS,
                   VECTOR_INNER_ERR_REPORT_TILIING(op_type, "more than 70 input are not supported"),
                   return false);
        V_CHECK_LT(dim_index, B_MAX_DIM_LEN,
                   VECTOR_INNER_ERR_REPORT_TILIING(op_type, "more than 16 dims are not supported"),
                   return false);
        run_info.AddTilingData(static_cast<int32_t>(input_shapes[operator_index][dim_index]));
      }
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_elewise_vars] error. Error message: %s", e.what());
    return false;
  }
  return true;
}

bool Broadcast::IsNeedDoubleBuffer() const {
  return ((s_pattern == Pattern::COMMON_BROADCAST) && (output_shape[1] >= max_available_ub));
}

bool Broadcast::DoTiling() {
  OP_LOGI(op_type.c_str(), "tiling running");
  bool ret = Init();
  ret = ret && GenerateOutputShape();
  ret = ret && CalcTiling();
  if (need_multi_core) {
    // cut block
    ret = ret && DoBlockTiling();
    if (ret && IsNeedDoubleBuffer()) {
      need_double_buffer = true;
      max_available_ub = max_available_ub_db;
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

bool Broadcast::CompletedShapes() {
    V_CHECK_LE(input_num, B_MAX_INPUT_NUMS,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "more than 70 input are not supported"),
               return false);
    for (size_t i = 0; i < input_num; i++) {
        input_shapes[i].fill(1LL);
        if (ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(i)->MutableShape().GetDimNum() >
            dim_len) {
            dim_len = ge::OpDescUtils::GetOpDescFromOperator(
              op_paras)->MutableInputDesc(i)->MutableShape().GetDimNum();
        }
    }
    V_CHECK_LE(dim_len, B_MAX_DIM_LEN,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "more than 16 dims are not supported"),
               return false);
    for (size_t i = 0; i < input_num; i++) {
        const ge::GeShape& shape = ge::OpDescUtils::GetOpDescFromOperator(
          op_paras)->MutableInputDesc(i)->MutableShape();
        size_t cur_dim_len = shape.GetDimNum();
        size_t start_index = dim_len - cur_dim_len;
        for (size_t j = 0; j < cur_dim_len; j++) {
            input_shapes[i][start_index++] = shape.GetDim(j);
        }
    }
    return true;
}

bool Broadcast::CompletedShapes(const std::vector<std::vector<int64_t>>& op_input_shapes) {
  V_CHECK_LE(input_num, B_MAX_INPUT_NUMS,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "more than 70 input are not supported"),
             return false);
  for (size_t i = 0; i < input_num; i++) {
    input_shapes[i].fill(1LL);
    dim_len = std::max(op_input_shapes[i].size(), dim_len);
  }
  V_CHECK_LE(dim_len, B_MAX_DIM_LEN,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "more than 16 dims are not supported"),
             return false);
  for (size_t i = 0; i < input_num; i++) {
    size_t cur_dim_len = op_input_shapes[i].size();
    size_t start_index = dim_len - cur_dim_len;
    for (size_t j = 0; j < cur_dim_len; j++) {
      input_shapes[i][start_index++] = op_input_shapes[i][j];
    }
  }
  return true;
}

bool Broadcast::GetOutputType(){
  V_OP_TILING_CHECK((op_paras.GetOutputsSize() != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output shape cannot be empty"),
                    return false);
  out_type = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableOutputDesc(0)->GetDataType();
  int64_t type_size = BGetElementByType(out_type);
  max_output_shape_size = ge::OpDescUtils::GetOpDescFromOperator(
      op_paras)->MutableOutputDesc(0)->MutableShape().GetDimNum();
  for (size_t i = 1; i < op_paras.GetOutputsSize(); i++) {
    int64_t cur_type_size = BGetElementByType(ge::OpDescUtils::GetOpDescFromOperator(
        op_paras)->MutableOutputDesc(i)->GetDataType());
    if (cur_type_size > type_size) {
      out_type = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableOutputDesc(i)->GetDataType();
      type_size = cur_type_size;
    }
    if (ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableOutputDesc(
        i)->MutableShape().GetDimNum() > max_output_shape_size) {
      max_output_shape_size = ge::OpDescUtils::GetOpDescFromOperator(
          op_paras)->MutableOutputDesc(i)->MutableShape().GetDimNum();
    }
  }
  return true;
}

bool Broadcast::GetType(){
    V_OP_TILING_CHECK((op_paras.GetInputsSize() != 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape cannot be empty"),
                      return false);
    in_type = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(0)->GetDataType();
    return GetOutputType();
}

bool Broadcast::CheckInputs(bool& is_pure_elementwise){
  for (size_t i = 0; i < dim_len; i++) {
    int64_t max_output = input_shapes[0][i];
    for (size_t j = 1; j < input_num; j++) {
      bool verify_broadcast = input_shapes[j][i] != 1 &&
                              (input_shapes[j][i] != max_output && max_output != 1);
      V_OP_TILING_CHECK((!verify_broadcast),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shapes [%s] cannot broadcast to shape [%s]",
                                                        std::to_string(input_shapes[j][i]).c_str(), std::to_string(
                                                          max_output).c_str()),
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

bool Broadcast::MatchConstShape(const std::vector<int64_t>& const_shapes, size_t& key_index) {
  if (!broadcast_compile_info.const_shapes_compile.first) {
   VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_const_shapes] error.");
   return false;
  }
  const std::vector<std::vector<int64_t>>& compile_const_shapes = broadcast_compile_info.const_shapes_compile.second;
  for (size_t i = 0; i < compile_const_shapes.size(); i++) {
    bool shape_equal = true;
    V_CHECK_EQ(const_shapes.size(), compile_const_shapes[i].size(),
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape and const shape not match"),
               return false);
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

bool Broadcast::CalcConstKey(const bool is_support_broadcast) {
  OP_LOGI(op_type.c_str(), "tiling running");
  size_t key_index = 0;
  bool ret = true;
  if (is_support_broadcast) {
    const int64_t min_broacast_num = 2;
    bool verify_input = input_num == min_broacast_num;
    V_OP_TILING_CHECK(verify_input,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "const unfold only support 2 dims"),
                      return false);
    std::array<int64_t, B_MAX_DIM_LEN> input_shape_x = input_shapes[0];
    std::array<int64_t, B_MAX_DIM_LEN> input_shape_y = input_shapes[1];
    std::vector<int64_t> const_shapes(dim_len, 0);
    for (size_t i = 0; i < dim_len; i++) {
      const_shapes[i] = static_cast<uint64_t>(input_shape_x[i]) & static_cast<uint64_t>(input_shape_y[i]);
    }
    ret = MatchConstShape(const_shapes, key_index);
  }
  if (ret) {
    if (!broadcast_compile_info.const_block_dims_compile.first) {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_const_block_dims] error.");
      return false;
    }
    const std::vector<int64_t>& const_block_dims = broadcast_compile_info.const_block_dims_compile.second;
    V_CHECK_GT(const_block_dims.size(), key_index,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "const_block_dims index out of range"),
               return false);
    block_dims = const_block_dims[key_index];
    const int64_t const_base_key = 100000000;
    key = const_base_key + key_index;
  }
  return ret;
}

bool Broadcast::IsEmptyTensor() {
  bool has_zero = false;
  for (size_t i = 0; i < op_paras.GetOutputsSize(); i++) {
    int64_t output_size = ge::OpDescUtils::GetOpDescFromOperator(
      op_paras)->MutableOutputDesc(i)->MutableShape().GetShapeSize();
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

bool Broadcast::WriteConstTiling() {
  OP_LOGD(op_type.c_str(), "tiling key:%lld", key);
  OP_LOGD(op_type.c_str(), "tiling block_dims:%lld", block_dims);
  run_info.SetBlockDim(static_cast<uint32_t>(block_dims));
  run_info.SetTilingKey(static_cast<uint32_t>(key));
  return true;
}

BroadcastCompileInfo::BroadcastCompileInfo(const std::string& op_type, const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_base_info")) {
    base_info_compile.first = true;
    base_info_compile.second =
      outer_compile_info.at("_base_info").get<std::unordered_map<std::string, std::vector<int64_t>>>();
  }
  if (!outer_compile_info.contains("_flag_info")) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "broadcast get _flag_info of compile_info error");
    return ;
  }
  flag_info_compile = outer_compile_info.at("_flag_info").get<std::vector<bool>>();
  if (!outer_compile_info.contains("_outs_uint1")) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "broadcast get _outs_uint1 of compile_info error");
    return ;
  }
  // pure_elewise cases: only_const_tiling && pure_elewise shape, or base_info contains key 100 && pure_elewise shape
  if (base_info_compile.second.count("100") || flag_info_compile.size() == 1) {
    pure_elewise_compile_info = ElewiseCompileInfo(op_type, outer_compile_info);
    pure_elewise_compile_info.SetBroadcastPattern(true);
  }
  outs_uint1_compile = outer_compile_info.at("_outs_uint1").get<bool>();
  if (outer_compile_info.contains("_elewise_vars")) {
    elewise_vars_compile.first = true;
    elewise_vars_compile.second = outer_compile_info.at(
      "_elewise_vars").get<std::unordered_map<std::string, std::vector<int64_t>>>();
  }
  if (outer_compile_info.contains("_const_block_dims")) {
    const_block_dims_compile.first = true;
    const_block_dims_compile.second = outer_compile_info.at("_const_block_dims").get<std::vector<int64_t>>();
  }

  if (outer_compile_info.contains("_const_shapes")) {
    const_shapes_compile.first = true;
    const_shapes_compile.second = outer_compile_info.at("_const_shapes").get<std::vector<std::vector<int64_t>>>();
  }

  if (outer_compile_info.contains("_fusion_index")) {
    fusion_index_compile.first = true;
    fusion_index_compile.second = outer_compile_info.at("_fusion_index").get<std::vector<std::vector<size_t>>>();
  }

  if (outer_compile_info.contains("_broadcast_axis")) {
    broadcast_axis_compile.first = true;
    broadcast_axis_compile.second = outer_compile_info.at("_broadcast_axis").get<std::vector<bool>>();
  }

  if (outer_compile_info.contains("_soc_version")) {
    soc_version.first = true;
    soc_version.second = outer_compile_info.at("_soc_version").get<std::string>();
  }
}

bool Broadcast::BroadcastTiling() {
  bool ret = GetType();
  input_num = op_paras.GetInputsSize();
  is_multi_output = op_paras.GetOutputsSize() > 1;
  ret = ret && CompletedShapes();
  bool is_empty_tensor = IsEmptyTensor();
  bool is_pure_elementwise = true;
  ret = ret && CheckInputs(is_pure_elementwise);
  if (!ret) {
      return ret;
  }
  bool is_const = false;
  bool is_support_broadcast = true;
  bool use_special_pattern = true;
  const int64_t special_pattern_threshold = 3;
  if (broadcast_compile_info.flag_info_compile.size() > special_pattern_threshold) {
    is_const = broadcast_compile_info.flag_info_compile[IS_CONST_INDEX];
    is_support_broadcast = broadcast_compile_info.flag_info_compile[IS_SUPPORT_BROADCAST_INDEX];
    use_special_pattern = broadcast_compile_info.flag_info_compile[USE_SPECIAL_PATTERN_INDEX];
  }
  if (is_const) {
    ret = CalcConstKey(is_support_broadcast);
    ret = ret && WriteConstTiling();
  } else if (is_empty_tensor) {
    key = INT32_MAX;
    block_dims = 1;
    ret = WriteConstTiling();
  } else if ((is_pure_elementwise && !(is_support_broadcast && !use_special_pattern)) || !is_support_broadcast) {
    OP_LOGD(op_type.c_str(), "broadcast turn to elewise_tiling");
    std::vector<std::vector<int64_t>> elewise_input_shapes(input_num, std::vector<int64_t>(dim_len));
    for (size_t i=0; i< input_num; i++) {
      for (size_t j=0; j<dim_len; j++) {
        elewise_input_shapes[i][j] = input_shapes[i][j];
      }
    }
    OpInfo custom_op_info(elewise_input_shapes, in_type);

    v3::Elewise elewise(op_type, op_paras, broadcast_compile_info.pure_elewise_compile_info, run_info);
    return ret && elewise.DoTiling(custom_op_info);
  } else {
    ret = ret && DoTiling();
    ret = ret && WriteTilingData();
  }
  return ret;
}

bool Broadcast::BroadcastTiling(const OpInfo& op_info) {
  bool ret = GetOutputType();
  in_type = op_info.GetInType();
  input_num = op_info.GetInputShape().size();
  is_multi_output = op_paras.GetOutputsSize() > 1;
  ret = ret&& CompletedShapes(op_info.GetInputShape());
  bool is_empty_tensor = IsEmptyTensor();
  bool is_pure_elementwise = true;
  ret = ret && CheckInputs(is_pure_elementwise);
  if (!ret) {
    return ret;
  }
  bool is_const = false;
  bool is_support_broadcast = true;
  bool use_special_pattern = true;
  constexpr uint32_t special_pattern_threshold = 3;

  if (broadcast_compile_info.flag_info_compile.size() > special_pattern_threshold) {
    is_const = broadcast_compile_info.flag_info_compile[IS_CONST_INDEX];
    is_support_broadcast = broadcast_compile_info.flag_info_compile[IS_SUPPORT_BROADCAST_INDEX];
    use_special_pattern = broadcast_compile_info.flag_info_compile[USE_SPECIAL_PATTERN_INDEX];
  }
  if (is_const) {
    ret = CalcConstKey(is_support_broadcast);
    ret = ret && WriteConstTiling();
  } else if (is_empty_tensor) {
    key = INT32_MAX;
    block_dims = 1;
    ret = WriteConstTiling();
  } else if ((is_pure_elementwise && !(is_support_broadcast && !use_special_pattern)) || !is_support_broadcast) {
    OP_LOGD(op_type.c_str(), "broadcast turn to elewise_tiling");
    std::vector<std::vector<int64_t>> elewise_input_shapes(input_num, std::vector<int64_t>(dim_len));
    for (size_t i=0; i< input_num; i++) {
      for (size_t j=0; j<dim_len; j++) {
        elewise_input_shapes[i][j] = input_shapes[i][j];
      }
    }
    OpInfo custom_op_info(elewise_input_shapes, in_type);

    v3::Elewise elewise(op_type, op_paras,  broadcast_compile_info.pure_elewise_compile_info, run_info);
    return ret && elewise.DoTiling(custom_op_info);
  } else {
    ret = ret && DoTiling();
    ret = ret && WriteTilingData();
  }
  return ret;
}
}  // namespace v3

bool BroadcastTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "broadcast tiling running");
  v3::Broadcast broadcast(op_type, op_paras, broadcast_compile_info, run_info);
  return broadcast.BroadcastTiling();
}

bool BroadcastTilingHandler::DoTiling(const ge::Operator& op_paras,
                                      utils::OpRunInfo& run_info,
                                      const OpInfo& op_info) const {
  OP_LOGD(op_type.c_str(), "broadcast custom tiling running");
  v3::Broadcast broadcast(op_type, op_paras, broadcast_compile_info, run_info);
  return broadcast.BroadcastTiling(op_info);
}

std::shared_ptr<AutoTilingHandler> CreateBroadcastTilingHandler(const std::string& op_type,
                                                                const std::string& pattern,
                                                                const nlohmann::json& parsed_compile_info) {
  return std::make_shared<BroadcastTilingHandler>(op_type, pattern, parsed_compile_info);
}

}  // namespace optiling