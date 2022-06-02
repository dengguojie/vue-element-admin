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
 * \file rl_tune.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RL_TUNE_H_
#define OPS_BUILT_IN_OP_TILING_RL_TUNE_H_

#include <vector>
#include <string>
#include "vector_tiling.h"

namespace optiling {
namespace v3 {
namespace rl {
#define LE(j, k) ((j) <= (k))
#define GE(j, k) ((j) >= (k))
#define EE(j, k) ((j) == (k))
#define NE(j, k) ((j) != (k))
#define LGE(j, k, z) (((j) >= (k)) && ((j) <= (z)))

constexpr int64_t DYNC_AXIS_MAX_NUM = 20;
constexpr int64_t RL_MAX_VARS_NUM = 256;
constexpr int64_t RL_TOTAL_SHAPE_DIM_LEN = 70 * 16;
constexpr int64_t RL_MAX_ATTR_SIZE = 70;

constexpr int LE_SYMBOL = 0;
constexpr int GE_SYMBOL = 1;
constexpr int EE_SYMBOL = 2;
constexpr int NE_SYMBOL = 3;
constexpr int LGE_SYMBOL = 4;

struct RlPattern {
  std::vector<int64_t> inputs_shape_bank;
  std::vector<int64_t> attr_bank;
};

struct RlBlockTilingInfo {
  int64_t block_split_axis{-1};
  std::vector<int64_t> bind_axes;
  std::string block_factor_name;
  int64_t core_num{-1};
  int64_t block_dim{-1};
};

struct RlUbTilingInfo {
  int64_t ub_split_axis{-1};
  std::vector<int64_t> ub_calc_axes;
  int64_t ub_count{-1};
};

struct RangeInfo {
  std::vector<std::vector<int>> dync_axis_inds;
  std::vector<int64_t> mod_val;
  std::vector<int> cmp_symbol;
  std::vector<std::vector<int64_t>> right_val;
};

struct RlBankInfo {
  RangeInfo range_info;
  // dynamic axis location pair, example: (input_num, dim_num)
  std::vector<std::pair<int64_t, int64_t>>dynamic_axis_loc;
  RlBlockTilingInfo rl_block_tiling_info;
  std::vector<RlUbTilingInfo> rl_ub_tiling_infos;
  std::vector<int64_t> workspace_info;
  uint64_t rl_kernel_key{0};
  std::vector<int64_t> rl_sch_vars;
};

void ParseRlBankInfo(const nlohmann::json& outer_compile_info,
                     std::pair<bool, std::vector<std::pair<RlPattern, std::vector<RlBankInfo>>>>& bank_info_pair);

inline bool CalcExpr(const RangeInfo& range_info, const std::array<int64_t, DYNC_AXIS_MAX_NUM>& vars_value) {
  bool result = true;
  int i = 0;
  for (const auto& val : range_info.right_val) {
    // *
    int64_t left_value = vars_value[range_info.dync_axis_inds[i][0]];
    for (size_t k = 1, len = range_info.dync_axis_inds[i].size(); k < len; k++) {
      left_value *= vars_value[range_info.dync_axis_inds[i][k]];
    }
    // %
    left_value = range_info.mod_val[i] > 0 ? left_value & (range_info.mod_val[i] - 1) : left_value;
    // compare and &&
    if (range_info.cmp_symbol[i] == LGE_SYMBOL) {  // >=&&<=
      result &= LGE(left_value, val[0], val[1]);
    } else if (range_info.cmp_symbol[i] == LE_SYMBOL) {  // <=
      result &= LE(left_value, val[0]);
    } else if (range_info.cmp_symbol[i] == GE_SYMBOL) {  // >=
      result &= GE(left_value, val[0]);
    } else if (range_info.cmp_symbol[i] == EE_SYMBOL) {  // ==
      result &= EE(left_value, val[0]);
    } else if (range_info.cmp_symbol[i] == NE_SYMBOL) {  // !=
      result &= NE(left_value, val[0]);
    }
    if (!result) {
      return result;
    }
    i++;
  }
  return result;
}

inline bool PatternMatch(const RlPattern& rl_pattern,
    const std::array<int64_t, RL_TOTAL_SHAPE_DIM_LEN>& inputs_shape, const size_t& inputs_shape_size,
    const std::array<int64_t, RL_MAX_ATTR_SIZE>& attr, const size_t& attr_size) {
  if (rl_pattern.inputs_shape_bank.size() != inputs_shape_size || rl_pattern.attr_bank.size() != attr_size) {
    return false;
  }
  for (size_t i = 0; i < inputs_shape_size; i++) {
    if (rl_pattern.inputs_shape_bank[i] != -1 && rl_pattern.inputs_shape_bank[i] != inputs_shape[i]) {
      return false;
    }
  }
  for (size_t i = 0; i < attr_size; i++) {
    if (rl_pattern.attr_bank[i] != attr[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace rl
}  // namespace v3
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RL_TUNE_H_
