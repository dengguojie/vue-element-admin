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
 * \file rl_tune.cc
 * \brief
 */
#include "rl_tune.h"

namespace optiling {
namespace v3 {
namespace rl {
namespace {
constexpr int DEC_NUM = 10;
constexpr int INPUT_INDEX = 2;
constexpr int DIM_INDEX = 1;
constexpr int KERNEL_KEY_INDEX = 3;
constexpr int SCH_VARS_INDEX = 4;
constexpr int CMP_SYMBOL_INDEX = 2;
constexpr int RIGHT_VAL_INDEX = 3;
constexpr int TILING_INFO_INDEX = 2;
constexpr int BLOCK_FACTOR_INDEX = 2;
constexpr int CORE_NUM_INDEX = 3;
constexpr int BLOCK_DIM_INDEX = 4;
constexpr int UB_COUNT_INDEX = 2;
constexpr size_t COMPUTE_PATTERN_LENGTH = 2;
}
void SplitToInt(const std::string& s, std::vector<int64_t>& sv, const char flag) {
  sv.clear();
  std::istringstream iss(s);
  std::string temp = "";
  while (getline(iss, temp, flag)) {
    sv.emplace_back(strtol(temp.c_str(), nullptr, DEC_NUM));
  }
}

void SplitToStr(const string& s, std::vector<string>& sv, const char flag) {
  sv.clear();
  std::istringstream iss(s);
  std::string temp = "";
  while (getline(iss, temp, flag)) {
    sv.emplace_back(temp);
  }
}

void PackageRlBankInfo(const nlohmann::json bank_info_json, const int idx, RlBankInfo& rl_bank_info) {
  // package rl bank info
  std::vector<std::pair<int64_t, int64_t>> dynamic_axis_loc;
  std::vector<std::string> dynamic_axis_name = bank_info_json[idx][1].get<std::vector<std::string>>();
  for (const auto& name_str : dynamic_axis_name) {
    std::vector<std::string> dynamic_axis_name_strs;
    SplitToStr(name_str, dynamic_axis_name_strs, '_');
    dynamic_axis_loc.emplace_back(std::make_pair(std::stoi(dynamic_axis_name_strs[INPUT_INDEX]),
                                                 std::stoi(dynamic_axis_name_strs[DIM_INDEX])));
  }
  rl_bank_info.dynamic_axis_loc = dynamic_axis_loc;
  rl_bank_info.rl_kernel_key = bank_info_json[idx][KERNEL_KEY_INDEX].get<uint64_t>();
  rl_bank_info.rl_sch_vars = bank_info_json[idx][SCH_VARS_INDEX].get<std::vector<int64_t>>();
  // package range info
  RangeInfo range_info;
  range_info.dync_axis_inds = bank_info_json[idx][0][0].get<std::vector<std::vector<int>>>();
  range_info.mod_val = bank_info_json[idx][0][1].get<std::vector<int64_t>>();
  range_info.cmp_symbol = bank_info_json[idx][0][CMP_SYMBOL_INDEX].get<std::vector<int>>();
  range_info.right_val = bank_info_json[idx][0][RIGHT_VAL_INDEX].get<std::vector<std::vector<int64_t>>>();
  rl_bank_info.range_info = range_info;
  // parse block tiling info
  RlBlockTilingInfo rl_block_tiling_info;
  rl_block_tiling_info.block_split_axis = bank_info_json[idx][TILING_INFO_INDEX][0][0].get<int64_t>();
  rl_block_tiling_info.bind_axes = bank_info_json[idx][TILING_INFO_INDEX][0][1].get<std::vector<int64_t>>();
  rl_block_tiling_info.block_factor_name =
      bank_info_json[idx][TILING_INFO_INDEX][0][BLOCK_FACTOR_INDEX].get<std::string>();
  rl_block_tiling_info.core_num = bank_info_json[idx][TILING_INFO_INDEX][0][CORE_NUM_INDEX].get<int64_t>();
  rl_block_tiling_info.block_dim = bank_info_json[idx][TILING_INFO_INDEX][0][BLOCK_DIM_INDEX].get<int64_t>();
  rl_bank_info.rl_block_tiling_info = rl_block_tiling_info;
  // parse ub tiling info
  std::vector<RlUbTilingInfo> rl_ub_tiling_infos;
  int size_count = 1;
  while (size_count < int(bank_info_json[idx][TILING_INFO_INDEX].size() - 1)) {
    RlUbTilingInfo rl_ub_tiling_info;
    rl_ub_tiling_info.ub_split_axis = bank_info_json[idx][TILING_INFO_INDEX][size_count][0].get<int64_t>();
    rl_ub_tiling_info.ub_calc_axes =
        bank_info_json[idx][TILING_INFO_INDEX][size_count][1].get<std::vector<int64_t>>();
    rl_ub_tiling_info.ub_count = bank_info_json[idx][TILING_INFO_INDEX][size_count][UB_COUNT_INDEX].get<int64_t>();
    rl_ub_tiling_infos.emplace_back(rl_ub_tiling_info);
    size_count++;
  }
  rl_bank_info.rl_ub_tiling_infos = rl_ub_tiling_infos;
  // parse workspace_info
  std::vector<int64_t> workspace_info =
      bank_info_json[idx][TILING_INFO_INDEX][size_count].get<std::vector<int64_t>>();
  rl_bank_info.workspace_info = workspace_info;
}

void ParseRlBankInfo(const nlohmann::json& outer_compile_info,
    std::pair<bool, std::vector<std::pair<rl::RlPattern, std::vector<rl::RlBankInfo>>>>& bank_info_pair) {
  if (outer_compile_info.contains("_bank_info")) {
    bank_info_pair.first = true;
    std::vector<std::pair<RlPattern, std::vector<RlBankInfo>>> bank_info_map;
    for (const auto& j : outer_compile_info.at("_bank_info").items()) {
      std::vector<RlBankInfo> rl_bank_infos;
      const nlohmann::json bank_info_json = j.value();
      for (int i = 0; i < (int)bank_info_json.size(); i++) {
        RlBankInfo rl_bank_info;
        // package rl_bank_info
        PackageRlBankInfo(bank_info_json, i, rl_bank_info);
        // save rl_bank_info to rl_bank_infos
        rl_bank_infos.emplace_back(rl_bank_info);
      }
      // parse pattern_str to RlPattern
      std::vector<std::string> compute_pattern;
      SplitToStr(j.key(), compute_pattern, '&');
      std::vector<int64_t> inputs_shape_bank;
      SplitToInt(compute_pattern[0], inputs_shape_bank, '_');
      std::vector<int64_t> attr_bank;
      if (compute_pattern.size() == COMPUTE_PATTERN_LENGTH) {
        SplitToInt(compute_pattern[1], attr_bank, '_');
      }
      RlPattern rl_pattern;
      rl_pattern.inputs_shape_bank = inputs_shape_bank;
      rl_pattern.attr_bank = attr_bank;
      // package different cpt_pattern:rl_bank_infos
      bank_info_map.emplace_back(std::make_pair(rl_pattern, rl_bank_infos));
    }
    bank_info_pair.second = bank_info_map;
  } else {
    bank_info_pair.first = false;
  }
}
}  // namespace rl
}  // namespace v3
}  // namespace optiling
