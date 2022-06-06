/* Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file group_norm.cc
 * \brief dynamic shape tiling of group_norm
 */
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"

namespace optiling {
struct TilingParams {
  int32_t tiling_mode;
  int32_t elem_num;
  int32_t hw_num;
  int32_t group_c;
  int32_t loop_m;
  int32_t last_m;
  int32_t loop_w;
  int32_t last_w;
  int32_t avg_ng;
  int32_t block_num;
  int32_t last_ng;
  int32_t shape_c;
  int32_t group_hw;
  int32_t hw;
};

void InitNormParams(TilingParams& params) {
  params.tiling_mode = 0;
  params.elem_num = 0;
  params.hw_num = 0;
  params.group_c = 0;
  params.loop_m = 0;
  params.last_m = 0;
  params.loop_w = 0;
  params.last_w = 0;
  params.avg_ng = 0;
  params.block_num = 0;
  params.last_ng = 0;
  params.shape_c = 0;
  params.group_hw = 0;
  params.hw = 0;
}

struct opInfo {
  int32_t core_num;
  int32_t num_groups;
};

bool GroupNormParseFunc(const std::string &op_type, const nlohmann::json &compile_info, opInfo &compile_value) {
  using namespace nlohmann;
  OP_TILING_CHECK(compile_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info_json is null"),
                  return false);

  const auto &all_vars = compile_info["vars"];
  // core num
  OP_TILING_CHECK(!GetCompileValue(all_vars, "core_num", compile_value.core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GroupNormParseFunc get core_num error"), return false);
  // ub size
  OP_TILING_CHECK(!GetCompileValue(all_vars, "num_groups", compile_value.num_groups),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GroupNormParseFunc get num_groups error"), return false);
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

void AddTilingParams(const TilingParams &params, utils::OpRunInfo &run_info) {
  run_info.AddTilingData(params.tiling_mode);
  run_info.AddTilingData(params.elem_num);
  run_info.AddTilingData(params.hw_num);
  run_info.AddTilingData(params.group_c);
  run_info.AddTilingData(params.loop_m);
  run_info.AddTilingData(params.last_m);
  run_info.AddTilingData(params.loop_w);
  run_info.AddTilingData(params.last_w);
  run_info.AddTilingData(params.avg_ng);
  run_info.AddTilingData(params.block_num);
  run_info.AddTilingData(params.last_ng);
  run_info.AddTilingData(params.shape_c);
  run_info.AddTilingData(params.group_hw);
  run_info.AddTilingData(params.hw);
}

void PrintNormParams(const std::string &op_type, const TilingParams &params) {
  GELOGD("op [%s] : params.tiling_mode=%d", op_type.c_str(), params.tiling_mode);
  GELOGD("op [%s] : params.elem_num=%d", op_type.c_str(), params.elem_num);
  GELOGD("op [%s] : params.hw_num=%d", op_type.c_str(), params.hw_num);
  GELOGD("op [%s] : params.group_c=%d", op_type.c_str(), params.group_c);
  GELOGD("op [%s] : params.loop_m=%d", op_type.c_str(), params.loop_m);
  GELOGD("op [%s] : params.last_m=%d", op_type.c_str(), params.last_m);
  GELOGD("op [%s] : params.loop_w=%d", op_type.c_str(), params.loop_w);
  GELOGD("op [%s] : params.last_w=%d", op_type.c_str(), params.last_w);
  GELOGD("op [%s] : params.avg_ng=%d", op_type.c_str(), params.avg_ng);
  GELOGD("op [%s] : params.block_num=%d", op_type.c_str(), params.block_num);
  GELOGD("op [%s] : params.last_ng=%d", op_type.c_str(), params.last_ng);
  GELOGD("op [%s] : params.shape_c=%d", op_type.c_str(), params.shape_c);
  GELOGD("op [%s] : params.group_hw=%d", op_type.c_str(), params.group_hw);
  GELOGD("op [%s] : params.hw=%d", op_type.c_str(), params.hw);
}

int32_t GroupNormDiv(int32_t value0, int32_t value1) {
  int32_t res = 0;
  if (value1 == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("GroupNormTiling", "GroupNormDiv, value1 cannot be zero");
    return 0;
  }
  res = (value0 + value1 - 1) / value1;
  return res;
}

int32_t GroupNormTail(int32_t value0, int32_t value1) {
  int32_t res = 0;
  if (value1 == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("GroupNormTiling", "GroupNormTail, value1 cannot be zero");
    return 0;
  }
  res = value0 % value1;
  if (res == 0) {
    res = value1;
  }
  return res;
}

int32_t GroupNormTilingMode(int32_t c1, int32_t group_c) {
  int32_t scale_n = 512;
  int32_t tiling_c_small = 0;
  int32_t tiling_c_mid = 1;
  int32_t tiling_c_big = 2;

  if (c1 <= scale_n) {
    return tiling_c_small;
  } else if (group_c <= scale_n) {
    return tiling_c_mid;
  } else {
    return tiling_c_big;
  }
}

bool GetGroupNormCompileParams(const std::string &op_type, const opInfo &op_compile_info_json, int32_t &core_num,
                               int32_t &num_groups) {
  core_num = op_compile_info_json.core_num;
  num_groups = op_compile_info_json.num_groups;
  return true;
}

void CalInfo5HD(TilingParams& tiling_params, int32_t core_num, int32_t num_groups, std::vector<int64_t> input_shape) {
  int32_t n = input_shape[0];
  int32_t c1 = input_shape[1];
  int32_t h = input_shape[2];
  int32_t w = input_shape[3];
  int32_t c0 = 16;
  int32_t ub_n = 512;
  int32_t group_c = c1 / num_groups;
  int32_t n_group = n * num_groups;
  int32_t group_hw = group_c * h * w;
  int32_t hw = h * w;

  tiling_params.tiling_mode = GroupNormTilingMode(c1, group_c);
  tiling_params.elem_num = group_c * h * w * c0;
  tiling_params.hw_num = h * w * c0;
  tiling_params.group_c = group_c;
  tiling_params.loop_m = GroupNormDiv(group_hw, ub_n);
  tiling_params.last_m = GroupNormTail(group_hw, ub_n);
  tiling_params.loop_w = GroupNormDiv(hw, ub_n);
  tiling_params.last_w = GroupNormTail(hw, ub_n);
  tiling_params.avg_ng = GroupNormDiv(n_group, core_num);
  tiling_params.block_num = GroupNormDiv(n_group, tiling_params.avg_ng);
  tiling_params.last_ng = n_group - tiling_params.avg_ng * (tiling_params.block_num - 1);
  tiling_params.shape_c = c1 * c0;
  tiling_params.group_hw = group_hw;
  tiling_params.hw = hw;
}

void CalInfoND(TilingParams& tiling_params, int32_t core_num, int32_t num_groups, std::vector<int64_t> input_shape,
               int32_t hw_num) {
  int32_t n = input_shape[0];
  int32_t c = input_shape[1];
  int32_t c0 = 16;
  int32_t ub_n = 512;
  int32_t group_c = c / num_groups;
  int32_t n_group = n * num_groups;
  int32_t group_hw = GroupNormDiv(group_c * hw_num, c0);
  int32_t hw = GroupNormDiv(hw_num, c0);
  int32_t align_c = GroupNormDiv(c, c0);
  int32_t align_group = GroupNormDiv(group_c, c0);

  tiling_params.tiling_mode = GroupNormTilingMode(align_c, align_group);
  tiling_params.elem_num = group_c * hw_num;
  tiling_params.hw_num = hw_num;
  tiling_params.group_c = group_c;
  tiling_params.loop_m = GroupNormDiv(group_hw, ub_n);
  tiling_params.last_m = GroupNormTail(group_hw, ub_n);
  tiling_params.loop_w = GroupNormDiv(hw, ub_n);
  tiling_params.last_w = GroupNormTail(hw, ub_n);
  tiling_params.avg_ng = GroupNormDiv(n_group, core_num);
  tiling_params.block_num = GroupNormDiv(n_group, tiling_params.avg_ng);
  tiling_params.last_ng = n_group - tiling_params.avg_ng * (tiling_params.block_num - 1);
  tiling_params.shape_c = c;
  tiling_params.group_hw = group_hw;
  tiling_params.hw = hw;
}

bool GroupNormTiling(const std::string &op_type, const ge::Operator &op_paras, const opInfo &op_compile_info_json,
                     utils::OpRunInfo &run_info) {
  GELOGI("GroupNormTiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  std::vector<int64_t> input_shape = operator_info->MutableInputDesc(0)->MutableShape().GetDims();
  ge::Format input_format = operator_info->MutableInputDesc(0)->GetFormat();

  int32_t input_dims = input_shape.size();
  int32_t row = 1;
  int32_t hw_num = 1;

  for (int i = 0; i < input_dims; i++) {
    row = row * input_shape[i];
  }

  for (int i = 2; i < input_dims; i++) {
    hw_num = hw_num * input_shape[i];
  }

  int32_t core_num = 0;
  int32_t num_groups = 0;
  bool flag = GetGroupNormCompileParams(op_type, op_compile_info_json, core_num, num_groups);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetGroupNormCompileParams failed.");
    return false;
  }

  GELOGI("op[%s] GetGroupNormCompileParams success.", op_type.c_str());

  TilingParams tiling_params;
  InitNormParams(tiling_params);

  if (num_groups == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("GroupNormTiling", "num_groups cannot be zero");
    return false;
  }

  if (input_format == FORMAT_NC1HWC0) {
    CalInfo5HD(tiling_params, core_num, num_groups, input_shape);
  } else {
    CalInfoND(tiling_params, core_num, num_groups, input_shape, hw_num);
  }

  AddTilingParams(tiling_params, run_info);
  PrintNormParams(op_type, tiling_params);

  run_info.SetBlockDim(tiling_params.block_num);
  GELOGI("GroupNormTiling end.");
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(GroupNorm, GroupNormTiling, GroupNormParseFunc, opInfo);
}  // namespace optiling
