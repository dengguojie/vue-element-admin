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
 * \file nll_loss.cc
 * \brief dynamic Nllloss tiling
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {

const int64_t BLOCK_SIZE = 32;

static int64_t GetFloorDiv(const int64_t l_value, const int64_t r_value) {
  if (r_value == 0) {
    return l_value;
  }

  return l_value / r_value;
}

static int64_t GetCeilDiv(const int64_t l_value, const int64_t r_value) {
  if (r_value == 0) {
    return l_value;
  }

  return (l_value + r_value - 1) / r_value;
}

static int64_t GetMod(const int64_t l_value, const int64_t r_value) {
  if (r_value == 0) {
    return l_value;
  }

  return l_value % r_value;
}

bool GetCompileParams(const std::string& op_type, const nlohmann::json& op_compile_info_json, int64_t& core_num,
                      int64_t& ub_size, std::string& reduction) {
  using namespace nlohmann;

  auto all_vars = op_compile_info_json["vars"];
  if (all_vars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: GetCompileParams, get core_num error.");
    return false;
  }
  core_num = all_vars["core_num"].get<std::int64_t>();

  if (all_vars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: GetCompileParams, get ub_size error.");
    return false;
  }
  ub_size = all_vars["ub_size"].get<std::int64_t>();

  if (all_vars.count("reduction") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: GetCompileParams, get reduction error.");
    return false;
  }
  reduction = all_vars["reduction"].get<std::string>();

  OP_LOGD(op_type.c_str(), "NLLLossTiling: GetCompileParams, core_num[%lld], ub_size[%lld], reduction[%s].", core_num,
          ub_size, reduction.c_str());

  return true;
}

bool CheckTensorShape(const std::string& op_type, const vector<int64_t>& x_shape, const vector<int64_t>& target_shape,
                      const vector<int64_t>& weight_shape) {
  int64_t x_dims = x_shape.size();

  if (x_dims <= 0 || x_dims > 2) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: input tensor x should be 1D or 2D.");
    return false;
  }

  if (target_shape.size() != 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: input tensor target should be 1D.");
    return false;
  }

  if (x_shape[0] != target_shape[0]) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: input tensor x[0] should be equal to target[0].");
    return false;
  }

  if (weight_shape.size() != 1) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: input tensor weight should be 1D.");
    return false;
  }

  if (x_shape.back() != weight_shape[0]) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: input tensor x[-1] should be equal to weight[0].");
    return false;
  }

  return true;
}

void SetLastCoreTilingData(const int64_t& max_line, const int64_t& n_size, int64_t& need_core_num,
                           int64_t& per_core_size, int64_t& per_core_loop_cnt, int64_t& per_core_left_size,
                           int64_t& last_core_size, int64_t& last_core_loop_cnt, int64_t& last_core_left_size) {
  need_core_num = 1;
  per_core_size = 0;
  per_core_loop_cnt = 0;
  per_core_left_size = 0;
  last_core_size = n_size;
  last_core_loop_cnt = GetFloorDiv(n_size, max_line);
  last_core_left_size = GetMod(n_size, max_line);
}

void SetCommonTilingData(const int64_t& max_line, const int64_t& per_core_size, const int64_t& last_core_size,
                         int64_t& per_core_loop_cnt, int64_t& per_core_left_size, int64_t& last_core_loop_cnt,
                         int64_t& last_core_left_size) {
  per_core_loop_cnt = GetFloorDiv(per_core_size, max_line);
  per_core_left_size = GetMod(per_core_size, max_line);
  last_core_loop_cnt = GetFloorDiv(last_core_size, max_line);
  last_core_left_size = GetMod(last_core_size, max_line);
}

int64_t GetMaxAligned(const int64_t& max_line, const int64_t& min_aligned, const int64_t& need_core_num,
                      const int64_t& n_size) {
  // ub limit
  int64_t max_aligned = max_line - GetMod(max_line, min_aligned);

  // the maximum amount of data that can be allocated per core
  int64_t limit_per_core_size = GetFloorDiv(n_size, need_core_num - 1);
  int64_t max_aligned_n_size = limit_per_core_size + min_aligned - GetMod(limit_per_core_size, min_aligned);
  return max_aligned < max_aligned_n_size ? max_aligned : max_aligned_n_size;
}

int64_t ChangeAligned(const int64_t& max_aligned, const int64_t& min_aligned, const int64_t& need_core_num,
                      int64_t& per_core_size, int64_t& last_core_size) {
  // data needs to be supplemented on each core for full division
  int64_t per_core_supple_data = GetMod(max_aligned - GetMod(per_core_size, max_aligned), max_aligned);

  // filled with data from the last core
  int64_t last_core_left_size = last_core_size - (need_core_num - 1) * per_core_supple_data;

  if (last_core_left_size > 0) {
    per_core_size += per_core_supple_data;
    last_core_size = last_core_left_size;
    return max_aligned;
  }

  // the minimum aligned is not divisible
  if (max_aligned == min_aligned) {
    return -1;
  }

  return ChangeAligned(max_aligned - min_aligned, min_aligned, need_core_num, per_core_size, last_core_size);
}

void RecursiveTiling(int64_t& ub_max_line, int64_t& need_core_num, int64_t& n_size, int64_t& c_size,
                     int64_t& per_core_size, int64_t& per_core_loop_cnt, int64_t& per_core_left_size,
                     int64_t& last_core_size, int64_t& last_core_loop_cnt, int64_t& last_core_left_size,
                     int64_t& min_aligned, bool is_left_data) {
  int64_t max_line = ub_max_line;

  // one core
  if (need_core_num == 1) {
    SetLastCoreTilingData(max_line, n_size, need_core_num, per_core_size, per_core_loop_cnt, per_core_left_size,
                          last_core_size, last_core_loop_cnt, last_core_left_size);
    return;
  }

  // each core equally divided data
  per_core_size = GetFloorDiv(n_size, need_core_num);
  last_core_size = per_core_size;
  int64_t left_size = GetMod(n_size, need_core_num);

  // use last core data supplement per core
  if (left_size > 0) {
    last_core_size -= GetMod((need_core_num - 1 - left_size), (need_core_num - 1));
  }

  // no data is allocated to the last core
  if (last_core_size <= 0) {
    need_core_num -= 1;
    return RecursiveTiling(ub_max_line, need_core_num, n_size, c_size, per_core_size, per_core_loop_cnt,
                           per_core_left_size, last_core_size, last_core_loop_cnt, last_core_left_size, min_aligned,
                           is_left_data);
  }

  // equally divided left data
  if (left_size > 0) {
    per_core_size += 1;
  }

  // core allow has left data
  if (is_left_data) {
    max_line = max_line - GetMod(max_line, min_aligned);
    int64_t line = ChangeAligned(min_aligned, min_aligned, need_core_num, per_core_size, last_core_size);
    if (line == -1) {
      need_core_num -= 1;
      return RecursiveTiling(ub_max_line, need_core_num, n_size, c_size, per_core_size, per_core_loop_cnt,
                             per_core_left_size, last_core_size, last_core_loop_cnt, last_core_left_size, min_aligned,
                             is_left_data);
    }
  } else {
    // integer multiple of min_aligned
    int64_t max_aligned = GetMaxAligned(max_line, min_aligned, need_core_num, n_size);

    // compute max_line when total x may exact division
    max_line = ChangeAligned(max_aligned, min_aligned, need_core_num, per_core_size, last_core_size);
    if (max_line == -1) {
      need_core_num -= 1;
      return RecursiveTiling(ub_max_line, need_core_num, n_size, c_size, per_core_size, per_core_loop_cnt,
                             per_core_left_size, last_core_size, last_core_loop_cnt, last_core_left_size, min_aligned,
                             is_left_data);
    }
  }

  SetCommonTilingData(max_line, per_core_size, last_core_size, per_core_loop_cnt, per_core_left_size,
                      last_core_loop_cnt, last_core_left_size);
}

void NLLLossCommonTiling(int64_t& bytes, int64_t& core_num, int64_t& ub_size, int64_t& need_core_num, int64_t& n_size,
                         int64_t& c_size, int64_t& per_core_size, int64_t& per_core_loop_cnt,
                         int64_t& per_core_left_size, int64_t& last_core_size, int64_t& last_core_loop_cnt,
                         int64_t& last_core_left_size, int64_t& ub_max_line, int64_t& min_aligned, bool is_left_data) {
  // one core
  if (ub_max_line < min_aligned) {
    SetLastCoreTilingData(ub_max_line, n_size, need_core_num, per_core_size, per_core_loop_cnt, per_core_left_size,
                          last_core_size, last_core_loop_cnt, last_core_left_size);
    return;
  }

  need_core_num = core_num;
  RecursiveTiling(ub_max_line, need_core_num, n_size, c_size, per_core_size, per_core_loop_cnt, per_core_left_size,
                  last_core_size, last_core_loop_cnt, last_core_left_size, min_aligned, is_left_data);
}

int64_t CalculUbSizeNormWeight(int64_t& x_size, int64_t& target_size, int64_t& weight_size, const int64_t& ub_size,
                               const int64_t& c_size, const int64_t& data_one_block) {
  int64_t ub_max_line;
  weight_size = GetCeilDiv(c_size, data_one_block) * data_one_block;
  // 4: x, target, valid_x, valid_weight 32byte aligned
  int64_t ub_remaining_size = ub_size - weight_size - 4 * data_one_block;
  // 3: target, valid_x, valid_weight, need one for per line
  ub_max_line = GetFloorDiv(ub_remaining_size, c_size + 3);
  x_size = GetCeilDiv(ub_max_line * c_size, data_one_block) * data_one_block;
  target_size = GetCeilDiv(ub_max_line, data_one_block) * data_one_block;
  return ub_max_line;
}

int64_t CalculUbSizeLargeWeight(int64_t& x_size, int64_t& target_size, int64_t& weight_size, const int64_t& ub_size,
                                const int64_t& c_size, const int64_t& data_one_block) {
  int64_t ub_max_line;
  x_size = data_one_block;
  weight_size = data_one_block;
  // 2: x_size and weight_size used one block
  // 3: target, valid_x, valid_weight equally divide the remaining space
  target_size = GetFloorDiv(ub_size - 2 * data_one_block, 3);
  // target_size 32byte unit
  ub_max_line = target_size - GetMod(target_size, data_one_block);
  return ub_max_line;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/attrs of the op
 * @param [in] op_info: compile stage generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool NLLLossTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                   utils::OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "NLLLossTiling start running.");
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed.");
    return false;
  }

  // get compile info
  int64_t bytes = 4;
  int64_t core_num = 0;
  int64_t ub_size = 0;
  int64_t tiling_mode = 0;
  int64_t need_core_num = 0;
  int64_t n_size = 0;
  int64_t c_size = 0;
  int64_t per_core_size = 0;
  int64_t per_core_loop_cnt = 0;
  int64_t per_core_left_size = 0;
  int64_t last_core_size = 0;
  int64_t last_core_loop_cnt = 0;
  int64_t last_core_left_size = 0;

  int64_t x_size = 0;
  int64_t target_size = 0;
  int64_t weight_size = 0;
  int64_t ub_max_line = 0;

  std::string reduction;
  int64_t min_aligned = 1;
  bool is_left_data = true;

  auto input_x_desc = operator_info->MutableInputDesc(0);
  auto input_target_desc = operator_info->MutableInputDesc(1);
  auto input_weight_desc = operator_info->MutableInputDesc(2);

  if (input_x_desc == nullptr || input_target_desc == nullptr || input_weight_desc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed.");
    return false;
  }

  std::vector<int64_t> x_shape = input_x_desc->MutableShape().GetDims();
  std::vector<int64_t> target_shape = input_target_desc->MutableShape().GetDims();
  std::vector<int64_t> weight_shape = input_weight_desc->MutableShape().GetDims();

  if (!GetCompileParams(op_type, op_info, core_num, ub_size, reduction)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: GetCompileParams error.");
    return false;
  }

  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  if (!CheckTensorShape(op_type, x_shape, target_shape, weight_shape)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: CheckTensorShape error.");
    return false;
  }

  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  // input x is 1D
  if (x_shape.size() == 1) {
    n_size = 1;
  } else {
    n_size = x_shape[0];
  }

  c_size = x_shape.back();

  int64_t data_one_block = GetFloorDiv(BLOCK_SIZE, bytes);
  ub_max_line = CalculUbSizeNormWeight(x_size, target_size, weight_size, ub_size, c_size, data_one_block);
  // c_size can move into ub
  if (ub_max_line >= 1) {
    tiling_mode = 1;
  } else {
    tiling_mode = 2;
    ub_max_line = CalculUbSizeLargeWeight(x_size, target_size, weight_size, ub_size, c_size, data_one_block);
  }

  if (reduction == "none") {
    min_aligned = GetFloorDiv(BLOCK_SIZE, bytes);
  }

  NLLLossCommonTiling(bytes, core_num, ub_size, need_core_num, n_size, c_size, per_core_size, per_core_loop_cnt,
                      per_core_left_size, last_core_size, last_core_loop_cnt, last_core_left_size, ub_max_line,
                      min_aligned, is_left_data);

  OP_LOGD(op_type.c_str(), "NLLLossTiling: n_size=%lld, c_size=%lld", n_size, c_size);
  OP_LOGD(op_type.c_str(), "NLLLossTiling: tiling_mode=%lld, need_core_num=%lld", tiling_mode, need_core_num);
  OP_LOGD(op_type.c_str(), "NLLLossTiling: per_core_size=%lld, per_core_loop_cnt=%lld, per_core_left_size=%lld",
          per_core_size, per_core_loop_cnt, per_core_left_size);
  OP_LOGD(op_type.c_str(), "NLLLossTiling: last_core_size=%lld, last_core_loop_cnt=%lld, last_core_left_size=%lld",
          last_core_size, last_core_loop_cnt, last_core_left_size);
  OP_LOGD(op_type.c_str(), "NLLLossTiling: x_size=%lld, target_size=%lld, weight_size=%lld", x_size, target_size,
          weight_size);

  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  // set tiling data
  run_info.AddTilingData(tiling_mode);
  run_info.AddTilingData(need_core_num);
  run_info.AddTilingData(n_size);
  run_info.AddTilingData(c_size);
  run_info.AddTilingData(per_core_size);
  run_info.AddTilingData(per_core_loop_cnt);
  run_info.AddTilingData(per_core_left_size);
  run_info.AddTilingData(last_core_size);
  run_info.AddTilingData(last_core_loop_cnt);
  run_info.AddTilingData(last_core_left_size);
  run_info.AddTilingData(x_size);
  run_info.AddTilingData(target_size);
  run_info.AddTilingData(weight_size);

  // block_dim, core num used in tik op
  // workspace, null for tik op
  run_info.SetBlockDim(need_core_num);

  PROFILING_TILING_END();
  OP_LOGD(op_type.c_str(), "NLLLossTiling run success.");

  return true;
}

// register tiling inferface of the Nllloss op
REGISTER_OP_TILING_FUNC_BUFFERED_V2(NLLLoss, NLLLossTiling);

}  // namespace optiling
