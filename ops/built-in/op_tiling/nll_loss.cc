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
 * \file nll_loss.cc
 * \brief dynamic Nllloss tiling
 */
#include <string>

#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"

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

bool CheckTensorShape(const std::string& op_type, const TeOpParas& op_paras) {
  if (op_paras.inputs.empty() || op_paras.inputs[0].tensor.empty() || op_paras.inputs[1].tensor.empty() ||
      op_paras.inputs[2].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: input shape error.");
    return false;
  }

  if (op_paras.outputs.empty() || op_paras.outputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: output shape error.");
    return false;
  }

  std::vector<int64_t> x_shape = op_paras.inputs[0].tensor[0].shape;
  std::vector<int64_t> target_shape = op_paras.inputs[1].tensor[0].shape;
  std::vector<int64_t> weight_shape = op_paras.inputs[2].tensor[0].shape;
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
                         int64_t& last_core_left_size, int64_t& unit_size, int64_t& min_aligned, bool is_left_data) {
  int64_t ub_part = 5;
  int64_t data_one_block = GetFloorDiv(BLOCK_SIZE, bytes);
  int64_t ub_x_size = GetFloorDiv(ub_size, ub_part);

  // dma move 32byte unit
  int64_t ub_max_line = GetFloorDiv(ub_x_size - GetMod(ub_x_size, data_one_block), unit_size);

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

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/attrs of the op
 * @param [in] op_info: compile stage generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool NLLLossTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                   OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "NLLLossTiling start running.");

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
  std::string reduction;
  int64_t unit_size = 1;
  int64_t min_aligned = 1;
  bool is_left_data = true;

  if (!GetCompileParams(op_type, op_info, core_num, ub_size, reduction)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: GetCompileParams error.");
    return false;
  }

  if (!CheckTensorShape(op_type, op_paras)) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "NLLLossTiling: CheckTensorShape error.");
    return false;
  }

  std::vector<int64_t> x_shape = op_paras.inputs[0].tensor[0].shape;

  // input x is 1D
  if (x_shape.size() == 1) {
    n_size = 1;
  } else {
    n_size = x_shape[0];
  }

  c_size = x_shape.back();

  int64_t ub_part = 5;
  int64_t ub_usable_size = GetFloorDiv(ub_size, ub_part);
  int64_t data_one_block = GetFloorDiv(BLOCK_SIZE, bytes);
  int64_t c_real_move_size = GetCeilDiv(c_size, data_one_block) * data_one_block;

  // c_size can move into ub
  if (ub_usable_size >= c_real_move_size) {
    tiling_mode = 1;
    unit_size = c_size;
  } else {
    tiling_mode = 2;
  }

  if (reduction == "none") {
    min_aligned = GetFloorDiv(BLOCK_SIZE, bytes);
  }

  NLLLossCommonTiling(bytes, core_num, ub_size, need_core_num, n_size, c_size, per_core_size, per_core_loop_cnt,
                      per_core_left_size, last_core_size, last_core_loop_cnt, last_core_left_size, unit_size,
                      min_aligned, is_left_data);

  OP_LOGD(op_type.c_str(), "NLLLossTiling: n_size=%lld, c_size=%lld", n_size, c_size);
  OP_LOGD(op_type.c_str(), "NLLLossTiling: tiling_mode=%lld, need_core_num=%lld", tiling_mode, need_core_num);
  OP_LOGD(op_type.c_str(), "NLLLossTiling: per_core_size=%lld, per_core_loop_cnt=%lld, per_core_left_size=%lld",
          per_core_size, per_core_loop_cnt, per_core_left_size);
  OP_LOGD(op_type.c_str(), "NLLLossTiling: last_core_size=%lld, last_core_loop_cnt=%lld, last_core_left_size=%lld",
          last_core_size, last_core_loop_cnt, last_core_left_size);

  // set tiling data
  ByteBufferPut(run_info.tiling_data, tiling_mode);
  ByteBufferPut(run_info.tiling_data, need_core_num);
  ByteBufferPut(run_info.tiling_data, n_size);
  ByteBufferPut(run_info.tiling_data, c_size);
  ByteBufferPut(run_info.tiling_data, per_core_size);
  ByteBufferPut(run_info.tiling_data, per_core_loop_cnt);
  ByteBufferPut(run_info.tiling_data, per_core_left_size);
  ByteBufferPut(run_info.tiling_data, last_core_size);
  ByteBufferPut(run_info.tiling_data, last_core_loop_cnt);
  ByteBufferPut(run_info.tiling_data, last_core_left_size);

  // block_dim, core num used in tik op
  run_info.block_dim = need_core_num;

  // workspace, null for tik op
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;

  OP_LOGD(op_type.c_str(), "NLLLossTiling run success.");

  return true;
}

// register tiling inferface of the Nllloss op
REGISTER_OP_TILING_FUNC_BUFFERED(NLLLoss, NLLLossTiling);

}  // namespace optiling
