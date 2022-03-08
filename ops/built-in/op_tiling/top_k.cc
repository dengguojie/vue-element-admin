/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file top_k.cc
 * \brief dynamic shape tiling of top_k
 */
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"

namespace optiling {
const std::string TOPK_OP_TYPE = "Topk";
const int32_t BATCH_COLS_PADDING_BASE = 1024;

struct TilingParams {
  int32_t need_core_num_input_scalar;
  int32_t row_num_input_scalar;
  int32_t col_num_input_scalar;
  int32_t k_num_input_scalar;
  int32_t loops_time_input_scalar;
  int32_t batch_num_input_scalar;
  int32_t rows_per_core_num_input_scalar;
  int32_t turning_num_input_scalar;
};

struct opInfo {
  int32_t core_num;
  int32_t k_num;
  int32_t max_k;
  int32_t batch_cols_padding;
  int32_t ub_size;
};

bool TopkParseFunc(const std::string &op_type, const nlohmann::json &compile_info, opInfo &compile_value) {
  using namespace nlohmann;
  OP_TILING_CHECK(compile_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info_json is null"),
                  return false);

  const auto &all_vars = compile_info["vars"];
  // core num
  OP_TILING_CHECK(!GetCompileValue(all_vars, "core_num", compile_value.core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TopkParseFunc get core_num error"), return false);
  OP_TILING_CHECK(!GetCompileValue(all_vars, "k_num", compile_value.k_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TopkParseFunc get k_num error"), return false);
  // k num
  OP_TILING_CHECK(!GetCompileValue(all_vars, "max_k", compile_value.max_k),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TopkParseFunc get max_k error"), return false);
  OP_TILING_CHECK(!GetCompileValue(all_vars, "k_num", compile_value.k_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TopkParseFunc get k_num error"), return false);

  // batch_cols_padding num
  OP_TILING_CHECK(!GetCompileValue(all_vars, "batch_cols_padding", compile_value.batch_cols_padding),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TopkParseFunc get batch_cols_padding error"), return false);

  // ub size
  OP_TILING_CHECK(!GetCompileValue(all_vars, "ub_size", compile_value.ub_size),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "TopkParseFunc get ub_size error"), return false);
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

void WriteTilingParams(const TilingParams &params, utils::OpRunInfo &run_info) {
  run_info.AddTilingData(params.need_core_num_input_scalar);
  run_info.AddTilingData(params.row_num_input_scalar);
  run_info.AddTilingData(params.col_num_input_scalar);
  run_info.AddTilingData(params.k_num_input_scalar);
  run_info.AddTilingData(params.loops_time_input_scalar);
  run_info.AddTilingData(params.batch_num_input_scalar);
  run_info.AddTilingData(params.rows_per_core_num_input_scalar);
  run_info.AddTilingData(params.turning_num_input_scalar);
}

void PrintTilingParams(const std::string &op_type, const TilingParams &params) {
  GELOGD("op [%s] : params.need_core_num_input_scalar=%d", op_type.c_str(), params.need_core_num_input_scalar);
  GELOGD("op [%s] : params.row_num_input_scalar=%d", op_type.c_str(), params.row_num_input_scalar);
  GELOGD("op [%s] : params.col_num_input_scalar=%d", op_type.c_str(), params.col_num_input_scalar);
  GELOGD("op [%s] : params.k_num_input_scalar=%d", op_type.c_str(), params.k_num_input_scalar);
  GELOGD("op [%s] : params.loops_time_input_scalar=%d", op_type.c_str(), params.loops_time_input_scalar);
  GELOGD("op [%s] : params.batch_num_input_scalar=%d", op_type.c_str(), params.batch_num_input_scalar);
  GELOGD("op [%s] : params.rows_per_core_num_input_scalar=%d", op_type.c_str(), params.rows_per_core_num_input_scalar);
  GELOGD("op [%s] : params.turning_num_input_scalar=%d", op_type.c_str(), params.turning_num_input_scalar);
}

int32_t GetLoopTimes(int32_t cols) {
  int32_t level = 0;
  // region proposal must has 16 fp16
  int32_t regions = (cols + 15) / 16;
  if (regions <= 1) {
    return level;
  }
  while (true) {
    level += 1;
    // 4 regions merge, 3 means ceil
    regions = (regions + 3) / 4;
    if (regions <= 1) {
      break;
    }
  }
  return level;
}

bool GetTopkCompileParams(const std::string &op_type, const opInfo &op_compile_info_json, int32_t &core_num,
                          int32_t &k_num, int32_t &batch_cols_padding, int32_t &ub_size, int32_t &max_k) {
  // core num
  core_num = op_compile_info_json.core_num;
  k_num = op_compile_info_json.k_num;
  max_k = op_compile_info_json.max_k;

  // batch_cols_padding num
  batch_cols_padding = op_compile_info_json.batch_cols_padding;

  // ub size
  ub_size = op_compile_info_json.ub_size;
  GELOGD(
      "op [%s] : GetTopkCompileParams, core_num[%d], k_num[%d], "
      "batch_cols_padding[%d], ub_size[%d].",
      TOPK_OP_TYPE.c_str(), core_num, k_num, batch_cols_padding, ub_size);
  return true;
}

bool GetConstValue(const ge::Operator &paras, const int32_t &idx, const ge::DataType &dtype, vector<int64_t> &values) {
  values.clear();
  if (!ops::GetConstIntData(paras, idx, values)) {
    return false;
  }

  return true;
}

void TopkTilingBase(const int32_t row, const int32_t col, const int32_t batch_cols_padding, const int32_t k_num,
                    const int32_t core_max, const std::string &op_type, utils::OpRunInfo &run_info) {
  int32_t rows_per_core;
  int32_t turning;
  int32_t need_core;
  int32_t batch;
  if (row <= core_max) {
    rows_per_core = 1;
    need_core = row;
    batch = 1;
    turning = core_max;
  } else {
    need_core = core_max;
    // cols_padding must be aligned 16
    int32_t cols_padding = ((col + 15) / 16) * 16;
    int32_t remain = row % core_max;
    // need +1 in op for mode2
    rows_per_core = (row + core_max - 1) / core_max;
    if (col <= batch_cols_padding / BATCH_COLS_PADDING_BASE * BATCH_COLS_PADDING_BASE) {
      batch = batch_cols_padding / cols_padding;
    } else {
      batch = 1;
    }
    turning = remain;
    if (remain == 0) {
      turning = core_max;
    }
  }
  // 16 is a limitation for k, 16 is used for data align
  if (k_num < 16 && k_num > 0) {
    // when k is not const, k_num use 0 as default value, and need to check k
    // value in py to update these two scalars.
    need_core = 1;
    rows_per_core = row;
  }
  int32_t loops = GetLoopTimes(col);
  TilingParams params{need_core, row, col, k_num, loops, batch, rows_per_core, turning};

  // write tiling params to run_info
  WriteTilingParams(params, run_info);
  // cout tiling params
  PrintTilingParams(op_type, params);

  run_info.SetBlockDim(need_core);
}

bool TopkTiling(const std::string &op_type, const ge::Operator &op_paras, const opInfo &op_compile_info_json,
                utils::OpRunInfo &run_info) {
  GELOGI("TopkTiling running.");
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  std::vector<int64_t> input_shape = operator_info->MutableInputDesc(0)->MutableShape().GetDims();
  int32_t input_dims = input_shape.size();
  int32_t row = 1;
  for (int i = 0; i < input_dims - 1; i++) {
    row = row * input_shape[i];
  }
  int32_t col = input_shape[input_dims - 1];
  int32_t core_max = 0;
  int32_t k_num = 0;
  int32_t batch_cols_padding = 0;
  int32_t ub_size = 0;
  int32_t max_k = 0;

  bool flag = GetTopkCompileParams(op_type, op_compile_info_json, core_max, k_num, batch_cols_padding, ub_size, max_k);
  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetTopkCompileParams failed.");
    return false;
  }
  GELOGI("op[%s] GetTopkCompileParams success.", op_type.c_str());

  std::vector<int64_t> values;
  GetConstValue(op_paras, 1, op_paras.GetInputDesc(1).GetDataType(), values);
  if (values.size() != 0) {
    int32_t k_element = values[0];
    if (k_element > max_k) {
      OP_LOGW(op_type, "AICORE does not support k[%d] > max_k[%d], should be transfered to AICPU.", k_element, max_k);
      return false;
    }
  }
  TopkTilingBase(row, col, batch_cols_padding, k_num, core_max, op_type, run_info);
  GELOGI("Topk_tiling end.");
  return true;
}
REGISTER_OP_TILING_V3_CUSTOM(TopKD, TopkTiling, TopkParseFunc, opInfo);
REGISTER_OP_TILING_V3_CUSTOM(TopKV2D, TopkTiling, TopkParseFunc, opInfo);
}  // namespace optiling
