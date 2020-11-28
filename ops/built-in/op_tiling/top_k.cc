/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/error_util.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {
const std::string TOPK_OP_TYPE = "Topk";

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

void WriteTilingParams(const TilingParams& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.need_core_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.row_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.col_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.k_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.loops_time_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.batch_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.rows_per_core_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.turning_num_input_scalar);
}

void PrintTilingParams(const std::string& op_type, const TilingParams& params) {
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
  int32_t regions = (cols + 15) / 16;
  if (regions <= 1) {
    return level;
  }
  while (true) {
    level += 1;
    regions = (regions + 3) / 4;
    if (regions <= 1) {
      break;
    }
  }
  return level;
}

bool GetTopkCompileParams(const std::string& op_type, const nlohmann::json& op_compile_info_json, int32_t& core_num,
                          int32_t& k_num, int32_t& batch_cols_padding, int32_t& ub_size) {
  using namespace nlohmann;
  if (op_compile_info_json == nullptr) {
    ge::OpsGetCompileParamsErrReport("Topk", "op_compile_info_json");
    OP_LOGE(op_type.c_str(), "op_compile_info_json is null");
    return false;
  }

  const auto& all_vars = op_compile_info_json["vars"];
  // core num
  if (all_vars.count("core_num") == 0) {
    ge::OpsGetCompileParamsErrReport("Topk", "core_num");
    OP_LOGE(op_type.c_str(), "core_num is null");
    return false;
  }
  core_num = all_vars["core_num"].get<std::int32_t>();

  // k num
  if (all_vars.count("k_num") == 0) {
    ge::OpsGetCompileParamsErrReport("Topk", "k_num");
    OP_LOGE(op_type.c_str(), "k_num is null");
    return false;
  }
  k_num = all_vars["k_num"].get<std::int32_t>();

  // batch_cols_padding num
  if (all_vars.count("batch_cols_padding") == 0) {
    ge::OpsGetCompileParamsErrReport("Topk", "batch_cols_padding");
    OP_LOGE(op_type.c_str(), "batch_cols_padding is null");
    return false;
  }
  batch_cols_padding = all_vars["batch_cols_padding"].get<std::int32_t>();

  // ub size
  if (all_vars.count("ub_size") == 0) {
    ge::OpsGetCompileParamsErrReport("Topk", "ub_size");
    OP_LOGE(op_type.c_str(), "ub_size is null");
    return false;
  }
  ub_size = all_vars["ub_size"].get<std::int32_t>();
  GELOGD("op [%s] : GetTopkCompileParams, core_num[%d], k_num[%d], batch_cols_padding[%d], ub_size[%d].",
         TOPK_OP_TYPE.c_str(), core_num, k_num, batch_cols_padding, ub_size);
  return true;
}

bool TopkTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_compile_info_json,
                OpRunInfo& run_info) {
  GELOGI("TopkTiling running.");
  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  int32_t input_dims = input_shape.size();
  int32_t row = 1;
  for (int i = 0; i < input_dims - 1; i++) {
    row = row * input_shape[i];
  }
  int32_t col = input_shape[input_dims - 1];
  int32_t need_core = 0;
  int32_t batch = 0;
  int32_t core_max = 0;
  int32_t k_num = 0;
  int32_t batch_cols_padding = 0;
  int32_t ub_size = 0;

  bool flag = GetTopkCompileParams(op_type, op_compile_info_json, core_max, k_num, batch_cols_padding, ub_size);
  if (!flag) {
    OP_LOGE("op[%s] GetTopkCompileParams failed.", op_type.c_str());
    return false;
  }
  GELOGI("op[%s] GetTopkCompileParams success.", op_type.c_str());

  int32_t rows_per_core = 0;
  int32_t turning = 0;
  int32_t cols_padding = 0;
  int32_t remain = 0;
  if (row < core_max) {
    rows_per_core = 1;
    need_core = row;
    batch = 1;
    turning = core_max;
  } else {
    need_core = core_max;
    cols_padding = ((col + 15) / 16) * 16;
    remain = row % core_max;
    // need +1 in op for mode2
    rows_per_core = (row + core_max - 1) / core_max;
    if (col <= 4096) {
      batch = batch_cols_padding / cols_padding;
    } else {
      batch = 1;
    }
    turning = remain;
    if (remain == 0) {
      turning = core_max;
    }
  }
  if (k_num < 16) {
    need_core = 1;
    rows_per_core = row;
  }
  int32_t loops = GetLoopTimes(col);
  TilingParams params{need_core, row, col, k_num, loops, batch, rows_per_core, turning};

  // write tiling params to run_info
  WriteTilingParams(params, run_info);
  // cout tiling params
  PrintTilingParams(op_type, params);

  run_info.block_dim = need_core;

  GELOGI("Topk_tiling end.");
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(TopKD, TopkTiling);
}  // namespace optiling