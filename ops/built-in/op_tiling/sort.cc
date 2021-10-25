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
 * \file sort.cc
 * \brief
 */
#include <string>
#include <math.h>

#include <nlohmann/json.hpp>
#include "op_tiling.h"

#include "../op_proto/util/error_util.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling
{

  const std::string SORT_OP_TYPE = "Sort";
  //  tiling mode 1
  const int64_t TILING_MODE_1 = 1;
  // tiling mode 1
  const int64_t TILING_MODE_2 = 2;

  const int32_t COL_PER_PART = 3600;

  const int64_t WORKSPACE_DIM = 3;
  const int64_t WORKSPACE_SIZE = 1073741824;

  struct SortTilingParams
  {
    int32_t tiling_mode_scalar;
    int32_t used_core_num_scalar;
    int32_t round_scalar;
    int32_t dim_num_scalar;
    int32_t dim_num_align_scalar;
    int32_t loop_times_scalar;
    int32_t batch_num_per_core_scalar;
    int32_t batch_tail_scalar;
    int32_t col_tail_loop_scalar;
    int32_t col_block_padding_scalar;
  };

  int32_t GetSortLoopTimes(int32_t cols)
  {
    OP_LOGD("GetSortLoopTimes is running");
    int32_t level = 0;
    int32_t regions = (cols + 15) / 16;
    if (regions <= 1)
    {
      return level + 1;
    }
    while (regions > 1)
    {
      level += 1;
      regions = (regions + 3) / 4;
      if (regions <= 1)
      {
        break;
      }
    }
    return level + 1;
  }

  // set tiling para
  void WriteTilingParams(const SortTilingParams& params, OpRunInfo& run_info) {
    OP_LOGD("WriteTilingParams is running");
    ByteBufferPut(run_info.tiling_data, params.tiling_mode_scalar);
    ByteBufferPut(run_info.tiling_data, params.used_core_num_scalar);
    ByteBufferPut(run_info.tiling_data, params.round_scalar);
    ByteBufferPut(run_info.tiling_data, params.dim_num_scalar);
    ByteBufferPut(run_info.tiling_data, params.dim_num_align_scalar);
    ByteBufferPut(run_info.tiling_data, params.loop_times_scalar);
    ByteBufferPut(run_info.tiling_data, params.batch_num_per_core_scalar);
    ByteBufferPut(run_info.tiling_data, params.batch_tail_scalar);
    ByteBufferPut(run_info.tiling_data, params.col_tail_loop_scalar);
    ByteBufferPut(run_info.tiling_data, params.col_block_padding_scalar);
  }

  // print tiling para
  void PrintTilingParams(const std::string& op_type, const SortTilingParams& params) {
    OP_LOGD("PrintTilingParams is running");

    OP_LOGD("op [%s] : params.tiling_mode_scalar=%d", op_type.c_str(), params.tiling_mode_scalar);
    OP_LOGD("op [%s] : params.used_core_num_scalar=%d", op_type.c_str(), params.used_core_num_scalar);
    OP_LOGD("op [%s] : params.round_scalar=%d", op_type.c_str(), params.round_scalar);
    OP_LOGD("op [%s] : params.dim_num_scalar=%d", op_type.c_str(), params.dim_num_scalar);
    OP_LOGD("op [%s] : params.dim_num_align_scalar=%d", op_type.c_str(), params.dim_num_align_scalar);
    OP_LOGD("op [%s] : params.loop_times_scalar=%d", op_type.c_str(), params.loop_times_scalar);
    OP_LOGD("op [%s] : params.batch_num_per_core_scalar=%d", op_type.c_str(), params.batch_num_per_core_scalar);
    OP_LOGD("op [%s] : params.batch_tail_scalar=%d", op_type.c_str(), params.batch_tail_scalar);
    OP_LOGD("op [%s] : params.col_tail_loop_scalar=%d", op_type.c_str(), params.col_tail_loop_scalar);
    OP_LOGD("op [%s] : params.col_block_padding_scalar=%d", op_type.c_str(), params.col_block_padding_scalar);
  }

  bool GetSortCompileParams(const std::string &op_type, const nlohmann::json &op_compile_info_json, int32_t &core_num,
                            int32_t &ub_size)
  {
    OP_LOGD("GetSortCompileParams is running");
    using namespace nlohmann;
    if (op_compile_info_json == nullptr)
    {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info_json is null");
      return false;
    }

    const auto &all_vars = op_compile_info_json["vars"];
    // core num
    if (all_vars.count("core_num") == 0)
    {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "core_num is null");
      return false;
    }
    core_num = all_vars["core_num"].get<std::int32_t>();
    // ub size
    if (all_vars.count("ub_size") == 0)
    {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_size is null");
      return false;
    }
    ub_size = all_vars["ub_size"].get<std::int32_t>();
    OP_LOGD("op [%s] : GetTopkCompileParams, core_num[%d], ub_size[%d].",
            SORT_OP_TYPE.c_str(), core_num, ub_size);
    return true;
  }

  bool SortTiling(const std::string &op_type, const TeOpParas &op_paras, const nlohmann::json &op_compile_info,
                  OpRunInfo &run_info)
  {
    using namespace ge;
    // tiling para
    OP_LOGD("SortTiling is running");
    int32_t tiling_mode = 1;
    int32_t need_core = 0;
    int32_t row = 1;
    int32_t col = 1;
    int32_t cols_padding = 0;
    int32_t loop_times = 1;
    int32_t rows_per_core = 0;
    int32_t remain = 0;
    int32_t col_tail_loop = 0;
    int32_t col_block_padding = 0;

    // other para
    int32_t core_max = 0;
    int32_t ub_size = 0;

    // get row and col
    std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
    int32_t input_dims = input_shape.size();
    col = input_shape[input_dims - 1];
    for (int i = 0; i < input_dims - 1; i++) {
      row = row * input_shape[i];
    }
    //  get loop time
    loop_times = GetSortLoopTimes(col);

    // get sort compile para
    bool flag = GetSortCompileParams(op_type, op_compile_info, core_max, ub_size);
    if (!flag)
    {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetSortCompileParams failed.");
      return false;
    }
    OP_LOGI("op[%s] GetSortCompileParams success.", op_type.c_str());

    cols_padding = ((col + 15) / 16) * 16;
    col_block_padding = ((col + 2047) / 2048) * 2048;
    if (col > 2048) {
       tiling_mode = 2;
       col_tail_loop = GetSortLoopTimes(cols_padding - COL_PER_PART);
    }
   // get other tiling para
    if (row < core_max) {
      rows_per_core = 0;
      need_core = row;
      remain = row;
    } else {
      need_core = core_max;
      rows_per_core = row / core_max;
      remain = row % core_max;
    }
    SortTilingParams params{tiling_mode, need_core, row, col, cols_padding, loop_times, rows_per_core, remain, col_tail_loop, col_block_padding};
    // write tiling params to run_info
    WriteTilingParams(params, run_info);
    // cout tiling params
    PrintTilingParams(op_type, params);
    run_info.block_dim = need_core;
    // workspace for tik op
    std::vector<int64_t> workspace(WORKSPACE_DIM,WORKSPACE_SIZE);
    run_info.workspaces = workspace;
    OP_LOGI("Sort Tiling end.");
    return true;
  }

  REGISTER_OP_TILING_FUNC_BUFFERED(Sort, SortTiling);
} // namespace optiling
