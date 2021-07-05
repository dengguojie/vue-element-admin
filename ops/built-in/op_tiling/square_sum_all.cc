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
 * \file square_sum_all.cc
 * \brief
 */
#include <math.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"

namespace optiling {

// Each core processing data num greater than the size we can get better performance from experience
const int64_t MINIMUM_DATA_NUM_EACH_CORE = 1024;
const int64_t VECTOR_PROCESS_BYTES = 256;

static int64_t GetCeilInt(int64_t value1, int64_t value2) { return (int64_t)(value1 + value2 - 1) / value2; }

struct SquareSumAllTilingParams {
  int32_t need_core_num_input_scalar = 0;
  int32_t data_num_each_core = 0;
  int32_t process_times_per_core = 0;
  int32_t process_times_remain_core = 0;
  int32_t every_process_data_num_per_core = 0;
  int32_t every_process_data_num_remain_core = 0;
  int32_t tail_num_per_core = 0;
  int32_t tail_num_remain_core = 0;
  int32_t reduce_sum_loop_per_core = 0;
  int32_t reduce_sum_loop_tail_per_core = 0;
  int32_t reduce_sum_loop_remain_core = 0;
  int32_t reduce_sum_loop_tail_remain_core = 0;
  int32_t burst_len_per_core = 0;
  int32_t burst_len_tail_per_core = 0;
  int32_t burst_len_remain_core = 0;
  int32_t burst_len_tail_remain_core = 0;
};

void SetRunParams(const SquareSumAllTilingParams& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.need_core_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.data_num_each_core);
  ByteBufferPut(run_info.tiling_data, params.process_times_per_core);
  ByteBufferPut(run_info.tiling_data, params.process_times_remain_core);
  ByteBufferPut(run_info.tiling_data, params.every_process_data_num_per_core);
  ByteBufferPut(run_info.tiling_data, params.every_process_data_num_remain_core);
  ByteBufferPut(run_info.tiling_data, params.tail_num_per_core);
  ByteBufferPut(run_info.tiling_data, params.tail_num_remain_core);
  ByteBufferPut(run_info.tiling_data, params.reduce_sum_loop_per_core);
  ByteBufferPut(run_info.tiling_data, params.reduce_sum_loop_tail_per_core);
  ByteBufferPut(run_info.tiling_data, params.reduce_sum_loop_remain_core);
  ByteBufferPut(run_info.tiling_data, params.reduce_sum_loop_tail_remain_core);
  ByteBufferPut(run_info.tiling_data, params.burst_len_per_core);
  ByteBufferPut(run_info.tiling_data, params.burst_len_tail_per_core);
  ByteBufferPut(run_info.tiling_data, params.burst_len_remain_core);
  ByteBufferPut(run_info.tiling_data, params.burst_len_tail_remain_core);
}

void PrintRunParams(const SquareSumAllTilingParams& params) {
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : needCoreNum=%d.", params.need_core_num_input_scalar);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : dataNumEachCore=%d.", params.data_num_each_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : processTimesPerCore=%d.", params.process_times_per_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : processTimesRemainCore=%d.", params.process_times_remain_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : everyProcDataPerCore=%d.", params.every_process_data_num_per_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : everyProcDataRemainCore=%d.",
          params.every_process_data_num_remain_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : tailNumPerCore=%d.", params.tail_num_per_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : tailNumRemainCore=%d.", params.tail_num_remain_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : loopPerCore=%d.", params.reduce_sum_loop_per_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : loopTailPerCore=%d.", params.reduce_sum_loop_tail_per_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : loopRemainCore=%d.", params.reduce_sum_loop_remain_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : loopTailRemainCore=%d.", params.reduce_sum_loop_tail_remain_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : burstLenPerCore=%d.", params.burst_len_per_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : burstLenTailPerCore=%d.", params.burst_len_tail_per_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : burstLenRemainCore=%d.", params.burst_len_remain_core);
  OP_LOGD("SquareSumAll", "op [SquareSumAllTiling] : burstLenTailRemainCore=%d.", params.burst_len_tail_remain_core);
}

int32_t CalcReduceSumLoop(int32_t& calc_num, int64_t& data_each_block, int64_t& dtype_bytes_size) {
  int64_t remain_num = calc_num;
  int64_t align_value = data_each_block * 2;
  int32_t loop_num = 0;
  while (remain_num > VECTOR_PROCESS_BYTES / dtype_bytes_size) {
    remain_num = ((remain_num / align_value) * align_value) / 2;
    loop_num++;
    if (remain_num <= VECTOR_PROCESS_BYTES / dtype_bytes_size) break;
  }
  return loop_num;
}

void CalcProcessParams(int64_t& ub_size, int64_t& dtype_bytes_size, int64_t& data_each_block,
                       int32_t& data_num_remain_core, SquareSumAllTilingParams& params) {
  int64_t ub_max_num = ub_size / dtype_bytes_size;
  int32_t process_data_num_each_core = params.data_num_each_core;
  int32_t every_process_data_num = (process_data_num_each_core > (ub_max_num / 2)) ? (ub_max_num / 2) :
                                   process_data_num_each_core;

  params.process_times_per_core = process_data_num_each_core / every_process_data_num;
  params.every_process_data_num_per_core = every_process_data_num;
  params.tail_num_per_core = process_data_num_each_core % every_process_data_num;
  params.reduce_sum_loop_per_core = CalcReduceSumLoop(every_process_data_num, data_each_block, dtype_bytes_size);
  params.reduce_sum_loop_tail_per_core = CalcReduceSumLoop(params.tail_num_per_core, data_each_block, dtype_bytes_size);
  params.burst_len_per_core = GetCeilInt(every_process_data_num, data_each_block);
  params.burst_len_tail_per_core = GetCeilInt(params.tail_num_per_core, data_each_block);

  // tail core
  int32_t process_data_extern_num = process_data_num_each_core + data_num_remain_core;
  every_process_data_num = (process_data_extern_num > (ub_max_num / 2)) ? (ub_max_num / 2) :
                           process_data_extern_num;

  params.process_times_remain_core = process_data_extern_num / every_process_data_num;
  params.every_process_data_num_remain_core = every_process_data_num;
  params.tail_num_remain_core = process_data_extern_num % every_process_data_num;
  params.reduce_sum_loop_remain_core = CalcReduceSumLoop(every_process_data_num, data_each_block, dtype_bytes_size);
  params.reduce_sum_loop_tail_remain_core = CalcReduceSumLoop(params.tail_num_remain_core, data_each_block,
                                                              dtype_bytes_size);
  params.burst_len_remain_core = GetCeilInt(every_process_data_num, data_each_block);
  params.burst_len_tail_remain_core = GetCeilInt(params.tail_num_remain_core, data_each_block);
}

static bool GetSquareSumAllCompileParams(const nlohmann::json& op_info, int64_t& ub_size, int64_t& core_num,
                                         int64_t& data_each_block, int64_t& dtype_bytes_size) {
  using namespace nlohmann;
  auto all_vars = op_info["vars"];

  if (all_vars.count("ub_size") == 0) {
    OP_LOGE("op [SquareSumAll]: GetSquareSumAllCompileParams, get ub_size error");
    return false;
  }
  ub_size = all_vars["ub_size"].get<std::int64_t>();

  if (all_vars.count("core_num") == 0) {
    OP_LOGE("op [SquareSumAll]: GetSquareSumAllCompileParams, get core_num error");
    return false;
  }
  core_num = all_vars["core_num"].get<std::int64_t>();

  if (all_vars.count("data_each_block") == 0) {
    OP_LOGE("op [SquareSumAll]: GetSquareSumAllCompileParams, get data_each_block error");
    return false;
  }
  data_each_block = all_vars["data_each_block"].get<std::int64_t>();

  if (all_vars.count("dtype_bytes_size") == 0) {
    OP_LOGE("op [SquareSumAll]: GetSquareSumAllCompileParams, get dtype_bytes_size error");
    return false;
  }
  dtype_bytes_size = all_vars["dtype_bytes_size"].get<std::int64_t>();

  return true;
}

// tiling function
bool SquareSumAllTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                        OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "op[SquareSumAllTiling] tiling running.");
  CHECK(op_paras.inputs.size() == 2, "op [%s] : op_paras.inputs.size() != 2", op_type.c_str());
  CHECK(!op_paras.inputs[0].tensor.empty(), "op [%s] : op_paras.inputs[0].tensor cannot be empty", op_type.c_str());
  CHECK(!op_paras.inputs[1].tensor.empty(), "op [%s] : op_paras.inputs[1].tensor cannot be empty", op_type.c_str());

  const std::vector<int64_t>& input_x_shape = op_paras.inputs[0].tensor[0].shape;
  const std::vector<int64_t>& input_y_shape = op_paras.inputs[1].tensor[0].shape;

  const std::string input_x_dtype = op_paras.inputs[0].tensor[0].dtype;
  const std::string input_y_dtype = op_paras.inputs[1].tensor[0].dtype;
  CHECK(input_x_dtype == input_y_dtype, "op [%s] : op_paras.inputs's dtype must be the same", op_type.c_str());

  // get compile info
  int64_t ub_size = 0;
  int64_t device_core_num = 0;
  int64_t data_each_block = 0;
  int64_t dtype_bytes_size = 0;
  if (!GetSquareSumAllCompileParams(op_info, ub_size, device_core_num, data_each_block, dtype_bytes_size)) {
    OP_LOGE(op_type.c_str(), "get compile info from nlohmann json failed.");
    return false;
  }

  int64_t input_x_num;
  if (input_x_shape.size() == 0) {
    input_x_num = 1;
  } else {
    input_x_num = std::accumulate(input_x_shape.begin(), input_x_shape.end(), 1, std::multiplies<int64_t>());
  }

  int64_t core_num = 0;
  if (input_x_num < data_each_block) {
    core_num = 1;
  } else {
    int64_t temp_num = GetCeilInt(input_x_num, MINIMUM_DATA_NUM_EACH_CORE);
    core_num = (temp_num < 32) ? temp_num : device_core_num;
  }

  SquareSumAllTilingParams params;
  params.need_core_num_input_scalar = core_num;
  params.data_num_each_core = input_x_num / core_num;
  int32_t data_num_remain_core = input_x_num % core_num;
  CalcProcessParams(ub_size, dtype_bytes_size, data_each_block, data_num_remain_core, params);
  SetRunParams(params, run_info);

  PrintRunParams(params);
  run_info.block_dim = core_num;
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;

  OP_LOGD(op_type.c_str(), "op[SquareSumAllTiling] tiling run success.");
  return true;
}

// register tiling interface of the SquareSumAll op.
REGISTER_OP_TILING_FUNC_BUFFERED(SquareSumAll, SquareSumAllTiling);
}  // namespace optiling
