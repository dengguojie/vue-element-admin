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
 * \file flatten.cc
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
using namespace ge;

static int64_t GetCeilInt(int64_t value1, int64_t value2) { 
  OP_TILING_CHECK(value2 == 0,
    VECTOR_INNER_ERR_REPORT_TILIING("Flattern", "In the GetCeilInt function, the divisor is 0"),
    return value1);
  return (int64_t)(value1 + value2 - 1) / value2; 
}

bool GetFlattenCompileParams(const nlohmann::json& op_compile_info, int64_t& core_num, int64_t& ub_size,
                             int64_t& block_size) {
  using namespace nlohmann;
  auto all_vars = op_compile_info["vars"];

  if (all_vars.count("core_num") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("Flattern", "GetFlattenCompileParams, get core_num error");
    return false;
  }
  core_num = all_vars["core_num"].get<std::int64_t>();

  if (all_vars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("Flattern", "GetFlattenCompileParams, get ub_size error");
    return false;
  }
  ub_size = all_vars["ub_size"].get<std::int64_t>();

  if (all_vars.count("block_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("Flattern", "GetFlattenCompileParams, get block_size error");
    return false;
  }
  block_size = all_vars["block_size"].get<std::int64_t>();

  return true;
}

bool FlattenTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_info,
                   OpRunInfo& run_info) {
  OP_LOGI(op_type.c_str(), "FlattenTiling running.");
  OP_TILING_CHECK(op_paras.inputs.empty(), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_paras.inputs cannot be empty."), return false);

  const std::vector<int64_t> src_shape = op_paras.inputs[0].tensor[0].shape;
  const std::vector<int64_t> dst_shape = op_paras.outputs[0].tensor[0].shape;

  int64_t data_size;
  if (src_shape.size() == 0) {
    data_size = 1;
  } else {
    data_size = std::accumulate(src_shape.begin(), src_shape.end(), 1, std::multiplies<int64_t>());
  }

  int64_t data_dst_size;
  if (dst_shape.size() == 0) {
    data_dst_size = 1;
  } else {
    data_dst_size = std::accumulate(dst_shape.begin(), dst_shape.end(), 1, std::multiplies<int64_t>());
  }

  if (data_size != data_dst_size) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The size of src and dst is not equal, can not use fuc flatten");
    return false;
  }

  int64_t core_num = 0;
  int64_t ub_size = 0;
  int64_t block_size = 0;
  if (!GetFlattenCompileParams(op_info, core_num, ub_size, block_size)) {
    return false;
  }

  int64_t core_number = core_num;
  if (data_size < block_size) {
    core_number = 1;
  }
  int64_t core_data = GetCeilInt(data_size, core_number);
  core_data = GetCeilInt(core_data, block_size) * block_size;
  int64_t core_used = GetCeilInt(data_size, core_data);
  int64_t core_last = core_data;
  if (data_size % core_data != 0) {
    core_last = data_size % core_data;
  }

  int64_t copy_loop = core_data / ub_size;
  int64_t copy_tail = core_data % ub_size;
  int64_t last_copy_loop = core_last / ub_size;
  int64_t last_copy_tail = core_last % ub_size;

  OP_LOGD(op_type.c_str(),
          "CompileParams, core_data = %d, core_used = %d, copy_loop = %d, copy_tail = %d, last_copy_loop = %d, "
          "last_copy_tail = %d",
          core_data, core_used, copy_loop, copy_tail, last_copy_loop, last_copy_tail);

  ByteBufferPut(run_info.tiling_data, core_data);
  ByteBufferPut(run_info.tiling_data, core_used);
  ByteBufferPut(run_info.tiling_data, copy_loop);
  ByteBufferPut(run_info.tiling_data, copy_tail);
  ByteBufferPut(run_info.tiling_data, last_copy_loop);
  ByteBufferPut(run_info.tiling_data, last_copy_tail);

  run_info.block_dim = core_used;
  std::vector<int64_t> workspace;
  run_info.workspaces = workspace;
  OP_LOGI(op_type.c_str(), "FlattenTiling run success.");
  return true;
}

// register tiling interface of the Flatten op.
REGISTER_OP_TILING_FUNC_BUFFERED(Flatten, FlattenTiling);
}  // namespace optiling
