/* Copyright (c) Huawei Technologies Co., Ltd. 2021 All rights reserved.
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
#include <nlohmann/json.hpp>
#include <string>
#include <algorithm>
#include <vector>
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "error_log.h"

namespace optiling {
struct InTopKTilingParams {
  int32_t row_num_input_scalar;
  int32_t col_num_input_scalar;
  int32_t need_core_num_input_scalar;
};

void InTopKWriteTilingParams (const InTopKTilingParams& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.row_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.col_num_input_scalar);
  ByteBufferPut(run_info.tiling_data, params.need_core_num_input_scalar);
}

void InTopKPrintTilingParams(const std::string& op_type, const InTopKTilingParams& params) {
  GELOGD("op [%s] : params.row_num_input_scalar=%d", op_type.c_str(), params.row_num_input_scalar);
  GELOGD("op [%s] : params.col_num_input_scalar=%d", op_type.c_str(), params.row_num_input_scalar);
  GELOGD("op [%s] : params.need_core_num_input_scalar=%d", op_type.c_str(), params.need_core_num_input_scalar);
}

bool InTopKDTiling(const std::string& op_type, const TeOpParas& op_paras, const nlohmann::json& op_compile_info_json,
                   OpRunInfo& run_info) {
  GELOGI("========================InTopKTiling running.====================");
  if (op_paras.inputs.empty() || op_paras.inputs[0].tensor.empty()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape cannot be empty");
    return false;
  }
  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  int32_t input_dims = input_shape.size();
  int32_t row = 1;
  for (int i = 0; i < input_dims - 1; i++) {
    row = row * input_shape[i];
  }
  int32_t col = input_shape[input_dims - 1];
  int32_t need_core = 0;
  int32_t core_num = 32;
  if (row <= core_num) {
    need_core = 1;
    } else {
      const auto& all_vars = op_compile_info_json["vars"];
      int32_t mini_cloud_core_nums = all_vars["mini_cloud_core_nums"].get<std::int32_t>();
      int32_t num = (row + core_num - 1) / core_num;
      if (num <= mini_cloud_core_nums) {
        need_core = num;
        } else {
          need_core = mini_cloud_core_nums;
      }
  }
  InTopKTilingParams params{row, col, need_core};
  InTopKWriteTilingParams(params, run_info);

  InTopKPrintTilingParams(op_type, params);
  run_info.block_dim = need_core;
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(InTopKD, InTopKDTiling);
}
