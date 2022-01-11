/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file trace.cc
 * \brief dynamic shape tiling of trace
 */

#include <vector>
#include <nlohmann/json.hpp>
#include "op_tiling_util.h"
#include "graph/debug/ge_log.h"

#include "op_log.h"
#include "error_log.h"
#include "graph/utils/op_desc_utils.h"
#include "vector_tiling_profiling.h"

namespace optiling {
const int mini_process_row = 32;

enum TilingMode {
  TILING_MODE_1 = 1,
  TILING_MODE_2
};

// align with 64B
struct TraceTilingparam {
  uint64_t input_h;
  uint64_t input_w;
  uint64_t tiling_mode;
  uint64_t need_core_num;
};

void TraceWriterTilingParams(const TraceTilingparam& params, OpRunInfo& run_info) {
  ByteBufferPut(run_info.tiling_data, params.input_h);
  ByteBufferPut(run_info.tiling_data, params.input_w);
  ByteBufferPut(run_info.tiling_data, params.tiling_mode);
  ByteBufferPut(run_info.tiling_data, params.need_core_num);
  return;
}

void TracePrintTilingParams(const std::string& op_type, const TraceTilingparam& params) {
  GELOGD("op [%s] : input_h = %d", op_type.c_str(), params.input_h);
  GELOGD("op [%s] : input_w = %d", op_type.c_str(), params.input_w);
  GELOGD("op [%s] : tiling_mode = %d", op_type.c_str(), params.tiling_mode);
  GELOGD("op [%s] : need_core_num = %d", op_type.c_str(), params.need_core_num);
  return;
}

bool TraceTiling(const std::string& op_type, const TeOpParas& op_paras,
                 const nlohmann::json& op_compile_info_json, OpRunInfo& run_info) {
  using namespace nlohmann;
  GELOGI("==================TraceTiling Running==================");
  if (op_paras.inputs.empty() || op_paras.inputs[0].tensor.empty()) {
    GELOGE(ge::FAILED, "input shape cannot be empty");
    return false;
  }

  TraceTilingparam tiling_param = {0};
  std::vector<int64_t> input_shape = op_paras.inputs[0].tensor[0].shape;
  int32_t input_dims = input_shape.size();
  tiling_param.input_h = input_shape[0];
  tiling_param.input_w = input_shape[input_dims - 1];
  std::string inputDtype = op_paras.inputs[0].tensor[0].dtype;
  auto& all_vars = op_compile_info_json["vars"];
  int64_t all_core_num = all_vars["core_num"].get<std::int64_t>();
  if (all_core_num == 0) {
    GELOGE(ge::FAILED, "get core num failed");
    return false;
  }

  int64_t matrix_order = std::min(tiling_param.input_h, tiling_param.input_w);
  // Each core processes at least 32 rows of data
  int64_t need_core_num = (matrix_order + mini_process_row - 1) / mini_process_row;
  if (need_core_num <= 1) {
    tiling_param.tiling_mode = TILING_MODE_1;
    tiling_param.need_core_num = 1;
  } else if (need_core_num > 1 && need_core_num < all_core_num) {
    tiling_param.tiling_mode = TILING_MODE_2;
    tiling_param.need_core_num = need_core_num;
  } else {
    tiling_param.tiling_mode = TILING_MODE_2;
    tiling_param.need_core_num = all_core_num;
  }

  TraceWriterTilingParams(tiling_param, run_info);
  TracePrintTilingParams(op_type, tiling_param);
  run_info.block_dim = tiling_param.need_core_num;
  return true;
}
REGISTER_OP_TILING_FUNC_BUFFERED(Trace, TraceTiling);
}