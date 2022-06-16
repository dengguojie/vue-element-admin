/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_V2_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GATHER_V2_H_
#include <cstdint>
#include <vector>
#include "auto_tiling_rt2.h"

namespace optiling {
struct GatherV2CompileInfo {
  std::shared_ptr<AutoTilingCompileInfo> dsl_compile_info;
  int64_t ub_size{1};
  int64_t l1_size{0};
  int64_t core_num{1};
  int64_t params_dsize{1};
  int64_t indices_dsize{1};
  bool is_tik{false};
  bool is_gather_v2{true};
  int64_t impl_mode{0};
};

struct GatherV2TilingParams {
  int64_t tiling_mode = 0;
  int64_t params_pre = 1;
  int64_t params_axis = 1;
  int64_t params_row = 1;
  int64_t indices_num = 1;
  int64_t cache_params = 0;
  int64_t need_core_num = 0;
  int64_t tail_process_core = 0;
  int64_t indices_num_each_core = 0;
  int64_t indices_num_remaining = 0;
  int64_t indices_loop_num = 0;
  int64_t indices_row_num_once = 0;
  int64_t indices_row_num_last = 0;
  int64_t row_num_once_ub = 0;
  int64_t row_num_once_tail_ub = 0;
  int64_t inner_loop_num = 0;
  int64_t row_num_last_ub = 0;
  int64_t row_num_last_tail_ub = 0;
  int64_t inner_loop_num_last = 0;
  int64_t params_total = 0;
  int64_t one_row_loop = 0;
  int64_t one_row_tail = 0;
  int64_t params_pre_each_core = 0;
  int64_t params_pre_remaining = 0;
  int64_t indices_row = 1;
  int64_t params_batch_each_core = 1;
  int64_t params_batch_remaining = 0;
  int64_t params_batch = 1;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GATHER_V2_RUNTIME2_H_

