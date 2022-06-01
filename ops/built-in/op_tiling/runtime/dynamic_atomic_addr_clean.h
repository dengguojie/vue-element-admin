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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_ATOMIC_ADDR_CLEAN_IMPL_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_ATOMIC_ADDR_CLEAN_IMPL_H_
#include <cstdint>
#include <vector>
#include "register/op_compile_info_base.h"

namespace optiling {
struct InputScalar {
  int32_t init_times_full_mask_full_repeat_time;
  int32_t ele_num_front_part;
  int32_t burst_len_last_part;
  int32_t repeat_time_last_part;
};

struct DynamicAtomicAddrCleanTilingData {
  // common input scalar
  int32_t select_key_input_scalar;
  int32_t need_core_num_input_scalar;
  int32_t ele_num_full_mask_full_repeat_time_input_scalar;
  int32_t burst_len_full_mask_full_repeat_time_input_scalar;

  // init input scalar
  // front core
  int32_t ele_num_front_core_input_scalar;
  InputScalar front_core_input_scalar;
  // last core
  int32_t ele_num_last_core_input_scalar;
  InputScalar last_core_input_scalar;
};

struct DynamicAtomicAddrCleanCompileInfo {
  uint32_t workspace_num = 0;
  uint32_t core_num = 0;
  uint32_t ub_size = 0;
  std::vector<int64_t> _workspace_index_list;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_ATOMIC_ADDR_CLEAN_IMPL_H_
