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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DROPOUTDOMASK_RUNTIME2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DROPOUTDOMASK_RUNTIME2_H_
#include <cstdint>
#include <vector>
#include "register/op_compile_info_base.h"

namespace optiling {
struct DropOutDoMaskTilingData {
  int64_t core_used_num;
  int64_t num_per_core;
  int64_t num_tail_core;
};

struct DropOutDoMaskCompileInfo {
  int64_t core_num;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DropOutDoMask_RUNTIME2_H_