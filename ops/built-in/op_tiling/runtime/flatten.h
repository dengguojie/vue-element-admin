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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FLATTEN_RUNTIME2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FLATTEN_RUNTIME2_H_
#include <cstdint>
#include <vector>
#include "register/op_compile_info_base.h"

namespace optiling {
struct FlattenTilingData {
  int64_t core_data;
  int64_t core_used;
  int64_t copy_loop;
  int64_t copy_tail;
  int64_t last_copy_loop;
  int64_t last_copy_tail;
};

struct FlattenCompileInfo {
  int64_t core_num;
  int64_t ub_size;
  int64_t block_size;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FLATTEN_RUNTIME2_H_
