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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDED_SLICE_V3_RUNTIME2_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDED_SLICE_V3_RUNTIME2_H

#include "register/op_compile_info_base.h"

namespace optiling {
struct StridedSliceV3CompileInfo {
  int32_t block_dim;
  int32_t ub_size;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_STRIDED_SLICE_V3_RUNTIME2_H