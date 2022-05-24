/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_V3_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_V3_H_
#include <cstdint>
#include <vector>
#include "register/op_compile_info_base.h"

namespace optiling {
struct DynamicRnnV3TilingData {
  int32_t sequenceLength;
  int32_t dynamicRnnBatch;
  int32_t chequeIndex;
};

struct DynamicRNNV3CompileInfo : public optiling::CompileInfoBase {
  std::vector<std::vector<int64_t>> tune_shape_list;
};
}  // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_V3_H_

