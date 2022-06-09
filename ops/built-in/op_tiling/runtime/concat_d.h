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

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONCAT_D_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONCAT_D_H_
#include <cstdint>
#include <vector>
#include "auto_tiling_rt2.h"

namespace optiling {
constexpr size_t MAX_CONCAT_NUM = 64;

struct ConcatDTilingInputInfo {
  int64_t inner_dims;
  int64_t output_index;
};

struct ConcatDTilingData{
  int64_t axis = 0;
  int64_t out_dims = 1;
  int64_t max_inner_dims = 0;
  int64_t min_inner_dims = 0;
  int64_t output_inner_length = 1;
  int64_t input_count = 0;
  int64_t reserve1 = 0;
  int64_t reserve2 = 0;

  // list of pair with inner_dims and output_idx
  ConcatDTilingInputInfo input_info[MAX_CONCAT_NUM];
};

struct ConcatDCompileInfo {
  std::shared_ptr<AutoTilingCompileInfo> dsl_compile_info;
  int32_t ori_axis{-1};
  int32_t core_num{1};
  int32_t input_size{1};
  bool is_tik{false};
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_CONCAT_D_H_
