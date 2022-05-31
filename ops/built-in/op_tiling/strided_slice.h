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
 * \file strided_slice.h
 * \brief dynamic shape tiling of strided_slice
 */

#ifndef CANN_OPS_BUILT_IN_OP_TILING_STRIDED_SLICE_H_
#define CANN_OPS_BUILT_IN_OP_TILING_STRIDED_SLICE_H_

#include <string>
#include <vector>
#include <map>

#include "op_tiling.h"

namespace optiling {
struct SliceParameters {
  std::vector<int64_t> input;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> end_list;
  std::vector<int64_t> stride_list;
  int64_t tiling_mode = 0;

  std::string to_string() const;
};

void SetSliceTilingData(const string& opType, SliceParameters& slice_params, utils::OpRunInfo& runInfo,
                        const ge::DataType& dtype, int32_t core_num, int32_t ub_size);

}  // namespace optiling

#endif  // CANN_OPS_BUILT_IN_OP_TILING_STRIDED_SLICE_H_