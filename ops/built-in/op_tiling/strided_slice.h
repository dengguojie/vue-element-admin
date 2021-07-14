/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

void SetSliceTilingData(const string& opType, SliceParameters& slice_params, OpRunInfo& runInfo,
                        const TeOpParas& opParas, int32_t core_num, int32_t ub_size);

static bool GetConstValue(const TeOpParas& paras, const string& name, const string& dtype,
                          std::vector<int64_t>& values) {
  values.clear();
  if (paras.const_inputs.count(name) == 0 || std::get<0>(paras.const_inputs.at(name)) == nullptr) {
    return false;
  }

  auto size = std::get<1>(paras.const_inputs.at(name));
  if (dtype == "int64") {
    int count = size / sizeof(int64_t);
    const int64_t* data_addr = reinterpret_cast<const int64_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i = 0; i < count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  } else if (dtype == "int32") {
    int count = size / sizeof(int32_t);
    const int32_t* data_addr = reinterpret_cast<const int32_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i = 0; i < count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  } else {
    return false;
  }

  return true;
}

static int64_t CalShapeMul(const std::vector<int64_t>& shape, int64_t start, int64_t end) {
  int64_t res = 1;
  for (; start <= end; start += 1) {
    res *= shape[start];
  }
  return res;
}

static int64_t CalVnchwUbSize(int64_t ub_size, int64_t dtype_size, int64_t byte_block) {
  int64_t block_element = byte_block / dtype_size;
  return (ub_size / dtype_size - block_element) / 2 / block_element * block_element;
}

static bool isShapeEqualExceptLast(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& output_shape,
                                   int64_t end) {
  for (int64_t i = 0; i <= end; i++) {
    if (input_shape[i] != output_shape[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace optiling

#endif  // CANN_OPS_BUILT_IN_OP_TILING_STRIDED_SLICE_H_
