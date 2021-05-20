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

#include "op_tiling.h"

namespace optiling {
struct SliceParameters {
  std::vector<int64_t> input;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> begin_list;
  std::vector<int64_t> end_list;
  std::vector<int64_t> stride_list;

  std::string to_string() const;
};

void SetSliceTilingData(const string& opType, SliceParameters& slice_params, OpRunInfo& runInfo);

static void SetRuningParams(const SliceParameters& params, OpRunInfo& runInfo) {
  int64_t mode_key = 0;
  int64_t shape_length = static_cast<int64_t>(params.input.size());
  ByteBufferPut(runInfo.tiling_data, mode_key);
  ByteBufferPut(runInfo.tiling_data, shape_length);
  const vector<int64_t>* tiling_params[] = {
      &params.input, &params.output_shape, &params.begin_list, &params.end_list, &params.stride_list,
  };

  for (auto item : tiling_params) {
    for (auto x : *item) {
      ByteBufferPut(runInfo.tiling_data, x);
    }
  }
}

static bool GetConstValue(const TeOpParas& paras,
                          const string& name,
                          const string& dtype,
                          std::vector<int64_t>& values) {
  values.clear();
  if (paras.const_inputs.count(name) == 0 || std::get<0>(paras.const_inputs.at(name)) == nullptr) {
    return false;
  }

  auto size = std::get<1>(paras.const_inputs.at(name));
  if (dtype == "int64") {
    int count = size / sizeof(int64_t);
    const int64_t *data_addr = reinterpret_cast<const int64_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i=0; i<count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  } else if (dtype == "int32") {
    int count = size / sizeof(int32_t);
    const int32_t *data_addr = reinterpret_cast<const int32_t*>(std::get<0>(paras.const_inputs.at(name)));
    for (int i=0; i<count; i++) {
      values.push_back(*data_addr);
      data_addr++;
    }
  } else {
    return false;
  }

  return true;
}

}

#endif  // CANN_OPS_BUILT_IN_OP_TILING_STRIDED_SLICE_H_
