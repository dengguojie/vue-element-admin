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

static void SetRuningParams(const SliceParameters& params, OpRunInfo& runInfo) {
  int64_t shape_length = static_cast<int64_t>(params.input.size());
  ByteBufferPut(runInfo.tiling_data, params.tiling_mode);
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
  for (size_t i = 0; i <= end; i++) {
    if (input_shape[i] != output_shape[i]) {
      return false;
    }
  }
  return true;
}

static void SetTilingMode(SliceParameters& parameters, int32_t core_num, const string& dtype, int32_t ub_size,
                          const std::string& opType) {
  map<string, int64_t> dtype_size_map = {
      {"int8", 1},    {"uint8", 1},  {"bool", sizeof(bool)},    {"int16", 2},   {"uint16", 2}, {"float16", 2},
      {"int32", 4},   {"uint32", 4}, {"float", sizeof(float)},  {"float32", 4}, {"int64", 8},  {"uint64", 8},
      {"float64", 8}, {"int64", 8},  {"double", sizeof(double)}};

  int64_t dtype_size = dtype_size_map[dtype];
  const int32_t BYTE_BLOCK = 32;
  const int32_t STRIDE_LIMIT = 65535 * BYTE_BLOCK;
  OP_LOGD(opType.c_str(), "param input/output tensor's data type: %s", dtype.c_str(), "dtype size: %lld", dtype_size);
  OP_LOGD(opType.c_str(), "param CalVnchwUbSize: %lld", CalVnchwUbSize(ub_size, dtype_size, BYTE_BLOCK));
  int64_t shape_len = parameters.output_shape.size();

  if (parameters.output_shape[shape_len - 1] * dtype_size < BYTE_BLOCK) {
    parameters.tiling_mode = 1;
  } else {
    parameters.tiling_mode = 2;
  }

  if (parameters.output_shape[shape_len - 1] * dtype_size < BYTE_BLOCK && shape_len >= 2 && dtype == "float16" &&
      CalShapeMul(parameters.output_shape, 0, shape_len - 3) % core_num == 0 &&
      parameters.output_shape[shape_len - 2] >= 16 &&
      parameters.input[shape_len - 1] * 256 <= CalVnchwUbSize(ub_size, dtype_size, BYTE_BLOCK)) {
    parameters.tiling_mode = 3;
  }

  if (shape_len >= 2 && parameters.output_shape[shape_len - 1] * dtype_size % BYTE_BLOCK == 0 &&
      parameters.input[shape_len - 1] * dtype_size % BYTE_BLOCK == 0 &&
      isShapeEqualExceptLast(parameters.input, parameters.output_shape, shape_len - 2) &&
      ub_size >= 2 * parameters.output_shape[shape_len - 1] * dtype_size &&
      (parameters.input[shape_len - 1] - parameters.output_shape[shape_len - 1]) * dtype_size <= STRIDE_LIMIT) {
    parameters.tiling_mode = 4;
  }
  OP_LOGD(opType.c_str(), "parameters.tiling_mode: %lld", parameters.tiling_mode);
}

}  // namespace optiling

#endif  // CANN_OPS_BUILT_IN_OP_TILING_STRIDED_SLICE_H_
