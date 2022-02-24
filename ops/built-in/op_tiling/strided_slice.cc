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
 * \file strided_slice.cpp
 * \brief dynamic shape tiling of strided_slice
 */

#include "strided_slice.h"

#include <string>
#include <map>
#include <cmath>
#include <utility>

#include <nlohmann/json.hpp>
#include "graph/debug/ge_log.h"

#include "op_log.h"
#include "../op_proto/util/error_util.h"
#include "../op_proto/util/op_common_util.h"
#include "../op_proto/strided_slice_infer_shape.h"
#include "error_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling_util.h"
#include "vector_tiling_profiling.h"

namespace optiling {
using namespace ge;

const int32_t BYTE_BLOCK = 32;
const size_t MAX_SUPPORTED_DIMS = 8;
constexpr int32_t SHAPE_LEN = 2;
constexpr int32_t TILING_FACTOR_2 = 2;
constexpr int32_t TILING_FACTOR_16 = 16;
constexpr int32_t BYTE_SIZE = 32;
constexpr int32_t TILING_MODE_5 = 5;
constexpr int32_t TILING_MODE_6 = 6;
constexpr int32_t TILING_MODE_7 = 7;
constexpr int32_t TILING_MODE_8 = 8;
constexpr int32_t TILING_FACTOR_256 = 256;
constexpr int32_t NUM_THREE = 3;
static const std::pair<int64_t, std::string> BEGIN_MASK_ATTR_INFO{0, "begin_mask"};
static const std::pair<int64_t, std::string> END_MASK_ATTR_INFO{1, "end_mask"};
static const std::pair<int64_t, std::string> ELLIPSIS_MASK_ATTR_INFO{2, "ellipsis_mask"};
static const std::pair<int64_t, std::string> NEW_AXIS_MASK_ATTR_INFO{3, "new_axis_mask"};
static const std::pair<int64_t, std::string> SHRINK_AXIS_MASK_ATTR_INFO{4, "shrink_axis_mask"};
static const uint32_t MASK_ATTR_DEFAULT_VALUE = 0;

std::string SliceParameters::to_string() const {
  string result = "input_shape:" + ops::to_string(input);
  result += " output_shape:" + ops::to_string(output_shape);
  result += " begin:" + ops::to_string(begin_list);
  result += " end:" + ops::to_string(end_list);
  result += " stride:" + ops::to_string(stride_list);
  result += " tiling_mode:" + std::to_string(tiling_mode);
  return result;
}

struct SliceMasks {
  uint32_t begin_mask = 0;
  uint32_t end_mask = 0;
  uint32_t ellipsis_mask = 0;
  uint32_t new_axis_mask = 0;
  uint32_t shrink_axis_mask = 0;
};

static string to_string(const ByteBuffer& tiling_data) {
  auto data = tiling_data.str();
  string result = "(";
  const int64_t* data_addr = reinterpret_cast<const int64_t*>(data.c_str());
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    result += std::to_string(*data_addr);
    data_addr++;
    result += ", ";
  }
  result += ")";

  return result;
}

static int64_t CalShapeMul(const std::vector<int64_t>& shape, int64_t start, int64_t end) {
  int64_t res = 1;
  for (; start <= end; start += 1) {
    res *= shape[start];
  }
  return res;
}

static int64_t CalVnchwUbSize(int64_t ub_size, int64_t dtype_size) {
  OP_TILING_CHECK(dtype_size == 0, VECTOR_INNER_ERR_REPORT_TILIING("StridedSlice", "dtype_size = 0 is not supported."),
                  return ub_size);
  int64_t block_element = BYTE_BLOCK / dtype_size;
  return (ub_size / dtype_size - block_element) / 2 / block_element * block_element;
}

static bool IsShapeEqualExceptLast(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& output_shape,
                                   int64_t end) {
  for (int64_t i = 0; i <= end; i++) {
    if (input_shape[i] != output_shape[i]) {
      return false;
    }
  }
  return true;
}

static void SetTilingMode(SliceParameters& parameters, int32_t core_num, const ge::DataType& dtype, int32_t ub_size,
                          const std::string& opType) {
  int64_t dtype_size = GetSizeByDataType(dtype);
  const int32_t STRIDE_LIMIT = 65535 * BYTE_BLOCK;
  OP_LOGD(opType.c_str(), "param input/output tensor's data type: %s", to_string(dtype).c_str(), "dtype size: %lld",
          dtype_size);
  OP_LOGD(opType.c_str(), "param CalVnchwUbSize: %lld", CalVnchwUbSize(ub_size, dtype_size));
  int64_t shape_len = parameters.output_shape.size();
  int64_t float16_type_size = 2;

  if (parameters.output_shape[shape_len - 1] * dtype_size < BYTE_BLOCK) {
    parameters.tiling_mode = 1;
  } else {
    parameters.tiling_mode = 2;
  }

  if (parameters.output_shape[shape_len - 1] * dtype_size < BYTE_BLOCK && shape_len >= SHAPE_LEN &&
      dtype == DT_FLOAT16 && CalShapeMul(parameters.output_shape, 0, shape_len - NUM_THREE) % core_num == 0 &&
      parameters.output_shape[shape_len - 2] >= 16 &&
      parameters.input[shape_len - 1] * TILING_FACTOR_256 <= CalVnchwUbSize(ub_size, dtype_size)) {
    parameters.tiling_mode = 3;
  }

  if (shape_len >= 2 && parameters.output_shape[shape_len - 1] * dtype_size % BYTE_BLOCK == 0 &&
      parameters.input[shape_len - 1] * dtype_size % BYTE_BLOCK == 0 &&
      IsShapeEqualExceptLast(parameters.input, parameters.output_shape, shape_len - SHAPE_LEN) &&
      ub_size >= 2 * parameters.output_shape[shape_len - 1] * dtype_size &&
      (parameters.input[shape_len - 1] - parameters.output_shape[shape_len - 1]) * dtype_size <= STRIDE_LIMIT) {
    parameters.tiling_mode = 4;
  }

  if (shape_len == SHAPE_LEN && IsShapeEqualExceptLast(parameters.input, parameters.output_shape, shape_len - \
      SHAPE_LEN) && parameters.output_shape[shape_len - 1] * dtype_size > BYTE_SIZE &&
      parameters.input[shape_len - 1] * BYTE_BLOCK * TILING_FACTOR_2 <= ub_size &&
      parameters.input[shape_len - 1] * dtype_size > BYTE_SIZE) {
    parameters.tiling_mode = TILING_MODE_6;
  }

  if (shape_len == SHAPE_LEN && IsShapeEqualExceptLast(parameters.input, parameters.output_shape, shape_len - \
      SHAPE_LEN) && parameters.output_shape[shape_len - 1] == 1 && BYTE_BLOCK * \
      (parameters.input[shape_len - 1] + 1) < ub_size) {
    parameters.tiling_mode = TILING_MODE_8;
  }

  int64_t multi_times = dtype_size / float16_type_size;
  int64_t input_inner_dims = parameters.input[shape_len - 1] * multi_times;
  int64_t output_inner_dims = parameters.output_shape[shape_len - 1] * multi_times;
  int64_t output_32bytes_align_rows = BYTE_BLOCK / float16_type_size;
  if (output_inner_dims > 0 && output_32bytes_align_rows % output_inner_dims == 0) {
    output_32bytes_align_rows = output_32bytes_align_rows / output_inner_dims;
  } else if (output_inner_dims % output_32bytes_align_rows == 0) {
    output_32bytes_align_rows = 1;
  }
  int64_t need_ub_size = input_inner_dims * TILING_FACTOR_16 * TILING_FACTOR_2;
  if (shape_len == SHAPE_LEN && IsShapeEqualExceptLast(parameters.input, parameters.output_shape, shape_len - \
      SHAPE_LEN) && need_ub_size * BYTE_BLOCK / dtype_size * output_32bytes_align_rows < ub_size &&
      dtype_size % float16_type_size == 0) {
    parameters.tiling_mode = TILING_MODE_5;
  }

  if (shape_len == 1 && parameters.stride_list[0] == 1) {
    parameters.tiling_mode = TILING_MODE_7;
  }

  OP_LOGD(opType.c_str(), "parameters.tiling_mode: %lld", parameters.tiling_mode);
}

static void SetRuningParams(const SliceParameters& params, utils::OpRunInfo& runInfo) {
  int64_t shape_length = static_cast<int64_t>(params.input.size());
  runInfo.AddTilingData(params.tiling_mode);
  runInfo.AddTilingData(shape_length);
  const vector<int64_t>* tiling_params[] = {
      &params.input, &params.output_shape, &params.begin_list, &params.end_list, &params.stride_list,
  };

  for (auto item : tiling_params) {
    for (auto x : *item) {
      runInfo.AddTilingData(x);
    }
  }
}

static bool GetStridedSliceSocParams(const std::string& opType, const ge::Operator& op_paras,
                                     const nlohmann::json& opCompileInfo, int32_t& core_num, uint32_t& begin_mask,
                                     uint32_t& end_mask, uint32_t& ellipsis_mask, uint32_t& new_axis_mask,
                                     uint32_t& shrink_axis_mask, int32_t& ub_size) {
  using namespace nlohmann;
  const auto& allVars = opCompileInfo["vars"];
  if (allVars.count("block_dim") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get block_dim error");
    return false;
  }
  core_num = allVars["block_dim"].get<std::int32_t>();

  if (allVars.count("ub_size") == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "GetCompileParams, get ub_size error");
    return false;
  }
  ub_size = allVars["ub_size"].get<std::int32_t>();

  ops::GetAttrValue(op_paras, BEGIN_MASK_ATTR_INFO, begin_mask, MASK_ATTR_DEFAULT_VALUE);
  ops::GetAttrValue(op_paras, END_MASK_ATTR_INFO, end_mask, MASK_ATTR_DEFAULT_VALUE);
  ops::GetAttrValue(op_paras, ELLIPSIS_MASK_ATTR_INFO, ellipsis_mask, MASK_ATTR_DEFAULT_VALUE);
  ops::GetAttrValue(op_paras, NEW_AXIS_MASK_ATTR_INFO, new_axis_mask, MASK_ATTR_DEFAULT_VALUE);
  ops::GetAttrValue(op_paras, SHRINK_AXIS_MASK_ATTR_INFO, shrink_axis_mask, MASK_ATTR_DEFAULT_VALUE);

  return true;
}

static void MakeSameDims(SliceParameters* parametersPtr) {
  auto& parameters = *parametersPtr;
  bool same_size = parameters.input.size() == parameters.begin_list.size() &&
                   parameters.input.size() == parameters.end_list.size() &&
                   parameters.input.size() == parameters.stride_list.size();
  if (!same_size) {
    return;
  }

  parameters.output_shape.clear();
  parameters.output_shape.reserve(parameters.input.size());
  for (size_t i = 0; i < parameters.input.size(); i++) {
    auto interval = parameters.end_list[i] - parameters.begin_list[i];
    auto stride_i = parameters.stride_list[i];
    if (stride_i == 0) {
      stride_i = 1;
    }
    int64_t output_size = interval / stride_i + (interval % stride_i != 0 ? 1 : 0);
    parameters.output_shape.push_back(output_size);
  }
}

static void MakePerformanceParams(SliceParameters& parameters) {
  SliceParameters perf_params;
  bool last_same = false;
  size_t perf_size = 0;
  for (size_t i = 0; i < parameters.input.size(); i++) {
    const auto input_shape_i = parameters.input[i];
    const auto output_shape_i = parameters.output_shape[i];
    const auto begin_i = parameters.begin_list[i];
    const auto end_i = parameters.end_list[i];
    const auto stride_i = parameters.stride_list[i];
    if (input_shape_i != output_shape_i) {
      perf_params.input.push_back(input_shape_i);
      perf_params.output_shape.push_back(output_shape_i);
      perf_params.begin_list.push_back(begin_i);
      perf_params.end_list.push_back(end_i);
      perf_params.stride_list.push_back(stride_i);
      last_same = false;
      perf_size++;
      continue;
    }

    if (!last_same) {
      last_same = true;
      perf_params.input.push_back(input_shape_i);
      perf_params.output_shape.push_back(output_shape_i);
      perf_params.begin_list.push_back(begin_i);
      perf_params.end_list.push_back(end_i);
      perf_params.stride_list.push_back(stride_i);
      perf_size++;
      continue;
    }

    const auto perf_index = perf_size - 1;
    perf_params.input[perf_index] *= input_shape_i;
    perf_params.output_shape[perf_index] *= output_shape_i;
    perf_params.begin_list[perf_index] = 0;
    perf_params.end_list[perf_index] = perf_params.input[perf_index];
    perf_params.stride_list[perf_index] = 1;
  }

  if (perf_params.input.size() > 1 && perf_params.input.back() == perf_params.output_shape.back() &&
      perf_params.stride_list[perf_params.input.size() - 2] == 1) {
    const auto last_second_index = perf_params.input.size() - 2;
    perf_params.input[last_second_index] *= perf_params.input.back();
    perf_params.output_shape[last_second_index] *= perf_params.output_shape.back();
    perf_params.begin_list[last_second_index] *= perf_params.input.back();
    perf_params.end_list[last_second_index] *= perf_params.input.back();
    perf_params.stride_list[last_second_index] = 1;

    perf_params.input.pop_back();
    perf_params.output_shape.pop_back();
    perf_params.begin_list.pop_back();
    perf_params.end_list.pop_back();
    perf_params.stride_list.pop_back();
  }

  parameters = perf_params;
}

/*
 * @brief: tiling function of StridedSlice
 * @param [in] opType: opType of the StridedSlice
 * @param [in] opParas: inputs/outputs/atts of the StridedSlice
 * @param [in] opCompileInfo: compile time generated info of the StridedSlice
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool StridedSliceTiling(const std::string& opType, const ge::Operator& opParas, const nlohmann::json& opCompileInfo,
                        utils::OpRunInfo& runInfo) {
  using namespace nlohmann;
  OP_LOGD(opType.c_str(), "StridedSliceTiling running.");
  PROFILING_TILING_INIT(opType.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(opParas);
  if (operator_info == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "operator_info is null");
    return false;
  }
  auto opdesc = operator_info->MutableInputDesc(0);
  if (opdesc == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "opdesc is null");
    return false;
  }

  struct SliceParameters slice_params = {};
  struct SliceMasks slice_masks = {};

  const std::vector<int64_t> input_shape = opdesc->MutableShape().GetDims();
  if (input_shape.size() > MAX_SUPPORTED_DIMS) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType,
                                    "StridedSliceTiling: input shape error, max supported dims is %zu, actual is %zu.",
                                    MAX_SUPPORTED_DIMS, input_shape.size());
    return false;
  }

  map<string, std::pair<int, vector<int64_t>&>> const_params = {
      {"begin", {1, slice_params.begin_list}},
      {"end", {2, slice_params.end_list}},
      {"strides", {3, slice_params.stride_list}},
  };

  for (auto& item : const_params) {
    auto& name = item.first;
    int index = item.second.first;
    auto& values = item.second.second;
    if (!ops::GetConstIntData(opParas, index, values)) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "Get %s values failed", name.c_str());
      return false;
    }
  }
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  for (size_t i = 0; i < slice_params.stride_list.size(); i++) {
    if (slice_params.stride_list[i] == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING(opType, "StridedSliceTiling: the dim of stride must be non-zero");
      return false;
    }

    if (slice_params.stride_list[i] != 1) {
      OP_LOGW(opType, "StridedSliceTiling: stride[%zu] must be 1, but it is %ld", i, slice_params.stride_list[i]);
      return false;
    }
  }

  int32_t core_num = 0;
  int32_t ub_size = 0;
  bool flag = GetStridedSliceSocParams(opType, opParas, opCompileInfo, core_num, slice_masks.begin_mask,
                                       slice_masks.end_mask, slice_masks.ellipsis_mask, slice_masks.new_axis_mask,
                                       slice_masks.shrink_axis_mask, ub_size);
  OP_LOGD(opType.c_str(), "param ub_size: %d", ub_size);

  if (!flag) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "StridedSliceTiling: get soc params error");
    return false;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  StridedSliceParams input_params = {
      input_shape,
      slice_params.begin_list,
      slice_params.end_list,
      slice_params.stride_list,
      vector<pair<int64_t, int64_t>>(),
      slice_masks.begin_mask,
      slice_masks.end_mask,
      slice_masks.ellipsis_mask,
      slice_masks.new_axis_mask,
      slice_masks.shrink_axis_mask,
      true,
      true,
      true,
  };

  vector<int64_t> output_shape;
  vector<pair<int64_t, int64_t>> output_ranges;

  if (!StridedSliceCommonInferShape(opType, input_params, output_shape, output_ranges)) {
    VECTOR_INNER_ERR_REPORT_TILIING(opType, "StridedSliceCommonInferShape failed.");
    return false;
  }

  slice_params.input = input_params.input_shape;
  slice_params.begin_list = input_params.begin;
  slice_params.end_list = input_params.end;
  slice_params.stride_list = input_params.strides;
  slice_params.output_shape = output_shape;

  MakeSameDims(&slice_params);
  OP_LOGD(opType.c_str(), "origin slice params: %s", slice_params.to_string().c_str());

  ge::DataType input_dtype = opdesc->GetDataType();
  SetSliceTilingData(opType, slice_params, runInfo, input_dtype, core_num, ub_size);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  runInfo.SetBlockDim(core_num);
  OP_LOGD(opType.c_str(), "tiling run success.");
  PROFILING_TILING_END();
  return true;
}

void SetSliceTilingData(const string& opType, SliceParameters& slice_params, utils::OpRunInfo& runInfo,
                        const ge::DataType& dtype, int32_t core_num, int32_t ub_size) {
  MakePerformanceParams(slice_params);
  // if
  OP_LOGD(opType.c_str(), "perf slice params: %s", slice_params.to_string().c_str());
  SetTilingMode(slice_params, core_num, dtype, ub_size, opType);
  OP_LOGD(opType.c_str(), "set tiling_mode params: %s", slice_params.to_string().c_str());

  SetRuningParams(slice_params, runInfo);
  OP_LOGD(opType.c_str(), "tiling param:%s", to_string(runInfo.GetAllTilingData()).c_str());
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(StridedSlice, StridedSliceTiling);
}  // namespace optiling
