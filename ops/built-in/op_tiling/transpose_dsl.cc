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
 * \file transpose_dsl.cpp
 * \brief
 */
#include "transpose_dsl.h"

#include <algorithm>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <numeric>

#include "graph/utils/op_desc_utils.h"
#include "vector_tiling.h"
#include "tiling_handler.h"
#include "auto_tiling_register.h"

namespace optiling {
namespace transpose {
namespace {
constexpr std::int64_t ELEMENT_IN_BLOCK_DEFAULT = 16;
constexpr std::int64_t ELEMENT_IN_BLOCK_B32 = 8;
constexpr std::int64_t ELEMENT_IN_BLOCK_B8 = 32;
constexpr std::int64_t ELEMENT_IN_BLOCK_B64 = 4;
constexpr std::int64_t NO_MULTI_BLOCK_BASE_KEY = 0;
constexpr std::int64_t CONST_TILING_KEY = 1000000;
constexpr std::int64_t ONE_DIMS_N_LAST_NO_CONV_BASE_KEY = 20000;
constexpr std::int64_t ONE_DIMS_N_LAST_ALIGN_BASE_KEY = 10000;
constexpr std::int64_t PURE_COPY_BASE_KEY = 3000000;
constexpr std::int64_t N_LAST_NO_CONV_BASE_KEY = 2020000;
constexpr std::int64_t N_LAST_ALIGN_BASE_KEY = 2030000;
constexpr std::int64_t GENERAL_BASE_KEY = 2000000;
constexpr std::int64_t BLOCK_AXIS_FACTOR = 100;
constexpr std::int64_t LOW_SPLIT_AXIS_FACTOR = 10;
constexpr std::int64_t B8_COEXISTING_QUANTITY_FACTOR = 32;
constexpr std::int64_t GT_B8_COEXISTING_QUANTITY_FACTOR = 16;
constexpr std::int64_t BLOCK_SIZE_BYTE = 32;
constexpr int64_t ALIGN_THRESHOLD = 512;
constexpr std::int64_t ONE_K_BYTES = 1024;
constexpr std::int64_t MULTI_CORE_EXPERIENCE = 64;
constexpr std::int64_t CROSS_COEXISTING_QUANTITY_FACTOR = 2;
constexpr std::int64_t BASE_COEXISTING_QUANTITY = 2;
constexpr std::int64_t LAST_TRANSPOSE_COEXISTING_QUANTITY = 4;
constexpr std::int64_t ALIGN_GT_B16_CROSS_UB_IN_FACTOR = 3;
constexpr std::int64_t ALIGN_CROSS_UB_IN_FACTOR = 2;
constexpr std::int64_t COMPILE_SHAPE_IS_ONE_FLAG = 2;
constexpr double CROSS_UB_IN_EXPERIENCE = 0.7;
}  // namespace

static const int64_t GetElementByType(const ge::DataType& dtype) {
  // element nums in one block, default, fp16, int16, uin16
  int64_t element_in_block = ELEMENT_IN_BLOCK_DEFAULT;
  if (dtype == ge::DataType::DT_FLOAT || dtype == ge::DataType::DT_INT32 || dtype == ge::DataType::DT_UINT32) {
    // element nums in one block by b32
    element_in_block = ELEMENT_IN_BLOCK_B32;
  } else if (dtype == ge::DataType::DT_INT8 || dtype == ge::DataType::DT_UINT8 || dtype == ge::DataType::DT_BOOL) {
    // element nums in one block by b8
    element_in_block = ELEMENT_IN_BLOCK_B8;
  } else if (dtype == ge::DataType::DT_INT64 || dtype == ge::DataType::DT_UINT64) {
    // element nums in one block by b64
    element_in_block = ELEMENT_IN_BLOCK_B64;
  }
  return element_in_block;
}

TransposeCompileInfo::TransposeCompileInfo(const nlohmann::json& json_compile_info) {
  Parse("TransposeDsl", json_compile_info);
}

bool TransposeCompileInfo::Parse(const char* op_type, const nlohmann::json& json_compile_info) {
  try {
    core_num = json_compile_info.at("_core_num");
    ub_size = json_compile_info.at("_ub_size");
    only_const_tiling = json_compile_info.at("_only_const_tiling");
    permute = json_compile_info.at("_permute").get<std::vector<int64_t>>();
    ori_permute = json_compile_info.at("_ori_permute").get<std::vector<int64_t>>();
    mergeable = json_compile_info.at("_mergeable").get<std::vector<int64_t>>();
    if (!only_const_tiling) {
      transpose_vars = json_compile_info.at("_transpose_vars").get<std::vector<bool>>();
    }
    is_const = json_compile_info.at("_is_const");
    if (is_const) {
      const_block_dims = json_compile_info.at("_const_dims");
    }
  } catch (...) {
    OP_LOGE(op_type, "Unknown Exception encountered when parsing Compile Info of op_type %s", op_type);
    return false;
  }
  return true;
}

template <typename T>
void Transpose<T>::GenerateShape(int64_t cur_dim_value, size_t cur_index, int64_t& real_index, bool& is_first_in) {
  if (c_info->mergeable[cur_index] == COMPILE_SHAPE_IS_ONE_FLAG) {
    return;
  } else if (c_info->mergeable[cur_index] == 1) {
    input_shapes[real_index] = input_shapes[real_index] * cur_dim_value;
  } else {
    if (is_first_in) {
      is_first_in = false;
    } else {
      real_index++;
    }
    input_shapes[real_index] = cur_dim_value;
  }
}

template <typename T>
bool Transpose<T>::GenerateOutputShape() {
  V_OP_TILING_CHECK((context->GetInputNums() != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape cannot be empty"), return false);
  const OpShape& shape = context->GetInputShape(0);
  size_t src_dim_len = shape.GetDimNum();
  V_OP_TILING_CHECK((src_dim_len == c_info->mergeable.size()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input dim length error"), return false);
  int64_t real_index = 0;
  bool is_first_in = true;
  for (size_t i = 0; i < src_dim_len; i++) {
    GenerateShape(shape.GetDim(i), i, real_index, is_first_in);
  }
  V_OP_TILING_CHECK(context->GetOutputDataType(0, dtype),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get inpute dtype error"),
                    return false);
  V_OP_TILING_CHECK((context->GetOutputNums() != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output shape cannot be empty"), return false);
  size_t dst_dim_len = context->GetOutputShape(0).GetDimNum();
  V_OP_TILING_CHECK((dst_dim_len == c_info->mergeable.size()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output dim length error"), return false);
  for (size_t i = 0; i < ori_permute.size(); i++) {
    output_shapes[i] = input_shapes[ori_permute[i]];
    max_dim_shape = std::max(max_dim_shape, output_shapes[i]);
  }
  return true;
}

template <typename T>
bool Transpose<T>::GenerateOutputShapeFromOp() {
  const auto inputs = op_info->GetInputShape();
  V_OP_TILING_CHECK((inputs != nullptr && !inputs->empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor is empty"), return false);
  const std::vector<int64_t>& op_input_shapes = inputs->at(0);
  size_t src_dim_len = op_input_shapes.size();
  V_OP_TILING_CHECK((src_dim_len == c_info->mergeable.size()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input dim length error"), return false);
  int64_t real_index = 0;
  bool is_first_in = true;
  for (size_t i = 0; i < src_dim_len; i++) {
    GenerateShape(op_input_shapes[i], i, real_index, is_first_in);
  }
  V_OP_TILING_CHECK(context->GetInputDataType(op_info, dtype),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get inpute dtype error"),
                    return false);
  for (size_t i = 0; i < ori_permute.size(); i++) {
    output_shapes[i] = input_shapes[ori_permute[i]];
    max_dim_shape = std::max(max_dim_shape, output_shapes[i]);
  }
  return true;
}

template <typename T>
bool Transpose<T>::CalcOutputSize() {
  V_OP_TILING_CHECK((!permute.empty()), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape cannot be empty"),
                    return false);
  output_size = std::accumulate(output_shapes.begin(), output_shapes.begin() + permute.size(), 1LL,
                                std::multiplies<int64_t>());
  V_CHECK_LE(output_size, INT32_MAX, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The output shape is too large"),
             return false);
  V_CHECK_GT(output_size, 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The output shape must be greater than 0"),
             return false);
  return true;
}

template <typename T>
bool Transpose<T>::CalcTiling() {
  if (max_dim_shape == output_size) {
    is_pure_copy = true;
    permute = {0};
    ori_permute = {0};
    input_shapes[0] = max_dim_shape;
    output_shapes[0] = max_dim_shape;
  }
  int64_t last_index = permute.size() - 1;
  is_last_transpose = permute[last_index] != last_index;
  int64_t coexisting_quantity = BASE_COEXISTING_QUANTITY;
  int64_t factor = (GetElementByType(dtype) == B8_COEXISTING_QUANTITY_FACTOR ? B8_COEXISTING_QUANTITY_FACTOR
                                                                             : GT_B8_COEXISTING_QUANTITY_FACTOR);
  int64_t cross_coexisting_quantity = (factor * CROSS_COEXISTING_QUANTITY_FACTOR) + coexisting_quantity;
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t bytes = BLOCK_SIZE_BYTE / ele_in_block;
  is_nlast_no_conv = !is_last_transpose && output_shapes[last_index] % ele_in_block == 0;
  int64_t cross_ub_size = c_info->ub_size;
  if (bytes == 1) {
    cross_ub_size -= ONE_K_BYTES;
  }
  if (is_last_transpose) {
    if (GetElementByType(dtype) < ELEMENT_IN_BLOCK_DEFAULT) {
      coexisting_quantity = LAST_TRANSPOSE_COEXISTING_QUANTITY;
    }
    max_available_ub = c_info->ub_size / coexisting_quantity / bytes / ele_in_block * ele_in_block;
    max_available_cross_ub = cross_ub_size / cross_coexisting_quantity / bytes / ele_in_block * ele_in_block;
  } else if (is_nlast_no_conv) {
    max_available_ub = c_info->ub_size / coexisting_quantity / bytes / ele_in_block * ele_in_block;
  } else {
    is_nlast_align =
        !is_last_transpose && (output_shapes[last_index] % ele_in_block == 0 ||
                               output_shapes[last_index] >= (ALIGN_THRESHOLD / (BLOCK_SIZE_BYTE / ele_in_block)));
    if (is_nlast_align) {
      max_available_ub = (c_info->ub_size - BLOCK_SIZE_BYTE) / bytes / ele_in_block * ele_in_block;
    } else {
      max_available_ub = cross_ub_size / cross_coexisting_quantity / bytes / ele_in_block * ele_in_block;
    }
  }
  if (is_pure_copy) {
    max_available_ub = c_info->ub_size / BASE_COEXISTING_QUANTITY / bytes / ele_in_block * ele_in_block;
  }
  const int64_t multi_core_threshold = MULTI_CORE_EXPERIENCE * GetElementByType(dtype);  // 2k
  if (output_size > multi_core_threshold) {
    need_multi_core = true;
  }
  return true;
}

template <typename T>
bool Transpose<T>::DoBlockTiling() {
  for (int64_t i = 0; i < high_ub_axis; i++) {
    if (!have_splits[i]) {
      if (block_dims * output_shapes[i] >= core_num) {
        block_axis = i;
        int64_t left_core = core_num / block_dims;
        block_factor = (output_shapes[i] + left_core - 1) / left_core;
        block_dims *= ((output_shapes[i] + block_factor - 1) / block_factor);
        break;
      } else {
        block_dims *= output_shapes[i];
      }
    }
  }
  if (block_axis == -1) {
    int64_t remainder_low = (output_shapes[low_ub_axis] + low_ub_factor - 1) / low_ub_factor;
    if (block_dims * remainder_low >= core_num) {
      block_axis = low_ub_axis;
      int64_t left_core = core_num / block_dims;
      block_factor = (remainder_low + left_core - 1) / left_core;
      block_dims *= ((remainder_low + block_factor - 1) / block_factor);
    } else {
      block_dims *= remainder_low;
      if (low_ub_axis == high_ub_axis) {
        block_axis = low_ub_axis;
        block_factor = 1;
      }
    }
  }
  if (block_axis == -1) {
    block_axis = high_ub_axis;
    int64_t remainder_low = (output_shapes[high_ub_axis] + high_ub_factor - 1) / high_ub_factor;
    int64_t left_core = core_num / block_dims;
    block_factor = (remainder_low + left_core - 1) / left_core;
    block_dims *= (remainder_low + block_factor - 1) / block_factor;
  }
  return true;
}

template <typename T>
bool Transpose<T>::InitCrossUbTilingParams(CrossUbTilingParams& tilingParams) {
  int64_t dim_len = ori_permute.size();
  V_OP_TILING_CHECK((dim_len > 0), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input dims must greater 0"), return false);
  tilingParams.input_index = dim_len - 1;
  tilingParams.output_index = dim_len - 1;
  tilingParams.input_ub_size = 1;
  tilingParams.output_ub_size = 1;
  input_available_ub = std::sqrt(max_available_ub);
  output_available_ub = max_available_ub / input_available_ub;
  int64_t ele_in_block = GetElementByType(dtype);
  tilingParams.cross_ub_in = std::sqrt(max_available_cross_ub) * CROSS_UB_IN_EXPERIENCE;
  tilingParams.cross_ub_out = std::sqrt(max_available_cross_ub);
  if (output_shapes[tilingParams.output_index] % ele_in_block == 0) {
    tilingParams.cross_ub_in = ele_in_block < ELEMENT_IN_BLOCK_DEFAULT ? ele_in_block * ALIGN_GT_B16_CROSS_UB_IN_FACTOR
                                                                       : ele_in_block * ALIGN_CROSS_UB_IN_FACTOR;
    tilingParams.cross_ub_out = tilingParams.cross_ub_in;
  }
  if (ele_in_block < ELEMENT_IN_BLOCK_DEFAULT) {
    output_available_ub = ele_in_block * GT_B8_COEXISTING_QUANTITY_FACTOR;
    input_available_ub = max_available_ub / output_available_ub;
  }
  tilingParams.can_update_ub = ele_in_block == ELEMENT_IN_BLOCK_DEFAULT;
  return true;
}

template <typename T>
bool Transpose<T>::DoUbTilingNoCross() {
  CrossUbTilingParams tilingParams;
  if (!InitCrossUbTilingParams(tilingParams)) {
    return false;
  }
  while (tilingParams.input_index >= 0 && tilingParams.output_index >= 0) {
    if (!InputCrossUbTiling(tilingParams)) {
      return false;
    }

    if (!OutputCrossUbTiling(tilingParams)) {
      return false;
    }

    bool no_available_ub =
        tilingParams.input_ub_size * input_shapes[tilingParams.input_index] >= input_available_ub &&
        tilingParams.output_ub_size * output_shapes[tilingParams.output_index] >= output_available_ub;
    if (no_available_ub) {
      bool had_split = have_splits[permute[tilingParams.input_index]] || have_splits[tilingParams.output_index];
      if (had_split) {
        continue;
      }
      if (permute[tilingParams.input_index] == tilingParams.output_index) {
        bool is_indivisible = tilingParams.cross_ub_in > tilingParams.input_ub_size &&
                              tilingParams.cross_ub_out > tilingParams.output_ub_size;
        if (is_indivisible) {
          return false;
        }
        CrossUbUpdateSameAxis(tilingParams);
        bool has_remainder =
            tilingParams.input_ub_size * input_shapes[tilingParams.input_index] < input_available_ub ||
            tilingParams.output_ub_size * output_shapes[tilingParams.output_index] < output_available_ub;
        if (has_remainder) {
          continue;
        }
      }
      low_ub_factor = input_available_ub / tilingParams.input_ub_size;
      high_ub_factor = output_available_ub / tilingParams.output_ub_size;
      low_ub_axis = permute[tilingParams.input_index];
      high_ub_axis = tilingParams.output_index;
      have_splits[low_ub_axis] = true;
      have_splits[high_ub_axis] = true;
      break;
    }
  }
  return true;
}

template <typename T>
void Transpose<T>::CrossUbUpdateSameAxis(CrossUbTilingParams& tilingParams) {
  int64_t ele_in_block = GetElementByType(dtype);
  if (tilingParams.input_ub_size >= tilingParams.output_ub_size) {
    input_available_ub = tilingParams.input_ub_size;
    tilingParams.input_index++;
    have_splits[permute[tilingParams.input_index]] = false;
    tilingParams.input_ub_size /= input_shapes[tilingParams.input_index];
    if (tilingParams.can_update_ub) {
      tilingParams.can_update_ub = false;
      output_available_ub = max_available_ub / ((input_available_ub + ele_in_block - 1) / ele_in_block * ele_in_block);
      output_available_ub = output_available_ub / ele_in_block * ele_in_block;
    }
  } else {
    output_available_ub = tilingParams.output_ub_size;
    tilingParams.output_index++;
    have_splits[tilingParams.output_index] = false;
    tilingParams.output_ub_size /= output_shapes[tilingParams.output_index];
    if (tilingParams.can_update_ub) {
      tilingParams.can_update_ub = false;
      input_available_ub = max_available_ub / ((output_available_ub + ele_in_block - 1) / ele_in_block * ele_in_block);
      input_available_ub = input_available_ub / ele_in_block * ele_in_block;
    }
  }
}

template <typename T>
bool Transpose<T>::OutputCrossUbTiling(CrossUbTilingParams& tilingParams) {
  if (have_splits[tilingParams.output_index]) {
    int64_t ele_in_block = GetElementByType(dtype);
    if (tilingParams.cross_ub_out > tilingParams.output_ub_size) {
      return false;
    }
    output_available_ub = tilingParams.output_ub_size;
    tilingParams.output_index++;
    have_splits[tilingParams.output_index] = false;
    tilingParams.output_ub_size /= output_shapes[tilingParams.output_index];
    if (tilingParams.can_update_ub) {
      tilingParams.can_update_ub = false;
      input_available_ub = max_available_ub / ((output_available_ub + ele_in_block - 1) / ele_in_block * ele_in_block);
      input_available_ub = input_available_ub / ele_in_block * ele_in_block;
    }
  } else if (tilingParams.output_ub_size * output_shapes[tilingParams.output_index] < output_available_ub) {
    tilingParams.output_ub_size *= output_shapes[tilingParams.output_index];
    have_splits[tilingParams.output_index] = true;
    tilingParams.output_index--;
  }
  return true;
}

template <typename T>
bool Transpose<T>::InputCrossUbTiling(CrossUbTilingParams& tilingParams) {
  if (have_splits[permute[tilingParams.input_index]]) {
    int64_t ele_in_block = GetElementByType(dtype);
    if (tilingParams.cross_ub_in > tilingParams.input_ub_size) {
      return false;
    }
    input_available_ub = tilingParams.input_ub_size;
    tilingParams.input_index++;
    have_splits[permute[tilingParams.input_index]] = false;
    tilingParams.input_ub_size /= input_shapes[tilingParams.input_index];
    if (tilingParams.can_update_ub) {
      tilingParams.can_update_ub = false;
      output_available_ub = max_available_ub / ((input_available_ub + ele_in_block - 1) / ele_in_block * ele_in_block);
      output_available_ub = output_available_ub / ele_in_block * ele_in_block;
    }
  } else if (tilingParams.input_ub_size * input_shapes[tilingParams.input_index] < input_available_ub) {
    tilingParams.input_ub_size *= input_shapes[tilingParams.input_index];
    have_splits[permute[tilingParams.input_index]] = true;
    tilingParams.input_index--;
  }
  return true;
}

template <typename T>
void Transpose<T>::DoStoreAlignBlockTiling() {
  if (!need_multi_core) {
    return;
  }
  for (int64_t i = 0; i < high_ub_axis; i++) {
    if (block_dims * output_shapes[i] >= core_num) {
      block_axis = i;
      int64_t left_core = core_num / block_dims;
      block_factor = (output_shapes[i] + left_core - 1) / left_core;
      block_dims *= ((output_shapes[i] + block_factor - 1) / block_factor);
      break;
    } else {
      block_dims *= output_shapes[i];
    }
  }
  if (block_axis == -1) {
    block_axis = high_ub_axis;
    int64_t remainder_low = (output_shapes[high_ub_axis] + high_ub_factor - 1) / high_ub_factor;
    int64_t left_core = core_num / block_dims;
    block_factor = (remainder_low + left_core - 1) / left_core;
    block_dims *= ((remainder_low + block_factor - 1) / block_factor);
  }
}

template <typename T>
void Transpose<T>::DoStoreAlignUbTiling(int64_t available_ub) {
  int64_t output_index = permute.size() - 1;
  int64_t output_ub_size = 1;
  int64_t last_output = output_shapes[output_index];
  int64_t ele_in_block = GetElementByType(dtype);
  output_shapes[output_index] = (last_output + ele_in_block - 1) / ele_in_block * ele_in_block;
  int64_t align_size = output_size / last_output * output_shapes[output_index];
  if (available_ub * core_num > align_size && align_size > available_ub && !is_pure_copy) {
    int64_t multi_core = 1;
    for (int64_t i = 0; i <= output_index; i++) {
      if (multi_core * output_shapes[i] >= core_num) {
        align_size /= (multi_core * output_shapes[i]);
        int64_t left_core = core_num / multi_core;
        int64_t factor = (output_shapes[i] + left_core - 1) / left_core;
        align_size *= factor;
        break;
      } else {
        multi_core *= output_shapes[i];
      }
    }
    available_ub = std::min(available_ub, align_size);
  }
  while (output_index >= 0) {
    if (output_ub_size * output_shapes[output_index] >= available_ub) {
      high_ub_factor = available_ub / output_ub_size;
      high_ub_axis = output_index;
      break;
    } else {
      output_ub_size *= output_shapes[output_index];
    }
    output_index--;
  }
  output_shapes[permute.size() - 1] = last_output;
  if (high_ub_axis == static_cast<int64_t>(permute.size() - 1)) {
    high_ub_factor = std::min(output_shapes[high_ub_axis], high_ub_factor);
  }
  if (high_ub_axis < 0) {
    high_ub_factor = output_shapes[0];
    high_ub_axis = 0;
    need_multi_core = false;
  }
  // update output for no overlap
  UbNoOverlap(output_ub_size);
}

template <typename T>
bool Transpose<T>::InitUbTilingParams(UbTilingParams& ubTilingParams, int64_t available_ub) {
  int64_t dim_len = ori_permute.size();
  V_OP_TILING_CHECK((dim_len > 0), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input dims must greater 0"), return false);
  ubTilingParams.input_index = dim_len - 1;
  ubTilingParams.output_index = dim_len - 1;
  ubTilingParams.cur_ub_size = available_ub;
  ubTilingParams.input_ub_size = 1;
  ubTilingParams.output_ub_size = 1;
  input_available_ub = std::sqrt(available_ub);
  output_available_ub = available_ub / input_available_ub;
  int64_t ele_in_block = GetElementByType(dtype);
  if (is_last_transpose && ele_in_block == ELEMENT_IN_BLOCK_B32 &&
      input_shapes[ubTilingParams.input_index] > ele_in_block) {
    input_available_ub = input_available_ub / ele_in_block * ele_in_block;
    output_available_ub = available_ub / input_available_ub;
  }
  ubTilingParams.no_update_cross = false;
  have_splits.fill(false);
  return true;
}

template <typename T>
bool Transpose<T>::DoUbTiling(int64_t available_ub) {
  UbTilingParams ubTilingParams;
  if (!InitUbTilingParams(ubTilingParams, available_ub)) {
    return false;
  }
  while (ubTilingParams.input_index >= 0 && ubTilingParams.output_index >= 0) {
    InputUbTiling(ubTilingParams);

    OutputUbTiling(ubTilingParams);

    bool no_available_ub =
        ubTilingParams.input_ub_size * input_shapes[ubTilingParams.input_index] >= input_available_ub &&
        ubTilingParams.output_ub_size * output_shapes[ubTilingParams.output_index] >= output_available_ub;
    if (no_available_ub) {
      bool had_split =
          have_splits[permute[ubTilingParams.input_index]] || have_splits[ubTilingParams.output_index];
      if (had_split) {
        continue;
      }
      bool split_finish = UbSplitSameAxis(ubTilingParams);
      if (split_finish) {
        low_ub_axis = permute[ubTilingParams.input_index];
        high_ub_axis = ubTilingParams.output_index;
        have_splits[low_ub_axis] = true;
        have_splits[high_ub_axis] = true;
        break;
      }
    }
  }
  if (low_ub_axis == -1) {
    need_multi_core = false;
    ubTilingParams.input_index = 0;
    ubTilingParams.output_index = 0;
    low_ub_axis = ubTilingParams.input_index;
    high_ub_axis = ubTilingParams.output_index;
    have_splits[low_ub_axis] = true;
    have_splits[high_ub_axis] = true;
    low_ub_factor = output_shapes[low_ub_axis];
    high_ub_factor = output_shapes[high_ub_axis];
  }
  return true;
}

template <typename T>
bool Transpose<T>::UbSplitSameAxis(UbTilingParams& tilingParams) {
  bool split_finish = true;
  if (permute[tilingParams.input_index] == tilingParams.output_index) {
    int64_t remainder_ub = tilingParams.cur_ub_size / (tilingParams.input_ub_size * tilingParams.output_ub_size);
    if (remainder_ub >= output_shapes[tilingParams.output_index]) {
      remainder_ub /= output_shapes[tilingParams.output_index];
      have_splits[tilingParams.output_index] = true;
      if (tilingParams.input_ub_size > tilingParams.output_ub_size) {
        input_available_ub = tilingParams.input_ub_size;
        output_available_ub =
            tilingParams.cur_ub_size / (input_available_ub * output_shapes[tilingParams.output_index]);
        tilingParams.no_update_cross = true;
      } else if (tilingParams.input_ub_size < tilingParams.output_ub_size) {
        output_available_ub = tilingParams.output_ub_size;
        input_available_ub =
            tilingParams.cur_ub_size / (output_available_ub * output_shapes[tilingParams.output_index]);
        tilingParams.no_update_cross = true;
      } else {
        input_available_ub = sqrt(remainder_ub);
        output_available_ub = remainder_ub / input_available_ub;
      }
      tilingParams.cur_ub_size /= output_shapes[tilingParams.output_index];
      tilingParams.input_index--;
      tilingParams.output_index--;
      split_finish = false;
    } else {
      low_ub_factor = remainder_ub;
      high_ub_factor = remainder_ub;
    }
  } else {
    low_ub_factor = input_available_ub / tilingParams.input_ub_size;
    high_ub_factor = output_available_ub / tilingParams.output_ub_size;
  }
  return split_finish;
}

template <typename T>
void Transpose<T>::OutputUbTiling(UbTilingParams& tilingParams) {
  if (have_splits[tilingParams.output_index]) {
    if (!tilingParams.no_update_cross) {
      output_available_ub = sqrt(tilingParams.cur_ub_size / output_shapes[tilingParams.output_index]);
      if (output_available_ub < tilingParams.output_ub_size) {
        output_available_ub = tilingParams.output_ub_size;
        input_available_ub =
            tilingParams.cur_ub_size / (output_available_ub * output_shapes[tilingParams.output_index]);
        tilingParams.no_update_cross = true;
      } else {
        input_available_ub = output_available_ub;
      }
      tilingParams.cur_ub_size /= output_shapes[tilingParams.output_index];
      tilingParams.input_ub_size /= output_shapes[tilingParams.output_index];
    }
    tilingParams.output_index--;
  } else if (tilingParams.output_ub_size * output_shapes[tilingParams.output_index] < output_available_ub) {
    tilingParams.output_ub_size *= output_shapes[tilingParams.output_index];
    have_splits[tilingParams.output_index] = true;
    tilingParams.output_index--;
  }
}

template <typename T>
void Transpose<T>::InputUbTiling(UbTilingParams& tilingParams) {
  if (have_splits[permute[tilingParams.input_index]]) {
    if (!tilingParams.no_update_cross) {
      input_available_ub = sqrt(tilingParams.cur_ub_size / input_shapes[tilingParams.input_index]);
      if (input_available_ub < tilingParams.input_ub_size) {
        input_available_ub = tilingParams.input_ub_size;
        output_available_ub = tilingParams.cur_ub_size / (input_available_ub * input_shapes[tilingParams.input_index]);
        tilingParams.no_update_cross = true;
      } else {
        output_available_ub = input_available_ub;
      }
      tilingParams.cur_ub_size /= input_shapes[tilingParams.input_index];
      tilingParams.output_ub_size /= input_shapes[tilingParams.input_index];
    }
    tilingParams.input_index--;
  } else if (tilingParams.input_ub_size * input_shapes[tilingParams.input_index] < input_available_ub) {
    tilingParams.input_ub_size *= input_shapes[tilingParams.input_index];
    have_splits[permute[tilingParams.input_index]] = true;
    tilingParams.input_index--;
  }
}

template <typename T>
void Transpose<T>::AdjustUbTiling() {
  if (!is_last_transpose && high_ub_axis == static_cast<int64_t>(permute.size() - 1)) {
    return;
  }
  AdjustTilingParams adjustTilingParams;
  adjustTilingParams.max_ub = max_available_ub;
  if (is_last_transpose && !first_last_transpose) {
    adjustTilingParams.max_ub = max_available_cross_ub;
  }
  adjustTilingParams.all_in_ub = 1;
  adjustTilingParams.input_in_ub = 1;
  adjustTilingParams.output_in_ub = 1;
  adjustTilingParams.input_index = ori_permute[low_ub_axis];
  CalcInUbSize(adjustTilingParams);

  AdjustInputFactor(adjustTilingParams);
  // update by output
  AdjustOutputFactor(adjustTilingParams);
  // update output for no overlap
  UbNoOverlap(adjustTilingParams.output_in_ub);
}

template <typename T>
void Transpose<T>::UbNoOverlap(int64_t output_in_ub) {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t tail = output_shapes[high_ub_axis] % high_ub_factor;
  if (tail > 0 && tail * output_in_ub < ele_in_block) {
    int64_t factor = high_ub_factor - 1;
    for (; factor > 0; factor--) {
      if (output_shapes[high_ub_axis] % factor == 0 ||
          (output_shapes[high_ub_axis] % factor) * output_in_ub >= ele_in_block) {
        break;
      }
    }
    if (factor == 1) {
      core_num = 1;
      return;
    }
    high_ub_factor = factor;
    if (low_ub_axis == high_ub_axis) {
      low_ub_factor = high_ub_factor;
    }
  }
}

template <typename T>
bool Transpose<T>::AdjustJudge(const AdjustTilingParams& adjustTilingParams, bool greater_input) const {
  int64_t input_index = adjustTilingParams.input_index;
  int64_t input_in_size = adjustTilingParams.input_in_ub;
  int64_t cur_input_ub_factor = low_ub_factor;
  int64_t dim_len = permute.size() - 1;
  while (cur_input_ub_factor == 1 || (input_index < dim_len && permute[input_index] > high_ub_axis)) {
    input_index++;
    cur_input_ub_factor = input_shapes[input_index];
    input_in_size /= cur_input_ub_factor;
  }
  int64_t output_index = high_ub_axis;
  int64_t output_in_size = adjustTilingParams.output_in_ub;
  int64_t cur_output_ub_factor = high_ub_factor;
  while (cur_output_ub_factor == 1 ||
         (output_index < dim_len && ori_permute[output_index] > adjustTilingParams.input_index)) {
    output_index++;
    cur_output_ub_factor = output_shapes[output_index];
    output_in_size /= cur_output_ub_factor;
  }
  bool no_same_axis = output_index != permute[input_index];
  if (greater_input) {
    return no_same_axis && input_in_size * cur_input_ub_factor >= output_in_size * cur_output_ub_factor;
  }
  return no_same_axis && output_in_size * cur_output_ub_factor > input_in_size * cur_input_ub_factor;
}

template <typename T>
void Transpose<T>::UpdateFactorOne(AdjustTilingParams& adjustTilingParams) {
  if (high_ub_factor != 1 || low_ub_factor != 1) {
    return;
  }
  int64_t dim_len = permute.size() - 1;
  while (low_ub_factor == 1 ||
         (adjustTilingParams.input_index < dim_len && permute[adjustTilingParams.input_index] > high_ub_axis)) {
    have_splits[permute[adjustTilingParams.input_index]] = false;
    adjustTilingParams.input_index++;
    low_ub_factor = input_shapes[adjustTilingParams.input_index];
    adjustTilingParams.input_in_ub /= low_ub_factor;
  }
  while (high_ub_factor == 1 ||
         (high_ub_axis < dim_len && permute[high_ub_axis] > adjustTilingParams.input_index)) {
    have_splits[high_ub_axis] = false;
    high_ub_axis++;
    high_ub_factor = output_shapes[high_ub_axis];
    adjustTilingParams.output_in_ub /= high_ub_factor;
  }
  low_ub_axis = permute[adjustTilingParams.input_index];
}

template <typename T>
void Transpose<T>::AdjustOutputFactor(AdjustTilingParams& adjustTilingParams) {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t output_no_continues = adjustTilingParams.all_in_ub / (adjustTilingParams.output_in_ub * high_ub_factor);
  int64_t output_align =
      (adjustTilingParams.output_in_ub * high_ub_factor + ele_in_block - 1) / ele_in_block * ele_in_block;
  if (output_align * output_no_continues > adjustTilingParams.max_ub && output_no_continues != 1) {
    // high_ub_factor is 1
    UpdateFactorOne(adjustTilingParams);
    if (AdjustJudge(adjustTilingParams, true)) {
      while (low_ub_factor == 1 || permute[adjustTilingParams.input_index] > high_ub_axis) {
        have_splits[permute[adjustTilingParams.input_index]] = false;
        adjustTilingParams.input_index++;
        low_ub_factor = input_shapes[adjustTilingParams.input_index];
      }
      int64_t new_factor = adjustTilingParams.max_ub / output_align / (output_no_continues / low_ub_factor);
      low_ub_factor = new_factor;
      low_ub_axis = permute[adjustTilingParams.input_index];
      if (low_ub_axis == high_ub_axis) {
        high_ub_factor = low_ub_factor;
      }
    } else {
      while (high_ub_factor == 1 || ori_permute[high_ub_axis] > adjustTilingParams.input_index) {
        have_splits[high_ub_axis] = false;
        high_ub_axis++;
        high_ub_factor = output_shapes[high_ub_axis];
        adjustTilingParams.output_in_ub /= high_ub_factor;
      }
      output_align =
          (adjustTilingParams.output_in_ub * high_ub_factor + ele_in_block - 1) / ele_in_block * ele_in_block;
      high_ub_factor = (output_align - ele_in_block) / adjustTilingParams.output_in_ub;
      if (low_ub_axis == high_ub_axis) {
        low_ub_factor = high_ub_factor;
      }
    }
  }
}

template <typename T>
void Transpose<T>::AdjustInputFactor(AdjustTilingParams& adjustTilingParams) {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t input_no_continues = adjustTilingParams.all_in_ub / (adjustTilingParams.input_in_ub * low_ub_factor);
  int64_t input_align =
      (adjustTilingParams.input_in_ub * low_ub_factor + ele_in_block - 1) / ele_in_block * ele_in_block;
  if (input_align * input_no_continues > adjustTilingParams.max_ub && input_no_continues != 1) {
    // low_ub_factor is 1
    UpdateFactorOne(adjustTilingParams);
    if (AdjustJudge(adjustTilingParams, false)) {
      while (high_ub_factor == 1 || ori_permute[high_ub_axis] > adjustTilingParams.input_index) {
        have_splits[high_ub_axis] = false;
        high_ub_axis++;
        high_ub_factor = output_shapes[high_ub_axis];
      }
      int64_t new_factor = adjustTilingParams.max_ub / input_align / (input_no_continues / high_ub_factor);
      adjustTilingParams.all_in_ub = adjustTilingParams.all_in_ub / high_ub_factor * new_factor;
      high_ub_factor = new_factor;
      if (low_ub_axis == high_ub_axis) {
        low_ub_factor = high_ub_factor;
      }
    } else {
      while (low_ub_factor == 1 || permute[adjustTilingParams.input_index] > high_ub_axis) {
        have_splits[permute[adjustTilingParams.input_index]] = false;
        adjustTilingParams.input_index++;
        low_ub_factor = input_shapes[adjustTilingParams.input_index];
        adjustTilingParams.input_in_ub /= low_ub_factor;
      }
      input_align = (adjustTilingParams.input_in_ub * low_ub_factor + ele_in_block - 1) / ele_in_block * ele_in_block;
      int64_t new_factor = (input_align - ele_in_block) / adjustTilingParams.input_in_ub;
      adjustTilingParams.all_in_ub = adjustTilingParams.all_in_ub / low_ub_factor * new_factor;
      low_ub_factor = new_factor;
      low_ub_axis = permute[adjustTilingParams.input_index];
      if (low_ub_axis == high_ub_axis) {
        high_ub_factor = low_ub_factor;
      }
    }
  }
}

template <typename T>
void Transpose<T>::CalcInUbSize(AdjustTilingParams& adjustTilingParams) const {
  size_t dim_len = ori_permute.size();
  for (size_t i = 0; i < dim_len; i++) {
    if (have_splits[i]) {
      if (i == static_cast<size_t>(low_ub_axis)) {
        adjustTilingParams.all_in_ub *= low_ub_factor;
      } else if (i == static_cast<size_t>(high_ub_axis)) {
        adjustTilingParams.all_in_ub *= high_ub_factor;
      } else {
        adjustTilingParams.all_in_ub *= output_shapes[i];
      }
    }
  }
  size_t input_split = adjustTilingParams.input_index + 1;
  for (; input_split < dim_len; input_split++) {
    adjustTilingParams.input_in_ub *= input_shapes[input_split];
  }
  for (size_t i = high_ub_axis + 1; i < dim_len; i++) {
    adjustTilingParams.output_in_ub *= output_shapes[i];
  }
}

template <typename T>
void Transpose<T>::CalcKey() {
  int64_t last_index = permute.size() - 1;
  if (is_pure_copy) {
    tiling_key = PURE_COPY_BASE_KEY;
  } else if (is_nlast_no_conv && high_ub_axis != last_index) {
    tiling_key = N_LAST_NO_CONV_BASE_KEY;
  } else if (is_nlast_align) {
    tiling_key = N_LAST_ALIGN_BASE_KEY;
  } else {
    tiling_key = GENERAL_BASE_KEY;
  }
  if (block_axis == -1) {
    tiling_key = NO_MULTI_BLOCK_BASE_KEY;
    if (is_nlast_no_conv) {
      tiling_key = ONE_DIMS_N_LAST_NO_CONV_BASE_KEY;
    } else if (is_nlast_align) {
      tiling_key = ONE_DIMS_N_LAST_ALIGN_BASE_KEY;
    }
  } else {
    if (is_nlast_align || is_pure_copy) {
      tiling_key = tiling_key + block_axis * BLOCK_AXIS_FACTOR + high_ub_axis;
    } else {
      tiling_key = tiling_key + block_axis * BLOCK_AXIS_FACTOR + low_ub_axis * LOW_SPLIT_AXIS_FACTOR + high_ub_axis;
    }
  }
}

template <typename T>
bool Transpose<T>::DoBlockAndUbTiling() {
  if (!need_multi_core) {
    return true;
  }
  bool ret = true;
  if (is_nlast_align || is_pure_copy) {
    DoStoreAlignUbTiling(max_available_ub);
    DoStoreAlignBlockTiling();
  } else {
    if (is_last_transpose) {
      first_last_transpose = DoUbTilingNoCross();
      if (!first_last_transpose) {
        ret = DoUbTiling(max_available_cross_ub);
      }
    } else {
      ret = DoUbTiling(max_available_ub);
    }
    if (need_multi_core) {
      if (ret) {
        AdjustUbTiling();
      }
      ret = ret && DoBlockTiling();
    }
  }
  return ret;
}

template <typename T>
void Transpose<T>::Init() {
  permute = c_info->permute;
  ori_permute = c_info->ori_permute;
  core_num = c_info->core_num;
}

template <typename T>
bool Transpose<T>::DoTiling() {
  Init();
  bool ret = true;
  if (op_info != nullptr) {
    ret = ret && GenerateOutputShapeFromOp();
  } else {
    ret = ret && GenerateOutputShape();
  }
  ret = ret && CalcOutputSize();
  ret = ret && CalcTiling();
  ret = ret && DoBlockAndUbTiling();
  if (!need_multi_core && is_pure_copy) {
    high_ub_axis = 0;
    block_axis = 0;
    block_factor = 1;
    high_ub_factor = max_dim_shape;
  }
  if (ret) {
    CalcKey();
  }
  return ret;
}

template <typename T>
void Transpose<T>::WriteConstTilingData() const {
  context->Append(static_cast<int32_t>(need_multi_core));
  context->Append(static_cast<int32_t>(block_axis));
  context->Append(static_cast<int32_t>(block_factor));
  if (!is_nlast_align && !is_pure_copy) {
    context->Append(static_cast<int32_t>(low_ub_axis));
    context->Append(static_cast<int32_t>(low_ub_factor));
  }
  context->Append(static_cast<int32_t>(high_ub_axis));
  context->Append(static_cast<int32_t>(high_ub_factor));
}

template <typename T>
bool Transpose<T>::WriteTilingData() const {
  OP_LOGD(op_type.c_str(), "tiling key:%lld", tiling_key);
  OP_LOGD(op_type.c_str(), "tiling block_dims:%lld", block_dims);
  OP_LOGD(op_type.c_str(), "tiling block_factor:%lld", block_factor);
  OP_LOGD(op_type.c_str(), "tiling low_ub_factor:%lld", low_ub_factor);
  OP_LOGD(op_type.c_str(), "tiling high_ub_factor:%lld", high_ub_factor);
  OP_LOGD(op_type.c_str(), "tiling block_axis:%lld", block_axis);
  OP_LOGD(op_type.c_str(), "tiling low_ub_axis:%lld", low_ub_axis);
  OP_LOGD(op_type.c_str(), "tiling high_ub_axis:%lld", high_ub_axis);

  context->SetBlockDim(static_cast<uint32_t>(block_dims));
  if (c_info->only_const_tiling) {
    WriteConstTilingData();
    return true;
  }
  context->SetTilingKey(tiling_key);
  if (is_pure_copy) {
    context->Append(static_cast<int32_t>(input_shapes[0]));
    context->Append(static_cast<int32_t>(block_factor));
    context->Append(static_cast<int32_t>(high_ub_factor));
    return true;
  }
  try {
    size_t dim_len = c_info->transpose_vars.size();
    V_OP_TILING_CHECK((dim_len == c_info->permute.size()),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "transpose variable error"), return false);
    for (size_t i = 0; i < dim_len; i++) {
      if (c_info->transpose_vars[i]) {
        context->Append(static_cast<int32_t>(input_shapes[i]));
      }
    }
  } catch (const std::exception& e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile_info[_transpose_vars] error. Error message: %s", e.what());
    return false;
  }
  if (need_multi_core) {
    context->Append(static_cast<int32_t>(block_factor));
    if (!is_nlast_align) {
      context->Append(static_cast<int32_t>(low_ub_factor));
    }
    if (low_ub_axis != high_ub_axis) {
      context->Append(static_cast<int32_t>(high_ub_factor));
    }
  }
  return true;
}

template <typename T>
void Transpose<T>::ProcessConst() const {
  context->SetTilingKey(CONST_TILING_KEY);
  context->SetBlockDim(static_cast<uint32_t>(c_info->const_block_dims));
}

template <typename T>
bool Transpose<T>::TransposeTiling() {
  op_type = context->GetOpType();
  c_info = dynamic_cast<const TransposeCompileInfo *>(context->GetCompileInfo());
  if (c_info->is_const) {
    ProcessConst();
    return true;
  }
  bool ret = DoTiling();
  ret = ret && WriteTilingData();
  return ret;
}
}  // namespace transpose

bool CreateTransposeDslTiling(gert::TilingContext* context, const OpInfoImpl* op_info) {
  OP_LOGD("TransposeDsl", "Enter Transpose Dsl");
  AutoTilingContext auto_tiling_context(context);
  if (op_info) {
    auto_tiling_context.SetCompileInfo(op_info->GetCompileInfo());
  }
  transpose::Transpose<AutoTilingContext> transpose(&auto_tiling_context, op_info);
  return transpose.TransposeTiling();
}

AutoTilingCompileInfo* CreateTransposeDslParser(const char* op_type, const nlohmann::json& json_compile_info) {
  auto compile_info = new transpose::TransposeCompileInfo();
  if (!compile_info->Parse(op_type, json_compile_info)) {
    return nullptr;
  }
  return compile_info;
}

bool TransposeDslTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD("TransposeDsl", "Enter Transpose Dsl");
  AutoTilingOp auto_tiling_op(op_type.c_str(), &op_paras, &compile_info, &run_info);
  transpose::Transpose<AutoTilingOp> transpose(&auto_tiling_op, nullptr);
  return transpose.TransposeTiling();
}

bool TransposeDslTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                         const OpInfo& op_info) const {
  OP_LOGD("TransposeDsl", "Enter Transpose Dsl OpInfo");
  AutoTilingOp auto_tiling_op(op_type.c_str(), &op_paras, &compile_info, &run_info);
  transpose::Transpose<AutoTilingOp> transpose(&auto_tiling_op, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  return transpose.TransposeTiling();
}

std::shared_ptr<AutoTilingHandler> CreateTransposeDslTilingHandler(const std::string& op_type,
                                                                   const std::string& pattern,
                                                                   const nlohmann::json& parsed_compile_info) {
  return std::make_shared<TransposeDslTilingHandler>(op_type, pattern, parsed_compile_info);
}

REGISTER_AUTO_TILING(SchPattern::TRANSPOSE, CreateTransposeDslTiling, CreateTransposeDslParser)
}  // namespace optiling
