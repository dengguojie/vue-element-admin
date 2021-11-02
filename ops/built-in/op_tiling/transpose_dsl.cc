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
#include "vector_tiling_log.h"
#include "vector_tiling.h"
#include "error_log.h"

namespace optiling {
namespace utils {
namespace transpose {
const std::int32_t ELEMENT_IN_BLOCK_DEFAULT = 16;
const std::int32_t ELEMENT_IN_BLOCK_B32 = 8;
const std::int32_t ELEMENT_IN_BLOCK_B8 = 32;
const std::int32_t ELEMENT_IN_BLOCK_B64 = 4;

const int64_t GetElementByType(const ge::DataType& dtype) {
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
}  // namespace transpose

bool Transpose::Init() {
  try {
    c_info.core_num = compile_info.at("_core_num");
    c_info.ub_size = compile_info.at("_ub_size");
    c_info.only_const_tiling = compile_info.at("_only_const_tiling");
    c_info.permute = compile_info.at("_permute").get<std::vector<int64_t>>();
    c_info.ori_permute = compile_info.at("_ori_permute").get<std::vector<size_t>>();
    c_info.mergeable = compile_info.at("_mergeable").get<std::vector<int64_t>>();
    if (!c_info.only_const_tiling) {
      c_info.transpose_vars = compile_info.at("_transpose_vars").get<std::vector<bool>>();
    }
  } catch (const std::exception& e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info params error. Error message: %s", e.what());
    return false;
  }
  V_OP_TILING_CHECK((op_paras.GetInputsSize() != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape cannot be empty"), return false);
  const ge::GeShape& shape = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(0)->MutableShape();
  size_t src_dim_len = shape.GetDimNum();
  V_OP_TILING_CHECK((src_dim_len == c_info.mergeable.size()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input dim length error"), return false);
  int64_t real_index = 0;
  bool is_first_in = true;
  for (size_t i = 0; i < src_dim_len; i++) {
    if (c_info.mergeable[i] == 2) {
      continue;
    } else if (c_info.mergeable[i] == 1) {
      input_shapes[real_index] = input_shapes[real_index] * shape.GetDim(i);
    } else {
      if (is_first_in) {
        is_first_in = false;
      } else {
        real_index++;
      }
      input_shapes[real_index] = shape.GetDim(i);
    }
  }
  V_OP_TILING_CHECK((op_paras.GetOutputsSize() != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output shape cannot be empty"), return false);
  const auto& out_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableOutputDesc(0);
  dtype = out_desc->GetDataType();
  size_t dst_dim_len = out_desc->MutableShape().GetDimNum();
  V_OP_TILING_CHECK((dst_dim_len == c_info.mergeable.size()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output dim length error"), return false);
  for (size_t i = 0; i < c_info.ori_permute.size(); i++) {
    output_shapes[i] = input_shapes[c_info.ori_permute[i]];
    max_dim_shape = std::max(max_dim_shape, output_shapes[i]);
  }
  return true;
}

bool Transpose::CalcTiling() {
  V_OP_TILING_CHECK((!c_info.permute.empty()), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape cannot be empty"),
                    return false);
  output_size = std::accumulate(output_shapes.begin(), output_shapes.begin() + c_info.permute.size(), 1LL,
                                std::multiplies<int64_t>());
  V_CHECK_LE(output_size, INT32_MAX, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The output shape is too large"),
             return false);
  V_CHECK_GT(output_size, 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The output shape must be greater than 0"),
             return false);
  if (max_dim_shape == output_size) {
    is_pure_copy = true;
    c_info.permute = {0};
    c_info.ori_permute = {0};
    input_shapes[0] = max_dim_shape;
    output_shapes[0] = max_dim_shape;
  }
  int64_t last_index = c_info.permute.size() - 1;
  is_last_transpose = c_info.permute[last_index] != last_index;
  int64_t coexisting_quantity = 2;
  int64_t factor = transpose::GetElementByType(dtype) == 32 ? 32 : 16;
  int64_t cross_coexisting_quantity = (factor * 2) + coexisting_quantity;
  int64_t ele_in_block = transpose::GetElementByType(dtype);
  int64_t bytes = transpose::BLOCK_SIZE / ele_in_block;
  is_nlast_no_conv = !is_last_transpose && output_shapes[last_index] % ele_in_block == 0;
  if (is_last_transpose) {
    if (transpose::GetElementByType(dtype) < 16) {
      coexisting_quantity = 4;
    }
    max_available_ub = c_info.ub_size / coexisting_quantity / bytes / ele_in_block * ele_in_block;
    max_available_cross_ub = c_info.ub_size / cross_coexisting_quantity / bytes / ele_in_block * ele_in_block;
  } else if (is_nlast_no_conv) {
    max_available_ub = c_info.ub_size / coexisting_quantity / bytes / ele_in_block * ele_in_block;
  } else {
    is_nlast_align =
        !is_last_transpose && (output_shapes[last_index] % ele_in_block == 0 ||
                               output_shapes[last_index] >= (transpose::ALIGN_THRESHOLD / (32 / ele_in_block)));
    if (is_nlast_align) {
      max_available_ub = (c_info.ub_size - 32) / bytes / ele_in_block * ele_in_block;
    } else {
      max_available_ub = c_info.ub_size / cross_coexisting_quantity / bytes / ele_in_block * ele_in_block;
    }
  }
  if (is_pure_copy) {
    max_available_ub = c_info.ub_size / 2 / bytes / ele_in_block * ele_in_block;
  }
  const int64_t multi_core_threshold = 64 * transpose::GetElementByType(dtype);  // 2k
  if (output_size > multi_core_threshold) {
    need_multi_core = true;
  }
  return true;
}

bool Transpose::DoBlockTiling() {
  for (int64_t i = 0; i < high_ub_axis; i++) {
    if (!have_splits[i]) {
      if (block_dims * output_shapes[i] >= c_info.core_num) {
        block_axis = i;
        block_factor = std::ceil(output_shapes[i] * 1.0 / (c_info.core_num / block_dims));
        block_dims *= std::ceil(output_shapes[i] * 1.0 / block_factor);
        break;
      } else {
        block_dims *= output_shapes[i];
      }
    }
  }
  if (block_axis == -1) {
    int64_t remainder_low = std::ceil(output_shapes[low_ub_axis] * 1.0 / low_ub_factor);
    if (block_dims * remainder_low >= c_info.core_num) {
      block_axis = low_ub_axis;
      block_factor = std::ceil(remainder_low * 1.0 / (c_info.core_num / block_dims));
      block_dims *= std::ceil(remainder_low * 1.0 / block_factor);
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
    int64_t remainder_low = std::ceil(output_shapes[high_ub_axis] * 1.0 / high_ub_factor);
    block_factor = std::ceil(remainder_low * 1.0 / (c_info.core_num / block_dims));
    block_dims *= std::ceil(remainder_low * 1.0 / block_factor);
  }
  return true;
}

bool Transpose::DoUbTilingNoCross() {
  transpose::CrossUbTilingParams tilingParams;
  int64_t dim_len = c_info.ori_permute.size();
  V_OP_TILING_CHECK((dim_len > 0), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input dims must greater 0"), return false);
  tilingParams.input_index = dim_len - 1;
  tilingParams.output_index = dim_len - 1;
  tilingParams.input_ub_size = 1;
  tilingParams.output_ub_size = 1;
  input_available_ub = std::sqrt(max_available_ub);
  output_available_ub = max_available_ub / input_available_ub;
  int64_t ele_in_block = transpose::GetElementByType(dtype);
  tilingParams.cross_ub_in = std::sqrt(max_available_cross_ub) * 0.7;
  tilingParams.cross_ub_out = std::sqrt(max_available_cross_ub);
  if (output_shapes[tilingParams.output_index] % ele_in_block == 0) {
    tilingParams.cross_ub_in = ele_in_block < 16 ? ele_in_block * 3 : ele_in_block * 2;
    tilingParams.cross_ub_out = tilingParams.cross_ub_in;
  }
  if (ele_in_block < 16) {
    output_available_ub = ele_in_block * 16;
    input_available_ub = max_available_ub / output_available_ub;
  }
  tilingParams.can_update_ub = ele_in_block == 16;
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
      bool had_split = have_splits[c_info.permute[tilingParams.input_index]] || have_splits[tilingParams.output_index];
      if (had_split) {
        continue;
      }
      if (c_info.permute[tilingParams.input_index] == tilingParams.output_index) {
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
      low_ub_axis = c_info.permute[tilingParams.input_index];
      high_ub_axis = tilingParams.output_index;
      have_splits[low_ub_axis] = true;
      have_splits[high_ub_axis] = true;
      break;
    }
  }
  return true;
}

void Transpose::CrossUbUpdateSameAxis(transpose::CrossUbTilingParams& tilingParams) {
  int64_t ele_in_block = transpose::GetElementByType(dtype);
  if (tilingParams.input_ub_size >= tilingParams.output_ub_size) {
    input_available_ub = tilingParams.input_ub_size;
    tilingParams.input_index++;
    have_splits[c_info.permute[tilingParams.input_index]] = false;
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

bool Transpose::OutputCrossUbTiling(transpose::CrossUbTilingParams& tilingParams) {
  if (have_splits[tilingParams.output_index]) {
    int64_t ele_in_block = transpose::GetElementByType(dtype);
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

bool Transpose::InputCrossUbTiling(transpose::CrossUbTilingParams& tilingParams) {
  if (have_splits[c_info.permute[tilingParams.input_index]]) {
    int64_t ele_in_block = transpose::GetElementByType(dtype);
    if (tilingParams.cross_ub_in > tilingParams.input_ub_size) {
      return false;
    }
    input_available_ub = tilingParams.input_ub_size;
    tilingParams.input_index++;
    have_splits[c_info.permute[tilingParams.input_index]] = false;
    tilingParams.input_ub_size /= input_shapes[tilingParams.input_index];
    if (tilingParams.can_update_ub) {
      tilingParams.can_update_ub = false;
      output_available_ub = max_available_ub / ((input_available_ub + ele_in_block - 1) / ele_in_block * ele_in_block);
      output_available_ub = output_available_ub / ele_in_block * ele_in_block;
    }
  } else if (tilingParams.input_ub_size * input_shapes[tilingParams.input_index] < input_available_ub) {
    tilingParams.input_ub_size *= input_shapes[tilingParams.input_index];
    have_splits[c_info.permute[tilingParams.input_index]] = true;
    tilingParams.input_index--;
  }
  return true;
}

bool Transpose::DoStoreAlignBlockTiling() {
  if (!need_multi_core) {
    return true;
  }
  for (int64_t i = 0; i < high_ub_axis; i++) {
    if (block_dims * output_shapes[i] >= c_info.core_num) {
      block_axis = i;
      block_factor = std::ceil(output_shapes[i] * 1.0 / (c_info.core_num / block_dims));
      block_dims *= std::ceil(output_shapes[i] * 1.0 / block_factor);
      break;
    } else {
      block_dims *= output_shapes[i];
    }
  }
  if (block_axis == -1) {
    block_axis = high_ub_axis;
    int64_t remainder_low = std::ceil(output_shapes[high_ub_axis] * 1.0 / high_ub_factor);
    block_factor = std::ceil(remainder_low * 1.0 / (c_info.core_num / block_dims));
    block_dims *= std::ceil(remainder_low * 1.0 / block_factor);
  }
  return true;
}

bool Transpose::DoStoreAlignUbTiling(int64_t available_ub) {
  int64_t output_index = c_info.permute.size() - 1;
  int64_t output_ub_size = 1;
  int64_t last_output = output_shapes[output_index];
  int64_t ele_in_block = transpose::GetElementByType(dtype);
  output_shapes[output_index] = (last_output + ele_in_block - 1) / ele_in_block * ele_in_block;
  int64_t align_size = output_size / last_output * output_shapes[output_index];
  if (available_ub * c_info.core_num > align_size && align_size > available_ub && !is_pure_copy) {
    int64_t multi_core = 1;
    for (int64_t i = 0; i <= output_index; i++) {
      if (block_dims * output_shapes[i] >= c_info.core_num) {
        align_size /= (multi_core * output_shapes[i]);
        int64_t factor = std::ceil(output_shapes[i] * 1.0 / (c_info.core_num / multi_core));
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
  output_shapes[c_info.permute.size() - 1] = last_output;
  if (high_ub_axis == static_cast<int64_t>(c_info.permute.size() - 1)) {
    high_ub_factor = std::min(output_shapes[high_ub_axis], high_ub_factor);
  }
  if (high_ub_axis < 0) {
    high_ub_factor = output_shapes[0];
    high_ub_axis = 0;
    need_multi_core = false;
  }
  // update output for no overlap
  UbNoOverlap(output_ub_size);
  return true;
}

bool Transpose::DoUbTiling(int64_t available_ub) {
  transpose::UbTilingParams ubTilingParams;
  int64_t dim_len = c_info.ori_permute.size();
  V_OP_TILING_CHECK((dim_len > 0), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input dims must greater 0"), return false);
  ubTilingParams.input_index = dim_len - 1;
  ubTilingParams.output_index = dim_len - 1;
  ubTilingParams.cur_ub_size = available_ub;
  ubTilingParams.input_ub_size = 1;
  ubTilingParams.output_ub_size = 1;
  input_available_ub = std::sqrt(available_ub);
  output_available_ub = available_ub / input_available_ub;
  int64_t ele_in_block = transpose::GetElementByType(dtype);
  if (is_last_transpose && ele_in_block == 8 && input_shapes[ubTilingParams.input_index] > ele_in_block) {
    input_available_ub = input_available_ub / ele_in_block * ele_in_block;
    output_available_ub = available_ub / input_available_ub;
  }
  ubTilingParams.no_update_cross = false;
  have_splits.fill(false);
  while (ubTilingParams.input_index >= 0 && ubTilingParams.output_index >= 0) {
    InputUbTiling(ubTilingParams);

    OutputUbTiling(ubTilingParams);

    bool no_available_ub =
        ubTilingParams.input_ub_size * input_shapes[ubTilingParams.input_index] >= input_available_ub &&
        ubTilingParams.output_ub_size * output_shapes[ubTilingParams.output_index] >= output_available_ub;
    if (no_available_ub) {
      bool had_split =
          have_splits[c_info.permute[ubTilingParams.input_index]] || have_splits[ubTilingParams.output_index];
      if (had_split) {
        continue;
      }
      bool split_finish = UbSplitSameAxis(ubTilingParams);
      if (split_finish) {
        low_ub_axis = c_info.permute[ubTilingParams.input_index];
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

bool Transpose::UbSplitSameAxis(transpose::UbTilingParams& tilingParams) {
  bool split_finish = true;
  if (c_info.permute[tilingParams.input_index] == tilingParams.output_index) {
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

void Transpose::OutputUbTiling(transpose::UbTilingParams& tilingParams) {
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

void Transpose::InputUbTiling(transpose::UbTilingParams& tilingParams) {
  if (have_splits[c_info.permute[tilingParams.input_index]]) {
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
    have_splits[c_info.permute[tilingParams.input_index]] = true;
    tilingParams.input_index--;
  }
}

void Transpose::AdjustUbTiling() {
  if (!is_last_transpose && high_ub_axis == static_cast<int64_t>(c_info.permute.size() - 1)) {
    return;
  }
  transpose::AdjustTilingParams adjustTilingParams;
  adjustTilingParams.max_ub = max_available_ub;
  if (is_last_transpose && !first_last_transpose) {
    adjustTilingParams.max_ub = max_available_cross_ub;
  }
  adjustTilingParams.all_in_ub = 1;
  adjustTilingParams.input_in_ub = 1;
  adjustTilingParams.output_in_ub = 1;
  adjustTilingParams.input_index = c_info.ori_permute[low_ub_axis];
  CalcInUbSize(adjustTilingParams);

  AdjustInputFactor(adjustTilingParams);
  // update by output
  AdjustOutputFactor(adjustTilingParams);
  // update output for no overlap
  UbNoOverlap(adjustTilingParams.output_in_ub);
}

void Transpose::UbNoOverlap(int64_t output_in_ub) {
  int64_t ele_in_block = transpose::GetElementByType(dtype);
  int64_t tail = output_shapes[high_ub_axis] % high_ub_factor;
  if (tail > 0 && tail * output_in_ub < ele_in_block) {
    int64_t factor = high_ub_factor - 1;
    for (; factor > 0; factor--) {
      if ((output_shapes[high_ub_axis] % factor) * output_in_ub >= ele_in_block) {
        break;
      }
    }
    high_ub_factor = factor;
    if (low_ub_axis == high_ub_axis) {
      low_ub_factor = high_ub_factor;
    }
  }
}

void Transpose::AdjustOutputFactor(transpose::AdjustTilingParams& adjustTilingParams) {
  int64_t ele_in_block = transpose::GetElementByType(dtype);
  int64_t output_no_continues = adjustTilingParams.all_in_ub / (adjustTilingParams.output_in_ub * high_ub_factor);
  int64_t output_align =
      (adjustTilingParams.output_in_ub * high_ub_factor + ele_in_block - 1) / ele_in_block * ele_in_block;
  if (output_align * output_no_continues > adjustTilingParams.max_ub && output_no_continues != 1) {
    // high_ub_factor is 1
    if (high_ub_factor == 1 &&
        (adjustTilingParams.input_in_ub * low_ub_factor) > (adjustTilingParams.output_in_ub * high_ub_factor)) {
      while (low_ub_factor == 1 || c_info.permute[adjustTilingParams.input_index] > high_ub_axis) {
        have_splits[c_info.permute[adjustTilingParams.input_index]] = false;
        adjustTilingParams.input_index++;
        low_ub_factor = input_shapes[adjustTilingParams.input_index];
        adjustTilingParams.input_in_ub /= low_ub_factor;
      }
      int64_t new_factor = adjustTilingParams.max_ub / output_align / (output_no_continues / low_ub_factor);
      low_ub_factor = new_factor;
      low_ub_axis = c_info.permute[adjustTilingParams.input_index];
      if (low_ub_axis == high_ub_axis) {
        high_ub_factor = low_ub_factor;
      }
    } else {
      if (high_ub_factor == 1) {
        while (high_ub_factor == 1 || c_info.permute[high_ub_axis] > adjustTilingParams.input_index) {
          have_splits[high_ub_axis] = false;
          high_ub_axis++;
          high_ub_factor = output_shapes[high_ub_axis];
          adjustTilingParams.output_in_ub /= high_ub_factor;
        }
      }
      high_ub_factor = (output_align - ele_in_block) / adjustTilingParams.output_in_ub;
      if (low_ub_axis == high_ub_axis) {
        low_ub_factor = high_ub_factor;
      }
    }
  }
}

void Transpose::AdjustInputFactor(transpose::AdjustTilingParams& adjustTilingParams) {
  int64_t ele_in_block = transpose::GetElementByType(dtype);
  int64_t input_no_continues = adjustTilingParams.all_in_ub / (adjustTilingParams.input_in_ub * low_ub_factor);
  int64_t input_align =
      (adjustTilingParams.input_in_ub * low_ub_factor + ele_in_block - 1) / ele_in_block * ele_in_block;
  if (input_align * input_no_continues > adjustTilingParams.max_ub && input_no_continues != 1) {
    // low_ub_factor is 1
    if (low_ub_factor == 1 &&
        (adjustTilingParams.output_in_ub * high_ub_factor) > (adjustTilingParams.input_in_ub * low_ub_factor)) {
      int64_t output_index = high_ub_axis;
      while (high_ub_factor == 1 || c_info.permute[output_index] > adjustTilingParams.input_index) {
        have_splits[high_ub_axis] = false;
        high_ub_axis++;
        high_ub_factor = output_shapes[high_ub_axis];
        adjustTilingParams.output_in_ub /= high_ub_factor;
      }
      int64_t new_factor = adjustTilingParams.max_ub / input_align / (input_no_continues / high_ub_factor);
      adjustTilingParams.all_in_ub = adjustTilingParams.all_in_ub / high_ub_factor * new_factor;
      high_ub_factor = new_factor;
      if (low_ub_axis == high_ub_axis) {
        low_ub_factor = high_ub_factor;
      }
    } else {
      if (low_ub_factor == 1) {
        while (low_ub_factor == 1 || c_info.permute[adjustTilingParams.input_index] > high_ub_axis) {
          have_splits[c_info.permute[adjustTilingParams.input_index]] = false;
          adjustTilingParams.input_index++;
          low_ub_factor = input_shapes[adjustTilingParams.input_index];
          adjustTilingParams.input_in_ub /= low_ub_factor;
        }
      }
      int64_t new_factor = (input_align - ele_in_block) / adjustTilingParams.input_in_ub;
      adjustTilingParams.all_in_ub = adjustTilingParams.all_in_ub / low_ub_factor * new_factor;
      low_ub_factor = new_factor;
      low_ub_axis = c_info.permute[adjustTilingParams.input_index];
      if (low_ub_axis == high_ub_axis) {
        high_ub_factor = low_ub_factor;
      }
    }
  }
}

void Transpose::CalcInUbSize(transpose::AdjustTilingParams& adjustTilingParams) {
  size_t dim_len = c_info.ori_permute.size();
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

void Transpose::CalcKey() {
  int64_t last_index = c_info.permute.size() - 1;
  if (is_pure_copy) {
    tiling_key = 3000000;
  } else if (is_nlast_no_conv && high_ub_axis != last_index) {
    tiling_key = 2020000;
  } else if (is_nlast_align) {
    tiling_key = 2030000;
  } else {
    tiling_key = 2000000;
  }
  if (block_axis == -1) {
    tiling_key = 0;
    if (is_nlast_no_conv) {
      tiling_key = 20000;
    } else if (is_nlast_align) {
      tiling_key = 10000;
    }
  } else {
    if (is_nlast_align || is_pure_copy) {
      tiling_key = tiling_key + block_axis * 100 + high_ub_axis;
    } else {
      tiling_key = tiling_key + block_axis * 100 + low_ub_axis * 10 + high_ub_axis;
    }
  }
}

bool Transpose::DoTiling() {
  bool ret = Init();
  ret = ret && CalcTiling();
  if (need_multi_core) {
    if (is_nlast_align || is_pure_copy) {
      ret = ret && DoStoreAlignUbTiling(max_available_ub);
      ret = ret && DoStoreAlignBlockTiling();
    } else {
      if (ret && is_last_transpose) {
        first_last_transpose = DoUbTilingNoCross();
        if (!first_last_transpose) {
          ret = DoUbTiling(max_available_cross_ub);
        }
      } else {
        ret = ret && DoUbTiling(max_available_ub);
      }
      if (need_multi_core) {
        if (ret) {
          AdjustUbTiling();
        }
        ret = ret && DoBlockTiling();
      }
    }
  }
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

bool Transpose::WriteTilingData(OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "tiling key:%lld", tiling_key);
  OP_LOGD(op_type.c_str(), "tiling block_dims:%lld", block_dims);
  OP_LOGD(op_type.c_str(), "tiling block_factor:%lld", block_factor);
  OP_LOGD(op_type.c_str(), "tiling low_ub_factor:%lld", low_ub_factor);
  OP_LOGD(op_type.c_str(), "tiling high_ub_factor:%lld", high_ub_factor);
  OP_LOGD(op_type.c_str(), "tiling block_axis:%lld", block_axis);
  OP_LOGD(op_type.c_str(), "tiling low_ub_axis:%lld", low_ub_axis);
  OP_LOGD(op_type.c_str(), "tiling high_ub_axis:%lld", high_ub_axis);

  run_info.SetBlockDim(static_cast<uint32_t>(block_dims));
  if (c_info.only_const_tiling) {
    run_info.AddTilingData(static_cast<int32_t>(need_multi_core));
    run_info.AddTilingData(static_cast<int32_t>(block_axis));
    run_info.AddTilingData(static_cast<int32_t>(block_factor));
    if (!is_nlast_align && !is_pure_copy) {
      run_info.AddTilingData(static_cast<int32_t>(low_ub_axis));
      run_info.AddTilingData(static_cast<int32_t>(low_ub_factor));
    }
    run_info.AddTilingData(static_cast<int32_t>(high_ub_axis));
    run_info.AddTilingData(static_cast<int32_t>(high_ub_factor));
    return true;
  }
  run_info.SetTilingKey(static_cast<uint32_t>(tiling_key));
  if (is_pure_copy) {
    run_info.AddTilingData(static_cast<int32_t>(input_shapes[0]));
    run_info.AddTilingData(static_cast<int32_t>(block_factor));
    run_info.AddTilingData(static_cast<int32_t>(high_ub_factor));
    return true;
  }
  try {
    size_t dim_len = c_info.transpose_vars.size();
    V_OP_TILING_CHECK((dim_len == c_info.permute.size()),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "transpose variable error"), return false);
    for (size_t i = 0; i < dim_len; i++) {
      if (c_info.transpose_vars[i]) {
        run_info.AddTilingData(static_cast<int32_t>(input_shapes[i]));
      }
    }
  } catch (const std::exception& e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile_info[_transpose_vars] error. Error message: %s", e.what());
    return false;
  }
  if (need_multi_core) {
    run_info.AddTilingData(static_cast<int32_t>(block_factor));
    if (!is_nlast_align) {
      run_info.AddTilingData(static_cast<int32_t>(low_ub_factor));
    }
    if (low_ub_axis != high_ub_axis) {
      run_info.AddTilingData(static_cast<int32_t>(high_ub_factor));
    }
  }
  return true;
}

}  // namespace utils

bool TransposeDsl(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& compile_info,
                  utils::OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "enter Transpose");
  try {
    bool is_const = compile_info.at("_is_const");
    if (is_const) {
      int32_t block_dims = compile_info.at("_const_dims");
      run_info.SetTilingKey(1000000);
      run_info.SetBlockDim(static_cast<uint32_t>(block_dims));
      return true;
    }
  } catch (const std::exception& e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile_info[_is_const or _const_dims] error. Error message: %s",
                                    e.what());
    return false;
  }
  utils::Transpose transpose(op_type, op_paras, compile_info);
  bool ret = transpose.DoTiling();
  ret = ret && transpose.WriteTilingData(run_info);
  return ret;
}

}  // namespace optiling
