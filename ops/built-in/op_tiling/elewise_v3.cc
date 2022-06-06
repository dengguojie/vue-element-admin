/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file elewise_v3.cc
 * \brief
 */
#include "elewise_v3.h"

#include <algorithm>
#include <unordered_map>

#include "tiling_handler.h"
#include "auto_tiling_register.h"
#include "graph/utils/op_desc_utils.h"
#include "rl_tune.h"

namespace optiling {
namespace v3 {
namespace {
const std::unordered_map<int64_t, int64_t> SPLIT_FACTORS {
  {1, 32767},
  {2, 32767},
  {4, 16383},
  {8, 8191},
};

const std::unordered_map<ElewisePattern, std::string> PATTERN_KEY {
  {ElewisePattern::CONST, "000"},
  {ElewisePattern::COMMON, "100"},
  {ElewisePattern::BROADCAST, "200"},
  {ElewisePattern::BROADCAST_SCALAR, "230"},
  {ElewisePattern::SCALAR_BROADCAST, "320"}
};

const std::unordered_map<ElewisePattern, uint64_t> TILING_BASE_KEY {
  {ElewisePattern::CONST, 100000000},
  {ElewisePattern::COMMON, 210000000},
  {ElewisePattern::BROADCAST, 220000000},
  {ElewisePattern::BROADCAST_SCALAR, 223000000},
  {ElewisePattern::SCALAR_BROADCAST, 232000000}
};

constexpr int64_t DOUBLE_BUFFER_SIZE = 2;
constexpr uint64_t CONST_TILING_KEY = 100000000;
constexpr uint32_t ELEWISE_FLAG_SIZE = 6;
constexpr int64_t ELEMENT_IN_BLOCK_DOUBLE = 4;
constexpr int64_t ELEMENT_IN_BLOCK_FLOAT = 8;
constexpr int64_t ELEMENT_IN_BLOCK_HALF = 16;
constexpr int64_t ELEMENT_IN_BLOCK_BOOL = 32;
constexpr int64_t ELEMENT_IN_BLOCK_BIT = 256;
constexpr int32_t PATTERN_AXIS_DIV_VALUE = 10;
constexpr int64_t MIN_DIM_CUT_INDEX = 10000;
constexpr int64_t MIN_BLOCK_CUT_INDEX = 20000;
constexpr int64_t MIN_UB_CUT_INDEX = 30000;
constexpr uint32_t BROADCAST_SCALAR_INPUT_NUM = 2;
}

static const int64_t GetElementByType(const ge::DataType& dtype) {
  // element nums in one block, default, fp16, int16, uin16
  constexpr int64_t one_bit_dtype_value = 100;
  if (dtype == ge::DataType::DT_FLOAT || dtype == ge::DataType::DT_INT32 || dtype == ge::DataType::DT_UINT32) {
    // element nums in one block by b32
    return ELEMENT_IN_BLOCK_FLOAT;
  } else if (dtype == ge::DataType::DT_INT8 || dtype == ge::DataType::DT_UINT8 || dtype == ge::DataType::DT_BOOL) {
    // element nums in one block by b8
    return ELEMENT_IN_BLOCK_BOOL;
  } else if (dtype == ge::DataType::DT_INT64 || dtype == ge::DataType::DT_UINT64) {
    // element nums in one block by b64
    return ELEMENT_IN_BLOCK_DOUBLE;
  }else if (dtype == one_bit_dtype_value) {
    // element nums in one block by uint1
    return ELEMENT_IN_BLOCK_BIT;
  } else {
    // element nums in one block by b16
    return ELEMENT_IN_BLOCK_HALF;
  }
}

ElewisePattern GetDispatchPattern(std::vector<std::vector<int64_t>> elewise_inputs,
                                  const uint32_t& classify_nums) {
  // remove same inputs of 2-D vector
  elewise_inputs.erase(unique(elewise_inputs.begin(), elewise_inputs.end()), elewise_inputs.end());
  /* elewise contains following four scenes:
   1. common: classify_nums <= 1 || all shape same
   2. broadcast: all shape can only contain two diff shapes && classify_nums > 2
   3. scalar_broadcast: all shape can only contian two diff shapes && left_multi_shape is one
   4. broadcast_scalar: all shape can only contian two diff shapes && right_multi_shape is one
  */
  constexpr uint32_t shape_diff_num = 2;
  constexpr uint32_t classify_diff_num = 2;
  if (classify_nums <= 1 || elewise_inputs.size() == 1) {
    return ElewisePattern::COMMON;
  }
  if (elewise_inputs.size() == shape_diff_num) {
    const int64_t left_align_size =
      std::accumulate(elewise_inputs[0].begin(), elewise_inputs[0].end(), 1LL, std::multiplies<int64_t>());
    const int64_t right_align_size =
      std::accumulate(elewise_inputs[1].begin(), elewise_inputs[1].end(), 1LL, std::multiplies<int64_t>());
    if (left_align_size == 1 || right_align_size == 1) {
      if (classify_nums > classify_diff_num) {
        return ElewisePattern::BROADCAST;
      }
      return left_align_size == 1 ? ElewisePattern::SCALAR_BROADCAST : ElewisePattern::BROADCAST_SCALAR;
    }
    return ElewisePattern::UNKNOWN;
  }
  return ElewisePattern::UNKNOWN;
}

template <typename T>
void Elewise<T>::SetBroadcastPattern(const ElewisePattern& pattern) {
  OP_LOGD(op_type, "Set pattern for elewise tiling!");
  if (pattern != ElewisePattern::UNKNOWN) {
    broadcast_dispatch = true;
    classify_pattern = pattern;
  }
}

template <typename T>
void Elewise<T>::GetOutputDtype() {
  V_OP_TILING_CHECK(context->GetOutputDataType(0, out_dtype),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get out dtype error"),
                    return);
  int64_t dtype_size = GetElementByType(out_dtype);
  for (uint32_t i = 1; i < context->GetOutputNums(); i++) {
    ge::DataType tmp_out_type;
    V_OP_TILING_CHECK(context->GetOutputDataType(i, tmp_out_type),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get out dtype error"),
                      return);
    int64_t cur_dtype_size = GetElementByType(tmp_out_type);
    if (cur_dtype_size > dtype_size) {
      out_dtype = tmp_out_type;
      dtype_size = cur_dtype_size;
    }
  }
}

template <typename T>
bool Elewise<T>::CheckCompileInfo() {
  // required compile_info check
  V_OP_TILING_CHECK((compile_info->classify_inputs_num > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "classify inputs num must be greater than zero!"),
                    return false);
  V_OP_TILING_CHECK((compile_info->flag_info_size > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "flag info size must be greater than zero!"),
                    return false);
  V_OP_TILING_CHECK((compile_info->ub_factor_align > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_factor_align must be greater than zero!"),
                    return false);
  return true;
}

template <typename T>
void Elewise<T>::GetCheckInputs(std::vector<uint32_t>& check_list) {
  if (op_info != nullptr) {
    // get shape from op_info, the shape type is vector
    for (uint32_t i = 0; i < input_num; i++) {
      const auto inputs_shapes = op_info->GetInputShape();
      V_OP_TILING_CHECK((inputs_shapes != nullptr),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "inputs shapes is empty"),
                        return);
      const std::vector<int64_t>& input_shape = inputs_shapes->at(i);
      const uint32_t shape_len = input_shape.size();
      const int64_t current_fuse_shape =
        std::accumulate(input_shape.begin(), input_shape.end(), 1LL, std::multiplies<int64_t>());
      input_fuse_shapes.emplace_back(current_fuse_shape);
      fuse_diff_shapes.emplace(current_fuse_shape);
      // scalar and fuse shape equals to one will not be add into check_list
      if (shape_len == 0 || (shape_len >= 1 && current_fuse_shape == 1)) {
        continue;
      }
      check_list.emplace_back(i);
    }
  }
  // get shape from context, the shape type is OpShape
  for (uint32_t i = 0; i < input_num; i++) {
    const OpShape& input_shape = context->GetInputShape(i);
    const uint32_t shape_len = context->GetInputShape(i).GetDimNum();
    const int64_t current_fuse_shape = input_shape.GetShapeSize();
    input_fuse_shapes.emplace_back(current_fuse_shape);
    fuse_diff_shapes.emplace(current_fuse_shape);
    // scalar and fuse shape equals to one will not be add into check_list
    if (shape_len == 0 || (shape_len >= 1 && current_fuse_shape == 1)) {
      continue;
    }
    check_list.emplace_back(i);
  }
}

template <typename T>
bool Elewise<T>::GetShapeUnderCheckCustom(std::vector<uint32_t>& check_list) {
 // check same custom inputs shape
  const auto inputs_shapes = op_info->GetInputShape();
  V_OP_TILING_CHECK((inputs_shapes != nullptr),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "inputs shapes is empty"),
                    return false);
  uint32_t min_len_index = check_list[0];
  uint32_t min_len = inputs_shapes->at(check_list[0]).size();
  // loop all custom input to get the true min_len and its index
  for (uint32_t i = 1; i < check_list.size(); i++) {
    const std::vector<int64_t>& check_shape = inputs_shapes->at(check_list[i]);
    const uint32_t check_len = check_shape.size();
    if (check_len < min_len) {
      min_len = check_len;
      min_len_index = check_list[i];
    }
  }
  const std::vector<int64_t>& min_shape = inputs_shapes->at(min_len_index);
  // broadcast dispatch shapes no need check again
  if (!broadcast_dispatch) {
    // custom input check rules: 1.from right to left, same dim_index with custom_min_shape must be same;
    // 2.index higher must be all 1.
    for (const auto& need_check_index : check_list) {
      const std::vector<int64_t>& need_check_shape = inputs_shapes->at(need_check_index);
      const uint32_t need_check_len = need_check_shape.size();
      uint32_t len_diff = need_check_len - min_len;
      for (uint32_t j = 0; j < len_diff; j++) {
        V_OP_TILING_CHECK((need_check_shape[j] == 1),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ele-custom long in_shape need be 1 on higher pos"),
                          return false);
      }
      for (uint32_t k = 0; k < min_len; k++) {
        V_OP_TILING_CHECK((need_check_shape[k + len_diff] == min_shape[k]),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ele-custom input shape must be equal on lower pos"),
                          return false);
      }
    }
  }
  out_shape = std::accumulate(min_shape.begin(), min_shape.end(), 1LL, std::multiplies<int64_t>());
  return true;
}

template <typename T>
bool Elewise<T>::GetShapeUnderCheck(std::vector<uint32_t>& check_list) {
  if (check_list.empty()) {
    const OpShape& output_shape = context->GetOutputShape(0);
    V_OP_TILING_CHECK((!output_shape.Empty()),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get output shape error"),
                      return false);
    out_shape = output_shape.GetShapeSize();
    return true;
  }
  if (op_info != nullptr) {
    return GetShapeUnderCheckCustom(check_list);
  }
  // check same len inputs shape
  uint32_t min_len_index = check_list[0];
  const OpShape& min_shape_check = context->GetInputShape(min_len_index);
  V_OP_TILING_CHECK((!min_shape_check.Empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input check shape error"),
                    return false);
  uint32_t min_len = min_shape_check.GetDimNum();
  // Loop all input to get the true min_len and its index
  for (uint32_t i = 1; i < check_list.size(); i++) {
    const OpShape& check_shape = context->GetInputShape(check_list[i]);
    V_OP_TILING_CHECK((!check_shape.Empty()),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input check shape error"),
                      return false);
    const uint32_t check_len = check_shape.GetDimNum();
    if (check_len < min_len) {
      min_len = check_len;
      min_len_index = check_list[i];
    }
  }
  const OpShape& min_shape = context->GetInputShape(min_len_index);
  // input check rules: 1.from right to left, same dim_index with min_shape must be same;
  // 2.index higher must be all 1.
  for (const auto& need_check_index : check_list) {
    const OpShape& need_check_shape = context->GetInputShape(need_check_index);
    V_OP_TILING_CHECK((!need_check_shape.Empty()),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input need check shape error"),
                      return false);
    const uint32_t need_check_len = need_check_shape.GetDimNum();
    uint32_t len_diff = need_check_len - min_len;
    for (uint32_t j = 0; j < len_diff; j++) {
      V_OP_TILING_CHECK((need_check_shape.GetDim(j) == 1),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise long input shape must be 1 on higher pos"),
                        return false);
    }
    for (uint32_t k = 0; k < min_len; k++) {
      V_OP_TILING_CHECK((need_check_shape.GetDim(k + len_diff) == min_shape.GetDim(k)),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise input shape must be equal on lower pos"),
                        return false);
    }
  }
  out_shape = min_shape.GetShapeSize();
  return true;
}

template <typename T>
bool Elewise<T>::GetInOutShapes() {
  // input shape check and get the output fuse shape
  std::vector<uint32_t> check_list;
  GetCheckInputs(check_list);
  return GetShapeUnderCheck(check_list);
}

template <typename T>
bool Elewise<T>::WriteKnownData() {
  OP_LOGD(op_type, "elewise known tiling key is:%llu and block_dims is:%lld", tiling_key, block_dims);
  context->SetBlockDim(static_cast<uint32_t>(block_dims));
  context->SetTilingKey(tiling_key);
  if (typeid(*context) == typeid(AutoTilingOp)) {
    return compile_info->varAttrWrap.WriteVarAttrs(tiling_key, op_type,
                                                   *context->GetOpParas(), *context->GetRunInfo());
  }
  return true;
}

template <typename T>
bool Elewise<T>::CalcConstKey() {
  const uint32_t const_shapes_size = compile_info->const_block_dims.second.size();
  constexpr uint32_t pure_elewise_const_size = 1;
  constexpr uint32_t broadcast_elewise_const_size = 2;
  if (const_shapes_size == pure_elewise_const_size) {
    block_dims = compile_info->const_block_dims.second[0];
    tiling_key = CONST_TILING_KEY;
  } else if (const_shapes_size == broadcast_elewise_const_size) {
    if (fuse_diff_shapes.size() == broadcast_elewise_const_size) {
      // inputs with diff shapes such as [1] and [4] will firstly add info during compiler time
      block_dims = compile_info->const_block_dims.second[0];
      tiling_key = CONST_TILING_KEY;
    } else {
      // inputs with diff shapes such as [4] and [4] will secondly add info during compiler time
      block_dims = compile_info->const_block_dims.second[1];
      tiling_key = CONST_TILING_KEY + 1;
    }
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The const key calc fail due to error const shapes size!");
    return false;
  }
  return true;
}

template <typename T>
bool Elewise<T>::ConstModeTiling() {
  OP_LOGD(op_type, "Enter into elewise const shape tiling.");
  return CalcConstKey() && WriteKnownData();
}

template <typename T>
bool Elewise<T>::EmptyModeTiling() {
  OP_LOGD(op_type, "Enter into elewise empty shape tiling.");
  block_dims = 1;
  tiling_key = INT32_MAX;
  return WriteKnownData();
}

template <typename T>
bool Elewise<T>::CalcPatternKey() {
  // broadcast dispatch set pattern for elewise, no need calculate again
  if (broadcast_dispatch) {
    return true;
  }
  if (compile_info->only_const_tiling) {
    classify_pattern = ElewisePattern::CONST;
  } else if (!compile_info->support_broadcast || fuse_diff_shapes.size() == 1) {
    classify_pattern = ElewisePattern::COMMON;
  } else if (compile_info->support_broadcast && compile_info->classify_inputs_num > BROADCAST_SCALAR_INPUT_NUM) {
    classify_pattern = ElewisePattern::BROADCAST;
  } else if (compile_info->absorbable_broadcast) {
    classify_pattern = input_fuse_shapes[0] == 1 ? ElewisePattern::SCALAR_BROADCAST : ElewisePattern::BROADCAST_SCALAR;
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The pattern key calc failed!");
    return false;
  }
  return true;
}

template <typename T>
bool Elewise<T>::ParseBaseInfo() {
  try {
    const auto& current_base_info = compile_info->base_info.second.at(PATTERN_KEY.at(classify_pattern));
    constexpr uint32_t base_info_size = 4;
    // broadcast base info size may be greater than 4
    V_OP_TILING_CHECK((current_base_info.size() >= base_info_size),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "base info size must be no less than four!"),
                      return false);
    constexpr uint32_t core_index = 0;
    constexpr uint32_t max_dtype_index = 1;
    constexpr uint32_t ub_available_index = 2;
    constexpr uint32_t ub_available_db_index = 3;
    core_num = current_base_info[core_index];
    max_dtype = current_base_info[max_dtype_index];
    max_available_ub = current_base_info[ub_available_index];
    max_available_ub_db = current_base_info[ub_available_db_index];
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_base_info] error. Error message: %s", e.what());
    return false;
  }
  return true;
}

template <typename T>
void Elewise<T>::CalcMultiCore() {
  const int64_t multi_core_threshold = GetElementByType(out_dtype) * core_num * DOUBLE_BUFFER_SIZE;
  if (out_shape < multi_core_threshold) {
    need_multi_core = false;
  }
}

template <typename T>
void Elewise<T>::DoBlockTiling() {
  int64_t cur_core = core_num;
  int64_t block_factor_align_size = compile_info->ub_factor_align;
  block_factor = std::ceil(out_shape * 1.0 / cur_core);
  block_factor = std::ceil(block_factor * 1.0 / block_factor_align_size) * block_factor_align_size;
  block_dims = std::ceil(out_shape * 1.0 / block_factor);
}

template <typename T>
bool Elewise<T>::DoUbTiling() {
  ub_factor = block_factor;
  int64_t limit = std::min(max_available_ub, SPLIT_FACTORS.at(max_dtype));
  if (need_double_buffer) {
    limit = std::min(max_available_ub_db, SPLIT_FACTORS.at(max_dtype));
  }
  if (limit < ub_factor) {
    int64_t ub_factor_align_size = compile_info->ub_factor_align;
    V_OP_TILING_CHECK((limit > 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                      "ub limit must be greater than zero, but it is [%ld]",
                                                      limit),
                      return false);
    int64_t ub_for_num = std::ceil(ub_factor * 1.0 / limit);
    int64_t adjust_factor = std::ceil(ub_factor * 1.0 / ub_for_num);
    int64_t align_factor = std::ceil(adjust_factor * 1.0 / ub_factor_align_size);
    ub_factor = align_factor * ub_factor_align_size;
    if (ub_factor > limit) {
      ub_factor = std::floor(adjust_factor * 1.0 / ub_factor_align_size) * ub_factor_align_size;
    }
  }
  return true;
}

template <typename T>
void Elewise<T>::CalcTilingKey() {
  constexpr uint64_t db_tiling_key = 10000;
  tiling_key = TILING_BASE_KEY.at(classify_pattern);
  if (need_double_buffer) {
    tiling_key += db_tiling_key;
  }
}

template <typename T>
bool Elewise<T>::WriteTilingData() const {
  OP_LOGD(op_type, "elewise tiling key is:%llu, block_dims is:%lld, block_factor is:%lld, ub_factor is:%lld",
          tiling_key, block_dims, block_factor, ub_factor);

  context->SetBlockDim(static_cast<uint32_t>(block_dims));
  if (compile_info->only_const_tiling) {
    int32_t double_buffer_num = need_double_buffer ? 1 : 0;
    constexpr int32_t elewise_block_axis = 0;
    constexpr int32_t elewise_ub_axis = 0;
    context->Append(static_cast<int32_t>(need_multi_core));
    context->Append(elewise_block_axis);
    context->Append(static_cast<int32_t>(block_factor));
    context->Append(elewise_ub_axis);
    context->Append(static_cast<int32_t>(ub_factor));
    context->Append(double_buffer_num);
    return true;
  }
  context->SetTilingKey(tiling_key);
  // Add elewise vars params
  try {
    const auto& var_list = compile_info->elewise_vars.second.at(std::to_string(tiling_key));
    for (const auto& var : var_list) {
      if (var >= MIN_UB_CUT_INDEX) {
        context->Append(static_cast<int32_t>(ub_factor));
      } else if (var >= MIN_BLOCK_CUT_INDEX) {
        context->Append(static_cast<int32_t>(block_factor));
      } else {
        context->Append(static_cast<int32_t>(input_fuse_shapes[var % MIN_DIM_CUT_INDEX]));
      }
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_elewise_vars] error. Error message: %s", e.what());
    return false;
  }
  if (typeid(*context) == typeid(AutoTilingOp)) {
    return compile_info->varAttrWrap.WriteVarAttrs(tiling_key, op_type,
                                                   *context->GetOpParas(), *context->GetRunInfo());
  }
  return true;
}

template <typename T>
bool Elewise<T>::WriteRlTilingData(const rl::RlBankInfo& rl_bank_info) const {
  OP_LOGD(op_type, "elewise rl tiling rl_block_dim is:%lld", rl_block_dim);
  OP_LOGD(op_type, "elewise rl tiling rl_kernel_key is:%lld", rl_bank_info.rl_kernel_key);
  OP_LOGD(op_type, "elewise rl tiling rl_block_factor is:%lld", rl_block_factor);
  OP_LOGD(op_type, "elewise rl tiling rl_ub_factor is:%lld", rl_ub_factor);

  context->SetBlockDim(static_cast<uint32_t>(rl_block_dim));
  context->SetTilingKey(rl_bank_info.rl_kernel_key);

  for (const auto& var_num : rl_bank_info.rl_sch_vars) {
    if (var_num >= MIN_UB_CUT_INDEX) {
      context->Append(static_cast<int32_t>(rl_ub_factor));
    } else if (var_num >= MIN_BLOCK_CUT_INDEX) {
      context->Append(static_cast<int32_t>(rl_block_factor));
    } else {
     context->Append(static_cast<int32_t>(input_fuse_shapes[var_num % MIN_DIM_CUT_INDEX]));
    }
  }
  return true;
}

template <typename T>
bool Elewise<T>::DoRlTiling(const rl::RlBankInfo& rl_bank_info) {
  OP_LOGD(op_type, "Enter into elewise rl tiling.");
  bool ret = true;
  // ub tiling
  // elewise only has one time ub split
  rl_ub_factor = rl_bank_info.rl_ub_tiling_infos[0].ub_count;
  // elewise factor need to align
  int64_t ele_in_block = GetElementByType(out_dtype);
  V_OP_TILING_CHECK((ele_in_block != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ele_in_block cannot be zero."),
                    return false);
  int64_t align_rl_ub_factor = std::floor(rl_ub_factor * 1.0 / ele_in_block) * ele_in_block;
  rl_ub_factor = std::max(ele_in_block, align_rl_ub_factor);
  V_OP_TILING_CHECK((rl_ub_factor != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "rl_ub_factor cannot be zero."),
                    return false);
  // Adjust the UB factor to avoid tail block less than 32 bytes
  int64_t ub_tail = out_shape % rl_ub_factor;
  if (ub_tail > 0 && ub_tail < ele_in_block) {
    int64_t ub_num = out_shape / rl_ub_factor;
    V_OP_TILING_CHECK((ub_num != 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_num cannot be zero."),
                      return false);
    int64_t ub_gap = std::ceil((ele_in_block - ub_tail) * 1.0 / ub_num);
    rl_ub_factor -= ub_gap;
  }
  // block tiling
  // not need to do block split, and no rl_block_factor
  if (rl_bank_info.rl_block_tiling_info.block_factor_name.empty()) {
    if (out_shape <= rl_bank_info.rl_block_tiling_info.core_num * ele_in_block) {
      // shape is less than core_num*ele_in_block_size, only enable single core
      rl_block_dim = 1;
      rl_ub_factor = out_shape;
    } else {
      int64_t ub_factor_max = rl_ub_factor;
      // try to enable all core, and next to do full split
      rl_ub_factor = std::ceil(out_shape * 1.0 / rl_bank_info.rl_block_tiling_info.core_num);
      // 32B align, and less equal ub_factor_max
      int64_t align_rl_ub_factor = std::ceil(rl_ub_factor * 1.0 / ele_in_block) * ele_in_block;
      rl_ub_factor = std::min(ub_factor_max, align_rl_ub_factor);
      rl_block_dim = std::ceil(out_shape * 1.0 / rl_ub_factor);
    }
  } else {  // need to do block split
    int64_t outer = std::ceil(out_shape * 1.0 / rl_ub_factor);
    rl_block_factor = std::ceil(outer * 1.0 / rl_bank_info.rl_block_tiling_info.core_num);
    V_OP_TILING_CHECK((rl_block_factor != 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "rl_block_factor cannot be zero."),
                      return false);
    rl_block_dim = std::ceil(outer * 1.0 / rl_block_factor);
  }
  return ret;
}

template <typename T>
bool Elewise<T>::TryMatchRlBank() {
  bool ret = true;
  // hit bank_info
  if (!compile_info->bank_info_pair.first) {
    return ret;
  }
  // hit compute_pattern
  // get shape and attr to calc cpt_pattern
  std::array<int64_t, rl::RL_TOTAL_SHAPE_DIM_LEN> inputs_shape{};
  for (uint32_t i = 0; i < input_num; i++) {
    inputs_shape[i] = out_shape;
  }
  int64_t pattern_id = -1;
  for (size_t j = 0; j < compile_info->bank_info_pair.second.size(); j++) {
    if (rl::PatternMatch(compile_info->bank_info_pair.second[j].first, inputs_shape, input_num, {}, 0)) {
      pattern_id = j;
      break;
    }
  }
  if (pattern_id < 0) {
    return ret;
  }
  // hit target range
  // calc vars_value by dynamic_axis_loc, elewise only have one dynamic axis
  std::array<int64_t, rl::DYNC_AXIS_MAX_NUM> vars_value{out_shape};
  for (const auto& rl_bank_info : compile_info->bank_info_pair.second[pattern_id].second) {
    if (rl::CalcExpr(rl_bank_info.range_info, vars_value)) {
      OP_LOGD(op_type, "Hit rl bank.");
      // do rl tiling
      ret = ret && DoRlTiling(rl_bank_info);
      // write rl tiling_data
      WriteRlTilingData(rl_bank_info);
      hit_rl_bank = true;
      return ret;
    }
  }
  return ret;
}

template <typename T>
bool Elewise<T>::SpecialModeTiling() {
  CalcPatternKey();
  bool ret = ParseBaseInfo();
  CalcMultiCore();
  if (need_multi_core) {
    DoBlockTiling();
    if (block_factor > std::min(max_available_ub, SPLIT_FACTORS.at(max_dtype))) {
      need_double_buffer = true;
    }
    ret = ret && DoUbTiling();
  } else {
    block_dims = 1;
    block_factor = out_shape;
    ub_factor = out_shape;
  }
  if (ret && !compile_info->only_const_tiling) {
    CalcTilingKey();
  }
  ret = WriteTilingData();
  return ret;
}

template <typename T>
bool Elewise<T>::DoTiling() {
  op_type = context->GetOpType();
  compile_info = dynamic_cast<const ElewiseCompileInfo *>(context->GetCompileInfo());
  input_num = op_info != nullptr ? op_info->GetInputShape()->size() : context->GetInputNums();
  GetOutputDtype();
  bool ret = CheckCompileInfo();
  ret = ret && GetInOutShapes();
  if (!ret) {
    OP_LOGE(op_type, "elewise custom tiling input infos get failed.");
    return ret;
  }
  // try to match rl bank
  if (TryMatchRlBank() && hit_rl_bank) {
    return ret;
  }
  // tiling dispatch to different classify mode
  if (compile_info->classify_const_mode) {
    ret = ConstModeTiling();
  } else if (out_shape == 0) {
    ret = EmptyModeTiling();
  } else {
    ret = SpecialModeTiling();
  }
  return ret;
}

void ElewiseCompileInfo::ParseClassifyNum(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_classify_inputs_num")) {
    classify_inputs_num = outer_compile_info.at("_classify_inputs_num").get<uint32_t>();
  }
}

void ElewiseCompileInfo::ParseFlagInfo(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_flag_info")) {
    const std::vector<bool>& input_flag_info = outer_compile_info.at("_flag_info").get<std::vector<bool>>();
    flag_info_size = input_flag_info.size();
    if (input_flag_info.size() > 0) {
      only_const_tiling = input_flag_info[0];
      constexpr uint32_t const_shapes_index = 1;
      constexpr uint32_t support_broadcast_index = 2;
      constexpr uint32_t absorbable_broadcast_index = 4;
      constexpr uint32_t elewise_flag_size = 6;
      // broadcast scene flag info size is seven
      if (flag_info_size >= elewise_flag_size) {
        classify_const_mode = input_flag_info[const_shapes_index];
        support_broadcast = input_flag_info[support_broadcast_index];
        absorbable_broadcast = input_flag_info[absorbable_broadcast_index];
      }
    }
  }
}

void ElewiseCompileInfo::ParseUbFactorAlign(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_ub_factor_align")) {
    ub_factor_align = outer_compile_info.at("_ub_factor_align").get<int64_t>();
  }
}

void ElewiseCompileInfo::ParseRequiredCompileInfo(const nlohmann::json& outer_compile_info) {
  ParseClassifyNum(outer_compile_info);
  ParseFlagInfo(outer_compile_info);
  ParseUbFactorAlign(outer_compile_info);
}

void ElewiseCompileInfo::ParseBaseInfo(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_base_info")) {
    base_info.first = true;
    base_info.second =
      outer_compile_info.at("_base_info").get<std::unordered_map<std::string, std::vector<int64_t>>>();
  }
}

void ElewiseCompileInfo::ParseConstCompileInfo(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_const_block_dims") && classify_const_mode) {
    const_block_dims.first = true;
    const_block_dims.second = outer_compile_info.at("_const_block_dims").get<std::vector<int64_t>>();
  }
}

void ElewiseCompileInfo::ParseElewiseVar(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_elewise_vars")) {
    elewise_vars.first = true;
    elewise_vars.second =
      outer_compile_info.at("_elewise_vars").get<std::unordered_map<std::string, std::vector<int64_t>>>();
  }
}

bool ElewiseCompileInfo::ParseVarsAttr(const nlohmann::json& outer_compile_info) {
  return varAttrWrap.ParseVarAttr(outer_compile_info);
}

bool ElewiseCompileInfo::ParseOptionalCompileInfo(const nlohmann::json& outer_compile_info) {
  if (ParseVarsAttr(outer_compile_info)) {
    ParseBaseInfo(outer_compile_info);
    ParseConstCompileInfo(outer_compile_info);
    ParseElewiseVar(outer_compile_info);
    return true;
  } else {
    return false;
  }
}

bool ElewiseCompileInfo::Parse(const char* op_type, const nlohmann::json& outer_compile_info) {
  OP_LOGD(op_type, "elewise compile info parse running");
  try {
    ParseRequiredCompileInfo(outer_compile_info);
    ParseOptionalCompileInfo(outer_compile_info);
    rl::ParseRlBankInfo(outer_compile_info, bank_info_pair);
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING("ElemWise", "parse compile_info error. Error message: %s", e.what());
    return false;
  }
  return true;
}

ElewiseCompileInfo::ElewiseCompileInfo(const string& op_type, const nlohmann::json& outer_compile_info) {
  OP_LOGD(op_type.c_str(), "elewise compile info parse running");
  ParseRequiredCompileInfo(outer_compile_info);
  ParseOptionalCompileInfo(outer_compile_info);
  rl::ParseRlBankInfo(outer_compile_info, bank_info_pair);
}
}  // namespace v3

bool CreateElewiseDslTiling(gert::TilingContext* context, const OpInfoImpl* op_info) {
  OP_LOGD("ElewiseDsl", "enter ElewiseDsl re2");
  AutoTilingContext auto_tiling_context(context);
  if (op_info != nullptr) {
    OP_LOGD(context->GetNodeType(), "Elewise rt2 tiling with op_info!");
    auto_tiling_context.SetCompileInfo(op_info->GetCompileInfo());
  }
  v3::Elewise<AutoTilingContext> elewise(&auto_tiling_context, op_info);
  return elewise.DoTiling();
}

AutoTilingCompileInfo* CreateElewiseDslParser(const char* op_type, const nlohmann::json& json_compile_info) {
  auto compile_info = new v3::ElewiseCompileInfo();
  if (!compile_info->Parse(op_type, json_compile_info)) {
    return nullptr;
  }
  return compile_info;
}

bool ElewiseTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "elewise old auto tiling running");
  AutoTilingOp auto_tiling_op(op_type.c_str(), &op_paras, &elewise_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&auto_tiling_op, nullptr);
  return elewise.DoTiling();
}

bool ElewiseTilingHandler::DoTiling(const ge::Operator& op_paras,
                                    utils::OpRunInfo& run_info,
                                    const OpInfo& op_info) const {
  OP_LOGD(op_type.c_str(), "elewise old custom tiling running");
  AutoTilingOp auto_tiling_op(op_type.c_str(), &op_paras, &elewise_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&auto_tiling_op, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  return elewise.DoTiling();
}

std::shared_ptr<AutoTilingHandler> CreateElewiseTilingHandler(const std::string& op_type,
                                                              const std::string& pattern,
                                                              const nlohmann::json& parsed_compile_info) {
  return std::make_shared<ElewiseTilingHandler>(op_type, pattern, parsed_compile_info);
}

REGISTER_AUTO_TILING(SchPattern::ELETWISE, CreateElewiseDslTiling, CreateElewiseDslParser)
}  // namespace optiling
