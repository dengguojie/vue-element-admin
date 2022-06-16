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
const std::unordered_map<int64_t, int64_t> SPLIT_FACTORS{
    {1, 32767},
    {2, 32767},
    {4, 16383},
    {8, 8191},
};
const std::unordered_map<ElewisePattern, std::string> PATTERN_KEY{
    {ElewisePattern::CONST, "000"},
    {ElewisePattern::COMMON, "100"},
    {ElewisePattern::BROADCAST, "200"},
    {ElewisePattern::BROADCAST_SCALAR, "230"},
    {ElewisePattern::SCALAR_BROADCAST, "320"},
    {ElewisePattern::NOT_ALL_FUSE, "111"},
};
const std::unordered_map<ElewisePattern, uint64_t> TILING_BASE_KEY{{ElewisePattern::CONST, 100000000},
                                                                   {ElewisePattern::COMMON, 210000000},
                                                                   {ElewisePattern::BROADCAST, 220000000},
                                                                   {ElewisePattern::BROADCAST_SCALAR, 223000000},
                                                                   {ElewisePattern::SCALAR_BROADCAST, 232000000},
                                                                   {ElewisePattern::NOT_ALL_FUSE, 211100000}};

constexpr int64_t DOUBLE_BUFFER_SIZE = 2;
constexpr uint64_t CONST_TILING_KEY = 100000000;
constexpr uint32_t ELEWISE_FLAG_SIZE = 6;
constexpr int64_t ELEMENT_IN_BLOCK_DOUBLE = 4;
constexpr int64_t ELEMENT_IN_BLOCK_FLOAT = 8;
constexpr int64_t ELEMENT_IN_BLOCK_HALF = 16;
constexpr int64_t ELEMENT_IN_BLOCK_BOOL = 32;
constexpr int64_t ELEMENT_IN_BLOCK_BIT = 256;
constexpr int32_t PATTERN_AXIS_DIV_VALUE = 10;
constexpr int64_t VAR_INDEX_NUM = 100;
constexpr int64_t MIN_DIM_CUT_INDEX = 10000;
constexpr int64_t MIN_BLOCK_CUT_INDEX = 20000;
constexpr int64_t MIN_UB_CUT_INDEX = 30000;
constexpr int64_t ORI_DIM_INDEX = 40000;
constexpr uint32_t BROADCAST_SCALAR_INPUT_NUM = 2;
constexpr int64_t BLOCK_NUM = 8;
constexpr int64_t MAX_REPEAT_TIMES = 8;

constexpr int64_t BIT_C0 = 256;
constexpr int64_t B8_C0 = 32;
constexpr int64_t B16_C0 = 16;
constexpr int64_t B32_C0 = 16;
constexpr int64_t B64_C0 = 4;
}

static const int64_t GetElementByType(const ge::DataType& dtype) {
  // element nums in one block, default, fp16, int16, uin16
  constexpr int64_t one_bit_dtype_value = 100;
  if (dtype == ge::DataType::DT_INT64 || dtype == ge::DataType::DT_UINT64) {
    // element nums in one block by b64
    return ELEMENT_IN_BLOCK_DOUBLE;
  } else if (dtype == ge::DataType::DT_FLOAT || dtype == ge::DataType::DT_INT32 || dtype == ge::DataType::DT_UINT32) {
    // element nums in one block by b32
    return ELEMENT_IN_BLOCK_FLOAT;
  } else if (dtype == ge::DataType::DT_FLOAT16 || dtype == ge::DataType::DT_INT16 || dtype == ge::DataType::DT_UINT16) {
    // element nums in one block by b16
    return ELEMENT_IN_BLOCK_HALF;
  } else if (dtype == ge::DataType::DT_INT8 || dtype == ge::DataType::DT_UINT8 || dtype == ge::DataType::DT_BOOL) {
    // element nums in one block by b8
    return ELEMENT_IN_BLOCK_BOOL;
  } else if (dtype == one_bit_dtype_value) {
    // element nums in one block by uint1
    return ELEMENT_IN_BLOCK_BIT;
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING("ElemWise", "The elewise pattern not support dtype!");
    return -1;
  }
}

static const int64_t GetC0Size(const ge::DataType& dtype) {
  constexpr int64_t one_bit_dtype_value = 100;
  if (dtype == ge::DataType::DT_INT64 || dtype == ge::DataType::DT_UINT64) {
    // element nums in one block by b64
    return B64_C0;
  } else if (dtype == ge::DataType::DT_FLOAT || dtype == ge::DataType::DT_INT32 || dtype == ge::DataType::DT_UINT32) {
    // element nums in one block by b32
    return B32_C0;
  } else if (dtype == ge::DataType::DT_FLOAT16 || dtype == ge::DataType::DT_INT16 || dtype == ge::DataType::DT_UINT16) {
    // element nums in one block by b16
    return B16_C0;
  } else if (dtype == ge::DataType::DT_INT8 || dtype == ge::DataType::DT_UINT8 || dtype == ge::DataType::DT_BOOL) {
    // element nums in one block by b8
    return B8_C0;
  } else if (dtype == one_bit_dtype_value) {
    // element nums in one block by uint1
    return BIT_C0;
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING("ElemWise", "The elewise pattern not support dtype!");
    return -1;
  }
}


ElewisePattern GetDispatchPattern(std::vector<std::vector<int64_t>> elewise_inputs,
                                  const uint32_t& classify_nums) {
  // remove same inputs of 2-D vector
  sort(elewise_inputs.begin(), elewise_inputs.end());
  elewise_inputs.erase(unique(elewise_inputs.begin(), elewise_inputs.end()), elewise_inputs.end());
  /* elewise contains following four scenes:
   1. common: classify_nums <= 1 || all shape same
   2. broadcast: all shape can only contain two diff shapes && classify_nums > 2
   3. scalar_broadcast: all shape can only contian two diff shapes && left_multi_shape is one
   4. broadcast_scalar: all shape can only contian two diff shapes && right_multi_shape is one
  */
  constexpr uint32_t shape_diff_num = 2;
  if (classify_nums <= 1 || elewise_inputs.size() == 1) {
    return ElewisePattern::COMMON;
  }
  if (elewise_inputs.size() == shape_diff_num) {
    const int64_t left_align_size =
      std::accumulate(elewise_inputs[0].begin(), elewise_inputs[0].end(), 1LL, std::multiplies<int64_t>());
    const int64_t right_align_size =
      std::accumulate(elewise_inputs[1].begin(), elewise_inputs[1].end(), 1LL, std::multiplies<int64_t>());
    if (left_align_size == 1 || right_align_size == 1) {
        return ElewisePattern::BROADCAST;
    }
    return ElewisePattern::UNKNOWN;
  }
  return ElewisePattern::UNKNOWN;
}

int64_t ElewiseCalcAlignCore(const int64_t& shape, const int64_t& core,
                             const int64_t& block_dims, const int64_t& half_core) {
  int64_t align_core = core;
  for (; align_core > 0; align_core--) {
    if (shape % align_core == 0) {
      break;
    }
  }
  return (block_dims * align_core) > half_core ? align_core : core;
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

template<typename T>
void Elewise<T>::MatchNotAllFuseTiling() {
  if (compile_info->elewise_pad_axis.first) {
    pad_c_axis = compile_info->elewise_pad_axis.second;
  }
  // only ori_c exists not aligned and contains_need_pad_compute be true will choose not all fuse tiling
  if (compile_info->only_const_tiling || compile_info->classify_const_mode) {
    if (compile_info->elewise_fused_index.first) {
      disable_all_fuse = true;
    }
    return ;
  }
  if (compile_info->elewise_fused_index.first) {
    int64_t pad_c_value = 1;
    bool ori_c_all_aligned = true;
    // elewie now only support ori_format NCHW/NHWC, and all ori_c shape same
    std::unordered_set<int64_t> pad_values;
    for (size_t i = 0; i < input_num; i++) {
      ge::DataType in_dtype;
      if (is_custom_tiling) {
        in_dtype = *op_info->GetInType();
      } else {
        V_OP_TILING_CHECK(context->GetInputDataType(i, in_dtype),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get in dtype error"),
                          return);
      }
      pad_c_value = context->GetOriginInputShape(i).GetDim(pad_c_axis);
      // check if ori_c is C0 aligned
      if (pad_c_value % GetC0Size(in_dtype) != 0) {
        ori_c_all_aligned = false;
      }
      pad_values.emplace(pad_c_value);
    }
    V_OP_TILING_CHECK((pad_values.size() <= 1),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise ori_c shape of all inputs must be equal."),
                      return);
    disable_all_fuse = compile_info->contains_need_pad_compute && !ori_c_all_aligned;
  }
}

template <typename T>
void Elewise<T>::GetOutputDtype() {
  V_OP_TILING_CHECK(context->GetOutputDataType(0, max_output_dtype),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get out dtype error"),
                    return);
  int64_t dtype_size = GetElementByType(max_output_dtype);
  for (size_t i = 1; i < context->GetOutputNums(); i++) {
    ge::DataType tmp_out_type;
    V_OP_TILING_CHECK(context->GetOutputDataType(i, tmp_out_type),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get tmp_out_type error"),
                      return);
    int64_t cur_dtype_size = GetElementByType(tmp_out_type);
    if (cur_dtype_size > dtype_size) {
      max_output_dtype = tmp_out_type;
      dtype_size = cur_dtype_size;
    }
  }
}

template <typename T>
void Elewise<T>::GetCheckInputs(std::vector<size_t>& check_list) {
  for (size_t i = 0; i < input_num; i++) {
    if (is_custom_tiling) {
      const std::vector<int64_t>& input_shape = (*(op_info->GetInputShape()))[i];
      const size_t shape_len = input_shape.size();
      const int64_t current_fuse_shape =
        std::accumulate(input_shape.begin(), input_shape.end(), 1LL, std::multiplies<int64_t>());
      input_fuse_shapes.emplace_back(current_fuse_shape);
      fuse_diff_shapes.emplace(current_fuse_shape);
      // scalar and fuse shape equals to one will not be add into check_list
      if (shape_len == 0 || (shape_len >= 1 && current_fuse_shape == 1)) {
        continue;
      }
      check_list.emplace_back(i);
    } else {
      const OpShape& input_shape = context->GetInputShape(i);
      const size_t& shape_len = input_shape.GetDimNum();
      // GE interface calc empty shape to zero, but we thought it was one
      const int64_t current_fuse_shape = (shape_len != 0 ? input_shape.GetShapeSize() : 1);
      input_fuse_shapes.emplace_back(current_fuse_shape);
      fuse_diff_shapes.emplace(current_fuse_shape);
      // scalar and fuse shape equals to one will not be add into check_list
      if (shape_len == 0 || (shape_len >= 1 && current_fuse_shape == 1)) {
        continue;
      }
      check_list.emplace_back(i);
    }
  }
}

template <typename T>
bool Elewise<T>::GetShapeUnderCheckCustom(std::vector<size_t>& check_list) {
 // check same custom inputs shape
  const auto inputs_shapes = op_info->GetInputShape();
  V_OP_TILING_CHECK((inputs_shapes != nullptr),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "inputs shapes is empty"),
                    return false);
  size_t min_len_index = check_list[0];
  size_t min_len = inputs_shapes->at(check_list[0]).size();
  // loop all custom input to get the true min_len and its index
  for (size_t i = 1; i < check_list.size(); i++) {
    const std::vector<int64_t>& check_shape = inputs_shapes->at(check_list[i]);
    const size_t check_len = check_shape.size();
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
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ele-custom longer shape need be 1 on higher pos"),
                          return false);
      }
      for (uint32_t k = 0; k < min_len; k++) {
        V_OP_TILING_CHECK((need_check_shape[k + len_diff] == min_shape[k]),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ele-custom all shape must be equal on lower pos"),
                          return false);
      }
    }
  }
  out_shape = std::accumulate(min_shape.begin(), min_shape.end(), 1LL, std::multiplies<int64_t>());
  return true;
}

template <typename T>
bool Elewise<T>::GetShapeUnderCheck(std::vector<size_t>& check_list) {
  if (check_list.empty()) {
    const OpShape& output_shape = context->GetOutputShape(0);
    V_OP_TILING_CHECK((!output_shape.Empty()),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get output shape error"),
                      return false);
    out_shape = output_shape.GetShapeSize();
    return true;
  }
  if (is_custom_tiling) {
    return GetShapeUnderCheckCustom(check_list);
  }
  // check same len inputs shape
  size_t min_len_index = check_list[0];
  const OpShape& min_shape_check = context->GetInputShape(min_len_index);
  V_OP_TILING_CHECK((!min_shape_check.Empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input check shape error"),
                    return false);
  uint32_t min_len = min_shape_check.GetDimNum();
  // Loop all input to get the true min_len and its index
  for (size_t i = 1; i < check_list.size(); i++) {
    const OpShape& check_shape = context->GetInputShape(check_list[i]);
    V_OP_TILING_CHECK((!check_shape.Empty()),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input check shape error"),
                      return false);
    const size_t& check_len = check_shape.GetDimNum();
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
    const size_t& need_check_len = need_check_shape.GetDimNum();
    size_t len_diff = need_check_len - min_len;
    for (size_t j = 0; j < len_diff; j++) {
      V_OP_TILING_CHECK((need_check_shape.GetDim(j) == 1),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise long input shape must be 1 on higher pos"),
                        return false);
    }
    for (size_t k = 0; k < min_len; k++) {
      V_OP_TILING_CHECK((need_check_shape.GetDim(k + len_diff) == min_shape.GetDim(k)),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise input shape must be equal on lower pos"),
                        return false);
    }
  }
  out_shape = min_shape.GetShapeSize();
  return true;
}

template <typename T>
void Elewise<T>::RefineNoFuseShapes() {
  V_OP_TILING_CHECK((compile_info->elewise_fused_index.first),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise not fuse must exist elewise_fused_index"),
                    return);
  fused_index_list = compile_info->elewise_fused_index.second;
  constexpr size_t fuse_index_list = 2;
  if (is_custom_tiling) {
    const std::vector<int64_t>& output_shape = (*(op_info->GetInputShape()))[0];
    for (const auto& index_list : fused_index_list) {
      if (index_list.size() == 1) {
        partial_fuse_out_shape.emplace_back(output_shape[index_list[0]]);
      } else if (index_list.size() == fuse_index_list) {
        int64_t multi_shape = std::accumulate(output_shape.begin() + index_list[0],
                                              output_shape.begin() + index_list[1] + 1,
                                              1LL, std::multiplies<int64_t>());
        partial_fuse_out_shape.emplace_back(multi_shape);
      } else {
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise fusion index calculates wrong.");
        return;
      }
    }
    return;
  }
  // get shape from ge operator
  const OpShape& output_shape = context->GetOutputShape(0);
  for (const auto& index_list : fused_index_list) {
    if (index_list.size() == 1) {
      partial_fuse_out_shape.emplace_back(output_shape.GetDim(index_list[0]));
    } else if (index_list.size() == fuse_index_list) {
      int64_t multi_shape = 1;
      for (size_t i = index_list[0]; i <= index_list[1]; i++) {
        multi_shape *= output_shape.GetDim(i);
      }
      partial_fuse_out_shape.emplace_back(multi_shape);
    } else {
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise fusion index calculates wrong.");
      return;
    }
  }
}

template <typename T>
bool Elewise<T>::GetNotAllFuseShapeUnderCheck() {
  if (is_custom_tiling) {
    const std::vector<int64_t> input_shape = op_info->GetInputShape()->at(0);
    const size_t& shape_len = input_shape.size();
    for (size_t i = 1; i < input_num; i++) {
      const std::vector<int64_t>& temp_input_shape = op_info->GetInputShape()->at(i);
      const size_t& temp_shape_len = temp_input_shape.size();
      V_OP_TILING_CHECK((shape_len == temp_shape_len),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise not all fuse only support same input len."),
                        return false);
      for (size_t j = 0; j < shape_len; j++) {
        V_OP_TILING_CHECK((input_shape[j] == temp_input_shape[j]),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise not_all_fuse only support all shape same."),
                          return false);
      }
    }
  } else {
    const OpShape& input_shape = context->GetInputShape(0);
    const size_t& shape_len = input_shape.GetDimNum();
    // this scene all shape and shape len must be equal
    for (size_t i = 1; i < input_num; i++) {
      const OpShape& temp_input_shape = context->GetInputShape(i);
      const size_t& temp_shape_len = temp_input_shape.GetDimNum();
      V_OP_TILING_CHECK((shape_len == temp_shape_len),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise not all fuse only support same input len."),
                        return false);
      for (size_t j = 0; j < shape_len; j++) {
        V_OP_TILING_CHECK((input_shape.GetDim(j) == temp_input_shape.GetDim(j)),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise not_all_fuse only support all shape same."),
                          return false);
      }
    }
  }
  // calc partial_fuse_out_shape
  if (compile_info->only_const_tiling || compile_info->classify_const_mode) {
    const OpShape& original_out_shape = context->GetOutputShape(0);
    for (size_t i = 0; i < original_out_shape.GetDimNum(); i++) {
      partial_fuse_out_shape.emplace_back(original_out_shape.GetDim(i));
    }
  } else {
    RefineNoFuseShapes();
  }
  out_shape = context->GetOutputShape(0).GetShapeSize();
  return true;
}

template <typename T>
bool Elewise<T>::GetInOutShapes() {
  if (disable_all_fuse) {
    return GetNotAllFuseShapeUnderCheck();
  }
  // input shape check and get the output fuse shape
  std::vector<size_t> input_check;
  GetCheckInputs(input_check);
  return GetShapeUnderCheck(input_check);
}

template <typename T>
bool Elewise<T>::WriteKnownData() {
  OP_LOGD(op_type, "elewise known tiling key is:%llu and block_dims is:%lld", tiling_key, block_dims);
  context->SetBlockDim(static_cast<uint32_t>(block_dims));
  context->SetTilingKey(tiling_key);
  return context->WriteVarAttrs(tiling_key);
}

template <typename T>
bool Elewise<T>::CalcConstKey() {
  const size_t& const_shapes_size = compile_info->const_block_dims.second.size();
  constexpr size_t pure_elewise_const_size = 1;
  constexpr size_t broadcast_elewise_const_size = 2;
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
    if (classify_pattern == ElewisePattern::BROADCAST) {
      if (compile_info->classify_inputs_num == BROADCAST_SCALAR_INPUT_NUM) {
        classify_pattern =
          input_fuse_shapes[0] == 1 ? ElewisePattern::SCALAR_BROADCAST : ElewisePattern::BROADCAST_SCALAR;
      }
    }
    return true;
  }
  if (compile_info->only_const_tiling) {
    classify_pattern = ElewisePattern::CONST;
  } else if (disable_all_fuse) {
    classify_pattern = ElewisePattern::NOT_ALL_FUSE;
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
  const int64_t multi_core_threshold = GetElementByType(max_output_dtype) * core_num * DOUBLE_BUFFER_SIZE;
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
  tiling_key += ub_axis;
}

template <typename T>
bool Elewise<T>::WriteTilingData() const {
  OP_LOGD(op_type,
          "tiling key:%llu, block_dims:%lld, block_axis:%lld, block_factor:%lld, ub_axis:%lld, ub_factor:%lld",
          tiling_key, block_dims, block_axis, block_factor, ub_axis, ub_factor);

  context->SetBlockDim(static_cast<uint32_t>(block_dims));
  if (compile_info->only_const_tiling) {
    int32_t double_buffer_num = need_double_buffer ? 1 : 0;
    context->Append(static_cast<int32_t>(need_multi_core));
    context->Append(static_cast<int32_t>(block_axis));
    context->Append(static_cast<int32_t>(block_factor));
    context->Append(static_cast<int32_t>(ub_axis));
    context->Append(static_cast<int32_t>(ub_factor));
    context->Append(double_buffer_num);
    return true;
  }
  context->SetTilingKey(tiling_key);
  // Add elewise vars params
  try {
    const auto& var_list = compile_info->elewise_vars.second.at(std::to_string(tiling_key));
    for (const auto& var : var_list) {
      if (var >= ORI_DIM_INDEX) {
        int64_t var_value = var;
        size_t in_index = static_cast<size_t>(var_value % ORI_DIM_INDEX % (pad_c_axis * VAR_INDEX_NUM));
        context->Append(static_cast<int32_t>(context->GetOriginInputShape(in_index).GetDim(pad_c_axis)));
      } else if (var >= MIN_UB_CUT_INDEX) {
        context->Append(static_cast<int32_t>(ub_factor));
      } else if (var >= MIN_BLOCK_CUT_INDEX) {
        context->Append(static_cast<int32_t>(block_factor));
      } else {
        if (disable_all_fuse) {
          context->Append(static_cast<int32_t>(partial_fuse_out_shape[var / VAR_INDEX_NUM % VAR_INDEX_NUM]));
        } else {
          context->Append(static_cast<int32_t>(input_fuse_shapes[var % MIN_DIM_CUT_INDEX]));
        }
      }
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_elewise_vars] error. Error message: %s", e.what());
    return false;
  }

  return context->WriteVarAttrs(tiling_key);
}

template <typename T>
void Elewise<T>::DoBlockTilingNotAllFuse() {
  int64_t cur_core = core_num;
  // multi core need more than half of cores
  int64_t half_core = core_num / 2;
  bool is_one_dim = partial_fuse_out_shape.size() == 1;
  // calc if need do block align
  const int64_t& block_align_threshold = GetElementByType(max_output_dtype) * BLOCK_NUM * MAX_REPEAT_TIMES * core_num;
  int64_t out_size =
    std::accumulate(partial_fuse_out_shape.begin(), partial_fuse_out_shape.end(), 1LL, std::multiplies<int64_t>());
  bool need_block_align = out_size <= block_align_threshold;

  for (size_t i = 0; i < partial_fuse_out_shape.size(); i++) {
    if (partial_fuse_out_shape[i] > cur_core) {
      int64_t align_core =
        need_block_align ? ElewiseCalcAlignCore(partial_fuse_out_shape[i], cur_core, block_dims, half_core) : cur_core;
      multi_core_output = partial_fuse_out_shape[i];
      block_axis = i;
      block_factor = std::ceil(partial_fuse_out_shape[i] * 1.0 / align_core);
      V_OP_TILING_CHECK((block_factor > 0),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "block_factor must be greater than zero."),
                        return);
      block_dims *= std::ceil(partial_fuse_out_shape[i] * 1.0 / block_factor);
      partial_fuse_out_shape[i] = block_factor;
      break;
    }
    if (need_block_align && cur_core % partial_fuse_out_shape[i] != 0 &&
        block_dims * partial_fuse_out_shape[i] > half_core) {
      multi_core_output = partial_fuse_out_shape[i];
      block_axis = i;
      block_factor = 1;
      block_dims *= partial_fuse_out_shape[i];
      partial_fuse_out_shape[i] = block_factor;
      if (!is_one_dim) {
        block_axis = i + 1;
        block_factor = partial_fuse_out_shape[i + 1];
        partial_fuse_out_shape[i] = multi_core_output;
        multi_core_output = partial_fuse_out_shape[i + 1];
      }
      break;
    }
    cur_core /= partial_fuse_out_shape[i];
    block_dims *= partial_fuse_out_shape[i];
  }
}

template <typename T>
void Elewise<T>::AdjustNotAllFuseUbTiling(const int64_t& under_ub_shape, const int64_t& limit) {
  if (block_axis == ub_axis) {
    int64_t ub_for_num = std::ceil(partial_fuse_out_shape[ub_axis] * 1.0 / ub_factor);
    V_OP_TILING_CHECK((ub_for_num > 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_for_num must be greater than zero."),
                      return);
    ub_factor = std::ceil(partial_fuse_out_shape[ub_axis] * 1.0 / ub_for_num);
  }
  int64_t shape_len = static_cast<int64_t>(partial_fuse_out_shape.size()) - 1;
  int64_t ele_in_block = GetElementByType(max_output_dtype);
  V_OP_TILING_CHECK((ele_in_block != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ele_in_block can not be zero."),
                    return);
  if (ub_axis == shape_len && ub_factor != partial_fuse_out_shape[shape_len]) {
    int64_t last_factor = ub_factor;
    int64_t align_factor = std::ceil(ub_factor * 1.0 / ele_in_block);
    ub_factor = align_factor * ele_in_block;
    if (ub_factor > limit) {
      ub_factor = std::floor(last_factor * 1.0 / ele_in_block) * ele_in_block;
    }
  }
  // adjust the ub factor to avoid tail block less than 32B
  V_OP_TILING_CHECK((ub_factor != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub_factor can not be zero."),
                    return);
  int64_t ub_tail = partial_fuse_out_shape[ub_axis] % ub_factor;
  if (ub_tail != 0 && (under_ub_shape * ub_tail < ele_in_block)) {
    V_OP_TILING_CHECK((under_ub_shape != 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "under_ub_shape can not be zero."),
                      return);
    int64_t need_tail = std::ceil(ele_in_block * 1.0 / under_ub_shape);
    int64_t ub_gap = std::ceil((need_tail - ub_tail) * 1.0 / (partial_fuse_out_shape[ub_axis] / ub_factor));
    ub_factor -= ub_gap;
  }
}

template <typename T>
void Elewise<T>::CheckUpdateUbTiling() {
  bool need_single_core = false;
  for (size_t i = 0; i < context->GetOutputNums(); i++) {
    ge::DataType each_out_dtype;
    V_OP_TILING_CHECK((context->GetOutputDataType(i, each_out_dtype)),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get out dtype error."),
                      return);
    int64_t ele_in_block = GetElementByType(each_out_dtype);
    int64_t cut_output = partial_fuse_out_shape[ub_axis];
    int64_t under_ub = std::accumulate(partial_fuse_out_shape.begin() + ub_axis + 1,
                                       partial_fuse_out_shape.end(), 1LL, std::multiplies<int64_t>());
    need_single_core = (cut_output % ub_factor != 0 && (cut_output % ub_factor) * under_ub < ele_in_block) ||
                       (cut_output % ub_factor == 0 && ub_factor * under_ub < ele_in_block);
    if (block_axis == ub_axis) {
      int64_t tail = multi_core_output % block_factor % ub_factor;
      need_single_core = need_single_core || (tail != 0 && tail * under_ub < ele_in_block);
    }
  }
  if (need_single_core) {
    partial_fuse_out_shape[block_axis] = multi_core_output;
    block_axis = 0;
    block_factor = partial_fuse_out_shape[block_axis];
    block_dims = 1;
  }
  int64_t max_tiling_core_num = core_num;
  if (need_single_core) {
    max_tiling_core_num = 1;
  }
  partial_fuse_out_shape[block_axis] = multi_core_output;
  int64_t shape_before_ub = std::accumulate(partial_fuse_out_shape.begin(),
                                            partial_fuse_out_shape.begin() + ub_axis, 1LL, std::multiplies<int64_t>());
  int64_t ub_split_out = std::ceil(partial_fuse_out_shape[ub_axis] * 1.0 / ub_factor);
  V_OP_TILING_CHECK((max_tiling_core_num != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "max_tiling_core_num can not be zero."),
                    return);
  block_factor = std::ceil(shape_before_ub * ub_split_out * 1.0 / max_tiling_core_num);
  V_OP_TILING_CHECK((block_factor != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "block_factor can not be zero."),
                    return);
  block_dims = std::ceil(shape_before_ub * ub_split_out * 1.0 / block_factor);
  block_axis = 0;
}

template<typename T>
void Elewise<T>::DoUbTilingNotAllFuse() {
  int64_t limit = std::min(max_available_ub, SPLIT_FACTORS.at(max_dtype));
  if (need_double_buffer) {
    limit = std::min(max_available_ub_db, SPLIT_FACTORS.at(max_dtype));
  }
  int64_t max_ub_shape = 1;
  int64_t shape_len = static_cast<int64_t>(partial_fuse_out_shape.size()) - 1;
  for (int64_t i = shape_len; i >= block_axis; i--) {
    if (partial_fuse_out_shape[i] >= limit) {
      ub_axis = i;
      ub_factor = limit;
      max_ub_shape *= ub_factor;
      break;
    }
    limit /= partial_fuse_out_shape[i];
    max_ub_shape *= partial_fuse_out_shape[i];
    ub_axis = i;
    ub_factor = partial_fuse_out_shape[i];
  }
  int64_t under_ub_shape = max_ub_shape / ub_factor;
  AdjustNotAllFuseUbTiling(under_ub_shape, limit);
  CheckUpdateUbTiling();
}

template <typename T>
void Elewise<T>::NotAllFuseTiling() {
  if (need_multi_core) {
    DoBlockTilingNotAllFuse();
    if (block_factor > std::min(max_available_ub, SPLIT_FACTORS.at(max_dtype))) {
      need_double_buffer = true;
    }
    DoUbTilingNotAllFuse();
    return ;
  }
  block_dims = 1;
  block_axis = 0;
  ub_axis = 0;
  block_factor = partial_fuse_out_shape[0];
  ub_factor = partial_fuse_out_shape[0];
}

template<typename T>
bool Elewise<T>::AllFuseTiling() {
  if (need_multi_core) {
    DoBlockTiling();
    if (block_factor > std::min(max_available_ub, SPLIT_FACTORS.at(max_dtype))) {
      need_double_buffer = true;
    }
    return DoUbTiling();
  }
  block_dims = 1;
  block_axis = 0;
  ub_axis = 0;
  block_factor = out_shape;
  ub_factor = out_shape;
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
  V_OP_TILING_CHECK((rl_bank_info.rl_ub_tiling_infos.size() > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "rl_ub_tiling_infos is empty."),
                    return false);
  rl_ub_factor = rl_bank_info.rl_ub_tiling_infos[0].ub_count;
  // elewise factor need to align
  int64_t ele_in_block = GetElementByType(max_output_dtype);
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
    if (rl::CalcExpr(rl_bank_info.range_info, op_type, vars_value, 1)) {
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
bool Elewise<T>::DoTiling() {
  compile_info = dynamic_cast<const ElewiseCompileInfo *>(context->GetCompileInfo());
  bool ret = CheckCompileInfo();
  op_type = context->GetOpType();
  is_custom_tiling = op_info != nullptr;
  input_num = context->GetInputNums(op_info);
  MatchNotAllFuseTiling();
  GetOutputDtype();
  ret = ret && GetInOutShapes();
  V_OP_TILING_CHECK(ret,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise tiling input check failed."),
                    return false);

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
    ret = CalcPatternKey();
    ret = ParseBaseInfo();
    CalcMultiCore();
    if (disable_all_fuse) {
      NotAllFuseTiling();
    } else {
      AllFuseTiling();
    }
    if (ret && !compile_info->only_const_tiling) {
      CalcTilingKey();
    }
    ret = WriteTilingData();
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

void ElewiseCompileInfo::ParseContainsPadCompute(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_contains_need_pad_compute")) {
    contains_need_pad_compute =
      outer_compile_info.at("_contains_need_pad_compute").get<bool>();
  }
}

void ElewiseCompileInfo::ParseFusedIndex(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_elewise_fused_index")) {
    elewise_fused_index.first = true;
    elewise_fused_index.second =
      outer_compile_info.at("_elewise_fused_index").get<std::vector<std::vector<size_t>>>();
  }
}

void ElewiseCompileInfo::ParsePadAxis(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_elewise_pad_axis")) {
    elewise_pad_axis.first = true;
    elewise_pad_axis.second =
      outer_compile_info.at("_elewise_pad_axis").get<size_t>();
  }
}

bool ElewiseCompileInfo::ParseVarsAttr(const nlohmann::json& outer_compile_info) {
  return var_attr_wrap.ParseVarAttr(outer_compile_info);
}

bool ElewiseCompileInfo::ParseOptionalCompileInfo(const nlohmann::json& outer_compile_info) {
  if (ParseVarsAttr(outer_compile_info)) {
    ParseBaseInfo(outer_compile_info);
    ParseConstCompileInfo(outer_compile_info);
    ParseElewiseVar(outer_compile_info);
    ParseContainsPadCompute(outer_compile_info);
    ParseFusedIndex(outer_compile_info);
    ParsePadAxis(outer_compile_info);
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

ElewiseCompileInfo::ElewiseCompileInfo(const std::string& op_type, const nlohmann::json& outer_compile_info) {
  OP_LOGD(op_type.c_str(), "elewise compile info parse running");
  ParseRequiredCompileInfo(outer_compile_info);
  ParseOptionalCompileInfo(outer_compile_info);
  rl::ParseRlBankInfo(outer_compile_info, bank_info_pair);
}
}  // namespace v3

bool CreateElewiseDslTiling(gert::TilingContext* context, const OpInfoImpl* op_info) {
  OP_LOGD("ElemWise", "enter ElewiseDsl re2");
  AutoTilingContext auto_tiling_context(context);
  if (op_info != nullptr) {
    OP_LOGD("ElemWise", "Elewise rt2 tiling with op_info!");
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

REGISTER_AUTO_TILING(SchPattern::ELETWISE, CreateElewiseDslTiling, CreateElewiseDslParser);
}  // namespace optiling
