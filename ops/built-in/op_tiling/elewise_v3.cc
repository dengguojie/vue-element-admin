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
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"

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

const int64_t GetElementByType(const ge::DataType& dtype) {
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
  } else {
    if (elewise_inputs.size() == shape_diff_num) {
      const int64_t left_align_size =
        std::accumulate(elewise_inputs[0].begin(), elewise_inputs[0].end(), 1LL, std::multiplies<int64_t>());
      const int64_t right_align_size =
        std::accumulate(elewise_inputs[1].begin(), elewise_inputs[1].end(), 1LL, std::multiplies<int64_t>());
      if (left_align_size == 1 || right_align_size == 1) {
        if (classify_nums > classify_diff_num) {
          return ElewisePattern::BROADCAST;
        } else {
          return left_align_size == 1 ? ElewisePattern::SCALAR_BROADCAST : ElewisePattern::BROADCAST_SCALAR;
        }
      } else {
          return ElewisePattern::UNKNOWN;
      }
    } else {
      return ElewisePattern::UNKNOWN;
    }
  }
}

void Elewise::SetBroadcastPattern(const ElewisePattern& pattern) {
  OP_LOGD("Set pattern for elewise tiling!");
  if (pattern != ElewisePattern::UNKNOWN) {
    broadcast_dispatch = true;
    classify_pattern = pattern;
  }
}

bool Elewise::CheckCompileInfo() {
  // required compile_info check
  V_CHECK_GT(compile_info.classify_inputs_num, 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise classify_inputs_num must be greater than zero!"),
             return false);
  V_CHECK_GT(compile_info.flag_info_size, 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise flag_info_size must be greater than zero!"),
             return false);
  V_CHECK_GT(compile_info.ub_factor_align, 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise ub_factor_align must be greater than zero!"),
             return false);
  return true;
}

void Elewise::GetOutputDtype() {
  out_dtype = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableOutputDesc(0)->GetDataType();
  int64_t dtype_size = GetElementByType(out_dtype);
  for (uint32_t i = 1; i < op_paras.GetOutputsSize(); i++) {
    int64_t cur_dtype_size =
      GetElementByType(ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableOutputDesc(i)->GetDataType());
    if (cur_dtype_size > dtype_size) {
      out_dtype = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableOutputDesc(i)->GetDataType();
      dtype_size = cur_dtype_size;
    }
  }
}

void Elewise::GetCheckInputs(std::vector<uint32_t>& check_list) {
  for (uint32_t i = 0; i < input_num; i++) {
    const ge::GeShape& input_shape =
      ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(i)->MutableShape();
    const uint32_t shape_len = input_shape.GetDimNum();
    // GE interface calc empty shape to zero, but we thought it was one
    const int64_t current_fuse_shape = (shape_len != 0 ? input_shape.GetShapeSize() : 1);
    input_fuse_shapes.emplace_back(current_fuse_shape);
    fuse_diff_shapes.emplace(current_fuse_shape);
    // scalar and fuse shape equals to one will not be add into check_list
    if (shape_len == 0 || (shape_len >= 1 && current_fuse_shape == 1)) {
      continue;
    } else {
      check_list.emplace_back(i);
    }
  }
}

void Elewise::GetCheckInputs(std::vector<uint32_t>& check_list, const OpInfo& op_info) {
  for (uint32_t i = 0; i < input_num; i++) {
    const std::vector<int64_t>& input_shape = op_info.GetInputShape()[i];
    const uint32_t shape_len = input_shape.size();
    const int64_t current_fuse_shape =
      std::accumulate(input_shape.begin(), input_shape.end(), 1LL, std::multiplies<int64_t>());
    input_fuse_shapes.emplace_back(current_fuse_shape);
    fuse_diff_shapes.emplace(current_fuse_shape);
    // scalar and fuse shape equals to one will not be add into check_list
    if (shape_len == 0 || (shape_len >= 1 && current_fuse_shape == 1)) {
      continue;
    } else {
      check_list.emplace_back(i);
    }
  }
}

bool Elewise::GetShapeUnderCheck(std::vector<uint32_t>& check_list) {
  // check same len inputs shape
  if (!check_list.empty()) {
    uint32_t min_len_index = check_list[0];
    uint32_t min_len =
      ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(min_len_index)->MutableShape().GetDimNum();
    // loop all input to get the true min_len and its index
    for (uint32_t i = 1; i < check_list.size(); i++) {
      const ge::GeShape& check_shape =
        ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(check_list[i])->MutableShape();
      const uint32_t check_len = check_shape.GetDimNum();
      if (check_len < min_len) {
        min_len = check_len;
        min_len_index = check_list[i];
      }
    }
    const ge::GeShape& min_shape =
      ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(min_len_index)->MutableShape();
    // input check rules: 1.from right to left, same dim_index with min_shape must be same;
    // 2.index higher must be all 1.
    for (uint32_t i = 0; i < check_list.size(); i++) {
      const ge::GeShape& need_check_shape =
        ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(check_list[i])->MutableShape();
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
  }
  return true;
}

bool Elewise::GetShapeUnderCheck(std::vector<uint32_t>& check_list, const OpInfo& op_info) {
  // check same custom inputs shape
  if (!check_list.empty()) {
    uint32_t min_len_index = check_list[0];
    uint32_t min_len = op_info.GetInputShape()[min_len_index].size();
    // loop all custom input to get the true min_len and its index
    for (uint32_t i = 1; i < check_list.size(); i++) {
      const std::vector<int64_t>& check_shape = op_info.GetInputShape()[check_list[i]];
      const uint32_t check_len = check_shape.size();
      if (check_len < min_len) {
        min_len = check_len;
        min_len_index = check_list[i];
      }
    }
    const std::vector<int64_t>& min_shape = op_info.GetInputShape()[min_len_index];
    // broadcast dispatch shapes no need check again
    if (!broadcast_dispatch) {
      // custom input check rules: 1.from right to left, same dim_index with custom_min_shape must be same;
      // 2.index higher must be all 1.
      for (uint32_t i = 0; i < check_list.size(); i++) {
        const std::vector<int64_t>& need_check_shape = op_info.GetInputShape()[check_list[i]];
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
  }
  return true;
}

bool Elewise::GetInOutShapes() {
  // input shape check and get the output fuse shape
  std::vector<uint32_t> input_check;
  GetCheckInputs(input_check);
  return GetShapeUnderCheck(input_check);
}

bool Elewise::GetInOutShapes(const OpInfo& op_info) {
  // custom input shape check and get the output fuse shape
  std::vector<uint32_t> input_check;
  GetCheckInputs(input_check, op_info);
  return GetShapeUnderCheck(input_check, op_info);
}

bool Elewise::WriteKnownData() {
  OP_LOGD(op_type.c_str(), "elewise known tiling key is:%llu and block_dims is:%lld", tiling_key, block_dims);
  run_info.SetBlockDim(static_cast<uint32_t>(block_dims));
  run_info.SetTilingKey(static_cast<uint32_t>(tiling_key));
  return compile_info.varAttrWrap.WriteVarAttrs(tiling_key, op_type, op_paras, run_info);
}

bool Elewise::CalcConstKey() {
  const uint32_t const_shapes_size = compile_info.const_block_dims.second.size();
  constexpr uint32_t pure_elewise_const_size = 1;
  constexpr uint32_t broadcast_elewise_const_size = 2;
  if (const_shapes_size == pure_elewise_const_size) {
    block_dims = compile_info.const_block_dims.second[0];
    tiling_key = CONST_TILING_KEY;
  } else if (const_shapes_size == broadcast_elewise_const_size) {
    if (fuse_diff_shapes.size() == broadcast_elewise_const_size) {
      // inputs with diff shapes such as [1] and [4] will firstly add info during compiler time
      block_dims = compile_info.const_block_dims.second[0];
      tiling_key = CONST_TILING_KEY;
    } else {
      // inputs with diff shapes such as [4] and [4] will secondly add info during compiler time
      block_dims = compile_info.const_block_dims.second[1];
      tiling_key = CONST_TILING_KEY + 1;
    }
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The const key calc fail due to error const shapes size!");
    return false;
  }
  return true;
}

bool Elewise::ConstModeTiling() {
  OP_LOGD(op_type.c_str(), "Enter into elewise const shape tiling.");
  return CalcConstKey() && WriteKnownData();
}

bool Elewise::EmptyModeTiling() {
  OP_LOGD(op_type.c_str(), "Enter into elewise empty shape tiling.");
  block_dims = 1;
  tiling_key = INT32_MAX;
  return WriteKnownData();
}

bool Elewise::CalcPatternKey() {
  // broadcast dispatch set pattern for elewise, no need calculate again
  if (broadcast_dispatch) {
    return true; 
  }
  if (compile_info.only_const_tiling) {
    classify_pattern = ElewisePattern::CONST;
  } else if (!compile_info.support_broadcast || fuse_diff_shapes.size() == 1) {
    classify_pattern = ElewisePattern::COMMON;
  } else if (compile_info.support_broadcast && compile_info.classify_inputs_num > BROADCAST_SCALAR_INPUT_NUM) {
    classify_pattern = ElewisePattern::BROADCAST;
  } else if (compile_info.absorbable_broadcast) {
    classify_pattern = input_fuse_shapes[0] == 1 ? ElewisePattern::SCALAR_BROADCAST : ElewisePattern::BROADCAST_SCALAR;
  } else {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The pattern key calc failed!");
    return false;
  }
  return true;
}

bool Elewise::ParseBaseInfo() {
  try {
    const auto& current_base_info = compile_info.base_info.second.at(PATTERN_KEY.at(classify_pattern));
    constexpr uint32_t base_info_size = 4;
    // broadcast base info size may be greater than 4
    V_CHECK_GE(current_base_info.size(), base_info_size,
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

void Elewise::CalcMultiCore() {
  const int64_t multi_core_threshold = GetElementByType(out_dtype) * core_num * DOUBLE_BUFFER_SIZE;
  if (out_shape < multi_core_threshold) {
    need_multi_core = false;
  }
}

void Elewise::DoBlockTiling() {
  int64_t cur_core = core_num;
  int64_t block_factor_align_size = compile_info.ub_factor_align;
  block_factor = std::ceil(out_shape * 1.0 / cur_core);
  block_factor = std::ceil(block_factor * 1.0 / block_factor_align_size) * block_factor_align_size;
  block_dims = std::ceil(out_shape * 1.0 / block_factor);
}

bool Elewise::DoUbTiling() {
  ub_factor = block_factor;
  int64_t limit = std::min(max_available_ub, SPLIT_FACTORS.at(max_dtype));
  if (need_double_buffer) {
    limit = std::min(max_available_ub_db, SPLIT_FACTORS.at(max_dtype));
  }
  if (limit < ub_factor) {
    int64_t ub_factor_align_size = compile_info.ub_factor_align;
    V_CHECK_GT(limit, 0,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub limit must be greater than zero, but it is [%ld]", limit),
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

void Elewise::CalcTilingKey() {
  constexpr uint64_t db_tiling_key = 10000;
  tiling_key = TILING_BASE_KEY.at(classify_pattern);
  if (need_double_buffer) {
    tiling_key += db_tiling_key;
  }
}

bool Elewise::WriteTilingData() const {
  OP_LOGD(op_type.c_str(), "elewise tiling key is:%lld, block_dims is:%lld, block_factor is:%lld, ub_factor is:%lld",
          tiling_key, block_dims, block_factor, ub_factor);

  run_info.SetBlockDim(static_cast<uint32_t>(block_dims));
  if (compile_info.only_const_tiling) {
    int32_t double_buffer_num = need_double_buffer ? 1 : 0;
    constexpr int32_t elewise_block_axis = 0;
    constexpr int32_t elewise_ub_axis = 0;
    run_info.AddTilingData(static_cast<int32_t>(need_multi_core));
    run_info.AddTilingData(elewise_block_axis);
    run_info.AddTilingData(static_cast<int32_t>(block_factor));
    run_info.AddTilingData(elewise_ub_axis);
    run_info.AddTilingData(static_cast<int32_t>(ub_factor));
    run_info.AddTilingData(double_buffer_num);
    return true;
  }
  run_info.SetTilingKey(static_cast<uint32_t>(tiling_key));
  // Add elewise vars params
  try {
    const auto& var_list = compile_info.elewise_vars.second.at(std::to_string(tiling_key));
    for (const auto& var : var_list) {
      if (var >= MIN_UB_CUT_INDEX) {
        run_info.AddTilingData(static_cast<int32_t>(ub_factor));
      } else if (var >= MIN_BLOCK_CUT_INDEX) {
        run_info.AddTilingData(static_cast<int32_t>(block_factor));
      } else {
        run_info.AddTilingData(static_cast<int32_t>(input_fuse_shapes[var % MIN_DIM_CUT_INDEX]));
      }
    }
  } catch (const std::exception &e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info[_elewise_vars] error. Error message: %s", e.what());
    return false;
  }
  return compile_info.varAttrWrap.WriteVarAttrs(tiling_key, op_type, op_paras, run_info);
}

bool Elewise::SpecialModeTiling() {
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
  if (ret && !compile_info.only_const_tiling) {
    CalcTilingKey();
  }
  ret = WriteTilingData();
  return ret;
}

bool Elewise::DoTiling() {
  bool ret = CheckCompileInfo();
  GetOutputDtype();
  input_num = op_paras.GetInputsSize();
  ret = ret && GetInOutShapes();
  if (!ret) {
    OP_LOGE(op_type.c_str(), "elewise tiling input infos get failed.");
    return ret;
  }
  GetOutputDtype();
  // tiling distribute to different classify mode
  if (compile_info.classify_const_mode) {
    ret = ConstModeTiling();
  } else if (out_shape == 0) {
    ret = EmptyModeTiling();
  } else {
    ret = SpecialModeTiling();
  }
  return ret;
}

bool Elewise::DoTiling(const OpInfo& op_info) {
  bool ret = CheckCompileInfo();
  GetOutputDtype();
  input_num = op_info.GetInputShape().size();
  ret = ret && GetInOutShapes(op_info);
  if (!ret) {
    OP_LOGE(op_type.c_str(), "elewise custom tiling input infos get failed.");
    return ret;
  }
  // tiling dispatch to different classify mode
  if (compile_info.classify_const_mode) {
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
    base_info.second = outer_compile_info.at("_base_info").get<std::unordered_map<std::string, std::vector<int64_t>>>();
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

ElewiseCompileInfo::ElewiseCompileInfo(const std::string& op_type, const nlohmann::json& outer_compile_info) {
  OP_LOGD(op_type.c_str(), "elewise compile info parse running");
  ParseRequiredCompileInfo(outer_compile_info);
  ParseOptionalCompileInfo(outer_compile_info);
}
}  // namespace v3

bool ElewiseTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "elewise tiling enter into elewise non_custom tiling!");
  v3::Elewise elewise(op_type, op_paras, elewise_compile_info, run_info);
  return elewise.DoTiling();
}

bool ElewiseTilingHandler::DoTiling(const ge::Operator& op_paras,
                                    utils::OpRunInfo& run_info,
                                    const OpInfo& op_info) const {
  OP_LOGD(op_type.c_str(), "elewise tiling enter into elewise custom tiling!");
  v3::Elewise elewise(op_type, op_paras, elewise_compile_info, run_info);
  return elewise.DoTiling(op_info);
}

std::shared_ptr<AutoTilingHandler> CreateElewiseTilingHandler(const std::string& op_type,
                                                              const std::string& pattern,
                                                              const nlohmann::json& parsed_compile_info) {
  return std::make_shared<ElewiseTilingHandler>(op_type, pattern, parsed_compile_info);
}
}  // namespace optiling
