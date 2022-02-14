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
constexpr int64_t ELEWISE_REPEAT_NUMS = 128;
constexpr int64_t ELEWISE_UINT1_REPEAT_NUMS = 256;
constexpr int64_t DOUBLE_BUFFER_SIZE = 2;
constexpr int64_t ELEWISE_MAX_INPUT_NUMS = 70;
constexpr int64_t KNOWN_PATTERN_KEY = 0;
constexpr int64_t COMMON_PATTERN_KEY = 1;
constexpr int64_t CONST_TILING_KEY = 100000000;
constexpr uint32_t ELEWISE_FLAG_SIZE = 6;
constexpr int64_t ELEMENT_IN_BLOCK_DOUBLE = 4;
constexpr int64_t ELEMENT_IN_BLOCK_FLOAT = 8;
constexpr int64_t ELEMENT_IN_BLOCK_HALF = 16;
constexpr int64_t ELEMENT_IN_BLOCK_BOOL = 32;
constexpr int64_t ELEMENT_IN_BLOCK_BIT = 256;
}

const int64_t Elewise::GetElementByType(const ge::DataType& dtype) const {
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

bool Elewise::CheckCompileInfo() {
  // required compile_info check
  V_CHECK_EQ(compile_info.has_outs_uint1, true,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise compile_info must include _outs_uint1."),
             return false);
  V_CHECK_EQ(compile_info.has_flag_info, true,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise compile_info must include flag_info"),
             return false);
  V_CHECK_GT(compile_info.flag_size, 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise flag_info can not be empty."),
             return false);
  if (compile_info.pattern_key == KNOWN_PATTERN_KEY || compile_info.pattern_key == COMMON_PATTERN_KEY) {
    V_CHECK_GT(compile_info.core_num, 0,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise base_info core_num can not be neg."),
               return false);
    V_CHECK_GT(compile_info.max_dtype, 0,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise base_info max_dtype can not be neg."),
               return false);
    V_OP_TILING_CHECK((SPLIT_FACTORS.find(compile_info.max_dtype) != SPLIT_FACTORS.end()),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise base_info max_dtype not in SPLIT_FACTORS"),
                      return false);
    V_CHECK_GT(compile_info.max_available_ub, 0,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise base_info max_available_ub can not be neg."),
               return false);
    V_CHECK_GT(compile_info.max_available_ub_db, 0,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise base_info max_available_ub_db can not be neg."),
               return false);
  }
  return true;
}

bool Elewise::CheckInOutNum() {
  const uint32_t input_num = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->GetAllInputsSize();
  V_CHECK_GT(input_num, 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise input num must be greater than zero"),
             return false);
  V_CHECK_LE(input_num, ELEWISE_MAX_INPUT_NUMS,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise is not support more than 70 inputs"),
             return false);
  V_CHECK_GT(op_paras.GetOutputsSize(), 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise output num must be greater than zero"),
             return false);
  return true;
}

bool Elewise::CheckInOutNum(const OpInfo& op_info) {
  V_CHECK_GT(op_info.GetInputShape().size(), 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise custom input num must be greater than zero"),
             return false);
  V_CHECK_LE(op_info.GetInputShape().size(), ELEWISE_MAX_INPUT_NUMS,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise is not support more than 70 custom inputs"),
             return false);
  V_CHECK_GT(op_paras.GetOutputsSize(), 0,
             VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise output num must be greater than zero"),
             return false);
  return true;
}

bool Elewise::GetShapeUnderCheck() {
  constexpr uint32_t default_len = 0;
  const uint32_t input_num = ge::OpDescUtils::GetOpDescFromOperator(op_paras)->GetAllInputsSize();
  std::vector<int64_t> input_check;
  uint32_t dim_len = default_len;
  // shape len check and prepare for len inputs shape check
  for (uint32_t i = 0; i < input_num; i++) {
    const ge::GeShape& input_shape =
      ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(i)->MutableShape();
    const uint32_t shape_len = input_shape.GetDimNum();
    if (shape_len == 0 || (shape_len == 1 && input_shape.GetDim(0) == 1)) {
      continue;
    } else {
      if (shape_len != dim_len && dim_len != default_len) {
        // No empty inputs empty has different len will be check
        OP_LOGE(op_type.c_str(), "elewise not support input different len when len is over zero.");
        return false;
      }
      dim_len = shape_len;
      input_check.emplace_back(i);
    }
  }
  // check same len inputs shape
  if (!input_check.empty()) {
    const ge::GeShape& first_shape =
      ge::OpDescUtils::GetOpDescFromOperator(op_paras)->MutableInputDesc(input_check[0])->MutableShape();
    const uint32_t first_len = first_shape.GetDimNum();
    for (uint32_t i = 1; i < input_check.size(); i++) {
      for (uint32_t j = 0; j < first_len; j++) {
        V_OP_TILING_CHECK((first_shape.GetDim(j) == ge::OpDescUtils::GetOpDescFromOperator(
                           op_paras)->MutableInputDesc(input_check[i])->MutableShape().GetDim(j)),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise all input shape must be same"),
                          return false);
      }
    }
    out_shape = first_shape.GetShapeSize();
  }
  return true;
}

bool Elewise::GetShapeUnderCheck(const OpInfo& op_info) {
  const uint32_t custom_input_num = op_info.GetInputShape().size();
  std::vector<int64_t> custom_input_check;
  constexpr uint32_t custom_default_len = 0;
  uint32_t custom_dim_len = custom_default_len;
  // shape len check and prepare for inputs shape check
  for (uint32_t i = 0; i < custom_input_num; i++) {
    const std::vector<int64_t>& custom_input_shape = op_info.GetInputShape()[i];
    const uint32_t custom_shape_len = custom_input_shape.size();
    if (custom_shape_len == 0 || (custom_shape_len == 1 && custom_input_shape[0] == 1)) {
      continue;
    } else {
      if (custom_shape_len != custom_dim_len && custom_dim_len != custom_default_len) {
        // No empty custom inputs empty has different len will be check
        OP_LOGE(op_type.c_str(), "elewise not support custom input diff len when len is over zero.");
        return false;
      }
      custom_dim_len = custom_shape_len;
      custom_input_check.emplace_back(static_cast<int64_t>(i));
    }
  }
  // check same custom inputs shape
  if (!custom_input_check.empty()) {
    V_CHECK_GT(static_cast<int64_t>(op_info.GetInputShape().size()), custom_input_check.back(),
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise custom input num may be out of index"),
               return false);
    const std::vector<int64_t>& custom_first_shape = op_info.GetInputShape()[custom_input_check[0]];
    const uint32_t custom_first_len = custom_first_shape.size();
    for (uint32_t i = 1; i < custom_input_check.size(); i++) {
      for (uint32_t j = 0; j < custom_first_len; j++) {
        V_OP_TILING_CHECK((custom_first_shape[j] == op_info.GetInputShape()[custom_input_check[i]][j]),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "elewise all custom input shape must be same"),
                          return false);
      }
    }
    out_shape =
      std::accumulate(custom_first_shape.begin(), custom_first_shape.end(), 1LL, std::multiplies<int64_t>());
  }
  return true;
}

void Elewise::GetShapeFromBroadcast(const OpInfo& op_info) {
  const uint32_t custom_input_num = op_info.GetInputShape().size();
  for (uint32_t i = 0; i < custom_input_num; i++) {
      int64_t shape_size = std::accumulate(op_info.GetInputShape()[i].begin(),
                                           op_info.GetInputShape()[i].end(), 1LL, std::multiplies<int64_t>());
      if (shape_size > out_shape) {
        out_shape = shape_size;
        break;
      }
  }
}

bool Elewise::GetOutShape() {
  // input and output number has limit
  bool ret = CheckInOutNum();
  if (!ret) {
    OP_LOGE(op_type.c_str(), "elewise input or output num is illegal!");
    return ret;
  }
  // broadcast pattern will be only changed by broadcast tiling distribution
  if (compile_info.broadcast_pattern) {
    OP_LOGE(op_type.c_str(), "elewise not custom dotiling only support broadcast_pattern false.");
    return false;
  }
  return GetShapeUnderCheck();
}

bool Elewise::GetOutShape(const OpInfo& op_info) {
  bool ret = CheckInOutNum(op_info);
  if (!ret) {
    OP_LOGE(op_type.c_str(), "elewise custom input or output num is illegal!");
    return ret;
  }
  if (compile_info.broadcast_pattern) {
    GetShapeFromBroadcast(op_info);
    return true;
  } else {
    return GetShapeUnderCheck(op_info);
  }
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

void Elewise::WriteKnownData() {
  OP_LOGD(op_type.c_str(), "elewise known tiling key is:%lld and block_dims is:%lld", tiling_key, block_dims);
  run_info.SetBlockDim(static_cast<uint32_t>(block_dims));
  run_info.SetTilingKey(static_cast<uint32_t>(tiling_key));
}

void Elewise::DoConstTiling() {
  OP_LOGD(op_type.c_str(), "Enter into elewise const shape tiling.");
  block_dims = compile_info.const_block_dims;
  tiling_key = CONST_TILING_KEY;
  WriteKnownData();
}

void Elewise::DoEmptyTiling() {
  OP_LOGD(op_type.c_str(), "Enter into elewise empty shape tiling.");
  block_dims = 1;
  tiling_key = INT32_MAX;
  WriteKnownData();
}

void Elewise::CalcMultiCore() {
  const int64_t multi_core_threshold = GetElementByType(out_dtype) * compile_info.core_num * DOUBLE_BUFFER_SIZE;
  if (out_shape < multi_core_threshold) {
    need_multi_core = false;
  }
}

void Elewise::DoBlockTiling() {
  int64_t cur_core = compile_info.core_num;
  int64_t elewise_align_size = compile_info.outs_uint1 ? ELEWISE_UINT1_REPEAT_NUMS : ELEWISE_REPEAT_NUMS;
  block_factor = std::ceil(out_shape * 1.0 / cur_core);
  block_factor = std::ceil(block_factor * 1.0 / elewise_align_size) * elewise_align_size;
  block_dims = std::ceil(out_shape * 1.0 / block_factor);
}

bool Elewise::DoUbTiling() {
  ub_factor = block_factor;
  int64_t limit = std::min(compile_info.max_available_ub, SPLIT_FACTORS.at(compile_info.max_dtype));
  if (need_double_buffer) {
    limit = std::min(compile_info.max_available_ub_db, SPLIT_FACTORS.at(compile_info.max_dtype));
  }
  if (limit < ub_factor) {
    int64_t elewise_align_size = compile_info.outs_uint1 ? ELEWISE_UINT1_REPEAT_NUMS : ELEWISE_REPEAT_NUMS;
    V_CHECK_GT(limit, 0,
               VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ub limit must be greater than zero, but it is [%ld]", limit),
               return false);
    int64_t ub_for_num = std::ceil(ub_factor * 1.0 / limit);
    int64_t adjust_factor = std::ceil(ub_factor * 1.0 / ub_for_num);
    int64_t align_factor = std::ceil(adjust_factor * 1.0 / elewise_align_size);
    ub_factor = align_factor * elewise_align_size;
    if (ub_factor > limit) {
      ub_factor = std::floor(adjust_factor * 1.0 / elewise_align_size) * elewise_align_size;
    }
  }
  return true;
}

void Elewise::CalcCommonKey() {
  constexpr int64_t common_tiling_key = 210000000;
  constexpr int64_t db_tiling_key = 10000;
  tiling_key = compile_info.use_special_pattern ? common_tiling_key : 0;
  if (need_double_buffer) {
    tiling_key += db_tiling_key;
  }
}

bool Elewise::DoCommonTiling() {
  OP_LOGD(op_type.c_str(), "Enter into elewise common shape tiling.");
  bool ret = true;
  CalcMultiCore();
  if (need_multi_core) {
    DoBlockTiling();
    if (block_factor > std::min(compile_info.max_available_ub, SPLIT_FACTORS.at(compile_info.max_dtype))) {
      need_double_buffer = true;
    }
    ret = ret && DoUbTiling();
  } else {
    block_dims = 1;
    block_factor = out_shape;
    ub_factor = out_shape;
  }
  if (ret && !compile_info.only_const_tiling) {
    CalcCommonKey();
  }
  WriteCommonData();
  return ret;
}

void Elewise::WriteCommonData() const {
  OP_LOGD(op_type.c_str(), "elewise tiling key is:%lld", tiling_key);
  OP_LOGD(op_type.c_str(), "elewise tiling block_dims is:%lld", block_dims);
  OP_LOGD(op_type.c_str(), "elewise tiling block_factor is:%lld", block_factor);
  OP_LOGD(op_type.c_str(), "elewise tiling ub_factor is:%lld", ub_factor);

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
  } else {
    constexpr uint32_t pure_elewise_var_size = 3;
    run_info.SetTilingKey(static_cast<uint32_t>(tiling_key));
    if (compile_info.elewise_vars_size == pure_elewise_var_size) {
      run_info.AddTilingData(static_cast<int32_t>(out_shape));
    }
    run_info.AddTilingData(static_cast<int32_t>(block_factor));
    run_info.AddTilingData(static_cast<int32_t>(ub_factor));
  }
}

bool Elewise::DoTiling() {
  bool ret = CheckCompileInfo();
  ret = ret && GetOutShape();
  if (!ret) {
    OP_LOGE(op_type.c_str(), "elewise tiling input infos check failed.");
    return ret;
  }
  GetOutputDtype();
  // elewise tiling compose of const, empty and common scene
  if (compile_info.is_const_shapes) {
    DoConstTiling();
  } else if (out_shape == 0) {
    DoEmptyTiling();
  } else {
    ret = DoCommonTiling();
  }
  return ret;
}

bool Elewise::DoTiling(const OpInfo& op_info) {
  bool ret = CheckCompileInfo();
  ret = ret && GetOutShape(op_info);
  if (!ret) {
    OP_LOGE(op_type.c_str(), "elewise custom tiling input infos check failed.");
    return ret;
  }
  GetOutputDtype();
  // elewise tiling compose of const, empty and common scene
  if (compile_info.is_const_shapes) {
    DoConstTiling();
  } else if (out_shape == 0) {
    DoEmptyTiling();
  } else {
    ret = DoCommonTiling();
  }
  return ret;
}

void ElewiseCompileInfo::ParseOutsUintOne(const nlohmann::json& outer_compile_info) {
  if (!outer_compile_info.contains("_outs_uint1")) {
    has_outs_uint1 = false;
  } else {
    outs_uint1 = outer_compile_info.at("_outs_uint1").get<bool>();
  }
}

void ElewiseCompileInfo::ParseFlagInfo(const nlohmann::json& outer_compile_info) {
  if (!outer_compile_info.contains("_flag_info")) {
    has_flag_info = false;
  } else {
    const std::vector<bool>& input_flag_info = outer_compile_info.at("_flag_info").get<std::vector<bool>>();
    flag_size = input_flag_info.size();
    if (input_flag_info.size() > 0) {
      only_const_tiling = input_flag_info[0];
      constexpr uint32_t common_flag_size = 6;
      constexpr uint32_t const_shapes_index = 1;
      constexpr uint32_t special_pattern_index = 3;
      // broadcast scene flag info size is seven
      if (input_flag_info.size() >= common_flag_size) {
        is_const_shapes = input_flag_info[const_shapes_index];
        use_special_pattern = input_flag_info[special_pattern_index];
      }
    }
  }
}

void ElewiseCompileInfo::ParseBaseInfo(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_base_info") && !outer_compile_info.at("_base_info").empty()) {
    constexpr uint32_t base_info_size = 4;
    constexpr uint32_t max_ub_index = 2;
    constexpr uint32_t max_ub_db_index = 3;
    constexpr int64_t elewise_known_key = 0;
    constexpr int64_t elewise_common_key = 1;
    const std::string known_str_key = "000";
    const std::string common_str_key = "100";
    const std::unordered_map<std::string, std::vector<int64_t>>& input_base_info =
      outer_compile_info.at("_base_info").get<std::unordered_map<std::string, std::vector<int64_t>>>();
    if (input_base_info.size() == 1) {
      if (flag_size >= ELEWISE_FLAG_SIZE && input_base_info.count(common_str_key) &&
          input_base_info.at(common_str_key).size() == base_info_size) {
        pattern_key = elewise_common_key;
        core_num = input_base_info.at(common_str_key)[0];
        max_dtype = input_base_info.at(common_str_key)[1];
        max_available_ub = input_base_info.at(common_str_key)[max_ub_index];
        max_available_ub_db = input_base_info.at(common_str_key)[max_ub_db_index];
      } else if (input_base_info.count(known_str_key) && input_base_info.at(known_str_key).size() == base_info_size) {
        pattern_key = elewise_known_key;
        core_num = input_base_info.at(known_str_key)[0];
        max_dtype = input_base_info.at(known_str_key)[1];
        max_available_ub = input_base_info.at(known_str_key)[max_ub_index];
        max_available_ub_db = input_base_info.at(known_str_key)[max_ub_db_index];
      }
    } else {
      if (input_base_info.count(common_str_key) && input_base_info.at(common_str_key).size() == base_info_size) {
        pattern_key = elewise_common_key;
        core_num = input_base_info.at(common_str_key)[0];
        max_dtype = input_base_info.at(common_str_key)[1];
        max_available_ub = input_base_info.at(common_str_key)[max_ub_index];
        max_available_ub_db = input_base_info.at(common_str_key)[max_ub_db_index];
      }
    }
  }
}

void ElewiseCompileInfo::ParseConstDims(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_const_block_dims") && is_const_shapes) {
    const std::vector<int64_t>& input_const_dims =
      outer_compile_info.at("_const_block_dims").get<std::vector<int64_t>>();
    if (!input_const_dims.empty()) {
      const_block_dims = input_const_dims[0];
    }
  }
}

void ElewiseCompileInfo::ParseElewiseVarSize(const nlohmann::json& outer_compile_info) {
  if (outer_compile_info.contains("_elewise_vars")&& !outer_compile_info.at("_elewise_vars").empty()) {
    const std::string tiling_key_str = "210000000";
    const std::string tiling_key_db_str = "210010000";
    const std::unordered_map<std::string, std::vector<int64_t>>& elewise_vars =
      outer_compile_info.at("_elewise_vars").get<std::unordered_map<std::string, std::vector<int64_t>>>();
    if (elewise_vars.count(tiling_key_str)) {
      elewise_vars_size = elewise_vars.at(tiling_key_str).size();
    } else if (elewise_vars.count(tiling_key_db_str)) {
      elewise_vars_size = elewise_vars.at(tiling_key_db_str).size();
    }
  }
}

void ElewiseCompileInfo::SetBroadcastPattern(const bool& is_broadcast_pattern) {
  broadcast_pattern = is_broadcast_pattern;
}

ElewiseCompileInfo::ElewiseCompileInfo(const std::string& op_type, const nlohmann::json& outer_compile_info) {
  OP_LOGD(op_type.c_str(), "elewise compile info parse running");
  ParseOutsUintOne(outer_compile_info);
  ParseFlagInfo(outer_compile_info);
  ParseBaseInfo(outer_compile_info);
  ParseConstDims(outer_compile_info);
  ParseElewiseVarSize(outer_compile_info);
}
}  // namespace v3

bool ElewiseTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD(op_type.c_str(), "elewise tiling running");
  v3::Elewise elewise(op_type, op_paras, elewise_compile_info, run_info);
  return elewise.DoTiling();
}

bool ElewiseTilingHandler::DoTiling(const ge::Operator& op_paras,
                                    utils::OpRunInfo& run_info,
                                    const OpInfo& op_info) const {
  OP_LOGD(op_type.c_str(), "elewise custom tiling running");
  v3::Elewise elewise(op_type, op_paras, elewise_compile_info, run_info);
  return elewise.DoTiling(op_info);
}

std::shared_ptr<AutoTilingHandler> CreateElewiseTilingHandler(const std::string& op_type,
                                                              const std::string& pattern,
                                                              const nlohmann::json& parsed_compile_info) {
  return std::make_shared<ElewiseTilingHandler>(op_type, pattern, parsed_compile_info);
}
}  // namespace optiling
