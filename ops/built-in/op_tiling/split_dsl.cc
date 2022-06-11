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
 * \file split_dsl.cc
 * \brief
 */

#include "split_dsl.h"

#include <cmath>
#include <numeric>

#include "graph/utils/op_desc_utils.h"
#include "vector_tiling.h"
#include "tiling_handler.h"
#include "auto_tiling_register.h"

namespace optiling {
namespace split {
namespace {
constexpr std::int32_t ELEMENT_IN_BLOCK_DEFAULT = 16;
constexpr std::int32_t ELEMENT_IN_BLOCK_B32 = 8;
constexpr std::int32_t ELEMENT_IN_BLOCK_B8 = 32;
constexpr std::int32_t ELEMENT_IN_BLOCK_B64 = 4;
constexpr std::int32_t SPLIT_DIM_LEN = 2;
constexpr std::int64_t ALL_ALIGN_NODE_NUMBERS = 1;
constexpr std::int64_t ALL_ALIGN_COPY_NODE_NUMBERS = 2;
constexpr std::int64_t HALF_CORE = 2;
constexpr std::int64_t BASE_NODE_NUMBERS = 2;
constexpr std::int64_t GENERAL_NODE_NUMBERS = 3;
constexpr std::int64_t ALL_ALIGN_COPY_EXPERIENCE = 12;
constexpr std::int64_t ONE_REPEAT_BLOCK_NUM = 8;
constexpr std::int64_t MULTI_CORE_THRESHOLD = 2048;
constexpr std::int64_t ROW_LIMIT_FACTOR = 16;
constexpr std::int64_t BASE_KEY = 3000000;
constexpr std::int64_t GENERAL_KEY = 2000000;
constexpr std::int64_t ALL_ALIGN_KEY = 4000000;
constexpr std::int64_t ALL_ALIGN_COPY_KEY = 4100000;
constexpr std::int64_t CUT_M_KEY = 5000000;
constexpr std::int64_t GENERAL_NONE_CUT_KEY = 0;
constexpr std::uint32_t CONST_TILING_KEY = 1000000;
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

SplitCompileInfo::SplitCompileInfo(const nlohmann::json& json_compile_info) {
  Parse("SplitDsl", json_compile_info);
}

bool SplitCompileInfo::Parse(const char* op_type, const nlohmann::json& json_compile_info) {
  try {
    is_const = json_compile_info.at("_is_const");
    if (is_const) {
      const_block_dims = json_compile_info.at("_const_dims");
    }
    core_num = json_compile_info.at("_core_num");
    ub_size = json_compile_info.at("_ub_size");
    ori_axis = json_compile_info.at("_ori_axis");
    only_const_tiling = json_compile_info.at("_only_const_tiling");
    is_avg_split = json_compile_info.at("_avg_split");
    if (!only_const_tiling) {
      split_is_const = json_compile_info.at("_split_is_const");
      split_vars = json_compile_info.at("_split_vars").get<std::vector<bool>>();
    }
  } catch (...) {
    OP_LOGE(op_type, "Unknown Exception encountered when parsing Compile Info of op_type %s", op_type);
    return false;
  }
  return true;
}

template <typename T>
bool Split<T>::GenerateBaseInput(const OpShape& shape, int64_t& axis) {
  V_OP_TILING_CHECK(context->GetInputDataType(static_cast<size_t>(0), dtype),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get inpute dtype error"),
                    return false);
  output_nums = context->GetOutputNums();
  is_single_output = output_nums == 1;
  V_OP_TILING_CHECK((output_nums <= MAX_INPUT_NUM),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output tensor is too much"), return false);
  int64_t dim_len = static_cast<int64_t>(shape.GetDimNum());
  axis = c_info->ori_axis < 0 ? c_info->ori_axis + dim_len : c_info->ori_axis;
  axis = is_single_output ? 0 : axis;
  V_OP_TILING_CHECK((axis >= 0 && axis < dim_len),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and compile shape not match"),
                    return false);
  int64_t fused_m = 1;
  int64_t fused_n = 1;
  for (int64_t j = 0; j < axis; j++) {
    fused_m *= shape.GetDim(j);
  }
  for (int64_t j = axis; j < dim_len; j++) {
    fused_n *= shape.GetDim(j);
  }
  input_shapes[0] = fused_m;
  input_shapes[1] = fused_n;
  is_empty = fused_m * fused_n == 0;
  return true;
}

template <typename T>
bool Split<T>::GenerateOutputShape() {
  int64_t axis = 0;
  const OpShape& shape = context->GetInputShape(0);
  V_OP_TILING_CHECK((!shape.Empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor can not be empty"), return false);
  if (!GenerateBaseInput(shape, axis)) {
    return false;
  }
  if (c_info->is_avg_split) {
    int64_t split_output = input_shapes[1] / output_nums;
    V_OP_TILING_CHECK((input_shapes[1] % output_nums == 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and split shape not match"),
                      return false);
    output_nums = 1;
    for (int64_t i = 0; i < output_nums; i++) {
      output_shapes[i][0] = input_shapes[0];
      output_shapes[i][1] = split_output;
    }
    min_split_shape = std::min(min_split_shape, split_output);
  } else {
    int64_t split_output = input_shapes[1] / shape.GetDim(axis);
    for (int64_t i = 0; i < output_nums; i++) {
      OpShape out_shape = context->GetOutputShape(i);
      output_shapes[i][0] = input_shapes[0];
      int64_t split_shape = split_output * out_shape.GetDim(axis);
      output_shapes[i][1] = split_shape;
      min_split_shape = split_shape != 0 ? std::min(min_split_shape, split_shape) : min_split_shape;
    }
  }
  if (shape.GetDimNum() == 0) {
    input_shapes[0] = 1;
    input_shapes[1] = 1;
    output_shapes[0].fill(1);
  }
  return true;
}

template <typename T>
bool Split<T>::GenerateBaseInputFromOp(const std::vector<int64_t>& shape, int64_t& ori_axis) {
  V_OP_TILING_CHECK(context->GetInputDataType(op_info, dtype),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get inpute dtype error"),
                    return false);
  output_nums = context->GetOutputNums();
  is_single_output = output_nums == 1;
  V_OP_TILING_CHECK((output_nums <= MAX_INPUT_NUM),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output tensor is too much"), return false);

  V_OP_TILING_CHECK((!shape.empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape is empty"), return false);
  auto dim_len = static_cast<int64_t>(shape.size());
  const std::vector<int64_t>* axes = op_info->GetAxes();
  V_OP_TILING_CHECK((axes != nullptr && !axes->empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor is too much"), return false);
  ori_axis = axes->at(0) < 0 ? axes->at(0) + dim_len : axes->at(0);
  ori_axis = is_single_output ? 0 : ori_axis;
  V_OP_TILING_CHECK((ori_axis >= 0 && ori_axis < dim_len),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and compile shape not match"),
                    return false);
  int64_t fused_m = 1;
  int64_t fused_n = 1;
  for (int64_t j = 0; j < ori_axis; j++) {
    fused_m *= shape[j];
  }
  for (int64_t j = ori_axis; j < dim_len; j++) {
    fused_n *= shape[j];
  }
  input_shapes[0] = fused_m;
  input_shapes[1] = fused_n;
  is_empty = fused_m * fused_n == 0;
  is_base = output_nums == 1;
  return true;
}

template <typename T>
bool Split<T>::GenerateOutputShapeFromOp() {
  const auto inputs = op_info->GetInputShape();
  V_OP_TILING_CHECK((inputs != nullptr && !inputs->empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor is empty"), return false);
  const std::vector<int64_t>& shape = inputs->at(0);
  int64_t ori_axis = 0;
  if (!GenerateBaseInputFromOp(shape, ori_axis)) {
    return false;
  }
  if (c_info->is_avg_split) {
    int64_t split_output = input_shapes[1] / output_nums;
    V_OP_TILING_CHECK((input_shapes[1] % output_nums == 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and split shape not match"),
                      return false);
    output_nums = 1;
    for (int64_t i = 0; i < output_nums; i++) {
      output_shapes[i][0] = input_shapes[0];
      output_shapes[i][1] = split_output;
    }
    min_split_shape = std::min(min_split_shape, split_output);
  } else {
    V_OP_TILING_CHECK((inputs->size() >= 1),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get split shape error"), return false);
    const std::vector<int64_t>& split_shape = inputs->at(1);
    V_OP_TILING_CHECK((output_nums == static_cast<int64_t>(split_shape.size())),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and split shape not match"),
                      return false);
    int64_t split_output = input_shapes[1] / shape[ori_axis];
    for (int64_t i = 0; i < output_nums; i++) {
      int64_t split_dim_out = is_single_output ? shape[ori_axis] : split_shape[i];
      output_shapes[i][0] = input_shapes[0];
      int64_t split_size = split_output * split_dim_out;
      output_shapes[i][1] = split_size;
      min_split_shape = split_size != 0 ? std::min(min_split_shape, split_size) : min_split_shape;
    }
  }
  if (shape.size() == 0) {
    input_shapes[0] = 1;
    input_shapes[1] = 1;
    output_shapes[0].fill(1);
  }
  return true;
}

template <typename T>
bool Split<T>::IsAllAlign() {
  int64_t ele_in_block = GetElementByType(dtype);
  if (input_shapes[1] % ele_in_block != 0) {
    return false;
  }
  for (int64_t i = 0; i < output_nums; i++) {
    if (output_shapes[i][1] % ele_in_block != 0) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool Split<T>::CalcSingleOutputInfo() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t bytes = BLOCK_SIZE_BYTES / ele_in_block;
  is_base = true;
  need_multi_core = true;
  max_available_ub = c_info->ub_size / BLOCK_SIZE_BYTES * BLOCK_SIZE_BYTES / bytes;
  return true;
}

template <typename T>
bool Split<T>::CalcAllAlignInfo() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t bytes = BLOCK_SIZE_BYTES / ele_in_block;
  max_available_ub = c_info->ub_size / ALL_ALIGN_COPY_NODE_NUMBERS / BLOCK_SIZE_BYTES * BLOCK_SIZE_BYTES / bytes;
  bool is_real_align_copy = input_shapes[1] <= max_available_ub &&
                            min_split_shape < ALL_ALIGN_COPY_EXPERIENCE * ele_in_block &&
                            input_shapes[0] * input_shapes[1] > max_available_ub &&
                            min_split_shape % (ONE_REPEAT_BLOCK_NUM * ele_in_block) != 0;
  if (is_real_align_copy) {
    col_limit = input_shapes[1];
    row_limit = max_available_ub / col_limit;
  } else {
    is_all_align = true;
    is_all_align_copy = false;
    max_available_ub = c_info->ub_size / ALL_ALIGN_NODE_NUMBERS / BLOCK_SIZE_BYTES * BLOCK_SIZE_BYTES / bytes;
    V_OP_TILING_CHECK((max_available_ub != 0),
                      VECTOR_INNER_ERR_REPORT_TILIING("Split", "max_available_ub cannot be zero."), return false);
    row_limit = std::min(ROW_LIMIT_FACTOR, input_shapes[0]);
    col_limit = max_available_ub / row_limit / ele_in_block * ele_in_block;
    if (input_shapes[1] < col_limit) {
      col_limit = input_shapes[1];
      row_limit = max_available_ub / col_limit;
      V_OP_TILING_CHECK((row_limit != 0),
                        VECTOR_INNER_ERR_REPORT_TILIING("Split", "row_limit cannot be zero."), return false);
      if (input_shapes[0] / row_limit < c_info->core_num) {
        row_limit = std::max(ROW_LIMIT_FACTOR, (input_shapes[0] + c_info->core_num - 1) / c_info->core_num);
      }
    }
  }
  need_multi_core = true;
  return true;
}

template <typename T>
bool Split<T>::CalcGeneralInfo() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t bytes = BLOCK_SIZE_BYTES / ele_in_block;
  max_available_ub = c_info->ub_size / GENERAL_NODE_NUMBERS / BLOCK_SIZE_BYTES * BLOCK_SIZE_BYTES / bytes;
  int64_t row_base = ROW_LIMIT_FACTOR * ele_in_block;
  row_limit = row_base;
  col_limit = max_available_ub / row_base / ele_in_block * ele_in_block;
  if (input_shapes[1] <= col_limit) {
    col_limit = input_shapes[1];
    if (ele_in_block == ELEMENT_IN_BLOCK_DEFAULT && input_shapes[0] != 1) {
      row_limit = max_available_ub / input_shapes[1] / row_base * row_base;
      only_cut_m = true;
    } else {
      is_base = true;
      max_available_ub = c_info->ub_size / BASE_NODE_NUMBERS / BLOCK_SIZE_BYTES * BLOCK_SIZE_BYTES / bytes;
    }
  } else {
    int64_t last_align = (min_split_shape * 2 + ele_in_block - 1) / ele_in_block * ele_in_block;
    bool is_general = c_info->is_avg_split && ele_in_block == ELEMENT_IN_BLOCK_DEFAULT && col_limit >= last_align &&
                      input_shapes[0] != 1;
    if (is_general) {
      col_limit = col_limit / min_split_shape * min_split_shape;
      last_align = (col_limit + ele_in_block - 1) / ele_in_block * ele_in_block;
      row_limit = max_available_ub / last_align / row_base * row_base;
    } else {
      is_base = true;
      max_available_ub = c_info->ub_size / BASE_NODE_NUMBERS / BLOCK_SIZE_BYTES * BLOCK_SIZE_BYTES / bytes;
    }
  }

  need_multi_core = input_shapes[0] > row_limit || input_shapes[1] > col_limit || is_all_align || is_base;
  only_cut_m = need_multi_core && only_cut_m;
  row_limit = only_cut_m ? row_base : row_limit;
  return true;
}

template <typename T>
bool Split<T>::CalcTiling() {
  if (is_single_output) {
    return CalcSingleOutputInfo();
  }
  is_all_align_copy = IsAllAlign();
  if (is_all_align_copy) {
    return CalcAllAlignInfo();
  }
  return CalcGeneralInfo();
}

template <typename T>
void Split<T>::DoBaseBlockTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t core_num = c_info->core_num;
  if (input_shapes[0] * input_shapes[1] < MULTI_CORE_THRESHOLD) {
    core_num = 1;
  }
  if (min_split_shape < ele_in_block && (input_shapes[0] * min_split_shape) / core_num < ele_in_block &&
      core_num != 1) {
    int64_t factor_left = (ele_in_block + min_split_shape - 1) / min_split_shape;
    core_num = std::max(input_shapes[0] / factor_left, 1L);
  }
  if (input_shapes[0] > core_num || min_split_shape / (core_num / input_shapes[0]) < ele_in_block) {
    block_factor = (input_shapes[0] + core_num - 1) / core_num;
    block_dims = (input_shapes[0] + block_factor - 1) / block_factor;
    if (c_info->is_avg_split) {
      avg_block_factor = output_shapes[0][1];
    }
  } else {
    block_factor = 1;
    block_dims = input_shapes[0];
    if (c_info->is_avg_split) {
      core_num = core_num / block_dims;
      avg_block_factor = (output_shapes[0][1] + core_num - 1) / core_num;
      block_dims *= (output_shapes[0][1] + avg_block_factor - 1) / avg_block_factor;
    }
  }
}

template <typename T>
void Split<T>::DoBaseUbTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  for (int64_t i = 0; i < output_nums; i++) {
    int64_t output_n = output_shapes[i][1];
    output_n = output_n == 0 ? 1 : output_n;
    if (c_info->is_avg_split) {
      output_n = avg_block_factor;
    }
    int64_t high_ub_factor = 1;
    int64_t low_ub_factor = 1;
    int64_t last_dim = (output_n + ele_in_block - 1) / ele_in_block * ele_in_block;
    if (last_dim >= max_available_ub) {
      high_ub_factor = std::min(max_available_ub, output_n);
    } else {
      high_ub_factor = output_n;
      low_ub_factor = std::min(block_factor, max_available_ub / last_dim);
    }
    high_ub_factors[i] = high_ub_factor;
    low_ub_factors[i] = low_ub_factor;
  }
}

template <typename T>
void Split<T>::DoUbTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t row_base = is_all_align || is_all_align_copy ? 1 : ROW_LIMIT_FACTOR * ele_in_block;
  low_ub_factors[0] = std::min(row_limit, ((input_shapes[0] + row_base - 1) / row_base * row_base));
  high_ub_factors[0] = std::min(col_limit, input_shapes[1]);
}

template <typename T>
void Split<T>::DoBlockTiling() {
  int64_t row_left = (input_shapes[0] + low_ub_factors[0] - 1) / low_ub_factors[0];
  if (row_left <= c_info->core_num / HALF_CORE && !(only_cut_m || is_all_align_copy)) {
    block_axis = 1;
    int64_t col_left = (input_shapes[1] + high_ub_factors[0] - 1) / high_ub_factors[0];
    int64_t core_left = c_info->core_num / row_left;
    block_factor = (col_left + core_left - 1) / core_left;
    block_dims = row_left * ((col_left + block_factor - 1) / block_factor);
  } else {
    block_axis = 0;
    block_factor = (row_left + c_info->core_num - 1) / c_info->core_num;
    block_dims = (row_left + block_factor - 1) / block_factor;
  }
}

template <typename T>
void Split<T>::CalcKey() {
  tiling_key = GENERAL_NONE_CUT_KEY;
  if (is_base) {
    tiling_key = BASE_KEY;
    return;
  } else if (only_cut_m) {
    tiling_key = CUT_M_KEY;
  } else if (need_multi_core) {
    tiling_key = GENERAL_KEY;
  }
  if (is_all_align) {
    tiling_key = ALL_ALIGN_KEY;
  } else if (is_all_align_copy) {
    tiling_key = ALL_ALIGN_COPY_KEY;
  }

  tiling_key += block_axis;
}

template <typename T>
bool Split<T>::DoTiling() {
  bool ret = true;
  if (op_info != nullptr) {
    ret = GenerateOutputShapeFromOp();
  } else {
    ret = GenerateOutputShape();
  }
  if (is_empty) {
    tiling_key = INT32_MAX;
    block_dims = 1;
    return ret;
  }
  ret = ret && CalcTiling();
  if (ret && need_multi_core) {
    if (is_base) {
      DoBaseBlockTiling();
      DoBaseUbTiling();
    } else {
      DoUbTiling();
      DoBlockTiling();
    }
  }
  if (ret) {
    CalcKey();
  }
  return ret;
}

template <typename T>
bool Split<T>::WriteConstTilingData() {
  size_t tiling_num = 0;
  tiling_data[tiling_num++] = static_cast<int32_t>(need_multi_core);
  tiling_data[tiling_num++] = static_cast<int32_t>(is_base);
  if (is_all_align_copy) {
    is_all_align = true;
    only_cut_m = true;
  }
  tiling_data[tiling_num++] = static_cast<int32_t>(is_all_align);
  tiling_data[tiling_num++] = static_cast<int32_t>(only_cut_m);
  tiling_data[tiling_num++] = static_cast<int32_t>(block_axis);
  tiling_data[tiling_num++] = static_cast<int32_t>(block_factor);
  tiling_data[tiling_num++] = static_cast<int32_t>(avg_block_factor);
  int64_t out_num = c_info->is_avg_split ? 1 : output_nums;
  for (int64_t i = 0; i < out_num; i++) {
    tiling_data[tiling_num++] = static_cast<int32_t>(low_ub_factors[i]);
  }
  for (int64_t i = 0; i < out_num; i++) {
    tiling_data[tiling_num++] = static_cast<int32_t>(high_ub_factors[i]);
  }
  for (size_t i = 0; i < tiling_num; i++) {
    if (!context->Append(tiling_data[i])) {
      return false;
    }
  }
  return true;
}

template <typename T>
void Split<T>::WriteShapeData(size_t& tiling_num) {
  for (size_t i = 0; i < c_info->split_vars.size(); i++) {
    if (c_info->split_vars[i]) {
      tiling_data[tiling_num++] = static_cast<int32_t>(input_shapes[i]);
    }
  }
  if (c_info->is_avg_split) {
    if (!c_info->split_is_const) {
      tiling_data[tiling_num++] = static_cast<int32_t>(output_shapes[0][1]);
    }
  } else {
    for (int64_t i = 0; i < output_nums; i++) {
      tiling_data[tiling_num++] = static_cast<int32_t>(output_shapes[i][1]);
    }
  }
}

template <typename T>
void Split<T>::WriteCutData(size_t& tiling_num) {
  if (is_base) {
    if (c_info->is_avg_split) {
      tiling_data[tiling_num++] = static_cast<int32_t>(block_factor);
      tiling_data[tiling_num++] = static_cast<int32_t>(avg_block_factor);
    } else {
      tiling_data[tiling_num++] = static_cast<int32_t>(block_factor);
    }
    for (int64_t i = 0; i < output_nums; i++) {
      tiling_data[tiling_num++] = static_cast<int32_t>(low_ub_factors[i]);
      tiling_data[tiling_num++] = static_cast<int32_t>(high_ub_factors[i]);
    }
  } else {
    tiling_data[tiling_num++] = static_cast<int32_t>(block_factor);
    if (is_all_align || is_all_align_copy) {
      tiling_data[tiling_num++] = static_cast<int32_t>(low_ub_factors[0]);
    }
    if (!(only_cut_m || is_all_align_copy)) {
      tiling_data[tiling_num++] = static_cast<int32_t>(high_ub_factors[0]);
    }
  }
}

template <typename T>
bool Split<T>::WriteTilingData() {
  OP_LOGD(op_type.c_str(), "tiling key:%ld", tiling_key);
  OP_LOGD(op_type.c_str(), "tiling block_dims:%ld", block_dims);
  OP_LOGD(op_type.c_str(), "tiling block_factor:%ld", block_factor);
  OP_LOGD(op_type.c_str(), "tiling low_ub_factor:%ld", low_ub_factors[0]);
  OP_LOGD(op_type.c_str(), "tiling high_ub_factor:%ld", high_ub_factors[0]);
  OP_LOGD(op_type.c_str(), "tiling block_axis:%ld", block_axis);

  context->SetBlockDim(static_cast<uint32_t>(block_dims));
  if (c_info->only_const_tiling) {
    WriteConstTilingData();
    return true;
  }
  context->SetTilingKey(tiling_key);
  if (is_empty) {
    return true;
  }
  size_t tiling_num = 0;
  V_OP_TILING_CHECK((c_info->split_vars.size() == SPLIT_DIM_LEN),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "split variable error"), return false);
  WriteShapeData(tiling_num);
  if (!need_multi_core) {
    for (size_t i = 0; i < tiling_num; i++) {
      if (!context->Append(tiling_data[i])) {
        return false;
      }
    }
    return true;
  }
  WriteCutData(tiling_num);
  for (size_t i = 0; i < tiling_num; i++) {
    if (!context->Append(tiling_data[i])) {
      return false;
    }
  }
  return true;
}

template <typename T>
void Split<T>::ProcessConst() const {
  context->SetTilingKey(CONST_TILING_KEY);
  context->SetBlockDim(static_cast<uint32_t>(c_info->const_block_dims));
}

template <typename T>
bool Split<T>::SplitTiling() {
  op_type = context->GetOpType();
  c_info = dynamic_cast<const SplitCompileInfo *>(context->GetCompileInfo());
  if (c_info->is_const) {
    ProcessConst();
    return true;
  }
  bool ret = DoTiling();
  ret = ret && WriteTilingData();
  return ret;
}
}  // namespace split

bool CreateSplitDslTiling(gert::TilingContext* context, const OpInfoImpl* op_info) {
  OP_LOGD("SplitDsl", "enter SplitDsl");
  AutoTilingContext auto_tiling_context(context);
  if (op_info) {
    auto_tiling_context.SetCompileInfo(op_info->GetCompileInfo());
  }
  split::Split<AutoTilingContext> split(&auto_tiling_context, op_info);
  return split.SplitTiling();
}

AutoTilingCompileInfo* CreateSplitDslParser(const char* op_type, const nlohmann::json& json_compile_info) {
  auto compile_info = new split::SplitCompileInfo();
  if (!compile_info->Parse(op_type, json_compile_info)) {
    return nullptr;
  }
  return compile_info;
}

bool SplitDslTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD("SplitDsl", "enter SplitDsl");
  AutoTilingOp auto_tiling_op(op_type.c_str(), &op_paras, &compile_info, &run_info);
  split::Split<AutoTilingOp> split(&auto_tiling_op, nullptr);
  return split.SplitTiling();
}

bool SplitDslTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                     const OpInfo& op_info) const {
  OP_LOGD("SplitDsl", "enter SplitDsl");
  AutoTilingOp auto_tiling_op(op_type.c_str(), &op_paras, &compile_info, &run_info);
  split::Split<AutoTilingOp> split(&auto_tiling_op, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  return split.SplitTiling();
}

std::shared_ptr<AutoTilingHandler> CreateSplitDslTilingHandler(const std::string& op_type, const std::string& pattern,
                                                               const nlohmann::json& parsed_compile_info) {
  return std::make_shared<SplitDslTilingHandler>(op_type, pattern, parsed_compile_info);
}

REGISTER_AUTO_TILING(SchPattern::SPLIT, CreateSplitDslTiling, CreateSplitDslParser);
}  // namespace optiling
