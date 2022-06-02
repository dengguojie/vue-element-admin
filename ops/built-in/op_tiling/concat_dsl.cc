/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file concat_dsl.cc
 * \brief
 */

#include "concat_dsl.h"

#include <cmath>
#include <numeric>

#include "graph/utils/op_desc_utils.h"
#include "vector_tiling.h"
#include "tiling_handler.h"
#include "auto_tiling_register.h"
#include "vector_tiling_rt2.h"

namespace optiling {
namespace concat {
namespace {
constexpr std::int32_t ELEMENT_IN_BLOCK_DEFAULT = 16;
constexpr std::int32_t ELEMENT_IN_BLOCK_B32 = 8;
constexpr std::int32_t ELEMENT_IN_BLOCK_B8 = 32;
constexpr std::int32_t ELEMENT_IN_BLOCK_B64 = 4;
constexpr std::int32_t ROW_ALIGN_FACTOR = 16;
constexpr std::int32_t COL_SPLIT_LIMIT_B8 = 96;
constexpr std::int32_t COL_SPLIT_LIMIT = 128;
constexpr std::int32_t ONE_REPEAT_BLOCK_NUM = 8;
constexpr std::int32_t HALF_REPEAT_BLOCK_NUM = 4;
constexpr std::int32_t HALF = 2;
constexpr std::int32_t GENERAL_NODE_NUMBERS = 2;
constexpr std::int32_t MULTI_CORE_EXPERIENCE = 24;
constexpr std::int32_t ONE_K_BYTES = 1024;
constexpr std::int64_t NO_MULTI_BLOCK_BASE_KEY = 0;
constexpr std::int64_t GENERAL_BASE_KEY = 2000000;
constexpr std::int64_t USE_ONE_CONCAT_BASE_KEY = 3000000;
constexpr std::int64_t READ_ALIGN_BASE_KEY = 4000000;
constexpr std::int64_t HALF_ALIGN_BASE_KEY = 5000000;
constexpr std::int64_t ALL_ONE_NO_CUT_BASE_KEY = 6000000;
constexpr std::int64_t GENERAL_NO_CUT_BASE_KEY = 2100000;
constexpr std::int64_t ONE_CONCAT_BASE_KEY = 3000001;
constexpr std::int64_t READ_ALIGN_NO_CUT_BASE_KEY = 4100000;
constexpr std::int64_t HALF_ALIGN_NO_CUT_BASE_KEY = 5100000;
constexpr std::size_t CONCAT_DIM_LEN = 2;
constexpr double COL_BLOCK_ALIGN_EXPERIENCE = 0.5;
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

ConcatCompileInfo::ConcatCompileInfo(const nlohmann::json& json_compile_info) {
  Parse("ConcatDsl", json_compile_info);
}

bool ConcatCompileInfo::Parse(const char* op_type, const nlohmann::json& json_compile_info) {
  try {
    is_const = json_compile_info.at("_is_const");
    if (is_const) {
      const_block_dims = json_compile_info.at("_const_dims");
    }
    core_num = json_compile_info.at("_core_num");
    ub_size = json_compile_info.at("_ub_size");
    ori_axis = json_compile_info.at("_ori_axis");
    only_const_tiling = json_compile_info.at("_only_const_tiling");
    if (!only_const_tiling) {
      concat_vars = json_compile_info.at("_concat_vars").get<std::vector<std::vector<bool>>>();
    }
    if (json_compile_info.contains("_align_vars")) {
      align_vars = json_compile_info.at("_align_vars").get<std::vector<size_t>>();
    }
  } catch (...) {
    OP_LOGE(op_type, "Unknown Exception encountered when parsing Compile Info of op_type %s", op_type);
    return false;
  }
  return true;
}

template <typename T>
bool Concat<T>::GenerateOutputShape() {
  V_OP_TILING_CHECK(context->GetOutputDataType(0, dtype),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get inpute dtype error"),
                    return false);
  int64_t output_m = 0;
  int64_t output_n = 0;
  input_nums = context->GetInputNums();
  V_OP_TILING_CHECK((input_nums <= MAX_INPUT_NUM), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor is too much"),
                    return false);
  int64_t axis = input_nums == 1 ? 0 : c_info->ori_axis;
  for (int64_t i = 0; i < input_nums; i++) {
    const OpShape& shape = context->GetInputShape(i);
    V_OP_TILING_CHECK((!shape.Empty()), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input shape error"),
                      return false);
    auto dim_len = static_cast<int64_t>(shape.GetDimNum());
    if (i == 0) {
      axis = axis < 0 ? axis + dim_len : axis;
      is_concat_zero = axis == 0;
    }
    V_OP_TILING_CHECK((axis >= 0 && axis < dim_len),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and compile shape not match"),
                      return false);
    int64_t cur_m = 1;
    int64_t cur_n = 1;
    for (int64_t j = 0; j < axis; j++) {
      cur_m *= shape.GetDim(j);
    }
    for (int64_t j = axis; j < dim_len; j++) {
      cur_n *= shape.GetDim(j);
    }
    if (i == 0) {
      output_m = cur_m;
    } else {
      V_OP_TILING_CHECK((cur_m == output_m),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "non-concat axis input shape is inconsistent"),
                        return false);
    }
    output_n += cur_n;
    input_shapes[i][0] = cur_m;
    input_shapes[i][1] = cur_n;
  }
  output_shapes[0] = output_m;
  output_shapes[1] = output_n;
  is_empty = output_m * output_n == 0;
  return true;
}

template <typename T>
bool Concat<T>::GenerateOutputShapeFromOp() {
  V_OP_TILING_CHECK(context->GetInputDataType(op_info, dtype),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get inpute dtype error"),
                    return false);
  int64_t output_m = 0;
  int64_t output_n = 0;
  input_nums = context->GetInputNums(op_info);
  OP_LOGD(op_type, "Concat input number is %lld:", input_nums);
  V_OP_TILING_CHECK((input_nums <= MAX_INPUT_NUM), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor is too much"),
                    return false);
  const std::vector<int64_t>* axes = op_info->GetAxes();
  V_OP_TILING_CHECK((axes != nullptr && !axes->empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "concat axis is empty"), return false);
  int64_t axis = input_nums == 1 ? 0 : axes->at(0);
  OP_LOGD(op_type, "Concat axis is %lld:", axis);
  const auto inputs = op_info->GetInputShape();
  V_OP_TILING_CHECK((inputs != nullptr),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input shape is empty"), return false);
  for (size_t i = 0; i < inputs->size(); i++) {
    const std::vector<int64_t>& shape = inputs->at(i);
    auto dim_len = static_cast<int64_t>(shape.size());
    if (i == 0) {
      axis = axis < 0 ? axis + dim_len : axis;
      is_concat_zero = axis == 0;
    }
    V_OP_TILING_CHECK((axis >= 0 && axis < dim_len),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and compile shape not match"),
                      return false);
    int64_t cur_m = 1LL;
    int64_t cur_n = 1LL;
    for (int64_t j = 0; j < axis; j++) {
      cur_m *= shape[j];
    }
    for (int64_t j = axis; j < dim_len; j++) {
      cur_n *= shape[j];
    }
    if (i == 0) {
      output_m = cur_m;
    } else {
      V_OP_TILING_CHECK((cur_m == output_m),
                        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "non-concat axis input shape is inconsistent"),
                        return false);
    }
    output_n += cur_n;
    input_shapes[i][0] = cur_m;
    input_shapes[i][1] = cur_n;
  }
  output_shapes[0] = output_m;
  output_shapes[1] = output_n;
  is_empty = output_m * output_n == 0;
  return true;
}

template <typename T>
bool Concat<T>::CalcTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t bytes = BLOCK_SIZE_BYTES / ele_in_block;
  if (output_shapes[0] == 1) {
    is_one_concat = true;
    need_multi_core = true;
    max_available_ub = (c_info->ub_size - BLOCK_SIZE_BYTES * (input_nums + 1)) / bytes / ele_in_block * ele_in_block;
  } else {
    int64_t row_limit = ROW_ALIGN_FACTOR;
    int64_t col_limit = ele_in_block == bytes ? COL_SPLIT_LIMIT_B8 : COL_SPLIT_LIMIT;
    if (output_shapes[0] > row_limit * ele_in_block || output_shapes[1] > col_limit) {
      need_multi_core = true;
    }
    int64_t coexisting_quantity = GENERAL_NODE_NUMBERS;
    max_available_ub = c_info->ub_size / coexisting_quantity / bytes / ele_in_block * ele_in_block;
  }
  return true;
}

template <typename T>
void Concat<T>::DoBlockTiling() {
  if (is_one_concat) {
    block_axis = 1;
    block_factor = (output_shapes[1] + c_info->core_num - 1) / c_info->core_num;
    block_dims *= (output_shapes[1] + block_factor - 1) / block_factor;
    return;
  }
  bool must_cut_zero_axis = output_shapes[0] >= c_info->core_num || no_align || read_align_no_ub;
  if (must_cut_zero_axis) {
    block_axis = 0;
    block_factor = (output_shapes[0] + c_info->core_num - 1) / c_info->core_num;
    block_dims = (output_shapes[0] + block_factor - 1) / block_factor;
  } else if (output_shapes[0] > (c_info->core_num / HALF)) {
    block_axis = 0;
    block_factor = 1;
    block_dims = output_shapes[0];
  } else {
    block_dims *= output_shapes[0];
  }
  if (block_axis != 0) {
    block_axis = 1;
    int64_t left_core = c_info->core_num / block_dims;
    block_factor = (output_shapes[1] + left_core - 1) / left_core;
    block_dims *= (output_shapes[1] + block_factor - 1) / block_factor;
  }
}

template <typename T>
void Concat<T>::DoUbTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t row_limit = ROW_ALIGN_FACTOR;
  int64_t col_limit = max_available_ub / (row_limit * ele_in_block) - HALF * ele_in_block;
  int64_t one_repeat = ONE_REPEAT_BLOCK_NUM * ele_in_block;
  bool is_col_block_align =
      (static_cast<double>(col_limit) / static_cast<double>(one_repeat)) - (col_limit / one_repeat) >=
      COL_BLOCK_ALIGN_EXPERIENCE;
  if (is_col_block_align) {
    col_limit = col_limit / ele_in_block * ele_in_block;
  } else {
    col_limit = col_limit / one_repeat * one_repeat;
  }
  int64_t ge_factor_n;
  int64_t lt_factor_n;
  CalcInputPattern(col_limit, ge_factor_n, lt_factor_n);
  if (all_concat_align) {
    DoAllAlignUbTiling();
    return;
  }
  factor_col = col_limit;
  real_factor_n = max_available_ub / (row_limit * ele_in_block);
  if (all_half_align) {
    factor_col *= (ele_in_block / HALF);
    real_factor_n = max_available_ub / (row_limit * HALF);
  }
  use_one_concat = (output_shapes[1] / factor_col >= output_shapes[0] / c_info->core_num) && ge_factor_n >= lt_factor_n;
  if (is_one_concat || use_one_concat) {
    all_half_align = false;
    DoOneConcatUbTiling();
    return;
  }
  if (output_shapes[1] <= real_factor_n) {
    DoNoAlignUbTiling(real_factor_n);
  } else {
    DoGeneralUbTiling(factor_col);
  }
}

template <typename T>
void Concat<T>::CalcInputPattern(int64_t col_limit, int64_t& ge_factor_n, int64_t& lt_factor_n) {
  ge_factor_n = 0;
  lt_factor_n = 0;
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t half_factor = ele_in_block / HALF;
  all_concat_align = true;
  all_half_align = true;
  for (int64_t i = 0; i < input_nums; i++) {
    if (input_shapes[i][1] >= col_limit) {
      ge_factor_n++;
    } else {
      lt_factor_n++;
    }
    if (input_shapes[i][1] % ele_in_block != 0) {
      all_concat_align = false;
    }
    if (input_shapes[i][1] % half_factor != 0) {
      all_half_align = false;
    }
  }
}

template <typename T>
void Concat<T>::DoGeneralUbTiling(int64_t factor_n) {
  V_OP_TILING_CHECK((factor_n != 0), VECTOR_INNER_ERR_REPORT_TILIING("Concat", "factor_n cannot be zero."), return);
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t row_limit = ROW_ALIGN_FACTOR;
  bool is_ub_factor_less_block = output_shapes[1] % factor_n != 0 && output_shapes[1] % factor_n < ele_in_block;
  if (is_ub_factor_less_block) {
    int64_t base_factor = factor_n - ele_in_block;
    bool is_find_factor = false;
    while (base_factor > factor_n / HALF) {
      is_find_factor = output_shapes[1] % base_factor == 0 || output_shapes[1] % base_factor >= ele_in_block;
      if (is_find_factor) {
        high_ub_factor = base_factor;
        break;
      }
      base_factor -= ele_in_block;
    }
    if (!is_find_factor) {
      base_factor = factor_n - 1;
      while (base_factor > ele_in_block) {
        is_find_factor = output_shapes[1] % base_factor == 0 || output_shapes[1] % base_factor >= ele_in_block;
        if (is_find_factor) {
          high_ub_factor = base_factor;
          break;
        }
        base_factor--;
      }
    }
  } else {
    high_ub_factor = output_shapes[1] > factor_n ? factor_n : output_shapes[1];
  }
  output_shapes[1] = (output_shapes[1] + high_ub_factor - 1) / high_ub_factor;

  int64_t align_factor = all_half_align ? HALF : ele_in_block;
  int64_t factor_m = row_limit * align_factor * (factor_n / high_ub_factor);
  DoUbSplitZeroAxis(factor_m);
}

template <typename T>
void Concat<T>::DoNoAlignUbTiling(int64_t factor_n) {
  no_align = true;
  all_one_concat = input_nums == output_shapes[1];
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t row_limit = ROW_ALIGN_FACTOR;
  int64_t align_factor = all_half_align ? HALF : ele_in_block;
  int64_t factor_m = row_limit * align_factor * (factor_n / output_shapes[1]);
  if (all_one_concat) {
    factor_m = max_available_ub / output_shapes[1];
    if (ele_in_block == BLOCK_SIZE_BYTES) {
      factor_m = factor_m / (ele_in_block * ele_in_block) * (ele_in_block * ele_in_block);
    } else {
      factor_m = factor_m / (row_limit * ele_in_block) * (row_limit * ele_in_block);
    }
  }
  output_shapes[1] = 1;
  DoUbSplitZeroAxis(factor_m);
}

template <typename T>
void Concat<T>::DoUbSplitZeroAxis(int64_t factor_m) {
  int64_t ele_in_block = GetElementByType(dtype);
  if (output_shapes[0] > factor_m) {
    low_ub_factor = factor_m;
  } else {
    low_ub_factor = (output_shapes[0] + ele_in_block - 1) / ele_in_block * ele_in_block;
  }
  output_shapes[0] = (output_shapes[0] + low_ub_factor - 1) / low_ub_factor;
}

template <typename T>
void Concat<T>::DoOneConcatUbTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t dtype_len = BLOCK_SIZE_BYTES / ele_in_block;
  max_available_ub = (c_info->ub_size - BLOCK_SIZE_BYTES * (input_nums + 1)) / dtype_len / ele_in_block * ele_in_block;
  int64_t core_size = 1;
  if (output_shapes[0] >= 1 && output_shapes[0] <= c_info->core_num / HALF) {
    core_size = c_info->core_num / output_shapes[0];
  }
  V_OP_TILING_CHECK((max_available_ub != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING("Concat", "max_available_ub cannot be zero."), return);
  if (output_shapes[1] / max_available_ub < core_size &&
      output_shapes[1] > HALF_REPEAT_BLOCK_NUM * ele_in_block * core_size) {
    max_available_ub = (output_shapes[1] + core_size - 1) / core_size;
    max_available_ub = (max_available_ub + ele_in_block - 1) / ele_in_block * ele_in_block;
  }
  high_ub_factor = min(max_available_ub, output_shapes[1]);
  ori_output_col = output_shapes[1];
  output_shapes[1] = (output_shapes[1] + high_ub_factor - 1) / high_ub_factor;
}

template <typename T>
void Concat<T>::DoAllAlignUbTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t dtype_len = BLOCK_SIZE_BYTES / ele_in_block;
  if (is_concat_zero) {
    max_available_ub = c_info->ub_size / dtype_len / ele_in_block * ele_in_block;
  } else {
    max_available_ub = c_info->ub_size / HALF / dtype_len / ele_in_block * ele_in_block;
  }
  if (output_shapes[1] <= max_available_ub && !is_one_concat) {
    read_align_no_ub = true;
    max_available_ub /= output_shapes[1];
    int64_t base_size = MULTI_CORE_EXPERIENCE * ONE_K_BYTES / dtype_len / output_shapes[1];
    int64_t max_core = c_info->core_num;
    if (base_size > 1) {
      max_core = (c_info->core_num + base_size - 1) / base_size;
    }
    low_ub_factor = min((output_shapes[0] + max_core - 1) / max_core, max_available_ub);
    output_shapes[1] = 1;
    output_shapes[0] = (output_shapes[0] + low_ub_factor - 1) / low_ub_factor;
  } else {
    int64_t row_factor = ROW_ALIGN_FACTOR;
    if (output_shapes[0] > row_factor) {
      low_ub_factor = row_factor;
    } else {
      low_ub_factor = output_shapes[0];
    }
    output_shapes[0] = (output_shapes[0] + low_ub_factor - 1) / low_ub_factor;
    int64_t core_size = 1;
    if (output_shapes[0] >= 1 && output_shapes[0] <= c_info->core_num / HALF) {
      core_size = c_info->core_num / output_shapes[0];
    }
    int64_t col_factor = (output_shapes[1] + core_size - 1) / core_size;
    col_factor = (col_factor + ele_in_block - 1) / ele_in_block * ele_in_block;
    row_factor = max_available_ub / low_ub_factor / ele_in_block * ele_in_block;
    high_ub_factor = min(row_factor, col_factor);
    output_shapes[1] = (output_shapes[1] + high_ub_factor - 1) / high_ub_factor;
  }
  is_one_concat = false;
}

template <typename T>
void Concat<T>::CalcFactor() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t cur_sum = 0;
  for (int64_t i = 0; i < input_nums - 1; i++) {
    cur_sum += input_shapes[i][1];
    if (cur_sum <= high_ub_factor) {
      align_factors[i] = 1;
    } else {
      align_factors[i] = ele_in_block;
    }
    cur_sum %= high_ub_factor;
  }
  cur_sum += input_shapes[input_nums - 1][1];
  bool is_last_input_no_align =
      (cur_sum == input_shapes[input_nums - 1][1] && cur_sum % ele_in_block != 0) || cur_sum > high_ub_factor;
  if (is_last_input_no_align) {
    align_factors[input_nums - 1] = ele_in_block;
  } else {
    align_factors[input_nums - 1] = 1;
  }
}

template <typename T>
void Concat<T>::CalcOffsets() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t cur_sum = 0;
  int64_t offset = 0;
  bool one_concat = is_one_concat || use_one_concat;
  if (one_concat) {
    low_ub_factor = 1;
  }
  for (int64_t i = 0; i < input_nums; i++) {
    cur_sum += input_shapes[i][1];
    if (cur_sum < high_ub_factor) {
      if (one_concat) {
        offset += (input_shapes[i][1] + ele_in_block - 1) / ele_in_block * ele_in_block;
      } else {
        offset += low_ub_factor * input_shapes[i][1];
      }
    } else {
      offset = ((cur_sum % high_ub_factor) + ele_in_block - 1) / ele_in_block * ele_in_block * low_ub_factor;
    }
    offsets[i] = offset;
    cur_sum %= high_ub_factor;
  }
}

template <typename T>
void Concat<T>::CalcKey() {
  if (is_one_concat) {
    tiling_key = ONE_CONCAT_BASE_KEY;
    return;
  }
  if (no_align) {
    tiling_key = GENERAL_NO_CUT_BASE_KEY;
    if (all_half_align) {
      tiling_key = HALF_ALIGN_NO_CUT_BASE_KEY;
    }
    if (all_one_concat) {
      tiling_key = ALL_ONE_NO_CUT_BASE_KEY;
    }
    return;
  }
  if (read_align_no_ub) {
    tiling_key = READ_ALIGN_NO_CUT_BASE_KEY;
  } else if (all_concat_align) {
    tiling_key = READ_ALIGN_BASE_KEY;
  } else if (use_one_concat) {
    tiling_key = USE_ONE_CONCAT_BASE_KEY;
  } else if (all_half_align) {
    tiling_key = HALF_ALIGN_BASE_KEY;
  } else {
    tiling_key = GENERAL_BASE_KEY;
  }
  if (need_multi_core) {
    tiling_key = tiling_key + block_axis;
  } else {
    tiling_key = NO_MULTI_BLOCK_BASE_KEY;
  }
}

template <typename T>
void Concat<T>::UpdateTiling() {
  if (is_one_concat) {
    block_dims = 1;
    block_axis = 1;
    block_factor = output_shapes[1];
  } else {
    use_one_concat = false;
    output_shapes[1] = ori_output_col;
    if (output_shapes[1] <= real_factor_n) {
      DoNoAlignUbTiling(real_factor_n);
    } else {
      DoGeneralUbTiling(factor_col);
    }
    block_axis = -1;
    block_factor = -1;
    block_dims = 1;
    DoBlockTiling();
  }
}

template <typename T>
bool Concat<T>::CheckZeroBlockTiling() const {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t cur_output = 0;
  for (int64_t i = input_nums - 1; i >= 0; i--) {
    if (cur_output < ele_in_block && input_shapes[i][1] < ele_in_block) {
      return true;
    }
    cur_output += input_shapes[i][1];
    if (cur_output > ele_in_block) {
      return false;
    }
  }
  return false;
}

template <typename T>
bool Concat<T>::CheckOneBlockTiling() const {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t block_inner_size = high_ub_factor * block_factor;
  int64_t cur_output = 0;
  for (int64_t i = 0; i < input_nums; i++) {
    bool is_last_one_concat = is_one_concat && i == input_nums - 1;
    if (is_last_one_concat) {
      continue;
    }
    cur_output += input_shapes[i][1];
    if (cur_output > block_inner_size) {
      int64_t left_size = cur_output % block_inner_size;
      bool is_overlap = left_size < ele_in_block && left_size != 0;
      if (is_overlap) {
        return true;
      }
      cur_output = left_size;
    } else {
      int64_t left_size = block_inner_size - cur_output;
      if (i == input_nums - 1) {
        left_size = 0;
      }
      if (left_size < ele_in_block && input_shapes[i][1] < ele_in_block) {
        return true;
      }
    }
  }
  return false;
}

template <typename T>
void Concat<T>::CheckAndUpdateTiling() {
  if (block_dims <= 1 || !(is_one_concat || use_one_concat)) {
    return;
  }
  bool is_overlap = false;
  if (block_axis == 0) {
    is_overlap = CheckZeroBlockTiling();
  } else {
    is_overlap = CheckOneBlockTiling();
  }
  if (is_overlap) {
    UpdateTiling();
  }
}

template <typename T>
bool Concat<T>::DoTiling() {
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
    DoUbTiling();
    DoBlockTiling();
    CheckAndUpdateTiling();
    bool is_need_calc_factor =
        !(is_one_concat || use_one_concat || c_info->only_const_tiling || no_align || all_concat_align);
    if (is_need_calc_factor) {
      CalcFactor();
    }
    bool is_need_calc_offset = !(c_info->only_const_tiling || no_align || read_align_no_ub);
    if (is_need_calc_offset) {
      CalcOffsets();
    }
  }
  if (ret) {
    CalcKey();
  }
  return ret;
}

template <typename T>
bool Concat<T>::WriteConstTilingData() {
  int64_t last_align_factor = GetElementByType(dtype);
  if (all_half_align) {
    last_align_factor /= HALF;
  }
  size_t tiling_num = 0;
  tiling_data[tiling_num++] = static_cast<int32_t>(need_multi_core);
  tiling_data[tiling_num++] = static_cast<int32_t>(is_one_concat || use_one_concat);
  tiling_data[tiling_num++] = static_cast<int32_t>(all_concat_align);
  tiling_data[tiling_num++] = static_cast<int32_t>(last_align_factor);
  tiling_data[tiling_num++] = static_cast<int32_t>(block_axis);
  tiling_data[tiling_num++] = static_cast<int32_t>(block_factor);
  tiling_data[tiling_num++] = static_cast<int32_t>(low_ub_factor);
  tiling_data[tiling_num++] = static_cast<int32_t>(high_ub_factor);

  for (size_t i = 0; i < tiling_num; i++) {
    if (!context->Append(tiling_data[i])) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool Concat<T>::WriteVar(size_t& tiling_num) {
  for (size_t i = 0; i < c_info->concat_vars.size(); i++) {
    V_OP_TILING_CHECK((c_info->concat_vars[i].size() == CONCAT_DIM_LEN),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "concat variable error"), return false);
    if (c_info->concat_vars[i][0]) {
      tiling_data[tiling_num++] = static_cast<int32_t>(input_shapes[i][0]);
    }
    if (c_info->concat_vars[i][1]) {
      tiling_data[tiling_num++] = static_cast<int32_t>(input_shapes[i][1]);
    }
  }
  return true;
}

template <typename T>
bool Concat<T>::WriteData(const size_t tiling_num) {
  for (size_t i = 0; i < tiling_num; i++) {
    if (!context->Append(tiling_data[i])) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool Concat<T>::WriteTilingData() {
  OP_LOGD(op_type.c_str(), "tiling key:%ld", tiling_key);
  OP_LOGD(op_type.c_str(), "tiling block_dims:%ld", block_dims);
  OP_LOGD(op_type.c_str(), "tiling block_factor:%ld", block_factor);
  OP_LOGD(op_type.c_str(), "tiling low_ub_factor:%ld", low_ub_factor);
  OP_LOGD(op_type.c_str(), "tiling high_ub_factor:%ld", high_ub_factor);
  OP_LOGD(op_type.c_str(), "tiling block_axis:%ld", block_axis);

  context->SetBlockDim(static_cast<uint32_t>(block_dims));
  if (c_info->only_const_tiling) {
    return WriteConstTilingData();
  }
  context->SetTilingKey(tiling_key);
  if (is_empty) {
    return true;
  }
  size_t tiling_num = 0;
  if (!WriteVar(tiling_num)) {
    return false;
  }
  if (need_multi_core) {
    tiling_data[tiling_num++] = static_cast<int32_t>(block_factor);
    bool one_concat = is_one_concat || use_one_concat;
    if (!one_concat) {
      tiling_data[tiling_num++] = static_cast<int32_t>(low_ub_factor);
    }
    bool unnecessary_factor_offset = no_align || read_align_no_ub;
    if (unnecessary_factor_offset) {
      return WriteData(tiling_num);
    }
    tiling_data[tiling_num++] = static_cast<int32_t>(high_ub_factor);
    bool need_align_factor = !(one_concat || all_concat_align);
    if (need_align_factor) {
      for (const auto& align_var : c_info->align_vars) {
        V_OP_TILING_CHECK((align_var < MAX_INPUT_NUM),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "concat factor error, input numbers too much"),
                          return false);
        tiling_data[tiling_num++] = static_cast<int32_t>(align_factors[align_var]);
      }
    }
    for (int64_t i = 0; i < input_nums - 1; i++) {
      tiling_data[tiling_num++] = static_cast<int32_t>(offsets[i]);
    }
  }
  return WriteData(tiling_num);
}

template <typename T>
void Concat<T>::ProcessConst() const {
  context->SetTilingKey(CONST_TILING_KEY);
  context->SetBlockDim(static_cast<uint32_t>(c_info->const_block_dims));
}

template <typename T>
bool Concat<T>::ConcatTiling() {
  op_type = context->GetOpType();
  c_info = dynamic_cast<const ConcatCompileInfo *>(context->GetCompileInfo());
  if (c_info->is_const) {
    ProcessConst();
    return true;
  }
  bool ret = DoTiling();
  ret = ret && WriteTilingData();
  return ret;
}
}  // namespace concat

bool CreateConcatDslTiling(gert::TilingContext* context, const OpInfoImpl* op_info) {
  OP_LOGD("ConcatDsl", "enter ConcatDsl");
  AutoTilingContext auto_tiling_context(context);
  if (op_info) {
    auto_tiling_context.SetCompileInfo(op_info->GetCompileInfo());
  }
  concat::Concat<AutoTilingContext> concat(&auto_tiling_context, op_info);
  return concat.ConcatTiling();
}

AutoTilingCompileInfo* CreateConcatDslParser(const char* op_type, const nlohmann::json& json_compile_info) {
  auto compile_info = new concat::ConcatCompileInfo();
  if (!compile_info->Parse(op_type, json_compile_info)) {
    return nullptr;
  }
  return compile_info;
}

bool ConcatDslTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  OP_LOGD("ConcatDsl", "enter ConcatDsl");
  AutoTilingOp auto_tiling_op(op_type.c_str(), &op_paras, &compile_info, &run_info);
  concat::Concat<AutoTilingOp> concat(&auto_tiling_op, nullptr);
  return concat.ConcatTiling();
}

bool ConcatDslTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                      const OpInfo& op_info) const {
  OP_LOGD("ConcatDsl", "enter ConcatDsl for OpInfo");
  AutoTilingOp auto_tiling_op(op_type.c_str(), &op_paras, &compile_info, &run_info);
  concat::Concat<AutoTilingOp> concat(&auto_tiling_op, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  return concat.ConcatTiling();
}

std::shared_ptr<AutoTilingHandler> CreateConcatDslTilingHandler(const std::string& op_type, const std::string& pattern,
                                                                const nlohmann::json& parsed_compile_info) {
  return std::make_shared<ConcatDslTilingHandler>(op_type, pattern, parsed_compile_info);
}

REGISTER_AUTO_TILING(SchPattern::CONCAT, CreateConcatDslTiling, CreateConcatDslParser)
}  // namespace optiling
