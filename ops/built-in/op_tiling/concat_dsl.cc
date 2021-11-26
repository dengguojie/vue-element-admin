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
 * \file concat_dsl.cc
 * \brief
 */

#include "concat_dsl.h"

#include <cmath>
#include <numeric>

#include "graph/utils/op_desc_utils.h"
#include "vector_tiling_log.h"
#include "vector_tiling.h"
#include "error_log.h"

namespace optiling {
namespace concat {
static constexpr std::int32_t ELEMENT_IN_BLOCK_DEFAULT = 16;
static constexpr std::int32_t ELEMENT_IN_BLOCK_B32 = 8;
static constexpr std::int32_t ELEMENT_IN_BLOCK_B8 = 32;
static constexpr std::int32_t ELEMENT_IN_BLOCK_B64 = 4;
static constexpr std::int32_t ROW_ALIGN_FACTOR = 16;
static constexpr std::int32_t COL_SPLIT_LIMIT_B8 = 96;
static constexpr std::int32_t COL_SPLIT_LIMIT = 128;
static constexpr std::int32_t ONE_REPEAT_BLOCK_NUM = 8;
static constexpr std::int32_t HALF_REPEAT_BLOCK_NUM = 4;
static constexpr std::int32_t HALF = 2;
static constexpr std::int32_t GENERAL_NODE_NUMBERS = 2;
static constexpr std::int32_t MULTI_CORE_EXPERIENCE = 24;
static constexpr std::int32_t ONE_K_BYTES = 1024;
static constexpr std::int64_t NO_MULTI_BLOCK_BASE_KEY = 0;
static constexpr std::int64_t GENERAL_BASE_KEY = 2000000;
static constexpr std::int64_t USE_ONE_CONCAT_BASE_KEY = 3000000;
static constexpr std::int64_t READ_ALIGN_BASE_KEY = 4000000;
static constexpr std::int64_t HALF_ALIGN_BASE_KEY = 5000000;
static constexpr std::int64_t ALL_ONE_NO_CUT_BASE_KEY = 6000000;
static constexpr std::int64_t GENERAL_NO_CUT_BASE_KEY = 2100000;
static constexpr std::int64_t ONE_CONCAT_BASE_KEY = 3000001;
static constexpr std::int64_t READ_ALIGN_NO_CUT_BASE_KEY = 4100000;
static constexpr std::int64_t HALF_ALIGN_NO_CUT_BASE_KEY = 5100000;
static constexpr std::size_t CONCAT_DIM_LEN = 2;
static constexpr double COL_BLOCK_ALIGN_EXPERIENCE = 0.5;

static OpInfo dummy_op_info(std::vector<std::vector<int64_t>>(), ge::DT_FLOAT16, std::vector<std::vector<int32_t>>());

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

static inline int64_t CeilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

static inline int64_t FloorAlign(int64_t a, int64_t b) {
  return a / b * b;
}

bool Concat::ParseCompileInfo() {
  try {
    c_info.core_num = compile_info.at("_core_num");
    c_info.ub_size = compile_info.at("_ub_size");
    c_info.ori_axis = compile_info.at("_ori_axis");
    c_info.only_const_tiling = compile_info.at("_only_const_tiling");
    if (!c_info.only_const_tiling) {
      c_info.concat_vars = compile_info.at("_concat_vars").get<std::vector<std::vector<bool>>>();
    }
    if (compile_info.contains("_align_vars")) {
      c_info.align_vars = compile_info.at("_align_vars").get<std::vector<size_t>>();
    }
  } catch (const std::exception& e) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get compile_info error. Error message: %s", e.what());
    return false;
  }
  return true;
}

bool Concat::GenerateOutputShape() {
  const auto& op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  dtype = op_desc->MutableOutputDesc(0)->GetDataType();
  int64_t output_m = 0;
  int64_t output_n = 0;
  input_nums = op_desc->GetAllInputsSize();
  V_OP_TILING_CHECK((input_nums <= MAX_INPUT_NUM), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor is too much"),
                    return false);
  for (int64_t i = 0; i < input_nums; i++) {
    const ge::GeShape& shape = op_desc->MutableInputDesc(i)->GetShape();
    auto dim_len = static_cast<int64_t>(shape.GetDimNum());
    V_OP_TILING_CHECK((c_info.ori_axis >= 0 && c_info.ori_axis < dim_len),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and compile shape not match"),
                      return false);
    int64_t cur_m = 1;
    int64_t cur_n = 1;
    for (int64_t j = 0; j < c_info.ori_axis; j++) {
      cur_m *= shape.GetDim(j);
    }
    for (int64_t j = c_info.ori_axis; j < dim_len; j++) {
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

bool Concat::GenerateOutputShapeFromOp() {
  dtype = op_info.GetInType();
  int64_t output_m = 0;
  int64_t output_n = 0;
  const std::vector<std::vector<int64_t>>& op_input_shapes = op_info.GetInputShape();
  input_nums = op_input_shapes.size();
  V_OP_TILING_CHECK((input_nums <= MAX_INPUT_NUM), VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor is too much"),
                    return false);
  const std::vector<std::vector<int32_t>>& axes = op_info.GetReduceAxes();
  V_OP_TILING_CHECK((!axes.empty() && !axes[0].empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor is too much"), return false);
  c_info.ori_axis = axes[0][0];
  for (int64_t i = 0; i < input_nums; i++) {
    const std::vector<int64_t>& shape = op_input_shapes[i];
    int64_t dim_len = shape.size();
    V_OP_TILING_CHECK((c_info.ori_axis >= 0 && c_info.ori_axis < dim_len),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and compile shape not match"),
                      return false);
    int64_t cur_m = 1LL;
    int64_t cur_n = 1LL;
    for (int64_t j = 0; j < c_info.ori_axis; j++) {
      cur_m *= shape[j];
    }
    for (int64_t j = c_info.ori_axis; j < dim_len; j++) {
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

bool Concat::CalcTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t bytes = BLOCK_SIZE / ele_in_block;
  if (output_shapes[0] == 1) {
    is_one_concat = true;
    need_multi_core = true;
    max_available_ub = FloorAlign((c_info.ub_size - BLOCK_SIZE * (input_nums + 1)) / bytes, ele_in_block);
  } else {
    int64_t row_limit = ROW_ALIGN_FACTOR;
    int64_t col_limit = ele_in_block == bytes ? COL_SPLIT_LIMIT_B8 : COL_SPLIT_LIMIT;
    if (output_shapes[0] > row_limit * ele_in_block || output_shapes[1] > col_limit) {
      need_multi_core = true;
    }
    int64_t coexisting_quantity = GENERAL_NODE_NUMBERS;
    max_available_ub = FloorAlign(c_info.ub_size / coexisting_quantity / bytes, ele_in_block);
  }
  return true;
}

void Concat::DoBlockTiling() {
  if (is_one_concat) {
    block_axis = 1;
    block_factor = CeilDiv(output_shapes[1], c_info.core_num);
    block_dims *= CeilDiv(output_shapes[1], block_factor);
    return;
  }
  bool must_cut_zero_axis = output_shapes[0] >= c_info.core_num || no_align || read_align_no_ub;
  if (must_cut_zero_axis) {
    block_axis = 0;
    block_factor = CeilDiv(output_shapes[0], c_info.core_num);
    block_dims = CeilDiv(output_shapes[0], block_factor);
  } else if (output_shapes[0] > (c_info.core_num / HALF)) {
    block_axis = 0;
    block_factor = 1;
    block_dims = output_shapes[0];
  } else {
    block_dims *= output_shapes[0];
  }
  if (block_axis != 0) {
    block_axis = 1;
    block_factor = CeilDiv(output_shapes[1], c_info.core_num / block_dims);
    block_dims *= CeilDiv(output_shapes[1], block_factor);
  }
}

void Concat::DoUbTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t row_limit = ROW_ALIGN_FACTOR;
  int64_t col_limit = max_available_ub / (row_limit * ele_in_block) - HALF * ele_in_block;
  int64_t one_repeat = ONE_REPEAT_BLOCK_NUM * ele_in_block;
  bool is_col_block_align =
      (static_cast<double>(col_limit) / static_cast<double>(one_repeat)) - (col_limit / one_repeat) >=
      COL_BLOCK_ALIGN_EXPERIENCE;
  if (is_col_block_align) {
    col_limit = FloorAlign(col_limit, ele_in_block);
  } else {
    col_limit = FloorAlign(col_limit, one_repeat);
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
  use_one_concat = (output_shapes[1] / factor_col >= output_shapes[0] / c_info.core_num) && ge_factor_n >= lt_factor_n;
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

void Concat::CalcInputPattern(int64_t col_limit, int64_t& ge_factor_n, int64_t& lt_factor_n) {
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

void Concat::DoGeneralUbTiling(const int64_t factor_n) {
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
  output_shapes[1] = CeilDiv(output_shapes[1], high_ub_factor);

  int64_t align_factor = all_half_align ? HALF : ele_in_block;
  int64_t factor_m = row_limit * align_factor * (factor_n / high_ub_factor);
  DoUbSplitZeroAxis(factor_m);
}

void Concat::DoNoAlignUbTiling(const int64_t factor_n) {
  no_align = true;
  all_one_concat = input_nums == output_shapes[1];
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t row_limit = ROW_ALIGN_FACTOR;
  int64_t align_factor = all_half_align ? HALF : ele_in_block;
  int64_t factor_m = row_limit * align_factor * (factor_n / output_shapes[1]);
  if (all_one_concat) {
    factor_m = max_available_ub / output_shapes[1];
    if (ele_in_block == BLOCK_SIZE) {
      factor_m = FloorAlign(factor_m, ele_in_block * ele_in_block);
    } else {
      factor_m = FloorAlign(factor_m, row_limit * ele_in_block);
    }
  }
  output_shapes[1] = 1;
  DoUbSplitZeroAxis(factor_m);
}

void Concat::DoUbSplitZeroAxis(const int64_t factor_m) {
  int64_t ele_in_block = GetElementByType(dtype);
  if (output_shapes[0] > factor_m) {
    low_ub_factor = factor_m;
  } else {
    low_ub_factor = CeilDiv(output_shapes[0], ele_in_block) * ele_in_block;
  }
  output_shapes[0] = CeilDiv(output_shapes[0], low_ub_factor);
}

void Concat::DoOneConcatUbTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t dtype_len = BLOCK_SIZE / ele_in_block;
  max_available_ub = FloorAlign((c_info.ub_size - BLOCK_SIZE * (input_nums + 1)) / dtype_len, ele_in_block);
  int64_t core_size = c_info.core_num;
  if (output_shapes[0] >= 1 && output_shapes[0] <= c_info.core_num / HALF) {
    core_size = c_info.core_num / output_shapes[0];
  } else {
    core_size = 1;
  }
  if (output_shapes[1] / max_available_ub < core_size &&
      output_shapes[1] > HALF_REPEAT_BLOCK_NUM * ele_in_block * core_size) {
    max_available_ub = CeilDiv(output_shapes[1], core_size);
    max_available_ub = CeilDiv(max_available_ub, ele_in_block) * ele_in_block;
  }
  high_ub_factor = min(max_available_ub, output_shapes[1]);
  ori_output_col = output_shapes[1];
  output_shapes[1] = CeilDiv(output_shapes[1], high_ub_factor);
}

void Concat::DoAllAlignUbTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t dtype_len = BLOCK_SIZE / ele_in_block;
  if (c_info.ori_axis == 0) {
    max_available_ub = FloorAlign(c_info.ub_size / dtype_len, ele_in_block);
  } else {
    max_available_ub = FloorAlign(c_info.ub_size / HALF / dtype_len, ele_in_block);
  }
  if (output_shapes[1] <= max_available_ub && !is_one_concat) {
    read_align_no_ub = true;
    max_available_ub /= output_shapes[1];
    int64_t base_size = MULTI_CORE_EXPERIENCE * ONE_K_BYTES / dtype_len / output_shapes[1];
    int64_t max_core = c_info.core_num;
    if (base_size > 1) {
      max_core = CeilDiv(c_info.core_num, base_size);
    }
    low_ub_factor = min(CeilDiv(output_shapes[0], max_core), max_available_ub);
    output_shapes[1] = 1;
    output_shapes[0] = CeilDiv(output_shapes[0], low_ub_factor);
  } else {
    int64_t row_factor = ROW_ALIGN_FACTOR;
    if (output_shapes[0] > row_factor) {
      low_ub_factor = row_factor;
    } else {
      low_ub_factor = output_shapes[0];
    }
    output_shapes[0] = CeilDiv(output_shapes[0], low_ub_factor);
    int64_t core_size = c_info.core_num;
    if (output_shapes[0] >= 1 && output_shapes[0] <= c_info.core_num / HALF) {
      core_size = c_info.core_num / output_shapes[0];
    } else {
      core_size = 1;
    }
    int64_t col_factor = CeilDiv(output_shapes[1], core_size);
    col_factor = CeilDiv(col_factor, ele_in_block) * ele_in_block;
    row_factor = FloorAlign(max_available_ub / low_ub_factor, ele_in_block);
    high_ub_factor = min(row_factor, col_factor);
    output_shapes[1] = (output_shapes[1] + high_ub_factor - 1) / high_ub_factor;
  }
  is_one_concat = false;
}

void Concat::CalcFactor() {
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

void Concat::CalcOffsets() {
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
        offset += CeilDiv(input_shapes[i][1], ele_in_block) * ele_in_block;
      } else {
        offset += low_ub_factor * input_shapes[i][1];
      }
    } else {
      offset = CeilDiv((cur_sum % high_ub_factor), ele_in_block) * ele_in_block * low_ub_factor;
    }
    offsets[i] = offset;
    cur_sum %= high_ub_factor;
  }
}

void Concat::CalcKey() {
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

void Concat::UpdateTiling() {
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

bool Concat::CheckZeroBlockTiling() {
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

bool Concat::CheckOneBlockTiling() {
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

void Concat::CheckAndUpdateTiling() {
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

bool Concat::DoTiling() {
  bool ret = ParseCompileInfo();
  if (has_op_info) {
    ret = ret && GenerateOutputShapeFromOp();
  } else {
    ret = ret && GenerateOutputShape();
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
        !(is_one_concat || use_one_concat || c_info.only_const_tiling || no_align || all_concat_align);
    if (is_need_calc_factor) {
      CalcFactor();
    }
    bool is_need_calc_offset = !(c_info.only_const_tiling || no_align || read_align_no_ub);
    if (is_need_calc_offset) {
      CalcOffsets();
    }
  }
  if (ret) {
    CalcKey();
  }
  return ret;
}

void Concat::WriteConstTilingData() const {
  int64_t last_align_factor = GetElementByType(dtype);
  if (all_half_align) {
    last_align_factor /= HALF;
  }
  run_info.AddTilingData(static_cast<int32_t>(need_multi_core));
  run_info.AddTilingData(static_cast<int32_t>(is_one_concat || use_one_concat));
  run_info.AddTilingData(static_cast<int32_t>(all_concat_align));
  run_info.AddTilingData(static_cast<int32_t>(last_align_factor));
  run_info.AddTilingData(static_cast<int32_t>(block_axis));
  run_info.AddTilingData(static_cast<int32_t>(block_factor));
  run_info.AddTilingData(static_cast<int32_t>(low_ub_factor));
  run_info.AddTilingData(static_cast<int32_t>(high_ub_factor));
}

bool Concat::WriteTilingData() const {
  OP_LOGD(op_type.c_str(), "tiling key:%ld", tiling_key);
  OP_LOGD(op_type.c_str(), "tiling block_dims:%ld", block_dims);
  OP_LOGD(op_type.c_str(), "tiling block_factor:%ld", block_factor);
  OP_LOGD(op_type.c_str(), "tiling low_ub_factor:%ld", low_ub_factor);
  OP_LOGD(op_type.c_str(), "tiling high_ub_factor:%ld", high_ub_factor);
  OP_LOGD(op_type.c_str(), "tiling block_axis:%ld", block_axis);

  run_info.SetBlockDim(static_cast<uint32_t>(block_dims));
  if (c_info.only_const_tiling) {
    WriteConstTilingData();
    return true;
  }
  run_info.SetTilingKey(tiling_key);
  if (is_empty) {
    return true;
  }
  for (size_t i = 0; i < c_info.concat_vars.size(); i++) {
    V_OP_TILING_CHECK((c_info.concat_vars[i].size() == CONCAT_DIM_LEN),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "concat variable error"), return false);
    if (c_info.concat_vars[i][0]) {
      run_info.AddTilingData(static_cast<int32_t>(input_shapes[i][0]));
    }
    if (c_info.concat_vars[i][1]) {
      run_info.AddTilingData(static_cast<int32_t>(input_shapes[i][1]));
    }
  }
  if (need_multi_core) {
    run_info.AddTilingData(static_cast<int32_t>(block_factor));
    bool one_concat = is_one_concat || use_one_concat;
    if (!one_concat) {
      run_info.AddTilingData(static_cast<int32_t>(low_ub_factor));
    }
    bool unnecessary_factor_offset = no_align || read_align_no_ub;
    if (unnecessary_factor_offset) {
      return true;
    }
    run_info.AddTilingData(static_cast<int32_t>(high_ub_factor));
    bool need_align_factor = !(one_concat || all_concat_align);
    if (need_align_factor) {
      for (const auto& align_var : c_info.align_vars) {
        V_OP_TILING_CHECK((align_var < MAX_INPUT_NUM),
                          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "concat factor error, input numbers too much"),
                          return false);
        run_info.AddTilingData(static_cast<int32_t>(align_factors[align_var]));
      }
    }
    for (int64_t i = 0; i < input_nums - 1; i++) {
      run_info.AddTilingData(static_cast<int32_t>(offsets[i]));
    }
  }
  return true;
}

bool Concat::ProcessConst(bool& is_const) {
  try {
    is_const = compile_info.at("_is_const");
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
  return true;
}

bool Concat::ConcatTiling() {
  bool is_const = false;
  bool ret = ProcessConst(is_const);
  if (is_const) {
    return ret;
  }
  ret = ret && DoTiling();
  ret = ret && WriteTilingData();
  return ret;
}

}  // namespace concat

bool ConcatDsl(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& compile_info,
               utils::OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "enter ConcatDsl");
  concat::Concat concat(op_type, op_paras, compile_info, concat::dummy_op_info, run_info, false);
  return concat.ConcatTiling();
}

bool ConcatDsl(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& compile_info,
               utils::OpRunInfo& run_info, const OpInfo& op_info) {
  OP_LOGD(op_type.c_str(), "enter ConcatDsl");
  concat::Concat concat(op_type, op_paras, compile_info, op_info, run_info, true);
  return concat.ConcatTiling();
}

bool ConcatDslTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  return ConcatDsl(op_type, op_paras, compile_info, run_info);
}

bool ConcatDslTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                      const OpInfo& op_info) const {
  return ConcatDsl(op_type, op_paras, compile_info, run_info, op_info);
}

std::shared_ptr<AutoTilingHandler> CreateConcatDslTilingHandler(const std::string& op_type, const std::string& pattern,
                                                                const nlohmann::json& parsed_compile_info) {
  return std::make_shared<ConcatDslTilingHandler>(op_type, pattern, parsed_compile_info);
}
}  // namespace optiling
