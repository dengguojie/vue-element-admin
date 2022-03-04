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
constexpr std::int64_t BASE_NODE_NUMBERS = 2;
constexpr std::int64_t GENERAL_NODE_NUMBERS = 3;
constexpr std::int64_t ALL_ALIGN_COPY_EXPERIENCE = 12;
constexpr std::int64_t ONE_REPEAT_BLOCK_NUM = 8;
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

CompileInfo::CompileInfo(const nlohmann::json& compile_info) {
  is_const = compile_info.at("_is_const");
  if (is_const) {
    const_block_dims = compile_info.at("_const_dims");
  }
  core_num = compile_info.at("_core_num");
  ub_size = compile_info.at("_ub_size");
  ori_axis = compile_info.at("_ori_axis");
  only_const_tiling = compile_info.at("_only_const_tiling");
  is_avg_split = compile_info.at("_avg_split");
  if (!only_const_tiling) {
    split_is_const = compile_info.at("_split_is_const");
    split_vars = compile_info.at("_split_vars").get<std::vector<bool>>();
  }
}

bool Split::GenerateOutputShape() {
  const auto& op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  dtype = op_desc->MutableOutputDesc(0)->GetDataType();
  output_nums = op_desc->GetAllOutputsDescSize();
  is_single_output = output_nums == 1;
  V_OP_TILING_CHECK((output_nums <= MAX_INPUT_NUM),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output tensor is too much"), return false);
  V_OP_TILING_CHECK((op_desc->GetAllInputsSize() != 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor can not be empty"), return false);
  const ge::GeShape& shape = op_desc->MutableInputDesc(0)->GetShape();
  auto dim_len = static_cast<int64_t>(shape.GetDimNum());
  int64_t axis = c_info.ori_axis < 0 ? c_info.ori_axis + dim_len : c_info.ori_axis;
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
  if (c_info.is_avg_split) {
    int64_t split_output = fused_n / output_nums;
    V_OP_TILING_CHECK((fused_n % output_nums == 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and split shape not match"),
                      return false);
    output_nums = 1;
    for (int64_t i = 0; i < output_nums; i++) {
      output_shapes[i][0] = fused_m;
      output_shapes[i][1] = split_output;
    }
    min_split_shape = std::min(min_split_shape, split_output);
  } else {
    int64_t split_output = fused_n / shape.GetDim(axis);
    for (int64_t i = 0; i < output_nums; i++) {
      const ge::GeShape& out_shape = op_desc->MutableOutputDesc(i)->GetShape();
      output_shapes[i][0] = fused_m;
      int64_t split_shape = split_output * out_shape.GetDim(axis);
      output_shapes[i][1] = split_shape;
      min_split_shape = split_shape != 0 ? std::min(min_split_shape, split_shape) : min_split_shape;
    }
  }
  if (dim_len == 0) {
    input_shapes[0] = 1;
    input_shapes[1] = 1;
    output_shapes[0].fill(1);
  }
  return true;
}

bool Split::GenerateOutputShapeFromOp() {
  const auto& op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  dtype = op_info.GetInType();
  output_nums = op_desc->GetAllOutputsDescSize();
  is_single_output = output_nums == 1;
  V_OP_TILING_CHECK((output_nums <= MAX_INPUT_NUM),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "output tensor is too much"), return false);
  const std::vector<std::vector<int64_t>>& op_input_shapes = op_info.GetInputShape();
  auto dim_len = static_cast<int64_t>(op_input_shapes[0].size());
  const std::vector<std::vector<int32_t>>& axes = op_info.GetReduceAxes();
  V_OP_TILING_CHECK((!axes.empty() && !axes[0].empty()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input tensor is too much"), return false);
  int64_t ori_axis = axes[0][0] < 0 ? axes[0][0] + dim_len : axes[0][0];
  ori_axis = is_single_output ? 0 : ori_axis;
  V_OP_TILING_CHECK((ori_axis >= 0 && ori_axis < dim_len),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and compile shape not match"),
                    return false);
  int64_t fused_m = 1;
  int64_t fused_n = 1;
  for (int64_t j = 0; j < ori_axis; j++) {
    fused_m *= op_input_shapes[0][j];
  }
  for (int64_t j = ori_axis; j < dim_len; j++) {
    fused_n *= op_input_shapes[0][j];
  }
  input_shapes[0] = fused_m;
  input_shapes[1] = fused_n;
  is_empty = fused_m * fused_n == 0;
  is_base = output_nums == 1;
  if (c_info.is_avg_split) {
    int64_t split_output = fused_n / output_nums;
    V_OP_TILING_CHECK((fused_n % output_nums == 0),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and split shape not match"),
                      return false);
    output_nums = 1;
    for (int64_t i = 0; i < output_nums; i++) {
      output_shapes[i][0] = fused_m;
      output_shapes[i][1] = split_output;
    }
    min_split_shape = std::min(min_split_shape, split_output);
  } else {
    V_OP_TILING_CHECK((output_nums == static_cast<int64_t>(op_input_shapes[1].size())),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "runtime shape and split shape not match"),
                      return false);
    int64_t split_output = fused_n / op_input_shapes[0][ori_axis];
    for (int64_t i = 0; i < output_nums; i++) {
      int64_t split_dim_out = is_single_output ? op_input_shapes[0][ori_axis] : op_input_shapes[1][i];
      output_shapes[i][0] = fused_m;
      int64_t split_shape = split_output * split_dim_out;
      output_shapes[i][1] = split_shape;
      min_split_shape = split_shape != 0 ? std::min(min_split_shape, split_shape) : min_split_shape;
    }
  }
  if (dim_len == 0) {
    input_shapes[0] = 1;
    input_shapes[1] = 1;
    output_shapes[0].fill(1);
  }
  return true;
}

bool Split::IsAllAlign() {
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

bool Split::CalcSingleOutputInfo() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t bytes = BLOCK_SIZE / ele_in_block;
  is_base = true;
  need_multi_core = true;
  max_available_ub = c_info.ub_size / BLOCK_SIZE * BLOCK_SIZE / bytes;
  return true;
}

bool Split::CalcAllAlignInfo() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t bytes = BLOCK_SIZE / ele_in_block;
  max_available_ub = c_info.ub_size / ALL_ALIGN_COPY_NODE_NUMBERS / BLOCK_SIZE * BLOCK_SIZE / bytes;
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
    max_available_ub = c_info.ub_size / ALL_ALIGN_NODE_NUMBERS / BLOCK_SIZE * BLOCK_SIZE / bytes;
    V_OP_TILING_CHECK((max_available_ub != 0),
                      VECTOR_INNER_ERR_REPORT_TILIING("Split", "max_available_ub cannot be zero."), return false);
    row_limit = std::min(ROW_LIMIT_FACTOR, input_shapes[0]);
    col_limit = max_available_ub / row_limit / ele_in_block * ele_in_block;
    if (input_shapes[1] < col_limit) {
      col_limit = input_shapes[1];
      row_limit = max_available_ub / col_limit;
      if (input_shapes[0] / row_limit < c_info.core_num) {
        row_limit = std::max(ROW_LIMIT_FACTOR, (input_shapes[0] + c_info.core_num - 1) / c_info.core_num);
      }
    }
  }
  need_multi_core = true;
  return true;
}

bool Split::CalcGeneralInfo() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t bytes = BLOCK_SIZE / ele_in_block;
  max_available_ub = c_info.ub_size / GENERAL_NODE_NUMBERS / BLOCK_SIZE * BLOCK_SIZE / bytes;
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
      max_available_ub = c_info.ub_size / BASE_NODE_NUMBERS / BLOCK_SIZE * BLOCK_SIZE / bytes;
    }
  } else {
    int64_t last_align = (min_split_shape * 2 + ele_in_block - 1) / ele_in_block * ele_in_block;
    bool is_general = c_info.is_avg_split && ele_in_block == ELEMENT_IN_BLOCK_DEFAULT && col_limit >= last_align &&
                      input_shapes[0] != 1;
    if (is_general) {
      col_limit = col_limit / min_split_shape * min_split_shape;
      last_align = (col_limit + ele_in_block - 1) / ele_in_block * ele_in_block;
      row_limit = max_available_ub / last_align / row_base * row_base;
    } else {
      is_base = true;
      max_available_ub = c_info.ub_size / BASE_NODE_NUMBERS / BLOCK_SIZE * BLOCK_SIZE / bytes;
    }
  }

  need_multi_core = input_shapes[0] > row_limit || input_shapes[1] > col_limit || is_all_align || is_base;
  only_cut_m = need_multi_core && only_cut_m;
  row_limit = only_cut_m ? row_base : row_limit;
  return true;
}

bool Split::CalcTiling() {
  if (is_single_output) {
    return CalcSingleOutputInfo();
  }
  is_all_align_copy = IsAllAlign();
  if (is_all_align_copy) {
    return CalcAllAlignInfo();
  }
  return CalcGeneralInfo();
}

void Split::DoBaseBlockTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t core_num = c_info.core_num;
  if (input_shapes[0] * input_shapes[1] < 2048) {
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
    if (c_info.is_avg_split) {
      avg_block_factor = output_shapes[0][1];
    }
  } else {
    block_factor = 1;
    block_dims = input_shapes[0];
    if (c_info.is_avg_split) {
      core_num = core_num / block_dims;
      avg_block_factor = (output_shapes[0][1] + core_num - 1) / core_num;
      block_dims *= (output_shapes[0][1] + avg_block_factor - 1) / avg_block_factor;
    }
  }
}

void Split::DoBaseUbTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  for (int64_t i = 0; i < output_nums; i++) {
    int64_t output_n = output_shapes[i][1];
    output_n = output_n == 0 ? 1 : output_n;
    if (c_info.is_avg_split) {
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

void Split::DoUbTiling() {
  int64_t ele_in_block = GetElementByType(dtype);
  int64_t row_base = is_all_align || is_all_align_copy ? 1 : ROW_LIMIT_FACTOR * ele_in_block;
  low_ub_factors[0] = std::min(row_limit, ((input_shapes[0] + row_base - 1) / row_base * row_base));
  high_ub_factors[0] = std::min(col_limit, input_shapes[1]);
}

void Split::DoBlockTiling() {
  int64_t row_left = (input_shapes[0] + low_ub_factors[0] - 1) / low_ub_factors[0];
  if (row_left <= c_info.core_num / 2 && !(only_cut_m || is_all_align_copy)) {
    block_axis = 1;
    int64_t col_left = (input_shapes[1] + high_ub_factors[0] - 1) / high_ub_factors[0];
    int64_t core_left = c_info.core_num / row_left;
    block_factor = (col_left + core_left - 1) / core_left;
    block_dims = row_left * ((col_left + block_factor - 1) / block_factor);
  } else {
    block_axis = 0;
    block_factor = (row_left + c_info.core_num - 1) / c_info.core_num;
    block_dims = (row_left + block_factor - 1) / block_factor;
  }
}

void Split::CalcKey() {
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

bool Split::DoTiling() {
  bool has_op_info = !op_info.GetInputShape().empty();
  bool ret;
  if (has_op_info) {
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

void Split::WriteConstTilingData() {
  int64_t tiling_num = 0;
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
  int64_t out_num = c_info.is_avg_split ? 1 : output_nums;
  for (int64_t i = 0; i < out_num; i++) {
    tiling_data[tiling_num++] = static_cast<int32_t>(low_ub_factors[i]);
  }
  for (int64_t i = 0; i < out_num; i++) {
    tiling_data[tiling_num++] = static_cast<int32_t>(high_ub_factors[i]);
  }
  run_info.AddTilingData(reinterpret_cast<char*>(&tiling_data[0]), sizeof(int32_t) * tiling_num);
}

void Split::WriteShapeData(int64_t& tiling_num) {
  for (size_t i = 0; i < c_info.split_vars.size(); i++) {
    if (c_info.split_vars[i]) {
      tiling_data[tiling_num++] = static_cast<int32_t>(input_shapes[i]);
    }
  }
  if (c_info.is_avg_split) {
    if (!c_info.split_is_const) {
      tiling_data[tiling_num++] = static_cast<int32_t>(output_shapes[0][1]);
    }
  } else {
    for (int64_t i = 0; i < output_nums; i++) {
      tiling_data[tiling_num++] = static_cast<int32_t>(output_shapes[i][1]);
    }
  }
}

void Split::WriteCutData(int64_t& tiling_num) {
  if (is_base) {
    if (c_info.is_avg_split) {
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

bool Split::WriteTilingData() {
  OP_LOGD(op_type.c_str(), "tiling key:%ld", tiling_key);
  OP_LOGD(op_type.c_str(), "tiling block_dims:%ld", block_dims);
  OP_LOGD(op_type.c_str(), "tiling block_factor:%ld", block_factor);
  OP_LOGD(op_type.c_str(), "tiling low_ub_factor:%ld", low_ub_factors[0]);
  OP_LOGD(op_type.c_str(), "tiling high_ub_factor:%ld", high_ub_factors[0]);
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
  int64_t tiling_num = 0;
  V_OP_TILING_CHECK((c_info.split_vars.size() == SPLIT_DIM_LEN),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "split variable error"), return false);
  WriteShapeData(tiling_num);
  if (!need_multi_core) {
    run_info.AddTilingData(reinterpret_cast<char*>(&tiling_data[0]), sizeof(int32_t) * tiling_num);
    return true;
  }
  WriteCutData(tiling_num);
  run_info.AddTilingData(reinterpret_cast<char*>(&tiling_data[0]), sizeof(int32_t) * tiling_num);
  return true;
}

void Split::ProcessConst() const {
  run_info.SetTilingKey(CONST_TILING_KEY);
  run_info.SetBlockDim(static_cast<uint32_t>(c_info.const_block_dims));
}

bool Split::SplitTiling() {
  if (c_info.is_const) {
    ProcessConst();
    return true;
  }
  bool ret = DoTiling();
  ret = ret && WriteTilingData();
  return ret;
}
}  // namespace split

bool SplitDsl(const std::string& op_type, const ge::Operator& op_paras, const split::CompileInfo& compile_info,
              utils::OpRunInfo& run_info) {
  OP_LOGD(op_type.c_str(), "enter SplitDsl");
  static std::vector<std::vector<int64_t>> dummy_input_shapes{};
  static OpInfo dummy_op_info(dummy_input_shapes, ge::DT_FLOAT16, std::vector<std::vector<int32_t>>());
  split::Split split(op_type, op_paras, compile_info, dummy_op_info, run_info);
  return split.SplitTiling();
}

bool SplitDsl(const std::string& op_type, const ge::Operator& op_paras, const split::CompileInfo& compile_info,
              utils::OpRunInfo& run_info, const OpInfo& op_info) {
  OP_LOGD(op_type.c_str(), "enter SplitDsl");
  split::Split split(op_type, op_paras, compile_info, op_info, run_info);
  return split.SplitTiling();
}

bool SplitDslTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info) const {
  return SplitDsl(op_type, op_paras, compile_info, run_info);
}

bool SplitDslTilingHandler::DoTiling(const ge::Operator& op_paras, utils::OpRunInfo& run_info,
                                     const OpInfo& op_info) const {
  return SplitDsl(op_type, op_paras, compile_info, run_info, op_info);
}

std::shared_ptr<AutoTilingHandler> CreateSplitDslTilingHandler(const std::string& op_type, const std::string& pattern,
                                                               const nlohmann::json& parsed_compile_info) {
  return std::make_shared<SplitDslTilingHandler>(op_type, pattern, parsed_compile_info);
}
}  // namespace optiling
