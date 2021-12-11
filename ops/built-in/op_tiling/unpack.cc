/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file unpack.cpp
 * \brief
 */
#include <algorithm>
#include <nlohmann/json.hpp>
#include <string>

#include "../op_proto/util/op_common_util.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "op_tiling.h"
#include "op_tiling_util.h"

namespace optiling {
constexpr int32_t kBlockSize{32};
constexpr int32_t kCalcMemSize{1024};
constexpr int32_t RightIndex{2};
enum TilingStrategy { SINGLE_OUTPUT = 0, SMALL_SHAPE, BIG_SHAPE, LESS_32B, LAST_DIM_SMALL };

// compile info
struct CompileInfo {
  int32_t axis;
  int32_t output_num;
  int32_t ub_size;
  int32_t core_num;
  int32_t dtype_size;
  bool is_special_tiling;
};

struct opInfo {
  int32_t axis;
  int32_t ub_size;
  int32_t output_num;
  int32_t core_num;
  bool is_special_tiling;
  std::map<std::string, std::vector<std::string>> vars;
};

bool UnpackParseFunc(const std::string& op_type, const nlohmann::json& compile_info, opInfo& compile_value) {
  using namespace nlohmann;
  OP_TILING_CHECK(compile_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "op_compile_info_json is null"),
                  return false);
  auto compile_vars = compile_info["compile_vars"];

  OP_TILING_CHECK(!GetCompileValue(compile_vars, "axis", compile_value.axis),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UnpackParseFunc get axis error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_vars, "ub_size", compile_value.ub_size),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UnpackParseFunc get ub_size error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_vars, "output_num", compile_value.output_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UnpackParseFunc get ksize error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_vars, "core_num", compile_value.core_num),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UnpackParseFunc get core_num error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_vars, "is_special_tiling", compile_value.is_special_tiling),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UnpackParseFunc get is_special_tiling error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "vars", compile_value.vars),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "UnpackParseFunc get vars error"), return false);
  OP_LOGI(op_type.c_str(), "GetCompileParams success.");
  return true;
}

const std::unordered_map<ge::DataType, int32_t> kDtypeSizeMap{
    {ge::DT_INT8, 1},   {ge::DT_UINT8, 1},   {ge::DT_BOOL, 1},  {ge::DT_INT64, 8},
    {ge::DT_UINT64, 8}, {ge::DT_FLOAT16, 2}, {ge::DT_INT16, 2}, {ge::DT_UINT16, 2},
    {ge::DT_FLOAT, 4},  {ge::DT_INT32, 4},   {ge::DT_UINT32, 4}};

void SetActualDimsInVars(const std::vector<int64_t>& input_shape, int32_t axis,
                         std::unordered_map<std::string, int32_t>& var_names) {
  int32_t left_dim = 1;
  int32_t right_dim = 1;
  int32_t input_dims = input_shape.size();

  for (int32_t i = 0; i < axis; i++) {
    left_dim *= input_shape[i];
  }
  for (int32_t i = axis + 1; i < input_dims; i++) {
    right_dim *= input_shape[i];
  }
  var_names["left_dim"] = left_dim;
  var_names["right_dim"] = right_dim;

  return;
}

bool CheckOpParams(const ge::Operator& op_paras, int32_t input_dims, int32_t& axis) {
  // check axis
  if (axis < -input_dims || axis >= input_dims) {
    VECTOR_INNER_ERR_REPORT_TILIING("Dynamic UnpackTiling", "Check axis is invalid.");
    return false;
  }

  axis = (axis >= 0) ? axis : axis + input_dims;
  return true;
}

void GetCompileParams(const opInfo& op_info, CompileInfo& compile_info) {
  compile_info.axis = op_info.axis;
  compile_info.ub_size = op_info.ub_size;
  compile_info.output_num = op_info.output_num;
  compile_info.core_num = op_info.core_num;
  compile_info.is_special_tiling = op_info.is_special_tiling;
}

void GetSingleOutputTilingParams(const std::vector<int64_t>& output_reshape, const CompileInfo& compile_info,
                                 std::unordered_map<std::string, int32_t>& var_names, TilingStrategy& key,
                                 int32_t& actual_block_num) {
  int64_t calc_size =
      std::accumulate(output_reshape.begin(), output_reshape.end(), (int64_t)1, std::multiplies<int64_t>());
  int32_t split_factor = 1;
  int32_t right_dim_in = 1;
  int32_t ub_limit = ((compile_info.ub_size / kBlockSize) * kBlockSize) / compile_info.dtype_size;
  actual_block_num = std::ceil(calc_size * compile_info.dtype_size * 1.0 / kCalcMemSize);
  if (actual_block_num >= compile_info.core_num) {
    actual_block_num = compile_info.core_num;
  }
  right_dim_in = std::ceil(calc_size * 1.0 / actual_block_num);
  if (right_dim_in <= ub_limit) {
    split_factor = right_dim_in;
  } else {
    split_factor = ub_limit;
  }
  key = SINGLE_OUTPUT;
  var_names["right_dim_in"] = right_dim_in;
  var_names["split_factor"] = split_factor;
  var_names["left_dim_out"] = actual_block_num;
}

int32_t GetSplitFactor(const int32_t ub_limit, const int64_t ori_last_ele, const int32_t ele_per_block) {
  int32_t split_factor = 1;
  for (int j = ub_limit; j > 0; j--) {
    if (ori_last_ele % j >= ele_per_block || ori_last_ele % j == 0) {
      split_factor = j;
      break;
    }
  }
  return split_factor;
}

void GetUbTiling(std::vector<int64_t>& output_reshape, const int32_t ub_limit, const int32_t dtype_size,
                 int32_t& split_axis, int32_t& split_factor) {
  int32_t ele_per_block = kBlockSize / dtype_size;
  int64_t ori_last_ele = output_reshape[RightIndex];

  if (output_reshape[RightIndex] % ele_per_block != 0) {
    output_reshape[RightIndex] = std::ceil(output_reshape[RightIndex] * 1.0 / ele_per_block) * ele_per_block;
  }
  int32_t len = static_cast<int32_t>(output_reshape.size()) - 1;
  for (int i = 0; i < len; i++) {
    int64_t ele_cnt =
        std::accumulate(output_reshape.begin() + i, output_reshape.end(), (int64_t)1, std::multiplies<int64_t>());
    if (ele_cnt <= ub_limit) {
      split_axis = i - 1;
      split_factor = ub_limit / ele_cnt;
      break;
    } else if (i == len - 1) {
      split_axis = i;
      if (ori_last_ele % ub_limit > 0 && ori_last_ele % ub_limit < ele_per_block) {
        split_factor = GetSplitFactor(ub_limit, ori_last_ele, ele_per_block);
      } else {
        split_factor = ub_limit;
      }
      break;
    }
  }
  if (split_axis < 0) {
    split_axis = 0;
    split_factor = output_reshape[0];
  }
}

void GetLeftDimOutForRightDimLess32B(int32_t left_dim, int32_t right_dim, int32_t dtype_size, int32_t& new_left_dim,
                                     int32_t& left_dim_out) {
  if (left_dim * right_dim * dtype_size >= kBlockSize) {
    while (right_dim * new_left_dim * dtype_size < kBlockSize) {
      left_dim_out--;
      if (left_dim_out == 0) {
        left_dim_out = 1;
      }
      new_left_dim = std::ceil(left_dim * 1.0 / left_dim_out);
      if (left_dim_out == 1) {
        break;
      }
    }
  } else {
    left_dim_out = 1;
    new_left_dim = left_dim;
  }
  return;
}

void GetMultiOutputTilingParams(const std::vector<int64_t>& output_reshape, const CompileInfo& compile_info,
                                int32_t& actual_block_num, TilingStrategy& key,
                                std::unordered_map<std::string, int32_t>& var_names) {
  int32_t left_dim = static_cast<int32_t>(output_reshape[0]);
  int32_t right_dim = static_cast<int32_t>(output_reshape[RightIndex]);
  int32_t core_num = compile_info.core_num;
  int32_t left_dim_out = 1;
  int32_t right_dim_in = 1;
  int32_t split_factor = 1;
  int32_t split_axis = 1;
  vector<int64_t> new_shape(output_reshape);
  int32_t ele_per_block = kBlockSize / compile_info.dtype_size;
  int32_t ub_limit = ((compile_info.ub_size / kBlockSize) * kBlockSize) / compile_info.dtype_size;

  if (left_dim <= core_num && right_dim * left_dim > kCalcMemSize / compile_info.dtype_size * core_num) {
    int32_t right_dim_out = core_num / left_dim;
    right_dim_in = std::ceil(right_dim * 1.0 / right_dim_out);
    left_dim_out = left_dim;
    if (right_dim % ele_per_block == 0) {
      while (right_dim_in % ele_per_block > 0 && right_dim_out > 1) {
        right_dim_out--;
        right_dim_in = std::ceil(right_dim * 1.0 / right_dim_out);
      }
    }
    actual_block_num = left_dim_out * right_dim_out;
    new_shape[0] = 1;
    new_shape[RightIndex] = right_dim_in;
  } else {
    left_dim_out = (left_dim > core_num) ? core_num : left_dim;
    right_dim_in = right_dim;
    int32_t new_left_dim = std::ceil(left_dim * 1.0 / left_dim_out);
    GetLeftDimOutForRightDimLess32B(left_dim, right_dim, compile_info.dtype_size, new_left_dim, left_dim_out);
    actual_block_num = left_dim_out;
    new_shape[0] = new_left_dim;
  }
  if (compile_info.is_special_tiling) {
    new_shape[1] = compile_info.output_num;
  }

  GetUbTiling(new_shape, ub_limit, compile_info.dtype_size, split_axis, split_factor);
  if (split_axis == 0 && right_dim_in * split_factor * compile_info.dtype_size < kBlockSize) {
    var_names["left_dim_out"] = 1;
    var_names["right_dim_in"] = right_dim_in;
    var_names["split_factor"] = split_factor;
    key = compile_info.is_special_tiling ? LAST_DIM_SMALL : LESS_32B;
    return;
  }
  key = (split_axis == 0) ? SMALL_SHAPE : BIG_SHAPE;
  if (compile_info.is_special_tiling) {
    key = LAST_DIM_SMALL;
  }
  var_names["right_dim_in"] = right_dim_in;
  var_names["left_dim_out"] = left_dim_out;
  var_names["split_factor"] = split_factor;
}

bool CheckRet(CompileInfo& compile_info, const ge::Operator& op_paras, const opInfo& op_info,
              const int32_t input_dims) {
  if (kDtypeSizeMap.count(op_paras.GetInputDesc(0).GetDataType()) == 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("UnpackTiling", "Invalid input dtype.");
    return false;
  }
  compile_info.dtype_size = kDtypeSizeMap.at(op_paras.GetInputDesc(0).GetDataType());
  GetCompileParams(op_info, compile_info);

  // check op input and output params
  bool ret = CheckOpParams(op_paras, input_dims, compile_info.axis);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING("UnpackTiling", "Check op params failed.");
    return ret;
  }
  return true;
}

/*
 * @brief: tiling function of op
 * @param [in] opType: opType of the op
 * @param [in] opParas: inputs/outputs/atts of the op
 * @param [in] op_info: compile time generated info of the op
 * @param [out] runInfo: result data
 * @return bool: success or not
 */
bool UnpackTiling(const std::string& op_type, const ge::Operator& op_paras, const opInfo& op_info,
                  utils::OpRunInfo& run_info) {
  GELOGI("op[%s] UnpackTiling running.", op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  std::vector<int64_t> input_shape = operator_info->MutableInputDesc(0)->MutableShape().GetDims();

  int32_t input_dims = input_shape.size();
  std::unordered_map<std::string, int32_t> var_names;
  CompileInfo compile_info;
  bool ret = CheckRet(compile_info, op_paras, op_info, input_dims);
  if (!ret) {
    return ret;
  }

  SetActualDimsInVars(input_shape, compile_info.axis, var_names);

  // turn the input shape into three dimensions [leftDim, inputAxis, rightDim]
  int64_t left_dim = 1;
  int64_t right_dim = 1;
  for (int32_t dim = 0; dim < compile_info.axis; dim++) {
    left_dim *= input_shape[dim];
  }
  for (int32_t dim = compile_info.axis + 1; dim < input_dims; dim++) {
    right_dim *= input_shape[dim];
  }
  std::vector<int64_t> output_reshape = {left_dim, 1, right_dim};

  bool is_special_tiling = compile_info.is_special_tiling;
  if (is_special_tiling) {
    int32_t out_num = compile_info.output_num;
    // each output requires 64B for reg buf
    compile_info.ub_size = compile_info.ub_size - 64 * out_num;
    compile_info.ub_size = compile_info.ub_size / (out_num + 1) * out_num;
  } else {
    compile_info.ub_size = compile_info.ub_size - 192 * compile_info.output_num;
  }

  // calc tiling para
  int32_t actual_block_num = 1;
  TilingStrategy key = SINGLE_OUTPUT;
  if (compile_info.output_num > 1) {
    GetMultiOutputTilingParams(output_reshape, compile_info, actual_block_num, key, var_names);
  } else {
    GetSingleOutputTilingParams(output_reshape, compile_info, var_names, key, actual_block_num);
  }

  run_info.SetBlockDim(actual_block_num);

  run_info.SetTilingKey(key);
  std::map<std::string, std::vector<std::string>> vars = op_info.vars;
  std::vector<std::string> all_vars = vars[std::to_string(key)];

  for (const auto& var : all_vars) {
    if (var_names.count(var) == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("UnpackTiling", "Get Compile var info failed!");
      return false;
    }
    run_info.AddTilingData(var_names[var]);
  }
  GELOGI("Unpack op Tiling end!!");
  return true;
}
// register tiling interface of the Unpack op.
REGISTER_OP_TILING_V3_CUSTOM(Unpack, UnpackTiling, UnpackParseFunc, opInfo);
}  // namespace optiling
