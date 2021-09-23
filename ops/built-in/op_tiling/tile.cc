/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file tile.cc
 * \brief
 */
#include <sstream>
#include <cctype>
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "vector_tiling.h"
#include "op_tiling_util.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"
#include <iostream>

namespace optiling {
bool TileTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                utils::OpRunInfo& run_info) {
  OP_TILING_CHECK((op_info.count("compile_shape") <= 0),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "compile info not contain [compile_shape]"), return false);
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);
  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 0 opdesc failed"),
                  return false);
  std::vector<int64_t> x_runtime_shape = input_desc->MutableShape().GetDims();
  ScalarToShape(x_runtime_shape);

  std::vector<int64_t> compile_shape = op_info["compile_shape"].get<std::vector<int64_t>>();
  std::vector<int64_t> multiples_value;

  OP_TILING_CHECK(!GetConstValue(op_paras, "multiples", multiples_value),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetConstValue multiples error!"), return false);
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  OP_TILING_CHECK(x_runtime_shape.size() != compile_shape.size(),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input0 shape size must equal to compile shape size!"),
                  return false);

  // align shape for multiples and input shapes
  uint64_t last_size = multiples_value.size();
  int64_t len_diff = last_size - compile_shape.size();
  OP_TILING_CHECK(
      (len_diff < 0),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "length of multiples should not be less than input_x's dimension"),
      return false);
  if (len_diff > 0) {
    x_runtime_shape.insert(x_runtime_shape.begin(), len_diff, 1);
    compile_shape.insert(compile_shape.begin(), len_diff, 1);
  }

  std::vector<int64_t> broadcast_input(last_size * 2, 1);
  std::vector<int64_t> broadcast_multiples(last_size * 2, 1);
  int pos = 0;
  for (uint64_t i = 0; i < last_size; i++) {
    if (compile_shape[i] != 1) {
      broadcast_multiples[pos] = multiples_value[i];
      pos++;
      broadcast_input[pos] = x_runtime_shape[i];
      broadcast_multiples[pos] = x_runtime_shape[i];
      pos++;
    } else {
      broadcast_multiples[pos] = multiples_value[i];
      pos++;
    }
  }
  broadcast_input.resize(pos);
  broadcast_multiples.resize(pos);

  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  vector<vector<int64_t>> inputshapes = {broadcast_multiples, broadcast_input};
  ge::DataType type = input_desc->GetDataType();
  OpInfo eletwise_info(inputshapes, type);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  bool ret = EletwiseTiling(op_type, op_paras, op_info, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

// register tiling interface of the Tile op.
REGISTER_OP_TILING_FUNC_BUFFERED_V2(Tile, TileTiling);
}  // namespace optiling
