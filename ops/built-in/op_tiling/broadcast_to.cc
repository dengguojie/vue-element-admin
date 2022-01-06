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
 * \file broadcast_to.cc
 * \brief
 */
#include <cctype>
#include <iostream>
#include <sstream>
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "vector_tiling.h"
#include "op_tiling_util.h"

namespace optiling {
struct BroadcastToCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
};

bool BroadcastToTiling(const std::string& op_type, const ge::Operator& op_paras,
                       const BroadcastToCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);
  std::vector<int64_t> x_runtime_shape = input_desc->MutableShape().GetDims();
  std::vector<int64_t> shape_value;

  input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);
  ge::DataType shape_dtype = input_desc->GetDataType();
  // input shape index is 1
  OP_TILING_CHECK(!ops::GetConstIntData(op_paras, 1, shape_value),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get input shape Const value error."), return false);
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  std::vector<int64_t> broadcast_shape = {};
  std::vector<int64_t> output_shape = {};

  // align shape for shape and input shapes
  int64_t len_diff = shape_value.size() - x_runtime_shape.size();
  OP_TILING_CHECK(
      (len_diff < 0),
      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "length of shape should not be less than input_x's dimension"),
      return false);
  int64_t const_value_front =
      std::accumulate(shape_value.begin(), shape_value.begin() + len_diff, 1, std::multiplies<int>());
  broadcast_shape.push_back(const_value_front);
  for (uint64_t i = 0; i < x_runtime_shape.size(); i++) {
    broadcast_shape.push_back(shape_value[len_diff + i]);
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  std::vector<std::vector<int64_t>> shapes = {x_runtime_shape, broadcast_shape};
  OpInfo eletwise_info(shapes, shape_dtype);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"), return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 BroadcastToCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  return true;
}

// register tiling interface of the BroadcastTo op.
REGISTER_OP_TILING_V3_CUSTOM(BroadcastTo, BroadcastToTiling, ParseJsonCompileInfo, BroadcastToCompileInfo);
}  // namespace optiling
