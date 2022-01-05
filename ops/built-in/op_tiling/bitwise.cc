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
 * \file bitwise.cc
 * \brief
 */
#include <sstream>
#include <cctype>
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "vector_tiling.h"
#include "op_tiling_util.h"

namespace optiling {
struct BitwizeCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
};

bool BitwiseTiling(const std::string& op_type, const ge::Operator& op_paras, const BitwizeCompileInfo& parsed_info,
                   utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);

  std::vector<int64_t> x1_broadcast_shape = input_desc->MutableShape().GetDims();
  ge::DataType shape_dtype = input_desc->GetDataType();

  input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);
  std::vector<int64_t> x2_broadcast_shape = input_desc->MutableShape().GetDims();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  int64_t len_diff = x1_broadcast_shape.size() - x2_broadcast_shape.size();
  if (len_diff > 0) {
    x2_broadcast_shape.insert(x2_broadcast_shape.begin(), len_diff, 1);
  } else if (len_diff < 0) {
    x1_broadcast_shape.insert(x1_broadcast_shape.begin(), std::abs(len_diff), 1);
  }

  // make the shape as same as dsl, the shape should add a dim (2,) when dtype from int32 to int16
  if (shape_dtype == ge::DT_INT32) {
    x1_broadcast_shape.push_back(2);
    x2_broadcast_shape.push_back(2);
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  vector<vector<int64_t>> input_shapes = {x1_broadcast_shape, x2_broadcast_shape};
  OpInfo eletwise_info(input_shapes, ge::DT_INT16);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"), return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 BitwizeCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  return true;
}

// register tiling interface of the BitwiseAnd op.
REGISTER_OP_TILING_V3_CUSTOM(BitwiseAnd, BitwiseTiling, ParseJsonCompileInfo, BitwizeCompileInfo);
REGISTER_OP_TILING_V3_CUSTOM(BitwiseOr, BitwiseTiling, ParseJsonCompileInfo, BitwizeCompileInfo);
REGISTER_OP_TILING_V3_CUSTOM(BitwiseXor, BitwiseTiling, ParseJsonCompileInfo, BitwizeCompileInfo);
}  // namespace optiling
