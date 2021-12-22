/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <iostream>
#include "error_log.h"
#include "op_log.h"
#include "op_tiling_util.h"
#include "error_log.h"
#include "vector_tiling.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
struct BiasAddCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  std::vector<int64_t> broadcast_bias_shape;
};

bool BiasAddTiling(const std::string& op_type, const ge::Operator& op_paras, const BiasAddCompileInfo& parsed_info,
                   utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  std::vector<int64_t> broadcast_bias_shape = parsed_info.broadcast_bias_shape;

  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 0 opdesc failed"), return false);
  const std::vector<int64_t> input_shape_x = input_desc->MutableShape().GetDims();
  OP_TILING_CHECK((broadcast_bias_shape.size() > input_shape_x.size()),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "shape of boardcast_bias is lager than shape of x."),
                  return false);

  for (size_t i = 0; i < broadcast_bias_shape.size(); i++) {
    broadcast_bias_shape[i] = broadcast_bias_shape[i] == -1 ? input_shape_x[i] : broadcast_bias_shape[i];
  }

  std::vector<std::vector<int64_t>> shapes = {input_shape_x, broadcast_bias_shape};
  ge::DataType type = input_desc->GetDataType();
  OpInfo eletwise_info(shapes, type);
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"), return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 BiasAddCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  // get core_num value
  OP_TILING_CHECK(!GetCompileValue(compile_info, "boardcast_bias_shape", parsed_info.broadcast_bias_shape),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get broadcast_bias_shape error"),
                  return false);
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(BiasAdd, BiasAddTiling, ParseJsonCompileInfo, BiasAddCompileInfo);
}  // namespace optiling
