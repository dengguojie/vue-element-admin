/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "error_log.h"
#include "op_tiling_util.h"
#include "vector_tiling.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {

bool SelectTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                  utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);
  auto input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 1 opdesc failed"), return false);

  std::vector<int64_t> x1_shape = input_desc->MutableShape().GetDims();
  ScalarToShape(x1_shape);
  input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 0 opdesc failed"), return false);

  std::vector<int64_t> condition_shape = input_desc->MutableShape().GetDims();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  if(condition_shape.size() < x1_shape.size()) condition_shape.resize(x1_shape.size(), 1);
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();
  vector<vector<int64_t>> inputshapes = {condition_shape, x1_shape};
  ge::DataType type = operator_info->MutableInputDesc(0)->GetDataType();
  OpInfo eletwise_info(inputshapes, type);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  bool ret = EletwiseTiling(op_type, op_paras, op_info, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

REGISTER_OP_TILING_FUNC_BUFFERED_V2(Select, SelectTiling);
}  // namespace optiling
