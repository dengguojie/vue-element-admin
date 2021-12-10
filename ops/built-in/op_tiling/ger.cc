/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#include <iostream>
#include "error_log.h"
#include "op_log.h"
#include "vector_tiling.h"
#include "op_tiling_util.h"

namespace optiling {

struct GerCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
};

bool GerTiling(const std::string& op_type, const ge::Operator& op_paras, const GerCompileInfo& parsed_info,
               utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);
  const std::vector<int64_t> shape_x1 = input_desc->MutableShape().GetDims();
  ge::DataType shape_dtype = input_desc->GetDataType();

  input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);
  const std::vector<int64_t> shape_x2 = input_desc->MutableShape().GetDims();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  std::vector<int64_t> shape_x1_new = shape_x1;
  std::vector<int64_t> shape_x2_new = shape_x2;

  shape_x1_new.push_back(1);
  shape_x2_new.insert(shape_x2_new.begin(), 1, 1);
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  std::vector<std::vector<int64_t>> shapes = {shape_x1_new, shape_x2_new};
  OpInfo eletwise_info(shapes, shape_dtype);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"), return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 GerCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(Ger, GerTiling, ParseJsonCompileInfo, GerCompileInfo);
}  // namespace optiling
