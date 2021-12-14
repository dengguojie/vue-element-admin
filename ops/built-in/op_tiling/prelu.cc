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

#include <iostream>
#include "vector_tiling.h"
#include "error_log.h"
#include "op_log.h"
#include "op_tiling_util.h"

using namespace std;

namespace optiling {
struct PReluCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  std::vector<int64_t> broadcast_weight_shape;
};

bool PReluTiling(const std::string& op_type, const ge::Operator& op_paras, const PReluCompileInfo& parsed_info,
                 utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  OP_LOGD(op_type, "Enter PReluTiling inputs size:%llu", op_paras.GetInputsSize());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);

  const std::vector<int64_t> input_shape_x = input_desc->MutableShape().GetDims();
  ge::DataType dtype = input_desc->GetDataType();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  std::vector<int64_t> broadcast_weight_shape = parsed_info.broadcast_weight_shape;

  if (broadcast_weight_shape.size() > input_shape_x.size()) {
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "shape of weight is lager than shape of x.");
    return false;
  }

  for (size_t i = 0; i < broadcast_weight_shape.size(); i++) {
    broadcast_weight_shape[i] = broadcast_weight_shape[i] == -1 ? input_shape_x[i] : broadcast_weight_shape[i];
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  vector<vector<int64_t>> input_shapes = {input_shape_x, broadcast_weight_shape};
  OpInfo eletwise_info(input_shapes, dtype);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"), return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 PReluCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "broadcast_weight_shape", parsed_info.broadcast_weight_shape),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get broadcast_weight_shape error"),
                  return false);
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(PRelu, PReluTiling, ParseJsonCompileInfo, PReluCompileInfo);
}  // namespace optiling
