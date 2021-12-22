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

#include <algorithm>
#include <unordered_map>
#include "error_log.h"
#include "vector_tiling.h"
#include "op_tiling_util.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
struct FillCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
};

bool FillTiling(const std::string& op_type, const ge::Operator& op_paras, const FillCompileInfo& parsed_info,
                utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);

  auto input_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 1 opdesc failed"), return false);

  auto output_desc = operator_info->MutableOutputDesc(0);
  OP_TILING_CHECK(output_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get output 0 opdesc failed"), return false);
  const std::vector<int64_t> input_value_shape = input_desc->MutableShape().GetDims();
  const std::vector<int64_t> output_shape = output_desc->MutableShape().GetDims();
  int64_t fused_output = std::accumulate(output_shape.begin(), output_shape.end(), 1ll, std::multiplies<int64_t>());

  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  std::vector<std::vector<int64_t>> tilingshapes = {{fused_output}, input_value_shape};
  ge::DataType type = input_desc->GetDataType();
  OpInfo eletwise_info(tilingshapes, type);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"), return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 FillCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(Fill, FillTiling, ParseJsonCompileInfo, FillCompileInfo);
}  // namespace optiling
