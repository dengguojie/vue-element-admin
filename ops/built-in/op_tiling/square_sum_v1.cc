/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file square_sum_v1.cc
 * \brief
 */
#include "../fusion_pass/common/fp16_t.hpp"
#include "error_log.h"
#include "vector_tiling.h"
#include "op_tiling_util.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"

namespace optiling {
using namespace ge;
using namespace std;

// define ignore_idx attr idx and name
static const std::pair<int64_t, std::string> AXIS_ATTR_INFO{0, "axis"};

struct CompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
};

bool SquareSumV1Tiling(const std::string& op_type, const ge::Operator& op_paras, const CompileInfo& parsed_info,
                       utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 0 opdesc failed"),
                  return false);

  auto input_shape = input_desc->MutableShape().GetDims();
  ge::DataType type = input_desc->GetDataType();
  int32_t input_shape_size = input_shape.size();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  std::vector<int32_t> axis;
  OP_TILING_CHECK(!ops::GetAttrValue(op_paras, AXIS_ATTR_INFO, axis),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get attr axis error"),
                  return false);

  if (axis.empty()) {
    for (int32_t i = 0; i < input_shape_size; i++) {
      axis.insert(axis.end(), i);
    }
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  vector<vector<int64_t>> input_shapes = {input_shape};
  vector<vector<int32_t>> input_axes = {axis};
  OpInfo eletwise_info(input_shapes, type, input_axes);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "tiling_handler is nullptr!"),
                  return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 CompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_REDUCE, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"),
                  return false);
  return true;
}

// register tiling interface of the Tile op.
REGISTER_OP_TILING_V3_CUSTOM(SquareSumV1, SquareSumV1Tiling, ParseJsonCompileInfo, CompileInfo);
}  // namespace optiling
