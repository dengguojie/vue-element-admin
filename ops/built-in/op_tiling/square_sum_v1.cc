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


const int32_t MINUS_ONE_AXIS = 1;
const int32_t MINUS_TOW_AXIS = 2;
const int32_t MINUS_ONE_AXIS_TO_NEWAXIS = 1;
const int32_t MINUS_TOW_AXIS_TO_NEWAXIS = 2;
// define ignore_idx attr idx and name
static const std::pair<int64_t, std::string> AXIS_ATTR_INFO{0, "axis"};

struct CompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
};

bool GetNewFormatAxis(const std::string& op_type, const int32_t len_input_ori, std::vector<int32_t>& axis) {
  std::vector<int32_t>new_axis{};
  OP_TILING_CHECK(len_input_ori == 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "input ori_shape size error!"),
                  return false);

  for (int32_t axle : axis) {
    axle = axle % len_input_ori;
    if (axle == len_input_ori - MINUS_ONE_AXIS) {
      new_axis.insert(new_axis.end(), len_input_ori - MINUS_TOW_AXIS_TO_NEWAXIS);
      new_axis.insert(new_axis.end(), len_input_ori + MINUS_ONE_AXIS_TO_NEWAXIS);
    } else if (axle == len_input_ori - MINUS_TOW_AXIS) {
      new_axis.insert(new_axis.end(), len_input_ori - MINUS_ONE_AXIS_TO_NEWAXIS);
      new_axis.insert(new_axis.end(), len_input_ori);
    } else {
      new_axis.insert(new_axis.end(), axle);
    }
  }
  axis = new_axis;

  return true;
}

bool SquareSumV1Tiling(const std::string& op_type, const ge::Operator& op_paras, const CompileInfo& parsed_info,
                       utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator failed!"), return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 0 opdesc failed"),
                  return false);

  auto input_shape = input_desc->MutableShape().GetDims();
  ge::DataType type = input_desc->GetDataType();
  ge::Format format = input_desc->GetFormat();
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

  if (format == ge::FORMAT_FRACTAL_NZ) {
    int32_t len_input_ori = input_desc->GetOriginShape().GetDimNum();
    OP_TILING_CHECK(!GetNewFormatAxis(op_type, len_input_ori, axis),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get new axis failed!"),
                    return false);
  }

  vector<vector<int64_t>> input_shapes = {input_shape};
  vector<vector<int32_t>> input_axes = {axis};
  OpInfo eletwise_info(input_shapes, type, input_axes);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "tiling_handler is null!"),
                  return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 CompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_REDUCE, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler failed!"),
                  return false);
  return true;
}

// register tiling interface of the Tile op.
REGISTER_OP_TILING_V3_CUSTOM(SquareSumV1, SquareSumV1Tiling, ParseJsonCompileInfo, CompileInfo);
}  // namespace optiling
