/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <iostream>
#include "vector_tiling.h"
#include "error_log.h"
#include "op_log.h"
#include "op_tiling_util.h"

using namespace std;

namespace optiling {

struct TileWithAxisCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  int64_t ori_axis_value;
  int64_t axis;
  int64_t tiles;
};

bool TileWithAxisTiling(const std::string& op_type, const ge::Operator& op_paras,
                        const TileWithAxisCompileInfo& parsed_info, utils::OpRunInfo& run_info) {
  std::cout << "Enter TileWithAxisTiling" << std::endl;
  PROFILING_TILING_INIT(op_type.c_str());

  int64_t ori_axis_value = parsed_info.ori_axis_value;
  int64_t axis = parsed_info.axis;
  int64_t tiles = parsed_info.tiles;
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);

  std::vector<int64_t> shape_x = input_desc->MutableShape().GetDims();
  ge::DataType dtype = input_desc->GetDataType();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();
  std::vector<int64_t> shape_y(shape_x);

  if (ori_axis_value != 1) {
    shape_x.insert(shape_x.begin() + axis, 1);
    shape_y.insert(shape_y.begin() + axis, tiles);
  } else {
    shape_y[axis] = tiles;
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  vector<vector<int64_t>> input_shapes = {shape_x, shape_y};
  OpInfo eletwise_info(input_shapes, dtype);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"),
                  return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 TileWithAxisCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "ori_axis_value", parsed_info.ori_axis_value),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get ori_axis_value error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "attr_axis", parsed_info.axis),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get attr_axis error"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "attr_tiles", parsed_info.tiles),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get ori_axis_value error"),
                  return false);
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(TileWithAxis, TileWithAxisTiling, ParseJsonCompileInfo, TileWithAxisCompileInfo);
}  // namespace optiling
