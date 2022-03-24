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
static const int64_t DIM_NUM_NDC1HWC0 = 6;
static const int64_t DIM_NUM_NC1HWC0 = 5;
static const int64_t NC1HWC0_DIM_INDEX_C0 = 4;
static const int64_t NDC1HWC0_DIM_INDEX_C0 = 5;
static const int64_t NDC1HWC0_DIM_INDEX_C1 = 2;

struct BiasAddCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  std::vector<int64_t> broadcast_bias_shape;
  bool is_unknown_rank;
};

static bool GetBiasShapeForUnknownRank(const std::string& op_type, const GeTensorDescPtr& input_x_desc,
                                       const GeTensorDescPtr& bias_desc, std::vector<int64_t>& broadcast_bias_shape) {
  const ge::Format x_format = input_x_desc->GetFormat();
  const GeShape& input_shape_x = input_x_desc->MutableShape();
  const GeShape& input_shape_bias = bias_desc->MutableShape();
  OP_TILING_CHECK((input_shape_bias.GetDimNum() < 1),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "the shape rank of input bias can not less 1"),
                  return false);
  const int32_t input_x_rank_num = input_shape_x.GetDimNum();
  // init the broadcast_bias_shape to all 1 shape
  broadcast_bias_shape.assign(input_x_rank_num, 1);
  switch (x_format) {
    case FORMAT_NC1HWC0:
      // when FORMAT_NC1HWC0, set broadcast_bias_shape = 1, C1, 1, 1, C0
      OP_TILING_CHECK(input_x_rank_num != DIM_NUM_NC1HWC0,
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type, "bias_add only support 5D when input is NC1HWC0"),
                      return false);
      broadcast_bias_shape[1] = input_shape_x.GetDim(1);
      broadcast_bias_shape[NC1HWC0_DIM_INDEX_C0] = input_shape_x.GetDim(NC1HWC0_DIM_INDEX_C0);
      break;
    case FORMAT_NDC1HWC0:
      // when FORMAT_NDC1HWC0, set broadcast_bias_shape = 1, 1, C1, 1, 1, C0
      OP_TILING_CHECK(
          input_x_rank_num != DIM_NUM_NDC1HWC0,
          VECTOR_INNER_ERR_REPORT_TILIING(op_type, "bias_add only support 6D when input format is NDC1HWC0"),
          return false);
      broadcast_bias_shape[NDC1HWC0_DIM_INDEX_C1] = input_shape_x.GetDim(NDC1HWC0_DIM_INDEX_C1);
      broadcast_bias_shape[NDC1HWC0_DIM_INDEX_C0] = input_shape_x.GetDim(NDC1HWC0_DIM_INDEX_C0);
      break;
    case FORMAT_NCDHW:
    case FORMAT_NCHW:
      // when FORMAT_NCDHW or FORMAT_NCHW, set broadcast_bias_shape = 1, C, 1 .....
      OP_TILING_CHECK(
          input_x_rank_num < 2,
          VECTOR_INNER_ERR_REPORT_TILIING(
              op_type, "the x shape rank must >= 2, when FORMAT_NCHW or FORMAT_NCDHW, but is %d", input_x_rank_num),
          return false);
      OP_TILING_CHECK(input_shape_x.GetDim(1) != input_shape_bias.GetDim(0),
                      VECTOR_INNER_ERR_REPORT_TILIING(
                          op_type, "data_format is NCDHW/NCHW, shape_bias must be equal to the second axis of shape_x"),
                      return false);
      broadcast_bias_shape[1] = input_shape_x.GetDim(1);
      break;
    default:
      // when FORMAT_NHWC/FORMAT_NDHWC/FORMAT_ND, set broadcast_bias_shape = 1, 1, ...., C
      OP_TILING_CHECK(input_x_rank_num < 2,
                      VECTOR_INNER_ERR_REPORT_TILIING(
                          op_type, "the x shape rank must >= 2, when FORMAT_NHWC/FORMAT_NDHWC/FORMAT_ND, but is %d",
                          input_x_rank_num),
                      return false);
      OP_TILING_CHECK((input_shape_x.GetDim(input_x_rank_num - 1) != input_shape_bias.GetDim(0)),
                      VECTOR_INNER_ERR_REPORT_TILIING(op_type,
                                                      "input format is FORMAT_NHWC/FORMAT_NDHWC/FORMAT_ND, shape_bias "
                                                      "must be equal to the last axis of shape_x"),
                      return false);
      broadcast_bias_shape[input_x_rank_num - 1] = input_shape_x.GetDim(input_x_rank_num - 1);
      break;
  }
  return true;
}

bool BiasAddTiling(const std::string& op_type, const ge::Operator& op_paras, const BiasAddCompileInfo& parsed_info,
                   utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  auto bias_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input x opdesc failed"),
                  return false);
  OP_TILING_CHECK(bias_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input bias opdesc failed"),
                  return false);

  std::vector<int64_t> broadcast_bias_shape;
  const GeShape& input_shape_x = input_desc->MutableShape();
  if (parsed_info.is_unknown_rank) {
    OP_TILING_CHECK(
        !GetBiasShapeForUnknownRank(op_type, input_desc, bias_desc, broadcast_bias_shape),
        VECTOR_INNER_ERR_REPORT_TILIING(op_type, "do GetBiasShapeForUnknownRank failed, will return false."),
        return false);
  } else {
    broadcast_bias_shape = parsed_info.broadcast_bias_shape;
    OP_TILING_CHECK((broadcast_bias_shape.size() > input_shape_x.GetDimNum()),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "shape of boardcast_bias is lager than shape of x."),
                    return false);
    for (size_t i = 0; i < broadcast_bias_shape.size(); i++) {
      broadcast_bias_shape[i] = broadcast_bias_shape[i] == -1 ? input_shape_x.GetDim(i) : broadcast_bias_shape[i];
    }
  }
  std::vector<std::vector<int64_t>> shapes = {input_shape_x.GetDims(), broadcast_bias_shape};
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
  // get parsed_info.is_unknown_rank value
  parsed_info.is_unknown_rank = false;
  GetCompileValue(compile_info, "is_unknown_rank", parsed_info.is_unknown_rank, false);
  // get core_num value
  OP_TILING_CHECK(!GetCompileValue(compile_info, "boardcast_bias_shape", parsed_info.broadcast_bias_shape),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get broadcast_bias_shape error"),
                  return false);
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(BiasAdd, BiasAddTiling, ParseJsonCompileInfo, BiasAddCompileInfo);
}  // namespace optiling
