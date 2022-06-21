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
#include "vector_tiling.h"
#include "op_log.h"
#include "op_tiling_util.h"

namespace optiling {
struct ScaleCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  std::vector<int64_t> boardcast_scale_shape;
  bool is_unknown_rank;
};

const std::pair<int64_t, std::string> SCALE_ATTR_INFO_AXIS(0, "axis");
const std::pair<int64_t, std::string> SCALE_ATTR_INFO_NUM_AXES(1, "num_axes");
const std::pair<int64_t, std::string> SCALE_ATTR_INFO_SCALE_FROM_BLOB(2, "scale_from_blob");

bool VarifyIfScaleFromBlob(const std::string& opType, const std::vector<int64_t>& xShape,
                           const std::vector<int64_t>& scaleShape, int32_t axis, int32_t numAxes) {
  int32_t xDimNum = xShape.size();
  int32_t scaleDimNum = scaleShape.size();
  if (numAxes == -1) {
    OP_TILING_CHECK(scaleDimNum != (xDimNum - axis),
                    VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "length_scale and scale_num must be equal"),
                    return false);
    for (int32_t i = 0; i < (xDimNum - axis); i++) {
      OP_TILING_CHECK(xShape[axis + i] != scaleShape[i],
                      VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(),
                                                      "dimensions shape x and shape scale must be equal"),
                      return false);
    }
  }
  
  OP_TILING_CHECK(numAxes == 0 && (scaleDimNum != 1 || scaleShape[0] != 1),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "scale must be a scalar"), return false);
  
  if (numAxes > 0) {
    OP_TILING_CHECK((numAxes + axis) > xDimNum,
                    VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "scale shape extends x shape when applied"),
                    return false);
    OP_TILING_CHECK(scaleDimNum != numAxes,
                    VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "length_scale and num_axes must be equal"),
                    return false);
  }
  return true;
}

bool Varify(const std::string& opType, const std::vector<int64_t>& xShape,
            const std::vector<int64_t>& scaleShape, int32_t axis, int32_t numAxes) {
  int32_t xDimNum = xShape.size();
  int32_t scaleDimNum = scaleShape.size();
  if (xDimNum == 1 || scaleShape[0] == 1) {
    return true;
  }
  OP_TILING_CHECK((scaleDimNum + axis) > xDimNum,
                  VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "scale shape extends x shape when applied"),
                  return false);
  for (int32_t i = 0; i < scaleDimNum; i++) {
    OP_TILING_CHECK(xShape[axis + i] != scaleShape[i],
                    VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(),
                                                    "dimensions shape x and shape scale must be equal"),
                    return false);
  }
  return true;
}

void GenerateBroadcastShapeIfScaleFromBlob(const std::vector<int64_t>& xShape, const std::vector<int64_t>& scaleShape,
                                           int32_t axis, int32_t numAxes, std::vector<int64_t>& broadcastShape) {
  int32_t xDimNum = xShape.size();
  // [1,] * xDimNum
  if (numAxes == 0) {
    broadcastShape.insert(broadcastShape.end(), xDimNum, 1);
    return;
  }
  // [1,] * axis + scaleShape
  broadcastShape.insert(broadcastShape.end(), axis, 1);
  broadcastShape.insert(broadcastShape.end(), scaleShape.begin(), scaleShape.end()); 
  if (numAxes == -1) {
    return;
  }
  // [1,] * axis + scaleShape + [1,] * (xDimNum - numAxes - axis) --- numAxes > 0
  broadcastShape.insert(broadcastShape.end(),  xDimNum - numAxes - axis, 1);
}

void GenerateBroadcastShape(const std::vector<int64_t>& xShape, const std::vector<int64_t>& scaleShape,
                            int32_t axis, int32_t numAxes, std::vector<int64_t>& broadcastShape) {
  int32_t xDimNum = xShape.size();
  int32_t scaleDimNum = scaleShape.size();
  // [1,] * xDimNum
  if (scaleDimNum == 1 && scaleShape[0] == 1) {
    broadcastShape.insert(broadcastShape.end(), xDimNum, 1);
    return;
  }
  // [1,] * axis + scaleShape + [1,] * (xDimNum - scaleDimNum -axis)
  broadcastShape.insert(broadcastShape.end(), axis, 1);
  broadcastShape.insert(broadcastShape.end(), scaleShape.begin(), scaleShape.end()); 
  broadcastShape.insert(broadcastShape.end(), xDimNum - scaleDimNum - axis, 1);
}

bool GetBroadcastShape(const std::string& opType, const ge::Operator& opParas, const std::vector<int64_t>& xShape,
                       const std::vector<int64_t>& scaleShape, std::vector<int64_t>& broadcastShape) {
  int32_t axis;
  OP_TILING_CHECK(!ops::GetAttrValue(opParas, SCALE_ATTR_INFO_AXIS, axis, 1),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "GetScaleShape, get axis error"),
                  return false);
  int32_t numAxes;
  OP_TILING_CHECK(!ops::GetAttrValue(opParas, SCALE_ATTR_INFO_NUM_AXES, numAxes, 1),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "GetScaleShape, get num_axes error"),
                  return false);
  bool scaleFromBlob;
  OP_TILING_CHECK(!ops::GetAttrValue(opParas, SCALE_ATTR_INFO_SCALE_FROM_BLOB, scaleFromBlob, true),
                  VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "GetScaleShape, get scale_from_blob error"),
                  return false);
  int32_t xDimNum = xShape.size();
  OP_TILING_CHECK(std::abs(axis) > xDimNum, VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "axis exceed x dim"),
                  return false);
  OP_TILING_CHECK(numAxes < -1, VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "num axes should be greater than -1"),
                  return false);
  if (axis < 0) {
    axis += xDimNum;
  }
  if (scaleFromBlob) {
    OP_TILING_CHECK(!VarifyIfScaleFromBlob(opType, xShape, scaleShape, axis, numAxes),
                    VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "check broadcast shape failed."), return false);
    GenerateBroadcastShapeIfScaleFromBlob(xShape, scaleShape, axis, numAxes, broadcastShape);
  } else {
    OP_TILING_CHECK(!Varify(opType, xShape, scaleShape, axis, numAxes),
                    VECTOR_INNER_ERR_REPORT_TILIING(opType.c_str(), "check broadcast shape failed."), return false);
    GenerateBroadcastShape(xShape, scaleShape, axis, numAxes, broadcastShape);
  }
  return true;
}

bool ScaleTiling(const std::string& op_type, const ge::Operator& op_paras, const ScaleCompileInfo& parsed_info,
                 utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  OP_LOGD("op [%s] Enter SCALETILING inputs size:%d", op_type.c_str(), op_paras.GetInputsSize());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get op_info failed."),
                  return false);

  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);

  const std::vector<int64_t> input_shape_x = input_desc->MutableShape().GetDims();
  auto input_scale_desc = operator_info->MutableInputDesc(1);
  OP_TILING_CHECK(input_scale_desc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input_desc failed."),
                  return false);
  const std::vector<int64_t> input_shape_scale = input_scale_desc->MutableShape().GetDims();
  ge::DataType dtype = input_desc->GetDataType();
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  std::vector<int64_t> boardcast_scale_shape;
  if (parsed_info.is_unknown_rank) {
    OP_TILING_CHECK(!GetBroadcastShape(op_type, op_paras, input_shape_x, input_shape_scale, boardcast_scale_shape),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get scale new shape failed."),
                    return false);
  } else {
    boardcast_scale_shape = parsed_info.boardcast_scale_shape;
    for (size_t i = 0; i < boardcast_scale_shape.size(); i++) {
      // print debug
      OP_LOGD(op_type, "SCALETILING boardcast_scale_shape i=%d value=%d", i, boardcast_scale_shape[i]);
      boardcast_scale_shape[i] = boardcast_scale_shape[i] == -1 ? input_shape_x[i] : boardcast_scale_shape[i];
    }
  }

  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  vector<vector<int64_t> > input_shapes = {input_shape_x, boardcast_scale_shape};
  OpInfo eletwise_info(input_shapes, dtype);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"), return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 ScaleCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  OP_TILING_CHECK(!GetCompileValue(compile_info, "is_unknown_rank", parsed_info.is_unknown_rank, false),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get is_unknown_rank error"),
                  return false);
  if (!parsed_info.is_unknown_rank) {
    OP_TILING_CHECK(!GetCompileValue(compile_info, "_boardcast_scale_shape", parsed_info.boardcast_scale_shape),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get boardcast_scale_shape error"),
                    return false);
  }
  return true;
}

REGISTER_OP_TILING_V3_CUSTOM(Scale, ScaleTiling, ParseJsonCompileInfo, ScaleCompileInfo);
}  // namespace optiling
