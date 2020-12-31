/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file random_ops_shape_fns.cpp
 * \brief
 */
#include "random_ops_shape_fns.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace ge {
graphStatus RandomShape(Operator& op, const std::string& shape_name, const std::string out_name) {
  Tensor tensor;
  if (op.GetInputConstData(shape_name, tensor) != GRAPH_SUCCESS) {
    std::string info = ": GetInputConstData failed.";
    OP_LOGE(op.GetName().c_str(), "%s", info.c_str());
    return GRAPH_FAILED;
  }
  Shape shape;
  if (MakeShapeFromShapeTensor(tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string info = ": MakeShapeFromShapeTensor failed.";
    OP_LOGE(op.GetName().c_str(), "%s", info.c_str());
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc(out_name);
  output_desc.SetShape(shape);
  return op.UpdateOutputDesc(out_name, output_desc);
}

graphStatus RandomShapeWithDataType(Operator& op, const std::string& shape_name, const std::string& date_type_attr_name,
                                    const std::string& out_name) {
  Tensor tensor;
  std::vector<std::string> input_infer_depends = {"shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  GeTensorDescPtr output_desc = op_desc->MutableOutputDesc(0);
  DataType type;
  if (op.GetAttr(date_type_attr_name.c_str(), type) != GRAPH_SUCCESS) {
    std::string info = ": Get dtype attr failed.";
    OP_LOGE(op_desc->GetName().c_str(), "%s", info.c_str());
    return GRAPH_FAILED;
  }
  if (op.GetInputConstData(shape_name, tensor) != GRAPH_SUCCESS) {
    output_desc->SetDataType(type);
    output_desc->SetShape(GeShape(UNKNOWN_RANK));
    output_desc->SetOriginShape(GeShape(UNKNOWN_RANK));
    return GRAPH_SUCCESS;
  }
  Shape shape;
  if (MakeShapeFromShapeTensor(tensor, shape, op_desc->GetName().c_str()) != GRAPH_SUCCESS) {
    std::string info = ": MakeShapeFromShapeTensor failed.";
    OP_LOGE(op_desc->GetName().c_str(), "%s", info.c_str());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> output_shape = shape.GetDims();
  output_desc->SetDataType(type);
  output_desc->SetShape(GeShape(output_shape));
  output_desc->SetOriginShape(GeShape(output_shape));
  return GRAPH_SUCCESS;
}
}  // namespace ge
