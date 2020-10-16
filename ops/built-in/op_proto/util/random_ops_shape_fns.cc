/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file random_ops_shape_fns.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include "random_ops_shape_fns.h"
#include "op_log.h"

namespace ge {
graphStatus RandomShape(Operator &op, const std::string &shape_name,
                        const std::string out_name) {
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

graphStatus RandomShapeWithDataType(Operator &op,
                                    const std::string &shape_name,
                                    const std::string &date_type_attr_name,
                                    const std::string &out_name) {
  Tensor tensor;
  if (op.GetInputConstData(shape_name, tensor) != GRAPH_SUCCESS) {
    std::string info =  ": GetInputConstData failed.";
    OP_LOGE(op.GetName().c_str(),"%s", info.c_str());
    return GRAPH_FAILED;
  }
  Shape shape;
  if (MakeShapeFromShapeTensor(tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string info = ": MakeShapeFromShapeTensor failed.";
    OP_LOGE(op.GetName().c_str(), "%s", info.c_str());
    return GRAPH_FAILED;
  }
  Operator::OpType type;
  if (op.GetAttr(date_type_attr_name, type) != GRAPH_SUCCESS) {
    std::string info =  ": get dtype attr failed.";
    OP_LOGE(op.GetName().c_str(), "%s", info.c_str());
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc(out_name);
  output_desc.SetDataType(type);
  output_desc.SetShape(shape);
  return op.UpdateOutputDesc(out_name, output_desc);
}
}