/* *
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file images_ops_shape_fns.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "images_ops_shape_fns.h"
#include "op_log.h"

namespace ge {
graphStatus ColorspaceShapeFn(Operator &op, const std::string output_name)
{
    Shape shape;
    graphStatus status = WithRankAtLeast(op.GetInputDesc(0), 1, shape, op.GetName().c_str());
    if (status != GRAPH_SUCCESS) {
        std::string info = ":input images must 1-D or higher rank.";
        OP_LOGE(op.GetName().c_str(),  "%s", info.c_str());
        return GRAPH_PARAM_INVALID;
    }
    int64_t dim = op.GetInputDesc(0).GetShape().GetDims().back();
    if (dim != 3) {
        std::string info =":input images last dimension must be size 3.";
        OP_LOGE(op.GetName().c_str(), "%s", info.c_str());
        return GRAPH_PARAM_INVALID;
    }
    TensorDesc desc = op.GetOutputDesc(output_name);
    desc.SetShape(Shape(shape));
    return op.UpdateOutputDesc(output_name, desc);
}

graphStatus ResizeShapeFn(Operator &op, const std::string input_name, const std::string size_input_name,
    const std::string output_name)
{
    Shape shape;
    graphStatus status = WithRank(op.GetInputDesc(0), 4, shape, op.GetName().c_str());
    if (status != GRAPH_SUCCESS) {
        std::string info =":input images must 4-D.";
        OP_LOGE(op.GetName().c_str(), "%s", info.c_str());
        return GRAPH_PARAM_INVALID;
    }
    auto dims = op.GetInputDesc(0).GetShape().GetDims();
    auto channel_dim = dims[3];
    TensorDesc input_td = op.GetInputDesc(0);
    if (input_td.GetFormat() == FORMAT_NCHW) {
        channel_dim = dims[1];
    }
    return SetOutputToSizedImage(op, dims[0], size_input_name, channel_dim, output_name);
}

graphStatus SetOutputToSizedImage(Operator &op, const int64_t batch_dim, const std::string size_input_name,
    const int64_t channel_dim, const std::string output_name)
{
    Shape size_shape;
    graphStatus status = WithRank(op.GetInputDesc(size_input_name), 1, size_shape, op.GetName().c_str());
    if (status != GRAPH_SUCCESS) {
        std::string info = ":input size must be 1-D.";
        OP_LOGE(op.GetName().c_str(), "%s", info.c_str());
        return GRAPH_PARAM_INVALID;
    }
    auto size_dims = op.GetInputDesc(size_input_name).GetShape().GetDims();
    if (size_dims[0] != 2) {
        std::string info =":input size must be 1-D tensor of 2 elements.";
        OP_LOGE(op.GetName().c_str(), "%s", info.c_str());
        return GRAPH_PARAM_INVALID;
    }
    Tensor size_tensor;
    status = op.GetInputConstData(size_input_name, size_tensor);
    if (status != GRAPH_SUCCESS) {
        std::string info =":get size tensor failed.";
        OP_LOGE(op.GetName().c_str(), "%s", info.c_str());
        return GRAPH_FAILED;
    }

    const int32_t *size_data = reinterpret_cast<const int32_t *>(size_tensor.GetData());

    int64_t size_width = static_cast<int64_t>(size_data[1]);
    int64_t size_height = static_cast<int64_t>(size_data[0]);
    TensorDesc td = op.GetOutputDesc(output_name);
    std::vector<int64_t> output_shape;

    TensorDesc input_td = op.GetInputDesc(0);
    if (input_td.GetFormat() == FORMAT_NCHW) {
        output_shape.push_back(batch_dim);
        output_shape.push_back(channel_dim);
        output_shape.push_back(size_height);
        output_shape.push_back(size_width);
    } else if (input_td.GetFormat() == FORMAT_NHWC) {
        output_shape.push_back(batch_dim);
        output_shape.push_back(size_height);
        output_shape.push_back(size_width);
        output_shape.push_back(channel_dim);
    } else {
        OP_LOGE(op.GetName().c_str(), "Not supported this format");
    }
    td.SetShape(Shape(output_shape));
    return op.UpdateOutputDesc(output_name, td);
}

graphStatus EncodeImageShapeFn(Operator &op)
{
    Shape unused_shape;
    if (WithRank(op.GetInputDesc(0), 3, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input rank must be 3 .");
        return GRAPH_FAILED;
    }

    Shape output_shape;
    (void)Scalar(output_shape);
    TensorDesc output_tensor = op.GetOutputDesc("contents");
    output_tensor.SetDataType(DT_STRING);
    output_tensor.SetShape(output_shape);
    return op.UpdateOutputDesc("contents", output_tensor);
}

template<typename T>
bool DimsAllEqualOrUnknown(std::initializer_list<T> && inputs, T unknown_dim_val)
{
  auto it = inputs.begin();
  for (; it != inputs.end() && (*it == unknown_dim_val); ++it) { }

  if (it == inputs.end()) {
    return true;
  }

  for (auto default_dim_val = *(it++); it != inputs.end(); ++it) {
    if (*it != default_dim_val && *it != unknown_dim_val) {
      return false;
    }
  }

  return true;
}

} // namespace ge