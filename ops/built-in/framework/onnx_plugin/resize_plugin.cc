/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "onnx_common.h"

using namespace ge;
namespace domi {
Status ChangeFormatResize(const ge::Operator& op, const int idx, ge::Format format, bool is_input) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "Get op_desc from operator failed.");
    return FAILED;
  }
  if (is_input) {
    ge::GeTensorDesc org_tensor = op_desc->GetInputDesc(idx);
    org_tensor.SetOriginFormat(format);
    org_tensor.SetFormat(format);
    auto ret = op_desc->UpdateInputDesc(idx, org_tensor);
    if (ret != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE(op.GetName().c_str(), "change input format failed.");
      return FAILED;
    }
  } else {
    ge::GeTensorDesc org_tensor_y = op_desc->GetOutputDesc(idx);
    org_tensor_y.SetOriginFormat(format);
    org_tensor_y.SetFormat(format);
    auto ret_y = op_desc->UpdateOutputDesc(idx, org_tensor_y);
    if (ret_y != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE(op.GetName().c_str(), "change output format failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ParseParamsResize(const Message *op_src, Operator &op_dst) {
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  std::string coordinate_transformation_mode_value;
  std::string mode_value;
  for (auto attr : node->attribute()) {
    if (attr.name() == "coordinate_transformation_mode" &&
        attr.type() == ge::onnx::AttributeProto::STRING) {
      coordinate_transformation_mode_value = attr.s();
      op_dst.SetAttr("coordinate_transformation_mode",
                     coordinate_transformation_mode_value);
    } else if (attr.name() == "mode" &&
               attr.type() == ge::onnx::AttributeProto::STRING) {
      mode_value = attr.s();
      op_dst.SetAttr("mode", mode_value);
    }
  }
  int input_size = node->input_size();
  op_dst.SetAttr("input_size", input_size);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  if (op_desc == nullptr){
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Get op desc failed.");
    return FAILED;
  }
  op_desc->AddDynamicInputDesc("x", input_size);
  op_desc->AddDynamicOutputDesc("y", 1);
  ge::AttrUtils::SetStr(op_desc, "original_type", "ai.onnx::11::Resize");
  return SUCCESS;
}

static Status ParseOpToGraphResize(const Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto data1 = op::Data("data1").set_attr_index(1);
  auto data2 = op::Data("data2").set_attr_index(2);
  auto data3 = op::Data("data3").set_attr_index(3);
  auto resize_x = op::Identity().set_input_x(data0);
  auto resize_roi = op::Identity().set_input_x(data1);
  auto resize_scales = op::Identity().set_input_x(data2);
  auto resize_sizes = op::Identity().set_input_x(data3);

  std::string coordinate_transformation_mode_value = "pytorch_half_pixel";
  if (op.GetAttr("coordinate_transformation_mode",coordinate_transformation_mode_value) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(),"Get attr coordinate transformation mode failed, set to default.");
  }
  bool half_pixel_centers = false;
  bool align_corners = false;
  if (coordinate_transformation_mode_value == "pytorch_half_pixel") {
    half_pixel_centers = true;
  } else if (coordinate_transformation_mode_value == "align_corners"){
    align_corners = true;
  }

  int input_size = 0;
  ge::Operator sizes;
  std::string mode_value;
  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  if (op.GetAttr("input_size", input_size) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get input_size from op failed");
    return FAILED;
  }
  if (input_size == 4) {
    sizes = op::Cast().set_input_x(resize_sizes).set_attr_dst_type(ge::DT_INT32);
  } else if (input_size == 3){
    int dtype = 0;
    int sizes_type = 3;
    auto resize = op::Shape().set_input_x(resize_x);
    auto resize_cast = op::Cast().set_input_x(resize).set_attr_dst_type(dtype);
    auto mul_sizes = op::Mul().set_input_x1(resize_cast).set_input_x2(resize_scales);
    sizes = op::Cast().set_input_x(mul_sizes).set_attr_dst_type(sizes_type);
  } else {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "The input_size is error.");
    return FAILED;
  }

  int64_t offsets = 2;
  int64_t size_num = 2;
  ge::TensorDesc tensorDesc;
  std::vector<int64_t> dims = {1};
  ge::Shape shape(dims);
  tensorDesc.SetShape(shape);
  tensorDesc.SetDataType(ge::DT_INT64);

  ge::Tensor tensor_offsets(tensorDesc, reinterpret_cast<uint8_t*>(&offsets), sizeof(int64_t));
  ge::Tensor tensor_size(tensorDesc, reinterpret_cast<uint8_t*>(&size_num), sizeof(int64_t));
  auto data_offsets = op::Const("data_offsets").set_attr_value(tensor_offsets);
  auto data_size = op::Const("data_size").set_attr_value(tensor_size);

  auto size = op::Slice().set_input_x(sizes).set_input_offsets(data_offsets).set_input_size(data_size);
  inputs.push_back(size);
  auto ret_resize_x = ChangeFormatResize(resize_x, 0, ge::FORMAT_NCHW, false);
  if (ret_resize_x != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "update resize_x format failed.");
    return FAILED;
  }
  if (op.GetAttr("mode", mode_value) != GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(),"Get attr mode failed, set to default.");
    return FAILED;
  }
  if (mode_value == "nearest") {
    auto resizeout_1 = op::ResizeNearestNeighborV2().set_input_x(resize_x)
                                                   .set_input_size(size)
                                                   .set_attr_align_corners(align_corners)
                                                   .set_attr_half_pixel_centers(half_pixel_centers);
    resizeout_1.AddControlInput(resize_roi);
    if (input_size == 4) {
      resizeout_1.AddControlInput(resize_scales);
    }
    output_indexs.emplace_back(resizeout_1, vector<std::size_t>{0});
  } else if (mode_value == "linear") {
    auto resizeout_2 = op::ResizeBilinearV2().set_input_x(resize_x)
                                            .set_input_size(size)
                                            .set_attr_align_corners(align_corners)
                                            .set_attr_half_pixel_centers(half_pixel_centers);
    resizeout_2.AddControlInput(resize_roi);
    if (input_size == 4) {
      resizeout_2.AddControlInput(resize_scales);
    }
    output_indexs.emplace_back(resizeout_2, vector<std::size_t>{0});
  } else {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "Unsupported interpolation mode");
    return FAILED;
  }
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::10::Resize",
                 "ai.onnx::11::Resize",
                 "ai.onnx::12::Resize",
                 "ai.onnx::13::Resize"})
  .ParseParamsFn(ParseParamsResize)
  .ParseOpToGraphFn(ParseOpToGraphResize)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
