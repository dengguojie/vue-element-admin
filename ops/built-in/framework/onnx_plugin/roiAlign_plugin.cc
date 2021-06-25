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

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

Status ParseParamsRoiAlign(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int output_height_value = 1;
  int output_width_value = 1;
  int sampling_ratio_value = 0;
  float spatial_scale_value = 1.0;
  
  int default_roi_end_mode_value = 0;
  op_dest.SetAttr("roi_end_mode", default_roi_end_mode_value);

  for (auto attr : node->attribute()) {
    if (attr.name() == "output_height" &&
        attr.type() == ge::onnx::AttributeProto::INT) {
      output_height_value = attr.i();
    } else if (attr.name() == "output_width") {
      output_width_value = attr.i();
    } else if (attr.name() == "sampling_ratio") {
      sampling_ratio_value = attr.i();
    } else if (attr.name() == "spatial_scale") {
      spatial_scale_value = attr.f();
    }
  }

  op_dest.SetAttr("pooled_height", output_height_value);
  op_dest.SetAttr("pooled_width", output_width_value);
  op_dest.SetAttr("sample_num", sampling_ratio_value);
  op_dest.SetAttr("spatial_scale", spatial_scale_value);
  op_dest.SetAttr("original_type", "ai.onnx::11::RoiAlign");
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", 3);
  op_desc->AddDynamicOutputDesc("y", 1);
  return SUCCESS;
}

Status ParseOpToGraphRoiAlign(const ge::Operator& op, Graph& graph) {
  int pooled_height = 1;
  op.GetAttr("pooled_height", pooled_height);
  int pooled_width = 1;
  op.GetAttr("pooled_width", pooled_width);
  int sample_num = 0;
  op.GetAttr("sample_num", sample_num);
  float spatial_scale = 1.0f;
  op.GetAttr("spatial_scale", spatial_scale);
  int roi_end_mode = 0;
  op.GetAttr("roi_end_mode", roi_end_mode);

  auto data0 = op::Data("data0").set_attr_index(0);
  auto data1 = op::Data("data1").set_attr_index(1);
  auto data2 = op::Data("data2").set_attr_index(2);

  int32_t concat_dim = 1;
  std::vector<int64_t> dims;
  ge::Shape shape(dims);
  TensorDesc tensor_desc(shape, ge::FORMAT_ND, ge::DT_INT32);
  ge::Tensor dim_tensor(tensor_desc, reinterpret_cast<uint8_t*>(&concat_dim), sizeof(int32_t));
  auto dim_const_op = op::Const("concat_dim").set_attr_value(dim_tensor);

  std::vector<int64_t> axes = {-1};
  auto unsqueeze_op = op::Unsqueeze("unsqueeze").set_input_x(data2).set_attr_axes(axes);
  auto cast_op = op::Cast("cast").set_input_x(unsqueeze_op).set_attr_dst_type(DT_FLOAT16);
  auto concat_op = op::Concat("concat").create_dynamic_input_x(2)
                                         .set_dynamic_input_x(0, cast_op)
                                         .set_dynamic_input_x(1, data1)
                                         .set_input_concat_dim(dim_const_op)
                                         .set_attr_N(2);
  
  auto roli_op = op::ROIAlign("roialign").set_input_features(data0)
                                         .set_input_rois(concat_op)
                                         .set_attr_spatial_scale(spatial_scale)
                                         .set_attr_pooled_height(pooled_height)
                                         .set_attr_pooled_width(pooled_width)
                                         .set_attr_sample_num(sample_num)
                                         .set_attr_roi_end_mode(roi_end_mode);
  std::vector<ge::Operator> inputs = {data0, data1, data2};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(roli_op, std::vector<size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}
// register ROIAlign op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::RoiAlign",
                 "ai.onnx::9::RoiAlign",
                 "ai.onnx::10::RoiAlign",
                 "ai.onnx::11::RoiAlign",
                 "ai.onnx::12::RoiAlign",
                 "ai.onnx::13::RoiAlign"})
  .ParseParamsFn(ParseParamsRoiAlign)
  .ParseOpToGraphFn(ParseOpToGraphRoiAlign)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
