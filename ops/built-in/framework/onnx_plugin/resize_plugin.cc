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
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"
#include "image_ops.h"

using namespace ge;
namespace domi {
int  INPUT_SIZES_FOUR = 4;
int  INPUT_SIZES_THREE = 3;
int  INPUT_SIZES_TWO = 2;

Status ParseParamsResize(const Message *op_src, Operator &op_dst) {
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int input_size = node->input_size();

  std::string coordinate_transformation_mode = "half_pixel";
  float cubic_coeff_a = -0.75;
  int exclude_outside = 0;
  float extrapolation_value = 0;
  std::string mode_value = "nearest";
  std::string nearest_mode = "round_prefer_floor";
  for (auto attr : node->attribute()) {
    if (attr.name() == "coordinate_transformation_mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      coordinate_transformation_mode = attr.s();
    }
    if (attr.name() == "cubic_coeff_a" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      cubic_coeff_a = attr.f();
    }
    if (attr.name() == "exclude_outside" && attr.type() == ge::onnx::AttributeProto::INT) {
      exclude_outside = attr.i();
    }
    if (attr.name() == "extrapolation_value" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      extrapolation_value = attr.f();
    }
    if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      mode_value = attr.s();
    }
    if (attr.name() == "nearest_mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      nearest_mode = attr.s();
    }
  }
  op_dst.SetAttr("input_size", input_size);
  op_dst.SetAttr("coordinate_transformation_mode", coordinate_transformation_mode);
  op_dst.SetAttr("cubic_coeff_a", cubic_coeff_a);
  op_dst.SetAttr("exclude_outside", exclude_outside);
  op_dst.SetAttr("extrapolation_value", extrapolation_value);
  op_dst.SetAttr("mode", mode_value);
  op_dst.SetAttr("nearest_mode", nearest_mode);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  if (op_desc == nullptr) {
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

  std::vector<float> roi_vector(1, 0);
  int64_t len_roi = roi_vector.size();
  std::vector<int64_t> dims_roi = {len_roi};
  ge::Tensor roi_tensor = Vec2Tensor(roi_vector, dims_roi, ge::DT_FLOAT, ge::FORMAT_ND);
  auto const_roi_tensor = op::Const().set_attr_value(roi_tensor);

  std::string coordinate_transformation_mode = "half_pixel";
  if (op.GetAttr("coordinate_transformation_mode", coordinate_transformation_mode) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get coordinate_transformation_mode from op failed.");
    return FAILED;
  }
  float cubic_coeff_a = -0.75;
  if (op.GetAttr("cubic_coeff_a", cubic_coeff_a) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get cubic_coeff_a from op failed.");
    return FAILED;
  }
  int exclude_outside = 0;
  if (op.GetAttr("exclude_outside", exclude_outside) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get exclude_outside from op failed.");
    return FAILED;
  }
  float extrapolation_value = 0;
  if (op.GetAttr("extrapolation_value", extrapolation_value) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get extrapolation_value from op failed.");
    return FAILED;
  }
  std::string mode_value = "nearest";
  if (op.GetAttr("mode", mode_value) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get mode_value from op failed.");
    return FAILED;
  }
  std::string nearest_mode = "round_prefer_floor";
  if (op.GetAttr("nearest_mode", nearest_mode) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get nearest_mode from op failed.");
    return FAILED;
  }
  int input_size = 0;
  if (op.GetAttr("input_size", input_size) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get input_size from op failed.");
    return FAILED;
  }

  std::vector<Operator> inputs {data0};
  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  if (input_size == INPUT_SIZES_FOUR) {
    inputs.push_back(data3);

    std::vector<float> scales_vector(1, 0);
    int64_t len_scales = scales_vector.size();
    std::vector<int64_t> dims_scales = {len_scales};
    ge::Tensor scales_tensor = Vec2Tensor(scales_vector, dims_scales, ge::DT_FLOAT, ge::FORMAT_ND);
    auto const_scales_tensor = op::Const().set_attr_value(scales_tensor);

    auto resizeout4 = op::Resize().set_input_x(resize_x)
                                  .set_input_roi(const_roi_tensor)
                                  .set_input_scales(const_scales_tensor)
                                  .set_input_sizes(resize_sizes)
                                  .set_attr_coordinate_transformation_mode(coordinate_transformation_mode)
                                  .set_attr_cubic_coeff_a(cubic_coeff_a)
                                  .set_attr_exclude_outside(exclude_outside)
                                  .set_attr_extrapolation_value(extrapolation_value)
                                  .set_attr_mode(mode_value)
                                  .set_attr_nearest_mode(nearest_mode);
    resizeout4.AddControlInput(resize_roi);
    resizeout4.AddControlInput(resize_scales);
    output_indexs.emplace_back(resizeout4, vector<std::size_t> {0});
  } else if (input_size == INPUT_SIZES_THREE) {
    inputs.push_back(data2);
    auto resizeout3 = op::Resize().set_input_x(resize_x)
                                  .set_input_roi(const_roi_tensor)
                                  .set_input_scales(resize_scales)
                                  .set_attr_coordinate_transformation_mode(coordinate_transformation_mode)
                                  .set_attr_cubic_coeff_a(cubic_coeff_a)
                                  .set_attr_exclude_outside(exclude_outside)
                                  .set_attr_extrapolation_value(extrapolation_value)
                                  .set_attr_mode(mode_value)
                                  .set_attr_nearest_mode(nearest_mode);
    resizeout3.AddControlInput(resize_roi);
    output_indexs.emplace_back(resizeout3, vector<std::size_t> {0});
  } else {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "The input_size is error.");
    return FAILED;
  }
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

Status ParseParamsResizeV10(const Message *op_src, Operator &op_dst) {
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int input_size = node->input_size();

  std::string coordinate_transformation_mode = "asymmetric";
  float cubic_coeff_a = -0.75;
  int exclude_outside = 0;
  float extrapolation_value = 0;
  std::string mode_value = "nearest";
  std::string nearest_mode = "round_prefer_floor";
  for (auto attr : node->attribute()) {
    if (attr.name() == "coordinate_transformation_mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      coordinate_transformation_mode = attr.s();
    }
    if (attr.name() == "cubic_coeff_a" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      cubic_coeff_a = attr.f();
    }
    if (attr.name() == "exclude_outside" && attr.type() == ge::onnx::AttributeProto::INT) {
      exclude_outside = attr.i();
    }
    if (attr.name() == "extrapolation_value" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      extrapolation_value = attr.f();
    }
    if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      mode_value = attr.s();
    }
    if (attr.name() == "nearest_mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      nearest_mode = attr.s();
    }
  }
  op_dst.SetAttr("input_size", input_size);
  op_dst.SetAttr("coordinate_transformation_mode", coordinate_transformation_mode);
  op_dst.SetAttr("cubic_coeff_a", cubic_coeff_a);
  op_dst.SetAttr("exclude_outside", exclude_outside);
  op_dst.SetAttr("extrapolation_value", extrapolation_value);
  op_dst.SetAttr("mode", mode_value);
  op_dst.SetAttr("nearest_mode", nearest_mode);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  if (op_desc == nullptr){
    ONNX_PLUGIN_LOGE(op_dst.GetName().c_str(), "Get op desc failed.");
    return FAILED;
  }
  op_desc->AddDynamicInputDesc("x", input_size);
  op_desc->AddDynamicOutputDesc("y", 1);
  ge::AttrUtils::SetStr(op_desc, "original_type", "ai.onnx::10::Resize");
  return SUCCESS;
}

static Status ParseOpToGraphResizeV10(const Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto data1 = op::Data("data1").set_attr_index(1);
  auto input_x = op::Identity().set_input_x(data0);
  auto input_scales = op::Identity().set_input_x(data1);

  std::vector<float> roi_vector(1, 0);
  int64_t len_roi = roi_vector.size();
  std::vector<int64_t> dims_roi = {len_roi};
  ge::Tensor roi_tensor = Vec2Tensor(roi_vector, dims_roi, ge::DT_FLOAT, ge::FORMAT_ND);
  auto const_roi_tensor = op::Const().set_attr_value(roi_tensor);

  std::vector<Operator> inputs {data0, data1};

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    ONNX_PLUGIN_LOGE("Resize", "Get op desc failed.");
    return FAILED;
  }

  std::string coordinate_transformation_mode = "asymmetric";
  if (op.GetAttr("coordinate_transformation_mode", coordinate_transformation_mode) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get coordinate_transformation_mode from op failed.");
    return FAILED;
  }
  float cubic_coeff_a = -0.75;
  if (op.GetAttr("cubic_coeff_a", cubic_coeff_a) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get cubic_coeff_a from op failed.");
    return FAILED;
  }
  int exclude_outside = 0;
  if (op.GetAttr("exclude_outside", exclude_outside) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get exclude_outside from op failed.");
    return FAILED;
  }
  float extrapolation_value = 0;
  if (op.GetAttr("extrapolation_value", extrapolation_value) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get extrapolation_value from op failed.");
    return FAILED;
  }
  std::string mode_value = "nearest";
  if (op.GetAttr("mode", mode_value) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get mode_value from op failed.");
    return FAILED;
  }
  std::string nearest_mode = "round_prefer_floor";
  if (op.GetAttr("nearest_mode", nearest_mode) != SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "get nearest_mode from op failed.");
    return FAILED;
  }

  auto resizeout = op::Resize().set_input_x(input_x)
                               .set_input_roi(const_roi_tensor)
                               .set_input_scales(input_scales)
                               .set_attr_coordinate_transformation_mode(coordinate_transformation_mode)
                               .set_attr_cubic_coeff_a(cubic_coeff_a)
                               .set_attr_exclude_outside(exclude_outside)
                               .set_attr_extrapolation_value(extrapolation_value)
                               .set_attr_mode(mode_value)
                               .set_attr_nearest_mode(nearest_mode);

  std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
  output_indexs.emplace_back(resizeout, vector<std::size_t> {0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::11::Resize",
                 "ai.onnx::12::Resize",
                 "ai.onnx::13::Resize"})
  .ParseParamsFn(ParseParamsResize)
  .ParseOpToGraphFn(ParseOpToGraphResize)
  .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::10::Resize"})
  .ParseParamsFn(ParseParamsResizeV10)
  .ParseOpToGraphFn(ParseOpToGraphResizeV10)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
