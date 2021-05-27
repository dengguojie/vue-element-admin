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
#include <string>
#include <vector>
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph.h"
#include "all_ops.h"

using namespace ge;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;
Status ChangeFormatInt8Conv(const ge::Operator& op, const int idx, ge::Format format, bool is_input) {
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    OP_LOGE("Int8Conv", "Get op_desc from operator failed.");
    return FAILED;
  }
  if (is_input) {
    ge::GeTensorDesc org_tensor = op_desc->GetInputDesc(idx);
    org_tensor.SetOriginFormat(format);
    org_tensor.SetFormat(format);
    auto ret = op_desc->UpdateInputDesc(idx, org_tensor);
    if (ret != ge::GRAPH_SUCCESS) {
      OP_LOGE("Int8Conv", "change input format failed.");
      return FAILED;
    }
  } else {
    ge::GeTensorDesc org_tensor_y = op_desc->GetOutputDesc(idx);
    org_tensor_y.SetOriginFormat(format);
    org_tensor_y.SetFormat(format);
    auto ret_y = op_desc->UpdateOutputDesc(idx, org_tensor_y);
    if (ret_y != ge::GRAPH_SUCCESS) {
      OP_LOGE("Int8Conv", "change output format failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ParseParamsInt8Conv(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("Int8Conv", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (op_desc == nullptr) {
    OP_LOGE("Int8Conv", "Get op_desc from operator failed.");
    return FAILED;
  }
  op_desc->AddDynamicInputDesc("x", 3);
  op_desc->AddDynamicOutputDesc("output", 1);
  ge::AttrUtils::SetStr(op_desc, "original_type", "ai.onnx::11::Int8Conv");

  int groups = 1;
  int64_t Y_zero_point = 0;
  float Y_scale = 0.0;
  std::string order = "NCHW";
  std::vector<float> scale = {};
  std::vector<int> strides = {};
  std::vector<int> pads = {};
  std::vector<int> dilations = {};

  for (const auto& attr : node->attribute()) {
    if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        strides.push_back(attr.ints(i));
      }
    } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        pads.push_back(attr.ints(i));
      }
    } else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        dilations.push_back(attr.ints(i));
      }
    } else if (attr.name() == "groups" && attr.type() == ge::onnx::AttributeProto::INT) {
      groups = attr.i();
    } else if (attr.name() == "Y_scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      Y_scale = attr.f();
    } else if (attr.name() == "Y_zero_point" && attr.type() == ge::onnx::AttributeProto::INT) {
      Y_zero_point = attr.i();
    } else if (attr.name().find("scale") != std::string::npos) {
      scale.push_back(attr.f());
    } else if (attr.name() == "order" && attr.type() == ge::onnx::AttributeProto::STRING) {
      order = attr.s();
    }
  }

  int temp_num = 1;
  if (order == "NCHW") {
    strides.insert(strides.begin(), 2, temp_num);
    dilations.insert(dilations.begin(), 2, temp_num);
  } else if (order == "NHWC") {
    strides.insert(strides.begin(), temp_num);
    strides.insert(strides.end(), temp_num);
    dilations.insert(dilations.begin(), temp_num);
    dilations.insert(dilations.end(), temp_num);
  } else {
    OP_LOGW("Int8Conv", "get attr order foramt is failed.");
  }

  op_dest.SetAttr("strides", strides);
  op_dest.SetAttr("pads", pads);
  op_dest.SetAttr("dilations", dilations);
  op_dest.SetAttr("groups", groups);
  op_dest.SetAttr("Y_scale", Y_scale);
  op_dest.SetAttr("Y_zero_point", Y_zero_point);
  op_dest.SetAttr("scale", scale[2]);
  op_dest.SetAttr("order", order);
  return SUCCESS;
}

static Status ParseOpToGraphInt8Conv(const ge::Operator& op, Graph& graph) {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto data1 = op::Data("data1").set_attr_index(1);
  auto data2 = op::Data("data2").set_attr_index(2);

  std::vector<int64_t> strides = {};
  if (op.GetAttr("strides", strides) != SUCCESS) {
    OP_LOGE("Int8Conv", "get strides from op failed");
    return FAILED;
  }
  std::vector<int64_t> pads = {};
  if (op.GetAttr("pads", pads) != SUCCESS) {
    OP_LOGE("Int8Conv", "get pads from op failed");
    return FAILED;
  }
  std::vector<int64_t> dilations = {};
  if (op.GetAttr("dilations", dilations) != SUCCESS) {
    OP_LOGE("Int8Conv", "get dilations from op failed");
    return FAILED;
  }
  int groups = 1;
  if (op.GetAttr("groups", groups) != SUCCESS) {
    OP_LOGE("Int8Conv", "get groups from op failed");
    return FAILED;
  }
  float Y_scale = 0.0;
  if (op.GetAttr("Y_scale", Y_scale) != SUCCESS) {
    OP_LOGE("Int8Conv", "get Y_scale from op failed");
    return FAILED;
  }
  int64_t Y_zero_point = 0;
  if (op.GetAttr("Y_zero_point", Y_zero_point) != SUCCESS) {
    OP_LOGE("Int8Conv", "get Y_zero_point from op failed");
    return FAILED;
  }
  float scale = 0.0;
  if (op.GetAttr("scale", scale) != SUCCESS) {
    OP_LOGE("Int8Conv", "get scale from op failed");
    return FAILED;
  }
  std::string order = "";
  if (op.GetAttr("order", order) != SUCCESS) {
    OP_LOGE("Int8Conv", "get order from op failed");
    return FAILED;
  }

  ge::Format filter_format = ge::FORMAT_NCHW;
  if (order == "NHWC") {
    filter_format = ge::FORMAT_HWCN;
  }
  std::map<std::string, ge::Format> kvmap = {{"NCHW", ge::FORMAT_NCHW}, {"NHWC", ge::FORMAT_NHWC}};
  auto order_map = kvmap.find(order);
  if (order_map == kvmap.end()){
    OP_LOGE("Int8Conv", "only support NCHW/NHWC, but got %s", order);
    return FAILED;
  }

  auto conv = op::Conv2D().set_input_x(data0)
                          .set_input_filter(data1)
                          .set_input_bias(data2)
                          .set_attr_strides(strides)
                          .set_attr_pads(pads)
                          .set_attr_dilations(dilations)
                          .set_attr_groups(groups)
                          .set_attr_data_format(order);

  // The x should be NCHW
  auto ret_x = ChangeFormatInt8Conv(conv, 0, order_map->second, true);
  if (ret_x != ge::GRAPH_SUCCESS) {
    OP_LOGE("Int8Conv", "update x format failed.");
    return FAILED;
  }
  // The filter should be NCHW/HWCN
  auto ret_w = ChangeFormatInt8Conv(conv, 1, filter_format, true);
  if (ret_w != ge::GRAPH_SUCCESS) {
    OP_LOGE("Int8Conv", "update filter format failed.");
    return FAILED;
  }
  // The output should be NCHW
  auto ret_y = ChangeFormatInt8Conv(conv, 0, order_map->second, false);
  if (ret_y != ge::GRAPH_SUCCESS) {
    OP_LOGE("Int8Conv", "update output format failed.");
    return FAILED;
  }

  ge::TensorDesc tensorDesc;
  std::vector<int64_t> dims = {1};
  ge::Shape shape(dims);
  tensorDesc.SetShape(shape);
  tensorDesc.SetDataType(DT_FLOAT16);
  ge::Tensor tensor_scale(tensorDesc, reinterpret_cast<uint8_t*>(&scale), sizeof(DT_FLOAT16));
  auto data_deq = op::Const("data_deq").set_attr_value(tensor_scale);
  auto ascendD = op::AscendDequant().set_input_x(conv)
                                    .set_input_deq_scale(data_deq)
                                    .set_attr_dtype(1);
  //The deq_data should be NC1HWC0
  auto ret_deq_x = ChangeFormatInt8Conv(ascendD, 0, order_map->second, true);
  if (ret_deq_x != ge::GRAPH_SUCCESS) {
    OP_LOGE("Intconv8", "update intput x format failed");
    return FAILED;
  }
  //The deq_data should be NC1HWC0
  auto ret_deq = ChangeFormatInt8Conv(ascendD, 1, order_map->second, true);
  if (ret_deq != ge::GRAPH_SUCCESS) {
    OP_LOGE("Intconv8", "update intput deq format failed");
    return FAILED;
  }
  //The deq_data should be NC1HWC0
  auto ret_deq_y = ChangeFormatInt8Conv(ascendD, 0, order_map->second, false);
  if (ret_deq_y != ge::GRAPH_SUCCESS) {
    OP_LOGE("Intconv8", "update output y format failed");
    return FAILED;
  }

  float offset = Y_zero_point;
  auto ascendQ = op::AscendQuant().set_input_x(ascendD)
                                  .set_attr_scale(Y_scale)
                                  .set_attr_offset(offset);
  //The deq_data should be NCHW
  auto ret_quant_x = ChangeFormatInt8Conv(ascendQ, 0, order_map->second, true);
  if (ret_quant_x != ge::GRAPH_SUCCESS) {
    OP_LOGE("Intconv8", "update intput x format failed");
    return FAILED;
  }
  //The deq_data should be NC1HWC0
  auto ret_quant_y = ChangeFormatInt8Conv(ascendQ, 0, ge::FORMAT_NC1HWC0, false);
  if (ret_quant_y != ge::GRAPH_SUCCESS) {
    OP_LOGE("Intconv8", "update output y format failed");
    return FAILED;
  }

  std::vector<ge::Operator> inputs{data0, data1, data2};
  std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
  outputs.emplace_back(ascendQ, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register Int8Conv op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Int8Conv",
                   "ai.onnx::9::Int8Conv",
                   "ai.onnx::10::Int8Conv",
                   "ai.onnx::11::Int8Conv",
                   "ai.onnx::12::Int8Conv",
                   "ai.onnx::13::Int8Conv"})
    .ParseParamsFn(ParseParamsInt8Conv)
    .ParseOpToGraphFn(ParseOpToGraphInt8Conv)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
