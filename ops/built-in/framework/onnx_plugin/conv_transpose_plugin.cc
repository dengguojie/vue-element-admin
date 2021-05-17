/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http:// www.apache.org/licenses/LICENSE-2.0
 */

#include "common/util/error_manager/error_manager.h"
#include "op_log.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

namespace domi {
static void ReportErr(const string &op_name, const string &desc,
                      const string &err_code) {
  std::map<string, string> err_map;
  err_map["op_name"] = op_name;
  err_map["description"] = desc;
  ErrorManager::GetInstance().ReportErrMessage(err_code, err_map);
}

static bool CheckOnnxAttr(const ge::onnx::NodeProto *p_node) {
  for (const auto &attr : p_node->attribute()) {
    if (attr.name() == "strides" || attr.name() == "dilations" ||
        attr.name() == "kernel_shape" || attr.name() == "output_padding") {
      const bool b_valide = (attr.type() == ge::onnx::AttributeProto::INTS &&
                             attr.ints_size() == 2);
      if (!b_valide) {
        ReportErr("ConvTransPose", "only ConvTransPose2D is supported now.",
                  "E50058");
        return false;
      }
    } else if (attr.name() == "pads") {
      const bool b_valide = (attr.type() == ge::onnx::AttributeProto::INTS &&
                             attr.ints_size() == 4);
      if (!b_valide) {
        ReportErr("ConvTransPose", "only ConvTransPose2D is supported now.",
                  "E50058");
        return false;
      }
    }
  }
  return true;
}

static bool CheckOnnxAttrValue(const ge::onnx::NodeProto *p_node) {
  for (const auto &attr : p_node->attribute()) {
    if (attr.name() == "dilations") {
      const bool b_valide = (attr.ints(0) == 1 && attr.ints(1) == 1);
      if (!b_valide) {
        ReportErr("ConvTransPose", "only dilations == 1 is supported now.",
                  "E50058");
        return false;
      }
    } else {
    }
  }
  return true;
}

static bool SetOpOriginFmt(ge::Operator &op_dst) {
  ge::TensorDesc x_t_d = op_dst.GetInputDesc("x");
  ge::TensorDesc w_t_d = op_dst.GetInputDesc("filter");
  ge::TensorDesc out_t_d = op_dst.GetOutputDesc("y");
  x_t_d.SetOriginFormat(ge::FORMAT_NCHW);
  x_t_d.SetFormat(ge::FORMAT_NCHW);
  w_t_d.SetOriginFormat(ge::FORMAT_NCHW);
  w_t_d.SetFormat(ge::FORMAT_NCHW);
  out_t_d.SetFormat(ge::FORMAT_NCHW);
  out_t_d.SetFormat(ge::FORMAT_NCHW);
  const ge::graphStatus x_res = op_dst.UpdateInputDesc("x", x_t_d);
  const ge::graphStatus w_res = op_dst.UpdateInputDesc("filter", w_t_d);
  const ge::graphStatus out_res = op_dst.UpdateOutputDesc("y", out_t_d);
  if (x_res != ge::GRAPH_SUCCESS) {
    ReportErr(op_dst.GetName(), "update x fmt failed.", "E50058");
    return false;
  }
  if (w_res != ge::GRAPH_SUCCESS) {
    ReportErr(op_dst.GetName(), "update w fmt failed", "E50058");
    return false;
  }
  if (out_res != ge::GRAPH_SUCCESS) {
    ReportErr(op_dst.GetName(), "update y fmt failed", "E50058");
    return false;
  }
  return true;
}

static bool SetDefaultAttr(ge::Operator &op_dst) {
  const std::vector<int32_t> strides_default = {1, 1, 1, 1};
  const std::vector<int32_t> dilations_default = {1, 1, 1, 1};
  const std::vector<int32_t> pads_default = {0, 0, 0, 0};
  const std::vector<int32_t> out_pads_default = {0, 0, 0, 0};
  const std::vector<int32_t> out_shape_default = {0, 0, 0, 0};
  const int32_t group_default = 1;
  const string fmt_default = "NCHW";
  const string auto_pad_default = "NOTSET";
  op_dst.SetAttr("strides", strides_default);
  op_dst.SetAttr("dilations", dilations_default);
  op_dst.SetAttr("pads", pads_default);
  op_dst.SetAttr("groups", group_default);
  op_dst.SetAttr("output_padding", out_pads_default);
  op_dst.SetAttr("input_size", out_shape_default);
  op_dst.SetAttr("data_format", fmt_default);
  op_dst.SetAttr("auto_pad", auto_pad_default);
  return true;
}

static void SetStrides(const ge::onnx::AttributeProto &src_attr,
                       ge::Operator &op_dst) {
  std::vector<int32_t> strides_list;
  if (src_attr.name() == "strides" &&
      src_attr.type() == ge::onnx::AttributeProto::INTS) {
    strides_list.push_back(1);
    strides_list.push_back(1);
    strides_list.push_back(src_attr.ints(0));
    strides_list.push_back(src_attr.ints(1));
    op_dst.SetAttr("strides", strides_list);
  }
}

static void SetDilations(const ge::onnx::AttributeProto &src_attr,
                         ge::Operator &op_dst) {
  std::vector<int32_t> dilations_list;
  if (src_attr.name() == "dilations" &&
      src_attr.type() == ge::onnx::AttributeProto::INTS) {
    dilations_list.push_back(1);
    dilations_list.push_back(1);
    dilations_list.push_back(src_attr.ints(0));
    dilations_list.push_back(src_attr.ints(1));
    op_dst.SetAttr("dilations", dilations_list);
  }
}

static void SetGroup(const ge::onnx::AttributeProto &src_attr,
                     ge::Operator &op_dst) {
  if (src_attr.name() == "group" &&
      src_attr.type() == ge::onnx::AttributeProto::INT) {
    op_dst.SetAttr("groups", src_attr.i());
  }
}

static void SetOutputPading(const ge::onnx::AttributeProto &src_attr,
                            ge::Operator &op_dst) {
  std::vector<int32_t> out_pads_list;
  if (src_attr.name() == "output_padding" &&
      src_attr.type() == ge::onnx::AttributeProto::INTS) {
    out_pads_list.push_back(0);
    out_pads_list.push_back(0);
    out_pads_list.push_back(src_attr.ints(0));
    out_pads_list.push_back(src_attr.ints(1));
    op_dst.SetAttr("output_padding", out_pads_list);
  }
}

static void SetPads(const ge::onnx::AttributeProto &src_attr, ge::Operator &op_dst) {
  static const int32_t onnx_y_begin = 0;
  static const int32_t onnx_x_begin = 1;
  static const int32_t onnx_y_end = 2;
  static const int32_t onnx_x_end = 3;
  std::vector<int32_t> pads_list;
  if (src_attr.name() == "pads" &&
      src_attr.type() == ge::onnx::AttributeProto::INTS) {
    // in onnx pads=[top, left, bottomm, right]
    // -> [top, bottom, left, right]
    pads_list.push_back(src_attr.ints(onnx_y_begin));
    pads_list.push_back(src_attr.ints(onnx_y_end));
    pads_list.push_back(src_attr.ints(onnx_x_begin));
    pads_list.push_back(src_attr.ints(onnx_x_end));
    op_dst.SetAttr("pads", pads_list);
  }
}

static void SetAutoPad(const ge::onnx::AttributeProto &src_attr, 
                       ge::Operator &op_dst, bool &is_set_auto_pad) {
  if (src_attr.name() == "auto_pad" &&
      src_attr.type() == ge::onnx::AttributeProto::STRING) {
    std::string auto_pad = src_attr.s();
    op_dst.SetAttr("auto_pad", auto_pad);
    is_set_auto_pad = true;
  }
}

static void SetOutputShape(const ge::onnx::AttributeProto &src_attr, 
                           ge::Operator &op_dst, bool &is_set_output_shape) {
  std::vector<int32_t> output_shape_list;
  if (src_attr.name() == "output_shape" &&
      src_attr.type() == ge::onnx::AttributeProto::INTS) {
    output_shape_list.push_back(src_attr.ints(0));
    output_shape_list.push_back(src_attr.ints(1));
    op_dst.SetAttr("output_shape", output_shape_list);
    is_set_output_shape = true;
  }
}

Status ParseParamsConv2DTranspose(const Message *op_src, ge::Operator &op_dst) {
  OP_LOGD(op_dst.GetName().c_str(), "Enter ParseParamsConv2DTranspose.");
  const ge::onnx::NodeProto *p_node =
      reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (p_node == nullptr) {
    OP_LOGE("Conv2DTranspose", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  if (!CheckOnnxAttr(p_node)) {
    OP_LOGE("Conv2DTranspose", "Check onnx attr failed.");
    return FAILED;
  }
  if (!CheckOnnxAttrValue(p_node)) {
    OP_LOGE("Conv2DTranspose", "Check onnx attr value failed.");
    return FAILED;
  }
  if (!SetOpOriginFmt(op_dst)) {
    OP_LOGE("Conv2DTranspose", "Update op format failed.");
    return FAILED;
  }
  if (!SetDefaultAttr(op_dst)) {
    OP_LOGE("Conv2DTranspose", "Set op default attr failed.");
    return FAILED;
  }
  
  bool is_set_auto_pad = false;
  bool is_set_output_shape = false;
  for (const auto &attr : p_node->attribute()) {
    SetDilations(attr, op_dst);
    SetStrides(attr, op_dst);
    SetGroup(attr, op_dst);
    SetOutputPading(attr, op_dst);
    SetPads(attr, op_dst);
    SetAutoPad(attr, op_dst, is_set_auto_pad);
    SetOutputShape(attr, op_dst, is_set_output_shape);
  }
  
  // when have output_shape and not have auto_pad, need set auto_pad as SAME_LOWER
  if (is_set_output_shape && !is_set_auto_pad) {
    op_dst.SetAttr("auto_pad", "SAME_LOWER");
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2DTransposeD")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::ConvTranspose",
                 "ai.onnx::9::ConvTranspose",
                 "ai.onnx::10::ConvTranspose",
                 "ai.onnx::11::ConvTranspose",
                 "ai.onnx::12::ConvTranspose",
                 "ai.onnx::13::ConvTranspose"})
  .ParseParamsFn(ParseParamsConv2DTranspose)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
