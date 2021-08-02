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
struct AvgLpPoolAttr {
  int p = 2;
  std::string auto_pad = "NOTSET";
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
};

struct AvgLpPoolTbeAttr {
  int p = 2;
  bool trans_2d = false;
  std::string padding_mode = "NOTSET";
  std::vector<int64_t> ksize;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
};

void AvgLpPoolMaybeChangeAttr(std::vector<int64_t>& value, int64_t length, int64_t num, bool transform_2d) {
  if (value.empty()) {
    value = std::vector<int64_t>(length, num);
  } else if (length == 4 && num != 0) {
    value.resize(length);
    value[3] = transform_2d ? 1 : value[1];
    value[2] = value[0];
    value[1] = 1;
    value[0] = 1;
  } else if (length == 4 && num == 0 && transform_2d) {
    value.resize(length);
    value[3] = 0;
    value[2] = 0;
  }
}

Status AvgUpdateAttrFromOnnx(const NodeProto* node, AvgLpPoolAttr& node_attr) {
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "auto_pad") {
      node_attr.auto_pad = attr.s();
    } else if (attr.name() == "kernel_shape") {
      for (int i = 0; i < attr.ints_size(); i++) {
        node_attr.kernel_shape.push_back(attr.ints(i));
      }
    } else if (attr.name() == "strides") {
      for (int i = 0; i < attr.ints_size(); i++) {
        node_attr.strides.push_back(attr.ints(i));
      }
    } else if (attr.name() == "pads") {
      unsigned int len = attr.ints_size();
      if (len & 1) {
        ONNX_PLUGIN_LOGE("AveragePool", "the length of pads must be even, such as [x1_begin, x2_begin...x1_end, x2_end,...]");
        return FAILED;
      }
      for (unsigned int i = 0; i < len / 2; i++) {
        node_attr.pads.push_back(attr.ints(i));
        node_attr.pads.push_back(attr.ints(i + len / 2));
      }
    } else if (attr.name() == "p") {
      node_attr.p = attr.i();
    }
  }
  return SUCCESS;
}

Status ParseParamsLpPool(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "reinterpret_cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Get OpDesc from operator failed.");
    return FAILED;
  }
  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("y", 1);
  op_dest.SetAttr("original_type", "ai.onnx::11::LpPool");

  AvgLpPoolAttr node_attr;
  if (AvgUpdateAttrFromOnnx(node, node_attr) != SUCCESS) {
    return FAILED;
  }

  int64_t dims = node_attr.kernel_shape.size();
  if (dims != 1 && dims != 2 && dims != 3) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "Only support 1D/2D/3D, but the length of kernel_shape is %ld", dims);
    return FAILED;
  }

  std::map<string, string> padding_mode = {
      {"NOTSET", "CALCULATED"}, {"SAME_UPPER", "SAME"}, {"SAME_LOWER", "SAME"}, {"VALID", "VALID"}};
  if (padding_mode.find(node_attr.auto_pad) == padding_mode.end()) {
    ONNX_PLUGIN_LOGE(op_dest.GetName().c_str(), "attr auto_pad[%s] only NOTSET/SAME_UPPER/SAME_LOWER/VALID", node_attr.auto_pad.c_str());
    return FAILED;
  }

  // set attr for AvgPoolV2
  bool trans = false;
  if (dims == 1) {
    dims = 2;
    trans = true;
  }

  AvgLpPoolMaybeChangeAttr(node_attr.kernel_shape, dims == 2 ? dims + 2 : dims, 1, trans);
  op_dest.SetAttr("ksize", node_attr.kernel_shape);

  AvgLpPoolMaybeChangeAttr(node_attr.strides, dims == 2 ? dims + 2 : dims, 1, trans);
  op_dest.SetAttr("strides", node_attr.strides);

  op_dest.SetAttr("padding_mode", padding_mode[node_attr.auto_pad]);
  op_dest.SetAttr("dims", dims);

  AvgLpPoolMaybeChangeAttr(node_attr.pads, dims * 2, 0, trans);
  op_dest.SetAttr("pads", node_attr.pads);

  op_dest.SetAttr("trans_2d", trans);
  op_dest.SetAttr("p", node_attr.p);
  return SUCCESS;
}

Status AvgUpdateTbeAttrFromOp(const Operator& op, AvgLpPoolTbeAttr& tbe_attr) {
  if (op.GetAttr("padding_mode", tbe_attr.padding_mode) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get padding_mode from op failed");
    return FAILED;
  };

  if (op.GetAttr("ksize", tbe_attr.ksize) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get ksize from op failed");
    return FAILED;
  };

  if (op.GetAttr("strides", tbe_attr.strides) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get strides from op failed");
    return FAILED;
  };

  if (op.GetAttr("pads", tbe_attr.pads) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get pads from op failed");
    return FAILED;
  };

  if (op.GetAttr("trans_2d", tbe_attr.trans_2d) != SUCCESS) {
    ONNX_PLUGIN_LOGW(op.GetName().c_str(), "get trans_2d from op failed, use default.");
    return FAILED;
  };

  if (op.GetAttr("p", tbe_attr.p) != SUCCESS) {
    ONNX_PLUGIN_LOGW(op.GetName().c_str(), "get p from op failed, use default.");
    return FAILED;
  }
  return SUCCESS;
}

void AvgLpPoolGenAicpuOp(Operator& op, std::vector<int64_t> ksize, std::vector<int64_t> strides, std::vector<int64_t> pads,
                   std::string padding_mode, Operator& input) {
  // aicpu only supports format=NHWC, so add permute operator to adjust the input
  std::vector<int64_t> ksize_transpose = {ksize[0], ksize[2], ksize[3], ksize[1]};
  std::vector<int64_t> strides_transpose = {strides[0], strides[2], strides[3], strides[1]};
  std::vector<int32_t> perm = {0, 2, 3, 1};
  auto tensor = Vec2Tensor(perm, {4}, ge::DT_INT32);
  auto const_perm = op::Const().set_attr_value(tensor);
  auto transposeIn = op::Transpose("permuteIn").set_input_x(input).set_input_perm(const_perm);

  std::vector<int32_t> pads_vector(8, 0);
  bool use_pad = false;
  for (size_t i = 0; i < pads.size(); i++) {
    pads_vector[i + 2] = static_cast<int32_t>(pads[i]);
    if (pads[i] != 0) {
      use_pad = true;
    }
  }
  if (use_pad) {
    int64_t len = pads_vector.size();
    TensorDesc tensorDesc(ge::Shape({len}), ge::FORMAT_NHWC, ge::DT_INT32);
    ge::Tensor pads_tensor = Vec2Tensor(pads_vector, {len}, ge::DT_INT32, ge::FORMAT_NHWC);
    auto paddings = op::Const("paddings").set_attr_value(pads_tensor);

    float tmp_const = 0.0;
    TensorDesc valueDesc(ge::Shape({1}), ge::FORMAT_NHWC, ge::DT_FLOAT);
    ge::Tensor values_tensor = Scalar2Tensor(tmp_const, {1}, ge::DT_INT32, ge::FORMAT_NHWC);
    auto constant_values = op::Const("constant_values").set_attr_value(values_tensor);

    auto padV2 =
        op::PadV2().set_input_x(transposeIn).set_input_paddings(paddings).set_input_constant_values(constant_values);

    op = op::AvgPool()
             .set_input_x(padV2)
             .set_attr_ksize(ksize_transpose)
             .set_attr_strides(strides_transpose)
             .set_attr_padding(padding_mode)
             .set_attr_data_format("NHWC");
  } else {
    op = op::AvgPool()
             .set_input_x(transposeIn)
             .set_attr_ksize(ksize_transpose)
             .set_attr_strides(strides_transpose)
             .set_attr_padding(padding_mode)
             .set_attr_data_format("NHWC");
  }
}

Status AvgLpPoolUpdateFormat(Operator& op, Format format) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  // update input format
  ge::GeTensorDesc orgTensorX = op_desc->GetInputDesc("x");
  orgTensorX.SetOriginFormat(format);
  orgTensorX.SetFormat(format);
  auto ret = op_desc->UpdateInputDesc("x", orgTensorX);
  if (ret != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "update input x format failed.");
    return FAILED;
  }
  ONNX_PLUGIN_LOGI(op.GetName().c_str(), "update input x format success, now is %d", op_desc->GetInputDesc("x").GetFormat());

  // update output format
  ge::GeTensorDesc orgTensorY = op_desc->GetOutputDesc("y");
  orgTensorY.SetOriginFormat(format);
  orgTensorY.SetFormat(format);
  ret = op_desc->UpdateOutputDesc("y", orgTensorY);
  if (ret != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "update output y format failed.");
    return FAILED;
  }
  ONNX_PLUGIN_LOGI(op.GetName().c_str(), "update output y format success, now is %d", op_desc->GetOutputDesc("y").GetFormat());
  return SUCCESS;
}

Status ParseOpToGraphLpPool(const Operator& op, Graph& graph) {
  int dims = 0;
  if (op.GetAttr("dims", dims) != SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "get dims from op failed");
    return FAILED;
  }

  AvgLpPoolTbeAttr tbe_attr;
  if (AvgUpdateTbeAttrFromOp(op, tbe_attr) != SUCCESS) {
    return FAILED;
  }

  ge::Operator data0 = op::Data("data0").set_attr_index(0);
  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t>>> outputs;
  
  float p_f = static_cast<float>(tbe_attr.p);
  auto power = op::Power().set_input_x(data0).set_attr_power(p_f).set_attr_scale(1).set_attr_shift(0);

  float mul_kw = 0;
  Operator output_op;
  if (dims == 2) {
    mul_kw = tbe_attr.ksize[2] * tbe_attr.ksize[3];
    Operator input = power;
    if (tbe_attr.trans_2d) {
      ge::Operator::OpListInt axes = {3};
      input = op::Unsqueeze("UnsqueezeX").set_input_x(input).set_attr_axes(axes);
    }

    if (tbe_attr.ksize[2] * tbe_attr.ksize[3] > 255 || (tbe_attr.strides[2] > 63 || tbe_attr.strides[3] > 63)) {
      ge::Operator aicpu_op;
      AvgLpPoolGenAicpuOp(aicpu_op, tbe_attr.ksize, tbe_attr.strides, tbe_attr.pads, "VALID", input);

      if (AvgLpPoolUpdateFormat(aicpu_op, ge::FORMAT_NHWC) != SUCCESS) {
        return FAILED;
      }
      std::vector<int32_t> perm = {0, 3, 1, 2};
      auto tensor = Vec2Tensor(perm, {4}, ge::DT_INT32);
      auto const_perm = op::Const().set_attr_value(tensor);
      output_op = op::Transpose("permuteOut").set_input_x(aicpu_op).set_input_perm(const_perm);
      if (tbe_attr.trans_2d) {
        ge::Operator::OpListInt axis = {3};
        output_op = op::Squeeze("SqueezeTranspose").set_input_x(output_op).set_attr_axis(axis);
      }
      // update output format
      auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(output_op);
      ge::GeTensorDesc orgTensorY = op_desc->GetOutputDesc("y");
      orgTensorY.SetOriginFormat(ge::FORMAT_NCHW);
      orgTensorY.SetFormat(ge::FORMAT_NCHW);
      auto ret = op_desc->UpdateOutputDesc("y", orgTensorY);
      if (ret != ge::GRAPH_SUCCESS) {
        ONNX_PLUGIN_LOGE(output_op.GetName().c_str(), "update output y format failed.");
        return FAILED;
      }
    } else {
      output_op = op::AvgPoolV2()
                      .set_input_x(input)
                      .set_attr_ksize(tbe_attr.ksize)
                      .set_attr_strides(tbe_attr.strides)
                      .set_attr_padding_mode(tbe_attr.padding_mode)
                      .set_attr_pads(tbe_attr.pads)
                      .set_attr_exclusive(false)
                      .set_attr_data_format("NCHW");
      if (tbe_attr.trans_2d) {
        ge::Operator::OpListInt axis = {3};
        output_op = op::Squeeze("SqueezeAvgpoolv2").set_input_x(output_op).set_attr_axis(axis);
        if (AvgLpPoolUpdateFormat(output_op, ge::FORMAT_NCHW) != SUCCESS) {
          return FAILED;
        }
      }
    }
  } else {
    mul_kw = tbe_attr.ksize[0] * tbe_attr.ksize[1] * tbe_attr.ksize[2];
    output_op = op::AvgPool3D()
                    .set_input_x(power)
                    .set_attr_ksize(tbe_attr.ksize)
                    .set_attr_strides(tbe_attr.strides)
                    .set_attr_pads(tbe_attr.pads)
                    .set_attr_data_format("NCDHW");
    if (AvgLpPoolUpdateFormat(output_op, ge::FORMAT_NCDHW) != SUCCESS) {
      return FAILED;
    }
  }
  
  auto muls = op::Muls().set_input_x(output_op).set_attr_value(mul_kw);
  float p1 = 1 / p_f;
  auto power1 = op::Power().set_input_x(muls).set_attr_power(p1).set_attr_scale(1).set_attr_shift(0);
  outputs.emplace_back(power1, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::LpPool",
                   "ai.onnx::9::LpPool", 
                   "ai.onnx::10::LpPool", 
                   "ai.onnx::11::LpPool",
                   "ai.onnx::12::LpPool", 
                   "ai.onnx::13::LpPool"})
    .ParseParamsFn(ParseParamsLpPool)
    .ParseOpToGraphFn(ParseOpToGraphLpPool)
    .ImplyType(ImplyType::TVM);
}  //  namespace domi
