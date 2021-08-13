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
struct AvgPoolAttr {
  std::string auto_pad = "NOTSET";
  int64_t ceil_mode = 0;
  int64_t count_include_pad = 0;  // 0 indicates not include pad
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
};

struct AvgTbeAttr {
  bool trans_2d = false;
  std::string padding_mode = "NOTSET";
  int64_t ceil_mode = 0;
  int64_t exclusive = 0;  // 0 indicates not include pad
  std::vector<int64_t> ksize;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
};

void AvgMaybeChangeAttr(std::vector<int64_t>& value, int64_t length, int64_t num, bool transform_2d) {
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

Status AvgUpdateAttrFromOnnx(const NodeProto* node, AvgPoolAttr& node_attr) {
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
      node_attr.auto_pad = attr.s();
    }

    if (attr.name() == "ceil_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
      node_attr.ceil_mode = attr.i();
    }

    if (attr.name() == "count_include_pad" && attr.type() == ge::onnx::AttributeProto::INT) {
      node_attr.count_include_pad = attr.i();
    }

    if (attr.name() == "kernel_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        node_attr.kernel_shape.push_back(attr.ints(i));
      }
    }
    if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        node_attr.strides.push_back(attr.ints(i));
      }
    }
    if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      unsigned int len = attr.ints_size();
      if (len & 1) {
        ONNX_PLUGIN_LOGE("AveragePool", "the length of pads must be even, such as [x1_begin, x2_begin...x1_end, x2_end,...]");
        return FAILED;
      }
      for (unsigned int i = 0; i < len / 2; i++) {
        node_attr.pads.push_back(attr.ints(i));
        node_attr.pads.push_back(attr.ints(i + len / 2));
      }
    }
  }
  return SUCCESS;
}

Status ParseParamsAveragePool(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    ONNX_PLUGIN_LOGE("AveragePool", "reinterpret_cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    ONNX_PLUGIN_LOGE("AveragePool", "Get OpDesc from operator failed.");
    return FAILED;
  }
  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("y", 1);
  op_dest.SetAttr("original_type", "ai.onnx::11::AveragePool");

  AvgPoolAttr node_attr;
  if (AvgUpdateAttrFromOnnx(node, node_attr) != SUCCESS) {
    return FAILED;
  }

  int64_t dims = node_attr.kernel_shape.size();
  if (dims != 1 && dims != 2 && dims != 3) {
    ONNX_PLUGIN_LOGE("AveragePool", "Only support 1D/2D/3D, but the length of kernel_shape is %ld", dims);
    return FAILED;
  }

  std::map<string, string> padding_mode = {
      {"NOTSET", "CALCULATED"}, {"SAME_UPPER", "SAME"}, {"SAME_LOWER", "SAME"}, {"VALID", "VALID"}};
  // set attr for AvgPoolV2
  bool trans = false;
  if (dims == 1) {
    dims = 2;
    trans = true;
  }

  AvgMaybeChangeAttr(node_attr.kernel_shape, dims == 2 ? dims + 2 : dims, 1, trans);
  op_dest.SetAttr("ksize", node_attr.kernel_shape);

  AvgMaybeChangeAttr(node_attr.strides, dims == 2 ? dims + 2 : dims, 1, trans);
  op_dest.SetAttr("strides", node_attr.strides);

  op_dest.SetAttr("padding_mode", padding_mode[node_attr.auto_pad]);
  op_dest.SetAttr("dims", dims);

  AvgMaybeChangeAttr(node_attr.pads, dims * 2, 0, trans);
  op_dest.SetAttr("pads", node_attr.pads);

  op_dest.SetAttr("ceil_mode", node_attr.ceil_mode);
  op_dest.SetAttr("exclusive", node_attr.count_include_pad);
  op_dest.SetAttr("trans_2d", trans);
  return SUCCESS;
}

Status AvgUpdateTbeAttrFromOp(const Operator& op, AvgTbeAttr& tbe_attr) {
  if (op.GetAttr("ceil_mode", tbe_attr.ceil_mode) != SUCCESS) {
    ONNX_PLUGIN_LOGE("AveragePool", "get ceil_mode from op failed");
    return FAILED;
  };
  if (op.GetAttr("padding_mode", tbe_attr.padding_mode) != SUCCESS) {
    ONNX_PLUGIN_LOGE("AveragePool", "get padding_mode from op failed");
    return FAILED;
  };
  if (op.GetAttr("ksize", tbe_attr.ksize) != SUCCESS) {
    ONNX_PLUGIN_LOGE("AveragePool", "get ksize from op failed");
    return FAILED;
  };
  if (op.GetAttr("strides", tbe_attr.strides) != SUCCESS) {
    ONNX_PLUGIN_LOGE("AveragePool", "get strides from op failed");
    return FAILED;
  };
  if (op.GetAttr("pads", tbe_attr.pads) != SUCCESS) {
    ONNX_PLUGIN_LOGE("AveragePool", "get pads from op failed");
    return FAILED;
  };
  if (op.GetAttr("exclusive", tbe_attr.exclusive) != SUCCESS) {
    ONNX_PLUGIN_LOGE("AveragePool", "get exclusive from op failed");
    return FAILED;
  };
  if (op.GetAttr("trans_2d", tbe_attr.trans_2d) != SUCCESS) {
    OP_LOGW("AveragePool", "get trans_2d from op failed, use default.");
  };
  return SUCCESS;
}

void AvgGenAicpuOp(Operator& op, std::vector<int64_t> ksize, std::vector<int64_t> strides, std::vector<int64_t> pads,
                   std::string padding_mode, Operator& input) {
  // aicpu only supports format=NHWC, so add permute operator to adjust the input
  std::vector<int64_t> ksize_transpose = {ksize[0], ksize[2], ksize[3], ksize[1]};
  std::vector<int64_t> strides_transpose = {strides[0], strides[2], strides[3], strides[1]};
  auto transposeIn = op::TransposeD("permuteIn").set_input_x(input).set_attr_perm({0, 2, 3, 1});

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
    std::vector<int64_t> dims_pad = {len};
    ge::Tensor pads_tensor = Vec2Tensor(pads_vector, dims_pad, ge::DT_INT32, ge::FORMAT_NHWC);
    auto paddings = op::Const("paddings").set_attr_value(pads_tensor);

    float tmp_const = 0.0f;
    std::vector<int64_t> dims = {1};
    ge::Tensor values_tensor = Scalar2Tensor(tmp_const, dims, ge::DT_FLOAT, ge::FORMAT_NHWC);
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

Status AvgUpdateFormat(Operator& op, Format format) {
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
  OP_LOGD(op.GetName().c_str(), "update input x format success, now is %d", op_desc->GetInputDesc("x").GetFormat());

  // update output format
  ge::GeTensorDesc orgTensorY = op_desc->GetOutputDesc("y");
  orgTensorY.SetOriginFormat(format);
  orgTensorY.SetFormat(format);
  ret = op_desc->UpdateOutputDesc("y", orgTensorY);
  if (ret != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE(op.GetName().c_str(), "update output y format failed.");
    return FAILED;
  }
  OP_LOGD(op.GetName().c_str(), "update output y format success, now is %d", op_desc->GetOutputDesc("y").GetFormat());
  return SUCCESS;
}

Status ParseOpToGraphAveragePool(const Operator& op, Graph& graph) {
  int dims = 0;
  if (op.GetAttr("dims", dims) != SUCCESS) {
    ONNX_PLUGIN_LOGE("AveragePool", "get dims from op failed");
    return FAILED;
  }

  AvgTbeAttr tbe_attr;
  if (AvgUpdateTbeAttrFromOp(op, tbe_attr) != SUCCESS) {
    return FAILED;
  }

  ge::Operator data0 = op::Data("data0").set_attr_index(0);
  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t>>> outputs;

  if (dims == 2) {
    if (tbe_attr.trans_2d) {
      ge::Operator::OpListInt axes = {3};
      data0 = op::Unsqueeze("UnsqueezeX").set_input_x(data0).set_attr_axes(axes);
    }
    if (tbe_attr.ksize[2] * tbe_attr.ksize[3] > 255 || (tbe_attr.strides[2] > 63 || tbe_attr.strides[3] > 63)) {
      ge::Operator aicpu_op;
      AvgGenAicpuOp(aicpu_op, tbe_attr.ksize, tbe_attr.strides, tbe_attr.pads, "VALID", data0);

      if (AvgUpdateFormat(aicpu_op, ge::FORMAT_NHWC) != SUCCESS) {
        return FAILED;
      }
      ge::Operator transposeOut = op::TransposeD("permuteOut").set_input_x(aicpu_op).set_attr_perm({0, 3, 1, 2});
      if (tbe_attr.trans_2d) {
        ge::Operator::OpListInt axis = {3};
        transposeOut = op::Squeeze("SqueezeTranspose").set_input_x(transposeOut).set_attr_axis(axis);
      }
      // update output format
      auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(transposeOut);
      ge::GeTensorDesc orgTensorY = op_desc->GetOutputDesc("y");
      orgTensorY.SetOriginFormat(ge::FORMAT_NCHW);
      orgTensorY.SetFormat(ge::FORMAT_NCHW);
      auto ret = op_desc->UpdateOutputDesc("y", orgTensorY);
      if (ret != ge::GRAPH_SUCCESS) {
        ONNX_PLUGIN_LOGE(transposeOut.GetName().c_str(), "update output y format failed.");
        return FAILED;
      }
      OP_LOGD(transposeOut.GetName().c_str(), "update output y format success, now is %d",
              op_desc->GetOutputDesc("y").GetFormat());
      outputs.emplace_back(transposeOut, std::vector<std::size_t>{0});
    } else {
      ge::Operator avgpoolv2 = op::AvgPoolV2()
                                   .set_input_x(data0)
                                   .set_attr_ksize(tbe_attr.ksize)
                                   .set_attr_strides(tbe_attr.strides)
                                   .set_attr_padding_mode(tbe_attr.padding_mode)
                                   .set_attr_pads(tbe_attr.pads)
                                   .set_attr_ceil_mode(tbe_attr.ceil_mode != 0)
                                   .set_attr_exclusive(tbe_attr.exclusive != 1)  // True indicates not include pad
                                   .set_attr_data_format("NCHW");
      if (tbe_attr.trans_2d) {
        ge::Operator::OpListInt axis = {3};
        avgpoolv2 = op::Squeeze("SqueezeAvgpoolv2").set_input_x(avgpoolv2).set_attr_axis(axis);
        if (AvgUpdateFormat(avgpoolv2, ge::FORMAT_NCHW) != SUCCESS) {
          return FAILED;
        }
      }
      outputs.emplace_back(avgpoolv2, std::vector<std::size_t>{0});
    }
  } else {
    auto avgpool3d = op::AvgPool3D()
                         .set_input_x(data0)
                         .set_attr_ksize(tbe_attr.ksize)
                         .set_attr_strides(tbe_attr.strides)
                         .set_attr_pads(tbe_attr.pads)
                         .set_attr_count_include_pad(tbe_attr.exclusive != 0)  // False indicates not include pad
                         .set_attr_ceil_mode(tbe_attr.ceil_mode)
                         .set_attr_data_format("NCDHW");
    if (AvgUpdateFormat(avgpool3d, ge::FORMAT_NCDHW) != SUCCESS) {
      return FAILED;
    }

    outputs.emplace_back(avgpool3d, std::vector<std::size_t>{0});
  }

  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::AveragePool","ai.onnx::9::AveragePool", "ai.onnx::10::AveragePool", "ai.onnx::11::AveragePool",
                   "ai.onnx::12::AveragePool", "ai.onnx::13::AveragePool"})
    .ParseParamsFn(ParseParamsAveragePool)
    .ParseOpToGraphFn(ParseOpToGraphAveragePool)
    .ImplyType(ImplyType::TVM);
}  //  namespace domi
