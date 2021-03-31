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
#include "graph.h"
#include "all_ops.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
struct AvgPoolAttr {
  std::string auto_pad = "NOTSET";
  int64_t ceil_mode = 0;
  int64_t count_include_pad = 0;
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
};

struct AvgTbeAttr {
  std::string padding_mode = "NOTSET";
  int64_t ceil_mode = 0;
  int64_t exclusive = 0;
  std::vector<int64_t> ksize;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
};

void AvgMaybeChangeAttr(std::vector<int64_t>& value, int64_t length, int64_t num) {
  if (value.empty()) {
    value = std::vector<int64_t>(length, num);
  } else if (length == 4 && num != 0) {
    value.resize(length);
    value[3] = value[1];
    value[2] = value[0];
    value[1] = 1;
    value[0] = 1;
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
      int len = attr.ints_size();
      if (len & 1) {
        OP_LOGE("AveragePool", "the length of pads must be even, such as [x1_begin, x2_begin...x1_end, x2_end,...]");
        return FAILED;
      }
      for (int i = 0; i < len / 2; i++) {
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
    OP_LOGE("AveragePool", "reinterpret_cast op_src to NodeProto failed.");
    return FAILED;
  }

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    OP_LOGE("AveragePool", "Get OpDesc from operator failed.");
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
  if (dims != 2 && dims != 3) {
    OP_LOGE("AveragePool", "Only support 2D/3D, but the length of kernel_shape is %ld", dims);
    return FAILED;
  }

  std::map<string, string> padding_mode = {
      {"NOTSET", "CALCULATED"}, {"SAME_UPPER", "SAME"}, {"SAME_LOWER", "SAME"}, {"VALID", "VALID"}};
  // set attr for AvgPoolV2
  AvgMaybeChangeAttr(node_attr.kernel_shape, dims == 2 ? dims + 2 : dims, 1);
  op_dest.SetAttr("ksize", node_attr.kernel_shape);

  AvgMaybeChangeAttr(node_attr.strides, dims == 2 ? dims + 2 : dims, 1);
  op_dest.SetAttr("strides", node_attr.strides);

  op_dest.SetAttr("padding_mode", padding_mode[node_attr.auto_pad]);
  op_dest.SetAttr("dims", dims);

  AvgMaybeChangeAttr(node_attr.pads, dims * 2, 0);
  op_dest.SetAttr("pads", node_attr.pads);

  op_dest.SetAttr("ceil_mode", node_attr.ceil_mode);
  op_dest.SetAttr("exclusive", node_attr.count_include_pad);
  return SUCCESS;
}

Status AvgUpdateTbeAttrFromOp(const Operator& op, AvgTbeAttr& tbe_attr) {
  if (op.GetAttr("ceil_mode", tbe_attr.ceil_mode) != SUCCESS) {
    OP_LOGE("AveragePool", "get ceil_mode from op failed");
    return FAILED;
  };
  if (op.GetAttr("padding_mode", tbe_attr.padding_mode) != SUCCESS) {
    OP_LOGE("AveragePool", "get padding_mode from op failed");
    return FAILED;
  };
  if (op.GetAttr("ksize", tbe_attr.ksize) != SUCCESS) {
    OP_LOGE("AveragePool", "get ksize from op failed");
    return FAILED;
  };
  if (op.GetAttr("strides", tbe_attr.strides) != SUCCESS) {
    OP_LOGE("AveragePool", "get strides from op failed");
    return FAILED;
  };
  if (op.GetAttr("pads", tbe_attr.pads) != SUCCESS) {
    OP_LOGE("AveragePool", "get pads from op failed");
    return FAILED;
  };
  if (op.GetAttr("exclusive", tbe_attr.exclusive) != SUCCESS) {
    OP_LOGE("AveragePool", "get exclusive from op failed");
    return FAILED;
  };
  return SUCCESS;
}

Status AvgUpdateFormat(Operator& op, Format format) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  // update input format
  ge::GeTensorDesc orgTensorX = op_desc->GetInputDesc("x");
  orgTensorX.SetOriginFormat(format);
  orgTensorX.SetFormat(format);
  auto ret = op_desc->UpdateInputDesc("x", orgTensorX);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update input x format failed.");
    return FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "update input x format success, now is %d", op_desc->GetInputDesc("x").GetFormat());

  // update output format
  ge::GeTensorDesc orgTensorY = op_desc->GetOutputDesc("y");
  orgTensorY.SetOriginFormat(format);
  orgTensorY.SetFormat(format);
  ret = op_desc->UpdateOutputDesc("y", orgTensorY);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output y format failed.");
    return FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "update output y format success, now is %d", op_desc->GetOutputDesc("y").GetFormat());
  return SUCCESS;
}

Status ParseOpToGraphAveragePool(const Operator& op, Graph& graph) {
  int dims = 0;
  if (op.GetAttr("dims", dims) != SUCCESS) {
    OP_LOGE("AveragePool", "get dims from op failed");
    return FAILED;
  }

  AvgTbeAttr tbe_attr;
  if (AvgUpdateTbeAttrFromOp(op, tbe_attr) != SUCCESS) {
    return FAILED;
  }

  auto data0 = op::Data("data0").set_attr_index(0);
  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t>>> outputs;

  if (dims == 2) {
    if (tbe_attr.ksize[2] * tbe_attr.ksize[3] > 255 || (tbe_attr.strides[2] > 63 || tbe_attr.strides[3] > 63)) {
      OP_LOGE("AveragePool", "not support ksize[2] %d * ksize[3] %d > 255 or strides[2] %d > 63 strides[3] %d > 63",
              tbe_attr.ksize[2], tbe_attr.ksize[3], tbe_attr.strides[2], tbe_attr.strides[3]);
      return FAILED;
    } else {
      auto avgpoolv2 = op::AvgPoolV2()
                           .set_input_x(data0)
                           .set_attr_ksize(tbe_attr.ksize)
                           .set_attr_strides(tbe_attr.strides)
                           .set_attr_padding_mode(tbe_attr.padding_mode)
                           .set_attr_pads(tbe_attr.pads)
                           .set_attr_ceil_mode(tbe_attr.ceil_mode == 0)
                           .set_attr_exclusive(tbe_attr.exclusive == 0)
                           .set_attr_data_format("NCHW");
      outputs.emplace_back(avgpoolv2, std::vector<std::size_t>{0});
    }
  } else {
    auto avgpool3d = op::AvgPool3D()
                         .set_input_x(data0)
                         .set_attr_ksize(tbe_attr.ksize)
                         .set_attr_strides(tbe_attr.strides)
                         .set_attr_pads(tbe_attr.pads)
                         .set_attr_count_include_pad(tbe_attr.exclusive == 0)
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
    .OriginOpType({"ai.onnx::9::AveragePool", "ai.onnx::10::AveragePool", "ai.onnx::11::AveragePool",
                   "ai.onnx::12::AveragePool", "ai.onnx::13::AveragePool"})
    .ParseParamsFn(ParseParamsAveragePool)
    .ParseOpToGraphFn(ParseOpToGraphAveragePool)
    .ImplyType(ImplyType::TVM);
}  //  namespace domi
