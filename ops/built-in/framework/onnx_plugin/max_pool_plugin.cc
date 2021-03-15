/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file max_pool_plugin.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph.h"
#include "all_ops.h"
#include "op_log.h"

using namespace ge;
namespace domi {
static const int OUTPUT_SIZE = 1;
struct TbeAttr {
  // public attr
  std::vector<int64_t> ksize;
  std::vector<int64_t> strides;
  std::string padding_mode;
  std::vector<int64_t> pads;
  int64_t ceil_mode;
  // for MaxPool3D
  std::vector<int64_t> dilation;
};

struct OnnxAttr {
  // according to onnx::maxpool
  std::string auto_pad = "NOTSET";
  int64_t ceil_mode = 0;
  std::vector<int64_t> dilations;
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
};

Status UpdateOnnxAttrFromOnnx(const ge::onnx::NodeProto* node, OnnxAttr& onnx_attr) {
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "kernel_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        onnx_attr.kernel_shape.push_back(attr.ints(i));
      }
    } else if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        onnx_attr.strides.push_back(attr.ints(i));
      }
    } else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
      onnx_attr.auto_pad = attr.s();
    } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      int len = attr.ints_size();
      if (len & 1) {
        OP_LOGE("MaxPool", "the length of pads must be even, such as [x1_begin, x2_begin...x1_end, x2_end,...]");
        return FAILED;
      }
      for (int i = 0; i < len / 2; i++) {
        onnx_attr.pads.push_back(attr.ints(i));
        onnx_attr.pads.push_back(attr.ints(i + len / 2));
      }
    } else if (attr.name() == "ceil_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
      onnx_attr.ceil_mode = attr.i();
    } else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        onnx_attr.dilations.push_back(attr.ints(i));
      }
    } else if (attr.name() == "storage_order" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
      OP_LOGE("MaxPool", "only support storage_order=0, but 1 is obtained now.");
      return FAILED;
    }
  }
  if (onnx_attr.kernel_shape.size() == 0) {
    OP_LOGE("MaxPool", "kernel_shape is required attribute, but NONE is obtained now.");
    return FAILED;
  }
  return SUCCESS;
}

void MaybeChangeAttr(std::vector<int64_t>& value, int64_t length, int64_t num) {
  if (value.empty()) {
    value = std::vector<int64_t>(length, num);
  } else if (length == 4) {
    value.resize(length);
    value[3] = value[1];
    value[2] = value[0];
    value[1] = 1;
    value[0] = 1;
  }
}

Status ParseParamsMaxPool(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("MaxPool", "[PLUGIN_MaxPool]--------------ParseParamsMaxPool  start---------------");
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE("MaxPool", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int op_output_size = node->output_size();
  if (op_output_size != OUTPUT_SIZE) {
    OP_LOGE("MaxPool", "The output of Indices is not support, transforming failed.");
    return FAILED;
  }

  // 1.add dynamic input and out
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    OP_LOGE("MaxPool", "Get OpDesc from operator failed.");
    return FAILED;
  }
  opDesc->AddDynamicInputDesc("x", 1);
  opDesc->AddDynamicOutputDesc("y", 1);
  // 2.set original_type
  op_dest.SetAttr("original_type", "ai.onnx::11::MaxPool");
  // 3.set attr if needed
  OnnxAttr node_attr;
  if (UpdateOnnxAttrFromOnnx(node, node_attr) != SUCCESS) {
    return FAILED;
  }

  int64_t dims = node_attr.kernel_shape.size();
  if (dims != 2 && dims != 3) {
    OP_LOGE("MaxPool", "Only support 2D/3D, but the length of kernel_shape is %ld", dims);
    return FAILED;
  }

  op_dest.SetAttr("dims", dims);
  op_dest.SetAttr("ceil_mode", node_attr.ceil_mode);
  std::map<string, string> padding_mode = {
      {"NOTSET", "CALCULATED"}, {"SAME_UPPER", "SAME"}, {"SAME_LOWER", "SAME"}, {"VALID", "VALID"}};
  op_dest.SetAttr("padding_mode", padding_mode[node_attr.auto_pad]);

  MaybeChangeAttr(node_attr.kernel_shape, dims == 2 ? dims + 2 : dims, 1);
  op_dest.SetAttr("ksize", node_attr.kernel_shape);

  MaybeChangeAttr(node_attr.strides, dims == 2 ? dims + 2 : dims, 1);
  op_dest.SetAttr("strides", node_attr.strides);

  MaybeChangeAttr(node_attr.pads, dims * 2, 0);
  op_dest.SetAttr("pads", node_attr.pads);

  MaybeChangeAttr(node_attr.dilations, dims * 2, 1);
  op_dest.SetAttr("dilation", node_attr.dilations);

  return SUCCESS;
}

Status UpdateTbeAttrFromOp(const Operator& op, TbeAttr& tbe_attr, int dims) {
  if (op.GetAttr("ceil_mode", tbe_attr.ceil_mode) != SUCCESS) {
    OP_LOGE("MaxPool", "get ceil_mode from op failed");
    return FAILED;
  };
  if (op.GetAttr("padding_mode", tbe_attr.padding_mode) != SUCCESS) {
    OP_LOGE("MaxPool", "get padding_mode from op failed");
    return FAILED;
  };
  if (op.GetAttr("ksize", tbe_attr.ksize) != SUCCESS) {
    OP_LOGE("MaxPool", "get ksize from op failed");
    return FAILED;
  };
  if (op.GetAttr("strides", tbe_attr.strides) != SUCCESS) {
    OP_LOGE("MaxPool", "get strides from op failed");
    return FAILED;
  };
  if (op.GetAttr("pads", tbe_attr.pads) != SUCCESS) {
    OP_LOGE("MaxPool", "get pads from op failed");
    return FAILED;
  };
  if (op.GetAttr("dilation", tbe_attr.dilation) != SUCCESS) {
    OP_LOGE("MaxPool", "get dilation from op failed");
    return FAILED;
  };
  return SUCCESS;
}

static Status ParseOpToGraphMaxPool(const Operator& op, Graph& graph) {
  int dims = 0;
  if (op.GetAttr("dims", dims) != SUCCESS) {
    OP_LOGE("MaxPool", "get dims from op failed");
    return FAILED;
  }
  TbeAttr tbe_attr;
  if (UpdateTbeAttrFromOp(op, tbe_attr, dims) != SUCCESS) {
    return FAILED;
  }

  auto data0 = op::Data("data0").set_attr_index(0);
  std::vector<Operator> inputs{data0};
  std::vector<std::pair<Operator, std::vector<size_t>>> outputs;

  if (dims == 2) {
    auto maxpoolv3 = op::MaxPoolV3()
                         .set_input_x(data0)
                         .set_attr_ksize(tbe_attr.ksize)
                         .set_attr_strides(tbe_attr.strides)
                         .set_attr_padding_mode(tbe_attr.padding_mode)
                         .set_attr_pads(tbe_attr.pads)
                         .set_attr_ceil_mode(tbe_attr.ceil_mode == 1)
                         .set_attr_data_format("NCHW");
    outputs.emplace_back(maxpoolv3, std::vector<std::size_t>{0});
  } else {
    auto maxpool3d = op::MaxPool3D()
                         .set_input_x(data0)
                         .set_attr_ksize(tbe_attr.ksize)
                         .set_attr_strides(tbe_attr.strides)
                         .set_attr_padding(tbe_attr.padding_mode)
                         .set_attr_pads(tbe_attr.pads)
                         .set_attr_dilation(tbe_attr.dilation)
                         .set_attr_ceil_mode(tbe_attr.ceil_mode)
                         .set_attr_data_format("NCDHW");

    // update input format
    ge::TensorDesc orgTensorX = maxpool3d.get_input_desc_x();
    orgTensorX.SetOriginFormat(ge::FORMAT_NCDHW);
    orgTensorX.SetFormat(ge::FORMAT_NCDHW);
    auto ret = maxpool3d.update_input_desc_x(orgTensorX);
    if (ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(maxpool3d.GetName().c_str(), "update input x format failed.");
      return FAILED;
    }
    OP_LOGI(maxpool3d.GetName().c_str(), "update input x format success, now is %d",
            maxpool3d.get_input_desc_x().GetFormat());

    // update output format
    ge::TensorDesc orgTensorY = maxpool3d.get_output_desc_y();
    orgTensorY.SetOriginFormat(ge::FORMAT_NCDHW);
    orgTensorY.SetFormat(ge::FORMAT_NCDHW);
    ret = maxpool3d.update_output_desc_y(orgTensorY);
    if (ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(maxpool3d.GetName().c_str(), "update input y format failed.");
      return FAILED;
    }
    OP_LOGI(maxpool3d.GetName().c_str(), "update input y format success, now is %d",
            maxpool3d.get_output_desc_y().GetFormat());

    outputs.emplace_back(maxpool3d, std::vector<std::size_t>{0});
  }

  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::9::MaxPool",
                   "ai.onnx::10::MaxPool",
                   "ai.onnx::11::MaxPool",
                   "ai.onnx::12::MaxPool",
                   "ai.onnx::13::MaxPool"})
    .ParseParamsFn(ParseParamsMaxPool)
    .ParseOpToGraphFn(ParseOpToGraphMaxPool)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
