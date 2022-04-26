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
#include "onnx_common.h"
#include "nn_pooling_ops.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "pad_ops.h"

using namespace ge;
namespace domi {
static const int OUTPUT_SIZE = 1;
static const int LEN_ATTR_1D = 1;
struct TbeAttr {
  bool trans_2d = false;
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
  bool trans_2d = false;
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
      if (attr.ints_size() == LEN_ATTR_1D) {
        onnx_attr.kernel_shape.push_back(1);
        onnx_attr.trans_2d = true;
      }
    } else if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        onnx_attr.strides.push_back(attr.ints(i));
      }
      if (attr.ints_size() == LEN_ATTR_1D) {
        onnx_attr.strides.push_back(1);
      }
    } else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
      onnx_attr.auto_pad = attr.s();
    } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      unsigned int len = attr.ints_size();
      if (len & 1) {
        ONNX_PLUGIN_LOGE("MaxPool", "the length of pads must be even, such as [x1_begin, x2_begin...x1_end, x2_end,...]");
        return FAILED;
      }
      for (unsigned int i = 0; i < len / 2; i++) {
        onnx_attr.pads.push_back(attr.ints(i));
        onnx_attr.pads.push_back(attr.ints(i + len / 2));
      }
      if (len / 2 == LEN_ATTR_1D) {
        onnx_attr.pads.push_back(0);
        onnx_attr.pads.push_back(0);
      }
    } else if (attr.name() == "ceil_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
      onnx_attr.ceil_mode = attr.i();
    } else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (int i = 0; i < attr.ints_size(); i++) {
        onnx_attr.dilations.push_back(attr.ints(i));
      }
      if (attr.ints_size() == LEN_ATTR_1D) {
        onnx_attr.dilations.push_back(attr.ints(0));
        onnx_attr.trans_2d = true;
      }
    } else if (attr.name() == "storage_order" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
      ONNX_PLUGIN_LOGE("MaxPool", "only support storage_order=0, but 1 is obtained now.");
      return FAILED;
    }
  }
  if (onnx_attr.kernel_shape.size() == 0) {
    ONNX_PLUGIN_LOGE("MaxPool", "kernel_shape is required attribute, but NONE is obtained now.");
    return FAILED;
  }
  return SUCCESS;
}

void MaybeChangeAttr(std::vector<int64_t>& value, int64_t length, int64_t num) {
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

Status ParseParamsMaxPool(const Message* op_src, ge::Operator& op_dest) {
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE("MaxPool", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int op_output_size = node->output_size();
  if (op_output_size != OUTPUT_SIZE) {
    ONNX_PLUGIN_LOGE("MaxPool", "The output of Indices is not support, transforming failed.");
    return FAILED;
  }

  // 1.add dynamic input and out
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    ONNX_PLUGIN_LOGE("MaxPool", "Get OpDesc from operator failed.");
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
    ONNX_PLUGIN_LOGE("MaxPool", "Only support 2D/3D, but the length of kernel_shape is %ld", dims);
    return FAILED;
  }

  op_dest.SetAttr("dims", dims);
  op_dest.SetAttr("ceil_mode", node_attr.ceil_mode);
  op_dest.SetAttr("trans_2d", node_attr.trans_2d);
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
    ONNX_PLUGIN_LOGE("MaxPool", "get ceil_mode from op failed");
    return FAILED;
  };
  if (op.GetAttr("padding_mode", tbe_attr.padding_mode) != SUCCESS) {
    ONNX_PLUGIN_LOGE("MaxPool", "get padding_mode from op failed");
    return FAILED;
  };
  if (op.GetAttr("ksize", tbe_attr.ksize) != SUCCESS) {
    ONNX_PLUGIN_LOGE("MaxPool", "get ksize from op failed");
    return FAILED;
  };
  if (op.GetAttr("strides", tbe_attr.strides) != SUCCESS) {
    ONNX_PLUGIN_LOGE("MaxPool", "get strides from op failed");
    return FAILED;
  };
  if (op.GetAttr("pads", tbe_attr.pads) != SUCCESS) {
    ONNX_PLUGIN_LOGE("MaxPool", "get pads from op failed");
    return FAILED;
  };
  if (op.GetAttr("dilation", tbe_attr.dilation) != SUCCESS) {
    ONNX_PLUGIN_LOGE("MaxPool", "get dilation from op failed");
    return FAILED;
  };
  if (op.GetAttr("trans_2d", tbe_attr.trans_2d) != SUCCESS) {
    OP_LOGW("MaxPool", "get trans_2d from op failed, use default.");
  };
  return SUCCESS;
}

void GenAicpuOp(Operator& op, std::vector<int64_t> ksize, std::vector<int64_t> strides, std::vector<int64_t> pads,
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

    float tmp_const = -65504.0f;
    std::vector<int64_t> dims = {1};
    ge::Tensor values_tensor = Scalar2Tensor(tmp_const, dims, ge::DT_FLOAT, ge::FORMAT_NHWC);
    auto constant_values = op::Const("constant_values").set_attr_value(values_tensor);

    auto padV2 =
        op::PadV2().set_input_x(transposeIn).set_input_paddings(paddings).set_input_constant_values(constant_values);

    op = op::MaxPool()
             .set_input_x(padV2)
             .set_attr_ksize(ksize_transpose)
             .set_attr_strides(strides_transpose)
             .set_attr_padding(padding_mode)
             .set_attr_data_format("NHWC");
  } else {
    op = op::MaxPool()
             .set_input_x(transposeIn)
             .set_attr_ksize(ksize_transpose)
             .set_attr_strides(strides_transpose)
             .set_attr_padding(padding_mode)
             .set_attr_data_format("NHWC");
  }
}

Status UpdateFormat(Operator& op, Format format) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  // update input format
  ge::GeTensorDesc orgTensorX = op_desc->GetInputDesc("x");
  orgTensorX.SetOriginFormat(format);
  orgTensorX.SetFormat(format);
  auto ret = op_desc->UpdateInputDesc("x", orgTensorX);
  if (ret != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "update input x format failed.");
    return FAILED;
  }
  OP_LOGD(TbeGetName(op).c_str(), "update input x format success, now is %d", op_desc->GetInputDesc("x").GetFormat());

  // update output format
  ge::GeTensorDesc orgTensorY = op_desc->GetOutputDesc("y");
  orgTensorY.SetOriginFormat(format);
  orgTensorY.SetFormat(format);
  ret = op_desc->UpdateOutputDesc("y", orgTensorY);
  if (ret != ge::GRAPH_SUCCESS) {
    ONNX_PLUGIN_LOGE(TbeGetName(op).c_str(), "update output y format failed.");
    return FAILED;
  }
  OP_LOGD(TbeGetName(op).c_str(), "update output y format success, now is %d", op_desc->GetOutputDesc("y").GetFormat());
  return SUCCESS;
}

static Status ParseOpToGraphMaxPool(const Operator& op, Graph& graph) {
  int dims = 0;
  if (op.GetAttr("dims", dims) != SUCCESS) {
    ONNX_PLUGIN_LOGE("MaxPool", "get dims from op failed");
    return FAILED;
  }
  TbeAttr tbe_attr;
  if (UpdateTbeAttrFromOp(op, tbe_attr, dims) != SUCCESS) {
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
    // because of the limitation of tbe operator, use aicpu instead
    if (tbe_attr.ksize[2] * tbe_attr.ksize[3] > 255 || (tbe_attr.strides[2] > 63 || tbe_attr.strides[3] > 63)) {
      Operator aicpu_op;
      GenAicpuOp(aicpu_op, tbe_attr.ksize, tbe_attr.strides, tbe_attr.pads, "VALID", data0);

      if (UpdateFormat(aicpu_op, ge::FORMAT_NHWC) != SUCCESS) {
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
      ge::Operator maxpoolv3 = op::MaxPoolV3()
                                   .set_input_x(data0)
                                   .set_attr_ksize(tbe_attr.ksize)
                                   .set_attr_strides(tbe_attr.strides)
                                   .set_attr_padding_mode(tbe_attr.padding_mode)
                                   .set_attr_pads(tbe_attr.pads)
                                   .set_attr_ceil_mode(tbe_attr.ceil_mode == 1)
                                   .set_attr_data_format("NCHW");
      if (tbe_attr.trans_2d) {
        ge::Operator::OpListInt axis = {3};
        maxpoolv3 = op::Squeeze("SqueezeMaxpoolv3").set_input_x(maxpoolv3).set_attr_axis(axis);
        if (UpdateFormat(maxpoolv3, ge::FORMAT_NCHW) != SUCCESS) {
          return FAILED;
        }
      }
      outputs.emplace_back(maxpoolv3, std::vector<std::size_t>{0});
    }
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
    if (UpdateFormat(maxpool3d, ge::FORMAT_NCDHW) != SUCCESS) {
      return FAILED;
    }

    outputs.emplace_back(maxpool3d, std::vector<std::size_t>{0});
  }

  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::MaxPool",
                   "ai.onnx::9::MaxPool",
                   "ai.onnx::10::MaxPool",
                   "ai.onnx::11::MaxPool",
                   "ai.onnx::12::MaxPool",
                   "ai.onnx::13::MaxPool"})
    .ParseParamsFn(ParseParamsMaxPool)
    .ParseOpToGraphFn(ParseOpToGraphMaxPool)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
