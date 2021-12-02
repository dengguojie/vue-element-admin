/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file conv_plugin.cc
 * \brief
 */
#include "onnx_common.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"

namespace domi {

using OpDesc = std::shared_ptr<ge::OpDesc>;
using namespace ge;
static const int INPUT_4D = 4;
static const int INPUT_5D = 5;
static const int INPUT_NUM_2 = 2;
static const int INPUT_NUM_3 = 3;
static const int ONNX_1D_ATTR_LEN = 1;
static const int ONNX_1D_ATTR_PAD_LEN = 2;

struct ConvAttr {
  std::vector<int64_t> dilations = {1, 1};
  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> pads;
  int64_t groups;
  std::string data_format;
  int dim_size;
  int input_num;
  bool trans_2d = false;
};

Status SetAttrToOp(const ge::onnx::NodeProto* node, ge::Operator& op) {
  // if attr is set in model, receive them with these var
  std::vector<int32_t> strides_list = {1, 1};
  std::vector<int32_t> dilations_list = {1, 1};
  std::vector<int32_t> pad_list;
  bool is_trans_2d = false;
  bool is_have_kernel_shape = false;
  int dim_size = INPUT_4D;
  // update attrs with model value
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == ONNX_1D_ATTR_LEN) {
        strides_list.push_back(1);
        is_trans_2d = true;
      }
      for (auto i = 0; i < attr.ints_size(); ++i) {
        strides_list.push_back(attr.ints(i));
      }
      op.SetAttr("strides", strides_list);
    } else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.ints_size() == ONNX_1D_ATTR_LEN) {
        dilations_list.push_back(1);
        is_trans_2d = true;
      }
      for (auto i = 0; i < attr.ints_size(); ++i) {
        dilations_list.push_back(attr.ints(i));
      }
      op.SetAttr("dilations", dilations_list);
    } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      // in onnx pads=[top, left, bottomm, right] -> [top, bottom, left, right]
      // in onnx pads=[head, top, left, tail, bottomm, right] -> [head, tail, top, bottom, left, right]
      unsigned int len = attr.ints_size();
      if (len & 1) {
        ONNX_PLUGIN_LOGE("Conv", "The value lenth of pads is odd, transform failed.");
        return FAILED;
      }
      if (attr.ints_size() == ONNX_1D_ATTR_PAD_LEN) {
        pad_list.push_back(0);
        pad_list.push_back(0);
        is_trans_2d = true;
      }
      for (unsigned int i = 0; i < len / 2; i++) {
        pad_list.push_back(attr.ints(i));
        pad_list.push_back(attr.ints(i + len / 2));
      }
      op.SetAttr("pads", pad_list);
    } else if (attr.name() == "group" && attr.type() == ge::onnx::AttributeProto::INT) {
      op.SetAttr("groups", attr.i());
    } else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
      op.SetAttr("auto_pad", attr.s());
    } else if (attr.name() == "kernel_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
      is_have_kernel_shape = true;
      is_trans_2d = attr.ints_size() >= 2 ? false : true;
      dim_size = attr.ints_size() > 2 ? INPUT_5D : INPUT_4D;
    }
  }

  if (!is_have_kernel_shape) {
    if (strides_list.size() == 2 && dilations_list.size() == 2 && pad_list.empty()) {
      ONNX_PLUGIN_LOGE(op.GetName().c_str(), "node must have attr (kernel_shape,pads,strides,dilations) at least one");
      return FAILED;
    }
    dim_size = (strides_list.size() == 5 || pad_list.size() == 6 || dilations_list.size() == 5) ? 5 : 4;
  }
  // kernel_shape属性暂时在TBE算子上没有相应的属性接收，所以没有设置它们
  // aicore算子暂时不支持auto_pad参数，接收后对其进行判断拦截处理

  op.SetAttr("dim_size", dim_size);
  op.SetAttr("trans_2d", is_trans_2d);

  return SUCCESS;
}

/*!
 * @brief Replace GE ParseParams fuction to process graph conv node attrs
 * @param op_src the source op info from onnx.
 * @param op the dest GE op.
 * @return status whether this operation success.
 */
Status ParseParamsConv(const Message* op_src, ge::Operator& op) {
  // Convert original onnx graph conv attrs to GE graph attrs
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    ONNX_PLUGIN_LOGE("Conv", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int n = node->input_size();
  op.SetAttr("input_num", n);
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->AddDynamicInputDesc("args", n);
  op_desc->AddDynamicOutputDesc("output", 1);
  op.SetAttr("original_type", "ai.onnx::11::Conv");

  if (SetAttrToOp(node, op) != SUCCESS) {
    ONNX_PLUGIN_LOGE("Conv", "set attr to operator failed");
    return FAILED;
  }

  return SUCCESS;
}

Status SetFormat(ge::Operator& op, const int& dims) {
  if (dims == INPUT_4D) {
    // The fmap should be NCHW
    auto ret_x = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCHW, true);
    if (ret_x != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE("Conv", "update fmap format failed.");
      return FAILED;
    }
    // The filter should be NCHW
    auto ret_w = ChangeFormatFromOnnx(op, 1, ge::FORMAT_NCHW, true);
    if (ret_w != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE("Conv", "update filter format failed.");
      return FAILED;
    }
    // The output should be NCHW
    auto ret_y = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCHW, false);
    if (ret_y != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE("Conv", "update output format failed.");
      return FAILED;
    }
  } else if (dims == INPUT_5D) {
    // The fmap should be NCDHW
    auto ret_x = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCDHW, true);
    if (ret_x != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE("Conv", "update fmap format failed.");
      return FAILED;
    }
    // The filter should be NCDHW
    auto ret_w = ChangeFormatFromOnnx(op, 1, ge::FORMAT_NCDHW, true);
    if (ret_w != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE("Conv", "update filter format failed.");
      return FAILED;
    }
    // The output should be NCDHW
    auto ret_y = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCDHW, false);
    if (ret_y != ge::GRAPH_SUCCESS) {
      ONNX_PLUGIN_LOGE("Conv", "update output format failed.");
      return FAILED;
    }
  } else {
    ONNX_PLUGIN_LOGE("Conv", "The input tensor is not 4D/5D, set format failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status GetConvAttr(const ge::Operator& op, ConvAttr& convAttr) {
  // check attr value, if value is null, then set default value here
  std::string pad_mode = "NOTSET";
  auto ret_strides = op.GetAttr("strides", convAttr.strides);
  auto ret_pads = op.GetAttr("pads", convAttr.pads);
  auto ret_dilations = op.GetAttr("dilations", convAttr.dilations);
  op.GetAttr("auto_pad", pad_mode);
  if (pad_mode != "NOTSET") {
    OP_LOGW("Conv",
            "The attr of auto_pad is not NOTSET, unsupported other value for now,transform failed, may cause precision "
            "error.");
  }
  if (ret_strides != SUCCESS && ret_pads != SUCCESS && ret_dilations != SUCCESS) {
    OP_LOGW("Conv",
            "get attr of strides or pads or dilations from op failed, can not distinguish 2D/3D, use default 2D,"
            " please set one of them obviously.");
  }
  if (op.GetAttr("dim_size", convAttr.dim_size) != SUCCESS) {
    ONNX_PLUGIN_LOGE("Conv", "get dim size from op failed");
    return FAILED;
  }
  if (op.GetAttr("input_num", convAttr.input_num) != SUCCESS) {
    ONNX_PLUGIN_LOGE("Conv", "get number of input from op failed");
    return FAILED;
  }
  if (op.GetAttr("groups", convAttr.groups) != SUCCESS)
    convAttr.groups = 1;

  if (op.GetAttr("data_format", convAttr.data_format) != SUCCESS) {
    // set data_format
    std::string data_format =
        (convAttr.strides.size() == 5 || convAttr.pads.size() == 6 || convAttr.dilations.size() == 5) ? "NCDHW"
                                                                                                      : "NCHW";
    convAttr.data_format = data_format;
  }

  if (op.GetAttr("trans_2d", convAttr.trans_2d) != SUCCESS) {
    OP_LOGW("Conv", "get the flag of convert 1d to 2d failed, use default.");
  }

  std::vector<int64_t> strides_list_default = {1, 1, 1, 1};
  std::vector<int64_t> dilations_list_default = {1, 1, 1, 1};
  std::vector<int64_t> pad_list_default = {0, 0, 0, 0};
  if (convAttr.dim_size == INPUT_5D) {
    strides_list_default.push_back(1);
    dilations_list_default.push_back(1);
    pad_list_default.push_back(0);
  }

  if (convAttr.strides.size() == 2)
    convAttr.strides = strides_list_default;
  if (convAttr.dilations.size() == 2)
    convAttr.dilations = dilations_list_default;
  if (convAttr.pads.size() == 0)
    convAttr.pads = pad_list_default;
  return SUCCESS;
}

static Status ParseOpToGraphConv(const ge::Operator& op, Graph& graph) {
  ConvAttr tbeAttr;
  if (GetConvAttr(op, tbeAttr) != SUCCESS) {
    ONNX_PLUGIN_LOGE("Conv", "get attr value failed.");
    return FAILED;
  }

  ge::Operator dataX = op::Data("dataX").set_attr_index(0);
  ge::Operator dataW = op::Data("dataW").set_attr_index(1);
  std::vector<Operator> inputs{dataX, dataW};
  std::vector<std::pair<Operator, std::vector<size_t>>> outputs;
  ge::Operator conv;
  ge::Operator dataB;
  if (tbeAttr.dim_size == INPUT_4D) {
    if (tbeAttr.trans_2d) {
      ge::Operator::OpListInt axes = {2};
      dataX = op::Unsqueeze("UnsqueezeX").set_input_x(dataX).set_attr_axes(axes);
      dataW = op::Unsqueeze("UnsqueezeW").set_input_x(dataW).set_attr_axes(axes);
    }
    switch (tbeAttr.input_num) {
      case INPUT_NUM_2:
        conv = op::Conv2D()
                   .set_input_x(dataX)
                   .set_input_filter(dataW)
                   .set_attr_strides(tbeAttr.strides)
                   .set_attr_pads(tbeAttr.pads)
                   .set_attr_dilations(tbeAttr.dilations)
                   .set_attr_groups(tbeAttr.groups)
                   .set_attr_data_format(tbeAttr.data_format);
        break;
      case INPUT_NUM_3:
        dataB = op::Data("dataB").set_attr_index(2);
        inputs.push_back(dataB);
        conv = op::Conv2D()
                   .set_input_x(dataX)
                   .set_input_filter(dataW)
                   .set_input_bias(dataB)
                   .set_attr_strides(tbeAttr.strides)
                   .set_attr_pads(tbeAttr.pads)
                   .set_attr_dilations(tbeAttr.dilations)
                   .set_attr_groups(tbeAttr.groups)
                   .set_attr_data_format(tbeAttr.data_format);
        break;
      default:
        ONNX_PLUGIN_LOGE("Conv", "the num of inputs is incorrect.");
        return FAILED;
    }
    if (SetFormat(conv, tbeAttr.dim_size) != SUCCESS) {
      ONNX_PLUGIN_LOGE("Conv", "set format for input and output failed.");
      return FAILED;
    }
    if (tbeAttr.trans_2d) {
      ge::Operator::OpListInt axis = {2};
      conv = op::Squeeze("SqueezeY").set_input_x(conv).set_attr_axis(axis);
    }
  } else if (tbeAttr.dim_size == INPUT_5D) {
    switch (tbeAttr.input_num) {
      case INPUT_NUM_2:
        conv = op::Conv3D()
                   .set_input_x(dataX)
                   .set_input_filter(dataW)
                   .set_attr_strides(tbeAttr.strides)
                   .set_attr_pads(tbeAttr.pads)
                   .set_attr_dilations(tbeAttr.dilations)
                   .set_attr_groups(tbeAttr.groups)
                   .set_attr_data_format(tbeAttr.data_format);
        break;
      case INPUT_NUM_3:
        dataB = op::Data("dataB").set_attr_index(2);
        inputs.push_back(dataB);
        conv = op::Conv3D()
                   .set_input_x(dataX)
                   .set_input_filter(dataW)
                   .set_input_bias(dataB)
                   .set_attr_strides(tbeAttr.strides)
                   .set_attr_pads(tbeAttr.pads)
                   .set_attr_dilations(tbeAttr.dilations)
                   .set_attr_groups(tbeAttr.groups)
                   .set_attr_data_format(tbeAttr.data_format);
        break;
      default:
        ONNX_PLUGIN_LOGE("Conv", "the num of inputs is incorrect.");
        return FAILED;
    }
    if (SetFormat(conv, tbeAttr.dim_size) != SUCCESS) {
      ONNX_PLUGIN_LOGE("Conv", "set format for input and output failed.");
      return FAILED;
    }
  } else {
    ONNX_PLUGIN_LOGE("Conv", "just support 4D or 5D input, transform failed.");
    return FAILED;
  }

  outputs.emplace_back(conv, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Conv", "ai.onnx::9::Conv", "ai.onnx::10::Conv", "ai.onnx::11::Conv", "ai.onnx::12::Conv",
                   "ai.onnx::13::Conv"})
    .ParseParamsFn(ParseParamsConv)
    .ParseOpToGraphFn(ParseOpToGraphConv)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
