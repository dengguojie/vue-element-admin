/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * convTranspose_plugin.cc
 *
 */
#include "onnx_common.h"
#include "../../op_proto/util/axis_util.h"
#include "../../op_proto/util/error_util.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"

namespace domi {

using OpDesc = std::shared_ptr<ge::OpDesc>;
using namespace ge;
static const int INPUT_4D = 4;
static const int INPUT_5D = 5;
static const int INPUT_NUM_2 = 2;
static const int INPUT_NUM_3 = 3;
static const int kIndex = 2;
static const int kLen2 = 2;
static const int kLen3 = 3;
bool is_set_output_shape = false;
bool is_set_auto_pad = false;
struct ConvTransposeAttr {
  std::vector<int64_t> dilations = {1, 1, 1, 1};
  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  int64_t groups = 1;
  std::string data_format = "NCHW";
  std::vector<int64_t> input_size = {0, 0, 0, 0};
  std::string auto_pad = "NOTSET";
  std::vector<int64_t> output_padding = {0, 0, 0, 0};
  int dim_size = 4;
  int input_num = 2;
  bool trans_2d = false;
};

Status AttrUpdate(std::vector<int32_t>& dst, std::vector<int32_t>& src, int offset, int count,
                  const ge::AscendString& op_name) {
  if ((int)src.size() < count) {
    ONNX_PLUGIN_LOGE(op_name.GetString(), "attr size[%d] should >= [%d]", (int)src.size(), count);
    return FAILED;
  }
  for (int i = 0; i < count; ++i) {
    dst[offset + i] = src[i];
  }
  return SUCCESS;
}

void SetIntListValue(const ge::onnx::AttributeProto &attr, std::vector<int32_t> &int_list) {
  for (auto i = 0; i < attr.ints_size(); ++i) {
    int_list.push_back(attr.ints(i));
  }
}

void GetPadList(const ge::onnx::AttributeProto &attr, std::vector<int32_t> &pad_list) {
  // in onnx pads=[top, left, bottomm, right] -> [top, bottom, left, right]
  // in onnx pads=[head, top, left, tail, bottomm, right] -> [head, tail, top, bottom, left, right]
  unsigned int len = attr.ints_size();
  for (unsigned int i = 0; i < len / 2; i++) {
    pad_list.push_back(attr.ints(i));
    pad_list.push_back(attr.ints(i + len / 2));
  }
}

void SetPadsAttr(std::vector<int32_t> &pad_list, int out_len, ge::Operator &op) {
  if (!pad_list.empty()) {
    for (int i = static_cast<int>(pad_list.size()); i < kLen2 * out_len; ++i) {
      auto it = pad_list.begin();
      pad_list.insert(it, 0);
    }
  }
  op.SetAttr("pads", pad_list);
}

void SetSingleValueAttr(const ge::onnx::AttributeProto &attr, ge::Operator &op) {
  if (attr.name() == "group" && attr.type() == ge::onnx::AttributeProto::INT) {
    op.SetAttr("groups", attr.i());
  } else if (attr.name() == "auto_pad" && attr.type() == ge::onnx::AttributeProto::STRING) {
    op.SetAttr("auto_pad", attr.s());
    is_set_auto_pad = true;
  }
}

Status SetAttrToOpConvTranspose(const ge::onnx::NodeProto *node, ge::Operator &op) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != ge::GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return FAILED);
  // if attr is set in model, receive them with these var
  std::vector<int32_t> strides_list = {1, 1};
  std::vector<int32_t> dilations_list = {1, 1};
  std::vector<int32_t> pad_list;
  std::vector<int32_t> out_pads_list;
  std::vector<int32_t> out_shape_list;
  bool is_trans_2d = false;
  bool is_have_kenel_shape = false;
  int dim_size = 4;
  // update attrs with model value
  for (const auto &attr : node->attribute()) {
    if (attr.type() == ge::onnx::AttributeProto::INTS) {
      if (attr.name() == "strides") {
        if (attr.ints_size() == 1) {
          strides_list.push_back(1);
          strides_list.push_back(attr.ints(0));
        } else {
          SetIntListValue(attr, strides_list);
        }
        op.SetAttr("strides", strides_list);
      } else if (attr.name() == "dilations") {
        if (attr.ints_size() == 1) {
          dilations_list.push_back(1);
        }
        SetIntListValue(attr, dilations_list);
        op.SetAttr("dilations", dilations_list);
      } else if (attr.name() == "pads") {
        unsigned int len = attr.ints_size();
        if (len & 1) {
          ONNX_PLUGIN_LOGE(op_name.GetString(), "The value lenth of pads is odd, transform failed.");
          return FAILED;
        }
        GetPadList(attr, pad_list);
      } else if (attr.name() == "output_padding") {
        SetIntListValue(attr, out_pads_list);
      } else if (attr.name() == "output_shape") {
        SetIntListValue(attr, out_shape_list);
        is_set_output_shape = true;
      } else if (attr.name() == "kernel_shape") {
        int len = attr.ints_size();
        is_have_kenel_shape = true;
        is_trans_2d = len == 1;
        dim_size = len >= kLen3 ? INPUT_5D : INPUT_4D;
      }
    } else {
      SetSingleValueAttr(attr, op);
    }
  }

  if (!is_have_kenel_shape) {
    ONNX_PLUGIN_LOGE(op_name.GetString(), "attr kenel_shape must have value");
    return FAILED;
  }

  int out_len = dim_size == INPUT_5D ? kLen3 : kLen2;
  if (!out_pads_list.empty()) {
    std::vector<int32_t> out_pads_list_new(out_len + kLen2, 0);
    if (AttrUpdate(out_pads_list_new, out_pads_list, kIndex, out_len, op_name) != SUCCESS) {
      ONNX_PLUGIN_LOGE(op_name.GetString(), "attr out_pads update fail");
      return FAILED;
    }
    op.SetAttr("output_padding", out_pads_list_new);
  }

  if (!out_shape_list.empty()) {
    std::vector<int32_t> out_shape_list_new(out_len, 0);
    if (AttrUpdate(out_shape_list_new, out_shape_list, 0, out_len, op_name) != SUCCESS) {
      ONNX_PLUGIN_LOGE(op_name.GetString(), "attr out_shape update fail");
      return FAILED;
    }
    op.SetAttr("output_shape", out_shape_list_new);
  }

  SetPadsAttr(pad_list, out_len, op);

  bool is_set_auto_pad_attr = is_set_output_shape && !is_set_auto_pad;
  if (is_set_auto_pad_attr) {
    op.SetAttr("auto_pad", "SAME_LOWER");
  }

  // kernel_shape属性暂时在TBE算子上没有相应的属性接收，所以没有设置它们
  // aicore算子暂时不支持auto_pad参数，接收后对其进行判断拦截处理

  op.SetAttr("dim_size", dim_size);
  op.SetAttr("trans_2d", is_trans_2d);

  return SUCCESS;
}

/*!
 * @brief Replace GE ParseParams fuction to process graph ConvTranspose node attrs
 * @param op_src the source op info from onnx.
 * @param op the dest GE op.
 * @return status whether this operation success.
 */
Status ParseParamsConvTranspose(const Message* op_src, ge::Operator& op) {
  // Convert original onnx graph ConvTranspose attrs to GE graph attrs
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int n = node->input_size();
  op.SetAttr("input_num", n);
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->AddDynamicInputDesc("args", n);
  op_desc->AddDynamicOutputDesc("output", 1);
  op.SetAttr("original_type", "ai.onnx::11::ConvTranspose");

  if (SetAttrToOpConvTranspose(node, op) != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "set attr to operator failed.");
    return FAILED;
  }

  return SUCCESS;
}

Status SetFormatConvTranspose(ge::Operator& op, const int& dims) {
  if (dims == INPUT_4D) {
    // The fmap should be NCHW
    auto ret_x = ChangeFormatFromOnnx(op, 1, ge::FORMAT_NCHW, true);
    if (ret_x != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update fmap format failed.");
      return FAILED;
    }
    // The filter should be NCHW
    auto ret_w = ChangeFormatFromOnnx(op, kIndex, ge::FORMAT_NCHW, true);
    if (ret_w != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update filter format failed.");
      return FAILED;
    }
    // The output should be NCHW
    auto ret_y = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCHW, false);
    if (ret_y != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update output format failed.");
      return FAILED;
    }
  } else if (dims == INPUT_5D) {
    // The fmap should be NCDHW
    auto ret_x = ChangeFormatFromOnnx(op, 1, ge::FORMAT_NCDHW, true);
    if (ret_x != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update fmap format failed.");
      return FAILED;
    }
    // The filter should be NCDHW
    auto ret_w = ChangeFormatFromOnnx(op, kIndex, ge::FORMAT_NCDHW, true);
    if (ret_w != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update filter format failed.");
      return FAILED;
    }
    // The output should be NCDHW
    auto ret_y = ChangeFormatFromOnnx(op, 0, ge::FORMAT_NCDHW, false);
    if (ret_y != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update output format failed.");
      return FAILED;
    }
  } else {
    CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "The input tensor is not 4D/5D, set format failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status GetConvTransposeAttr(const ge::Operator& op, ConvTransposeAttr& convTransposeAttr) {
  // check attr value, if value is null, then set default value here
  std::string pad_mode = "NOTSET";
  op.GetAttr("strides", convTransposeAttr.strides);
  op.GetAttr("pads", convTransposeAttr.pads);
  op.GetAttr("dilations", convTransposeAttr.dilations);
  op.GetAttr("auto_pad", pad_mode);
  op.GetAttr("trans_2d", convTransposeAttr.trans_2d);
  auto ret_output_padding = op.GetAttr("output_padding", convTransposeAttr.output_padding);
  if (op.GetAttr("dim_size", convTransposeAttr.dim_size) != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "get dim size from op failed");
    return FAILED;
  }
  if (op.GetAttr("input_num", convTransposeAttr.input_num) != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "get number of input from op failed.");
    return FAILED;
  }
  if (op.GetAttr("groups", convTransposeAttr.groups) != SUCCESS)
    convTransposeAttr.groups = 1;

  if (op.GetAttr("data_format", convTransposeAttr.data_format) != SUCCESS) {
    // set data_format
    std::string data_format = convTransposeAttr.dim_size == INPUT_5D ? "NCDHW" : "NCHW";
    convTransposeAttr.data_format = data_format;
  }
  if (ret_output_padding != SUCCESS) {
    if (convTransposeAttr.dim_size == INPUT_5D) {
      std::vector<int64_t> output_padding_list = {0, 0, 0, 0, 0};
      convTransposeAttr.output_padding = output_padding_list;
    } else {
      std::vector<int64_t> output_padding_list = {0, 0, 0, 0};
      convTransposeAttr.output_padding = output_padding_list;
    }
  }
  std::vector<int64_t> strides_list_default = {1, 1, 1, 1};
  std::vector<int64_t> dilations_list_default = {1, 1, 1, 1};
  std::vector<int64_t> pad_list_default = {0, 0, 0, 0};
  std::vector<int64_t> input_size = {0, 0, 0, 0};
  if (convTransposeAttr.dim_size == INPUT_5D) {
    strides_list_default.push_back(1);
    dilations_list_default.push_back(1);
    pad_list_default.push_back(0);
    pad_list_default.push_back(0);
    input_size.push_back(0);
  }
  convTransposeAttr.input_size = input_size;
  if ((int)convTransposeAttr.strides.size() != convTransposeAttr.dim_size)
    convTransposeAttr.strides = strides_list_default;
  if ((int)convTransposeAttr.dilations.size() != convTransposeAttr.dim_size)
    convTransposeAttr.dilations = dilations_list_default;
  if ((int)convTransposeAttr.pads.size() == 0)
    convTransposeAttr.pads = pad_list_default;
  return SUCCESS;
}

static Status ParseOpToGraphConvTranspose(const ge::Operator& op, Graph& graph) {
  ConvTransposeAttr tbeAttr;
  if (GetConvTransposeAttr(op, tbeAttr) != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "get attr value failed.");
    return FAILED;
  }

  ge::Operator dataX = op::Data("dataX").set_attr_index(0);
  ge::Operator dataW = op::Data("dataW").set_attr_index(1);
  std::vector<Operator> inputs{dataX, dataW};
  std::vector<std::pair<Operator, std::vector<size_t>>> outputs;
  ge::Operator convTranspose;
  ge::Operator dataB;

  std::vector<int64_t> dims = {(int)tbeAttr.input_size.size()};
  auto input_size_tensor = Vec2Tensor(tbeAttr.input_size, dims, ge::DT_INT64);
  auto const_input_size = op::Const().set_attr_value(input_size_tensor);

  if (tbeAttr.dim_size == INPUT_4D) {
    if (tbeAttr.trans_2d) {
      ge::Operator::OpListInt axes = {2};
      dataX = op::Unsqueeze("UnsqueezeX").set_input_x(dataX).set_attr_axes(axes);
      dataW = op::Unsqueeze("UnsqueezeW").set_input_x(dataW).set_attr_axes(axes);
    }
    switch (tbeAttr.input_num) {
      case INPUT_NUM_2:
        convTranspose = op::Conv2DTranspose()
                            .set_input_x(dataX)
                            .set_input_filter(dataW)
                            .set_input_input_size(const_input_size)
                            .set_attr_strides(tbeAttr.strides)
                            .set_attr_pads(tbeAttr.pads)
                            .set_attr_dilations(tbeAttr.dilations)
                            .set_attr_groups(tbeAttr.groups)
                            .set_attr_output_padding(tbeAttr.output_padding)
                            .set_attr_data_format(tbeAttr.data_format);
        break;
      case INPUT_NUM_3:
        dataB = op::Data("dataB").set_attr_index(2);
        inputs.push_back(dataB);
        convTranspose = op::Conv2DTranspose()
                            .set_input_x(dataX)
                            .set_input_filter(dataW)
                            .set_input_input_size(const_input_size)
                            .set_input_bias(dataB)
                            .set_attr_strides(tbeAttr.strides)
                            .set_attr_pads(tbeAttr.pads)
                            .set_attr_dilations(tbeAttr.dilations)
                            .set_attr_groups(tbeAttr.groups)
                            .set_attr_output_padding(tbeAttr.output_padding)
                            .set_attr_data_format(tbeAttr.data_format);
        break;
      default:
        CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "the num of inputs is incorrect.");
        return FAILED;
    }
    if (SetFormatConvTranspose(convTranspose, tbeAttr.dim_size) != SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "set format for input and output failed.");
      return FAILED;
    }
    if (tbeAttr.trans_2d) {
      ge::Operator::OpListInt axis = {2};
      convTranspose = op::Squeeze("SqueezeY").set_input_x(convTranspose).set_attr_axis(axis);
    }
  } else if (tbeAttr.dim_size == INPUT_5D) {
    switch (tbeAttr.input_num) {
      case INPUT_NUM_2:
        convTranspose = op::Conv3DTranspose()
                            .set_input_x(dataX)
                            .set_input_filter(dataW)
                            .set_input_input_size(const_input_size)
                            .set_attr_strides(tbeAttr.strides)
                            .set_attr_pads(tbeAttr.pads)
                            .set_attr_dilations(tbeAttr.dilations)
                            .set_attr_groups(tbeAttr.groups)
                            .set_attr_output_padding(tbeAttr.output_padding)
                            .set_attr_data_format(tbeAttr.data_format);
        break;
      case INPUT_NUM_3:
        dataB = op::Data("dataB").set_attr_index(2);
        inputs.push_back(dataB);
        convTranspose = op::Conv3DTranspose()
                            .set_input_x(dataX)
                            .set_input_filter(dataW)
                            .set_input_bias(dataB)
                            .set_input_input_size(const_input_size)
                            .set_attr_strides(tbeAttr.strides)
                            .set_attr_pads(tbeAttr.pads)
                            .set_attr_dilations(tbeAttr.dilations)
                            .set_attr_groups(tbeAttr.groups)
                            .set_attr_output_padding(tbeAttr.output_padding)
                            .set_attr_data_format(tbeAttr.data_format);
        break;
      default:
        CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "the num of inputs is incorrect.");
        return FAILED;
    }
    if (SetFormatConvTranspose(convTranspose, tbeAttr.dim_size) != SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "set format for input and output failed.");
      return FAILED;
    }
  } else {
    CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "just support 4D or 5D input, transform failed.");
    return FAILED;
  }

  outputs.emplace_back(convTranspose, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::ConvTranspose"),
                   ge::AscendString("ai.onnx::9::ConvTranspose"),
                   ge::AscendString("ai.onnx::10::ConvTranspose"),
                   ge::AscendString("ai.onnx::11::ConvTranspose"),
                   ge::AscendString("ai.onnx::12::ConvTranspose"),
                   ge::AscendString("ai.onnx::13::ConvTranspose")})
    .ParseParamsFn(ParseParamsConvTranspose)
    .ParseOpToGraphFn(ParseOpToGraphConvTranspose)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
