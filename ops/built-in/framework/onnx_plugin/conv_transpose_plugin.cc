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
 *convTranspose_plugin.cc
 *
 */
#include <string>
#include <vector>

#include "register/register.h"
#include "graph/operator.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "graph/utils/op_desc_utils.h"

#include "graph.h"
#include "all_ops.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {

using OpDesc = std::shared_ptr<ge::OpDesc>;
using namespace ge;
static const int INPUT_4D = 4;
static const int INPUT_5D = 5;
static const int INPUT_NUM_2 = 2;
static const int INPUT_NUM_3 = 3;
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
};

Status SetAttrToOpConvTranspose(const ge::onnx::NodeProto* node, ge::Operator& op) {
  // if attr is set in model, receive them with these var
  std::vector<int32_t> strides_list = {1, 1};
  std::vector<int32_t> dilations_list = {1, 1};
  std::vector<int32_t> pad_list;
  int dim_size = 4;
  // update attrs with model value
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "strides" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (auto i = 0; i < attr.ints_size(); ++i) {
        strides_list.push_back(attr.ints(i));
      }
      op.SetAttr("strides", strides_list);
    } else if (attr.name() == "dilations" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (auto i = 0; i < attr.ints_size(); ++i) {
        dilations_list.push_back(attr.ints(i));
      }
      op.SetAttr("dilations", dilations_list);
    } else if (attr.name() == "pads" && attr.type() == ge::onnx::AttributeProto::INTS) {
      // in onnx pads=[top, left, bottomm, right] -> [top, bottom, left, right]
      // in onnx pads=[head, top, left, tail, bottomm, right] -> [head, tail, top, bottom, left, right]
      unsigned int len = attr.ints_size();
      if (len & 1) {
        CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "The value lenth of pads is odd, transform failed.");
        return FAILED;
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
      is_set_auto_pad = true;
    } else if (attr.name() == "output_padding" && attr.type() == ge::onnx::AttributeProto::INTS) {
      dim_size = (strides_list.size() == 5 || pad_list.size() == 6 || dilations_list.size() == 5) ? 5 : 4;
      if (dim_size == 4) {
        std::vector<int64_t> out_pads_list;
        out_pads_list.push_back(0);
        out_pads_list.push_back(0);
        out_pads_list.push_back(attr.ints(0));
        out_pads_list.push_back(attr.ints(1));
        op.SetAttr("output_padding", out_pads_list);
      } else {
        std::vector<int64_t> out_pads_list;
        out_pads_list.push_back(0);
        out_pads_list.push_back(0);
        out_pads_list.push_back(attr.ints(0));
        out_pads_list.push_back(attr.ints(1));
        out_pads_list.push_back(attr.ints(2));
        op.SetAttr("output_padding", out_pads_list);
      }
    } else if (attr.name() == "output_shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
      dim_size = (strides_list.size() == 5 || pad_list.size() == 6 || dilations_list.size() == 5) ? 5 : 4;
      if (dim_size == 4) {
        std::vector<int64_t> out_shape_list;
        out_shape_list.push_back(attr.ints(0));
        out_shape_list.push_back(attr.ints(1));
        op.SetAttr("output_shape", out_shape_list);
      } else {
        std::vector<int64_t> out_shape_list;
        out_shape_list.push_back(attr.ints(0));
        out_shape_list.push_back(attr.ints(1));
        out_shape_list.push_back(attr.ints(2));
        op.SetAttr("output_shape", out_shape_list);
      }
      is_set_output_shape = true;
    }
  }
  if (is_set_output_shape && !is_set_auto_pad) {
    op.SetAttr("auto_pad", "SAME_LOWER");
  }
  // kernel_shape属性暂时在TBE算子上没有相应的属性接收，所以没有设置它们
  // aicore算子暂时不支持auto_pad参数，接收后对其进行判断拦截处理

  dim_size = (strides_list.size() == 5 || pad_list.size() == 6 || dilations_list.size() == 5) ? 5 : 4;
  op.SetAttr("dim_size", dim_size);

  return SUCCESS;
}

Status ChangeFormatConvTranspose(OpDesc& op_dsc, const int idx, ge::Format format, bool is_input) {
  if (is_input) {
    ge::GeTensorDesc org_tensor = op_dsc->GetInputDesc(idx);
    org_tensor.SetOriginFormat(format);
    org_tensor.SetFormat(format);
    auto ret = op_dsc->UpdateInputDesc(idx, org_tensor);
    if (ret != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "change input format failed.");
      return FAILED;
    }
  } else {
    ge::GeTensorDesc org_tensor_y = op_dsc->GetOutputDesc(idx);
    org_tensor_y.SetOriginFormat(format);
    org_tensor_y.SetFormat(format);
    auto ret_y = op_dsc->UpdateOutputDesc(idx, org_tensor_y);
    if (ret_y != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "change output format failed.");
      return FAILED;
    }
  }
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
  OpDesc op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_dsc == nullptr) {
    CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "get op desc failed.");
    return FAILED;
  }
  if (dims == INPUT_4D) {
    // The fmap should be NCHW
    auto ret_x = ChangeFormatConvTranspose(op_dsc, 0, ge::FORMAT_NCHW, true);
    if (ret_x != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update fmap format failed.");
      return FAILED;
    }
    // The filter should be NCHW
    auto ret_w = ChangeFormatConvTranspose(op_dsc, 1, ge::FORMAT_NCHW, true);
    if (ret_w != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update filter format failed.");
      return FAILED;
    }
    // The output should be NCHW
    auto ret_y = ChangeFormatConvTranspose(op_dsc, 0, ge::FORMAT_NCHW, false);
    if (ret_y != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update output format failed.");
      return FAILED;
    }
  } else if (dims == INPUT_5D) {
    // The fmap should be NCDHW
    auto ret_x = ChangeFormatConvTranspose(op_dsc, 0, ge::FORMAT_NCDHW, true);
    if (ret_x != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update fmap format failed.");
      return FAILED;
    }
    // The filter should be NCDHW
    auto ret_w = ChangeFormatConvTranspose(op_dsc, 1, ge::FORMAT_NCDHW, true);
    if (ret_w != ge::GRAPH_SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN("ConvTranspose", "update filter format failed.");
      return FAILED;
    }
    // The output should be NCDHW
    auto ret_y = ChangeFormatConvTranspose(op_dsc, 0, ge::FORMAT_NCDHW, false);
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
  auto ret_strides = op.GetAttr("strides", convTransposeAttr.strides);
  auto ret_pads = op.GetAttr("pads", convTransposeAttr.pads);
  auto ret_dilations = op.GetAttr("dilations", convTransposeAttr.dilations);
  op.GetAttr("auto_pad", pad_mode);
  auto ret_output_padding = op.GetAttr("output_padding", convTransposeAttr.output_padding);
  if (ret_strides != SUCCESS && ret_pads != SUCCESS && ret_dilations != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(
        "ConvTranspose",
        "get attr of strides or pads or dilations from op failed, can not distinguish 2D/3D, use default 2D,"
        " please set one of them obviously.");
  }
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
    std::string data_format = (convTransposeAttr.strides.size() == 5 || convTransposeAttr.pads.size() == 6 ||
                               convTransposeAttr.dilations.size() == 5)
                                  ? "NCDHW"
                                  : "NCHW";
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
    input_size.push_back(0);
  }
  convTransposeAttr.input_size = input_size;
  if (convTransposeAttr.strides.size() == 2)
    convTransposeAttr.strides = strides_list_default;
  if (convTransposeAttr.dilations.size() == 2)
    convTransposeAttr.dilations = dilations_list_default;
  if (convTransposeAttr.pads.size() == 0)
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
  if (tbeAttr.dim_size == INPUT_4D) {
    switch (tbeAttr.input_num) {
      case INPUT_NUM_2:
        convTranspose = op::Conv2DTransposeD()
                            .set_input_x(dataX)
                            .set_input_filter(dataW)
                            .set_attr_strides(tbeAttr.strides)
                            .set_attr_pads(tbeAttr.pads)
                            .set_attr_dilations(tbeAttr.dilations)
                            .set_attr_input_size(tbeAttr.input_size)
                            .set_attr_groups(tbeAttr.groups)
                            .set_attr_output_padding(tbeAttr.output_padding)
                            .set_attr_data_format(tbeAttr.data_format);
        break;
      case INPUT_NUM_3:
        dataB = op::Data("dataB").set_attr_index(2);
        inputs.push_back(dataB);
        convTranspose = op::Conv2DTransposeD()
                            .set_input_x(dataX)
                            .set_input_filter(dataW)
                            .set_input_bias(dataB)
                            .set_attr_strides(tbeAttr.strides)
                            .set_attr_pads(tbeAttr.pads)
                            .set_attr_dilations(tbeAttr.dilations)
                            .set_attr_input_size(tbeAttr.input_size)
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

  } else if (tbeAttr.dim_size == INPUT_5D) {
    switch (tbeAttr.input_num) {
      case INPUT_NUM_2:
        convTranspose = op::Conv3DTransposeD()
                            .set_input_x(dataX)
                            .set_input_filter(dataW)
                            .set_attr_strides(tbeAttr.strides)
                            .set_attr_pads(tbeAttr.pads)
                            .set_attr_dilations(tbeAttr.dilations)
                            .set_attr_input_size(tbeAttr.input_size)
                            .set_attr_groups(tbeAttr.groups)
                            .set_attr_output_padding(tbeAttr.output_padding)
                            .set_attr_data_format(tbeAttr.data_format);
        break;
      case INPUT_NUM_3:
        dataB = op::Data("dataB").set_attr_index(2);
        inputs.push_back(dataB);
        convTranspose = op::Conv3DTransposeD()
                            .set_input_x(dataX)
                            .set_input_filter(dataW)
                            .set_input_bias(dataB)
                            .set_attr_strides(tbeAttr.strides)
                            .set_attr_pads(tbeAttr.pads)
                            .set_attr_dilations(tbeAttr.dilations)
                            .set_attr_input_size(tbeAttr.input_size)
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
    .OriginOpType({"ai.onnx::8::ConvTranspose", "ai.onnx::9::ConvTranspose", "ai.onnx::10::ConvTranspose",
                   "ai.onnx::11::ConvTranspose", "ai.onnx::12::ConvTranspose", "ai.onnx::13::ConvTranspose"})
    .ParseParamsFn(ParseParamsConvTranspose)
    .ParseOpToGraphFn(ParseOpToGraphConvTranspose)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
