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
 * \file int8_conv_relu_plugin.cc
 * \brief
 */
#include <string>
#include <vector>
#include <map>

#include "register/register.h"
#include "graph/operator.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "graph/utils/op_desc_utils.h"

#include "graph.h"
#include "all_ops.h"
#include "op_log.h"

namespace domi {

using OpDesc = std::shared_ptr<ge::OpDesc>;
using namespace ge;
static const int INPUT_4D = 4;
static const int INPUT_NUM_2 = 2;
static const int INPUT_NUM_3 = 3;

struct Int8ConvAttr {
  std::vector<int64_t> dilations;
  std::vector<int64_t> strides;
  std::vector<int64_t> kernels;
  std::vector<int64_t> pads;
  float ascend_dequant_scale;
  int ascend_dequant_offset;
  float ascend_quant_scale;
  int ascend_quant_offset;
  int groups;
  std::string data_format;
  int dim_size;
  int input_num;
};

Status SetAttrToInt8Op(const ge::onnx::NodeProto* node, ge::Operator& op) {
  // if attr is set in model, receive them with these var
  std::vector<int32_t> strides_list;
  std::vector<int32_t> dilations_list;
  std::vector<int32_t> pad_list;
  std::vector<int32_t> kernel_list;
  std::map<string, float> scale_map;
  std::map<string, int> offset_map;

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
        OP_LOGE("Int8ConvRelu", "The value lenth of pads is odd, transform failed.");
        return FAILED;
      }
      for (unsigned int i = 0; i < len / 2; i++) {
        pad_list.push_back(attr.ints(i));
        pad_list.push_back(attr.ints(i + len / 2));
      }
      op.SetAttr("pads", pad_list);
    } else if (attr.name() == "group" && attr.type() == ge::onnx::AttributeProto::INT) {
      op.SetAttr("groups", attr.i());
    } else if (attr.name() == "kernels" && attr.type() == ge::onnx::AttributeProto::INTS) {
      for (auto i = 0; i < attr.ints_size(); ++i) {
        kernel_list.push_back(attr.ints(i));
      }
      op.SetAttr("kernels", kernel_list);
    } else if (attr.name() == "order" && attr.type() == ge::onnx::AttributeProto::STRING) {
      op.SetAttr("data_format", attr.s());
    } else if (attr.name().find("scale") != std::string::npos && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale_map[attr.name()] = attr.f();
    } else if (attr.name().find("zero_point") != std::string::npos && attr.type() == ge::onnx::AttributeProto::INT) {
      offset_map[attr.name()] = attr.i();
    }
  }
  if (scale_map.size() < 4) {
    OP_LOGW("Int8ConvRelu", "The number of scales is less then 4.");
  }
  unsigned int s_conter = 0;
  for (auto iter = scale_map.begin(); iter != scale_map.end(); ++iter) {
    if (s_conter == 3) {
      op.SetAttr("ascend_dequant_scale", iter->second);
    } else if (s_conter == 0) {
      op.SetAttr("ascend_quant_scale", iter->second);
    }
    ++s_conter;
  }
  if (offset_map.size() < 4) {
    OP_LOGW("Int8ConvRelu", "The number of offset is less then 4.");
  }
  unsigned int o_counter = 0;
  for (auto iter = offset_map.begin(); iter != offset_map.end(); ++iter) {
    if (o_counter == 3) {
      if (iter->second != 0) {
        OP_LOGW("Int8ConvRelu", "The offset of operator AscendDequant in NPU must 0.");
      }
      op.SetAttr("ascend_dequant_offset", 0);
    } else if (o_counter == 0) {
      op.SetAttr("ascend_quant_offset", static_cast<float>(iter->second));
    }
    ++o_counter;
  }
  int dim_size = 4;
  op.SetAttr("dim_size", dim_size);

  return SUCCESS;
}

Status ChangeInt8Format(OpDesc& op_dsc, const size_t idx, ge::Format& format, bool is_input) {
  if (is_input) {
    ge::GeTensorDesc org_tensor = op_dsc->GetInputDesc(idx);
    org_tensor.SetOriginFormat(format);
    org_tensor.SetFormat(format);
    auto ret = op_dsc->UpdateInputDesc(idx, org_tensor);
    if (ret != ge::GRAPH_SUCCESS) {
      OP_LOGE("Int8ConvRelu", "change input of idx %d format failed.", idx);
      return FAILED;
    }
  } else {
    ge::GeTensorDesc org_tensor_y = op_dsc->GetOutputDesc(idx);
    org_tensor_y.SetOriginFormat(format);
    org_tensor_y.SetFormat(format);
    auto ret_y = op_dsc->UpdateOutputDesc(idx, org_tensor_y);
    if (ret_y != ge::GRAPH_SUCCESS) {
      OP_LOGE("Int8ConvRelu", "change output of idx %d format failed.", idx);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ParseParamsInt8ConvRelu(const Message* op_src, ge::Operator& op) {
  // Convert original onnx graph conv attrs to GE graph attrs
  const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
  if (nullptr == node) {
    OP_LOGE("Int8ConvRelu", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int n = node->input_size();
  op.SetAttr("input_num", n);
  OpDesc op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->AddDynamicInputDesc("args", n);
  op_desc->AddDynamicOutputDesc("output", 1);
  op.SetAttr("original_type", "ai.onnx::11::Int8ConvRelu");

  if (SetAttrToInt8Op(node, op) != SUCCESS) {
    OP_LOGE("Int8ConvRelu", "set attr to operator failed");
    return FAILED;
  }

  return SUCCESS;
}

Status SetInt8Format(ge::Operator& op, const int& dims, ge::Format& format) {
  OpDesc op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_dsc == nullptr) {
    OP_LOGE("Int8ConvRelu", "get op desc failed.");
    return FAILED;
  }
  size_t input_num = op_dsc->GetInputsSize();
  std::string op_name = op_dsc->GetName();
  if (dims == INPUT_4D) {
    for (size_t i = 0; i < input_num; ++i) {
      // The input should be NCHW or NHWC
      auto ret_x = ChangeInt8Format(op_dsc, i, format, true);  // ge::FORMAT_NCHW
      if (ret_x != ge::GRAPH_SUCCESS) {
        OP_LOGE("Int8ConvRelu", "update %s input format failed.", op_name.c_str());
        return FAILED;
      }
    }
    // The output should be NCHW or NHWC
    auto ret_y = ChangeInt8Format(op_dsc, 0, format, false);
    if (ret_y != ge::GRAPH_SUCCESS) {
      OP_LOGE("Int8ConvRelu", "update %s output format failed.", op_name.c_str());
      return FAILED;
    }
  } else {
    OP_LOGE("Int8ConvRelu", "The input tensor is not 4D, set format failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status GetInt8ConvAttr(const ge::Operator& op, Int8ConvAttr& convAttr) {
  // check attr value, if value is null, then set default value here
  auto ret_strides = op.GetAttr("strides", convAttr.strides);
  auto ret_pads = op.GetAttr("pads", convAttr.pads);
  auto ret_dilations = op.GetAttr("dilations", convAttr.dilations);
  auto ret_kernels = op.GetAttr("kernels", convAttr.kernels);

  if (ret_strides != SUCCESS && ret_pads != SUCCESS && ret_dilations != SUCCESS && ret_kernels != SUCCESS) {
    OP_LOGW("Int8ConvRelu", "get attr of kernels from op failed, data of filter is missing,please set it obviously.");
  }
  if (op.GetAttr("dim_size", convAttr.dim_size) != SUCCESS) {
    OP_LOGE("Int8ConvRelu", "get dim size from op failed");
    return FAILED;
  }
  if (op.GetAttr("input_num", convAttr.input_num) != SUCCESS) {
    OP_LOGE("Int8ConvRelu", "get number of input from op failed");
    return FAILED;
  }
  if (op.GetAttr("groups", convAttr.groups) != SUCCESS)
    convAttr.groups = 1;

  if (op.GetAttr("data_format", convAttr.data_format) != SUCCESS) {
    // set default data_format
    std::string data_format = "NCHW";
    convAttr.data_format = data_format;
  }

  if (op.GetAttr("ascend_dequant_scale", convAttr.ascend_dequant_scale) != SUCCESS) {
    OP_LOGW("Int8ConvRelu", "get the attr of ascendDequant scale failed.");
  }
  if (op.GetAttr("ascend_dequant_offset", convAttr.ascend_dequant_offset) != SUCCESS) {
    OP_LOGW("Int8ConvRelu", "get the attr of ascendDequant offset failed.");
  }
  if (op.GetAttr("ascend_quant_scale", convAttr.ascend_quant_scale) != SUCCESS) {
    OP_LOGW("Int8ConvRelu", "get the attr of ascendQuant scale failed.");
  }
  if (op.GetAttr("ascend_quant_offset", convAttr.ascend_quant_offset) != SUCCESS) {
    OP_LOGW("Int8ConvRelu", "get the attr of ascendQuant offset failed.");
  }
  unsigned int stride_size = convAttr.strides.size();
  unsigned int dilation_size = convAttr.dilations.size();
  if (stride_size == 2) {
    if (convAttr.data_format.find("C") == 1) {
      convAttr.strides.insert(convAttr.strides.begin(), 1);
      convAttr.strides.insert(convAttr.strides.begin(), 1);
    } else if (convAttr.data_format.find("C") == (convAttr.data_format.size() - 1)) {
      convAttr.strides.insert(convAttr.strides.begin(), 1);
      convAttr.strides.push_back(1);
    } else if (convAttr.data_format.find("C") == std::string::npos) {
      OP_LOGW("Int8ConvRelu", "the format of operater is incorrect.");
    }
  }
  if (dilation_size == 2) {
    if (convAttr.data_format.find("C") == 1) {
      convAttr.dilations.insert(convAttr.dilations.begin(), 1);
      convAttr.dilations.insert(convAttr.dilations.begin(), 1);
    } else if (convAttr.data_format.find("C") == (convAttr.data_format.size() - 1)) {
      convAttr.dilations.insert(convAttr.dilations.begin(), 1);
      convAttr.dilations.push_back(1);
    } else if (convAttr.data_format.find("C") == std::string::npos) {
      OP_LOGW("Int8ConvRelu", "the format of operater is incorrect.");
    }
  }
  std::vector<int64_t> strides_list_default = {1, 1, 1, 1};
  std::vector<int64_t> dilations_list_default = {1, 1, 1, 1};
  std::vector<int64_t> pad_list_default = {0, 0, 0, 0};

  if (convAttr.strides.empty())
    convAttr.strides = strides_list_default;
  if (convAttr.dilations.empty())
    convAttr.dilations = dilations_list_default;
  if (convAttr.pads.empty())
    convAttr.pads = pad_list_default;
  return SUCCESS;
}

static Status ParseOpToGraphInt8ConvRelu(const ge::Operator& op, Graph& graph) {
  Int8ConvAttr tbeAttr;
  if (GetInt8ConvAttr(op, tbeAttr) != SUCCESS) {
    OP_LOGE("Int8ConvRelu", "get attr value failed.");
    return FAILED;
  }
  std::map<string, ge::Format> format_map = {
      {"NCHW", ge::FORMAT_NCHW}, {"NHWC", ge::FORMAT_NHWC}, {"NCDHW", ge::FORMAT_NCDHW}, {"NDHWC", ge::FORMAT_NDHWC}};
  ge::Operator dataX = op::Data("dataX").set_attr_index(0);
  ge::Operator dataW = op::Data("dataW").set_attr_index(1);
  std::vector<Operator> inputs{dataX, dataW};
  std::vector<std::pair<Operator, std::vector<size_t>>> outputs;
  ge::Operator conv;
  ge::Operator dataB;
  if (tbeAttr.dim_size == INPUT_4D) {
    switch (tbeAttr.input_num) {
      case INPUT_NUM_2:
        conv = op::Conv2D("Int8ConvReluConv2D")
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
        conv = op::Conv2D("Int8ConvReluConv2D")
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
        OP_LOGE("Int8ConvRelu", "the num of inputs is incorrect.");
        return FAILED;
    }
    if (SetInt8Format(conv, tbeAttr.dim_size, format_map[tbeAttr.data_format]) != SUCCESS) {
      OP_LOGE("Int8ConvRelu", "set format for input and output of conv failed.");
      return FAILED;
    }
  } else {
    OP_LOGE("Int8ConvRelu", "just support 4D input, transform failed.");
    return FAILED;
  }

  ge::Shape shape({1});
  TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
  ge::Tensor scale_tensor(tensorDesc, reinterpret_cast<uint8_t*>(&tbeAttr.ascend_dequant_scale), sizeof(ge::DT_FLOAT));
  auto const_op = op::Const("scale").set_attr_value(scale_tensor);
  ge::Operator ascend_deq = op::AscendDequant("Int8ConvReluAscendDeq")
                                .set_input_x(conv)
                                .set_input_deq_scale(const_op)
                                .set_attr_sqrt_mode(false)
                                .set_attr_relu_flag(false)
                                .set_attr_dtype(DT_FLOAT16);
  if (SetInt8Format(ascend_deq, tbeAttr.dim_size, format_map[tbeAttr.data_format]) != SUCCESS) {
    OP_LOGE("Int8ConvRelu", "set format for input and output of ascend dequant failed.");
    return FAILED;
  }
  ge::Operator relu = op::Relu("Int8ConvReluRelu").set_input_x(ascend_deq);
  if (SetInt8Format(relu, tbeAttr.dim_size, format_map[tbeAttr.data_format]) != SUCCESS) {
    OP_LOGE("Int8ConvRelu", "set format for input and output of relu failed.");
    return FAILED;
  }
  ge::Operator ascend_quant = op::AscendQuant("Int8ConvReluAscendQuant")
                                  .set_input_x(relu)
                                  .set_attr_scale(tbeAttr.ascend_quant_scale)
                                  .set_attr_offset(tbeAttr.ascend_quant_offset)
                                  .set_attr_sqrt_mode(false)
                                  .set_attr_round_mode("Round");
  if (SetInt8Format(ascend_quant, tbeAttr.dim_size, format_map[tbeAttr.data_format]) != SUCCESS) {
    OP_LOGE("Int8ConvRelu", "set format for input and output of ascend quant failed.");
    return FAILED;
  }
  outputs.emplace_back(ascend_quant, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::Int8ConvRelu", "ai.onnx::9::Int8ConvRelu", "ai.onnx::10::Int8ConvRelu",
                   "ai.onnx::11::Int8ConvRelu", "ai.onnx::12::Int8ConvRelu", "ai.onnx::13::Int8ConvRelu"})
    .ParseParamsFn(ParseParamsInt8ConvRelu)
    .ParseOpToGraphFn(ParseOpToGraphInt8ConvRelu)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
