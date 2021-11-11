/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file dilation2d_backprop_filter_plugin.cc
 * \brief
 */
#include "op_log.h"
#include "register/register.h"
#include "tensor.h"

namespace domi {
Status ParseDilation2DBackpropFilter(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);

  ge::TensorDesc input_tensor = op.GetInputDesc("x");
  input_tensor.SetOriginFormat(ge::FORMAT_NHWC);
  input_tensor.SetFormat(ge::FORMAT_NHWC);
  auto ret = op.UpdateInputDesc("x", input_tensor);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update input format failed.");
    return FAILED;
  }

  ge::TensorDesc filter_tensor = op.GetInputDesc("filter");
  filter_tensor.SetOriginFormat(ge::FORMAT_NHWC);
  filter_tensor.SetFormat(ge::FORMAT_NHWC);
  auto ret_filter = op.UpdateInputDesc("filter", filter_tensor);
  if (ret_filter != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update filter format failed.");
    return FAILED;
  }

  ge::TensorDesc out_backprop_tensor = op.GetInputDesc("out_backprop");
  out_backprop_tensor.SetOriginFormat(ge::FORMAT_NHWC);
  out_backprop_tensor.SetFormat(ge::FORMAT_NHWC);
  auto ret_out_backprop = op.UpdateInputDesc("out_backprop", out_backprop_tensor);
  if (ret_out_backprop != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update out_backprop format failed.");
    return FAILED;
  }

  ge::TensorDesc output_tensor = op.GetOutputDesc("y");
  output_tensor.SetOriginFormat(ge::FORMAT_NHWC);
  output_tensor.SetFormat(ge::FORMAT_NHWC);
  auto ret_output = op.UpdateOutputDesc("y", output_tensor);
  if (ret_output != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update output format failed.");
    return FAILED;
  }

  std::string padding;
  if (op.GetAttr("padding", padding) == ge::GRAPH_SUCCESS) {
    op.SetAttr("padding_mode", padding);
  }

  return SUCCESS;
}

// register Dilation2DBackpropFilter op to GE
REGISTER_CUSTOM_OP("Dilation2DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Dilation2DBackpropFilter")
    .ParseParamsFn(ParseDilation2DBackpropFilter)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
