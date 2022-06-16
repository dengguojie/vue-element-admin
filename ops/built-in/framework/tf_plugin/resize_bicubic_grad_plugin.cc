/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file resize_bicubic_grad_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "tensor.h"

#include "op_log.h"

namespace domi {
Status ParseResizeBicubicGrad(const Message* op_src, ge::Operator& op) {
  if (AutoMappingFn(op_src, op) != SUCCESS) {
    return FAILED;
  }

  ge::TensorDesc input_tensor = op.GetInputDesc("grads");
  input_tensor.SetOriginFormat(ge::FORMAT_NHWC);
  input_tensor.SetFormat(ge::FORMAT_NHWC);
  ge::TensorDesc input_tensor1 = op.GetInputDesc("original_image");
  input_tensor1.SetOriginFormat(ge::FORMAT_NHWC);
  input_tensor1.SetFormat(ge::FORMAT_NHWC);
  auto ret0 = op.UpdateInputDesc("grads", input_tensor);
  auto ret1 = op.UpdateInputDesc("original_image", input_tensor1);
  if ((ret0 != ge::GRAPH_SUCCESS) || (ret1 != ge::GRAPH_SUCCESS)) {
    OP_LOGE(TbeGetName(op).c_str(), "update input format failed.");
    return FAILED;
  }
  ge::TensorDesc output_tensor = op.GetOutputDesc("y");
  output_tensor.SetOriginFormat(ge::FORMAT_NHWC);
  output_tensor.SetFormat(ge::FORMAT_NHWC);
  auto ret_output = op.UpdateOutputDesc("y", output_tensor);
  if (ret_output != ge::GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "update output format failed.");
    return FAILED;
  }
  return SUCCESS;
}
// register ResizeBicubicGrad op to GE
REGISTER_CUSTOM_OP("ResizeBicubicGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ResizeBicubicGrad")
    .ParseParamsFn(ParseResizeBicubicGrad)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
