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
 * \file crop_and_resize_grad_boxes_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "tensor.h"

#include "op_log.h"

namespace domi {
Status ParseCropAndResizeGradBoxes(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);

  ge::TensorDesc input_tensor0 = op.GetInputDesc("grads");
  input_tensor0.SetOriginFormat(ge::FORMAT_NHWC);
  input_tensor0.SetFormat(ge::FORMAT_NHWC);
  auto ret0 = op.UpdateInputDesc("grads", input_tensor0);
  ge::TensorDesc input_tensor1 = op.GetInputDesc("images");
  input_tensor1.SetOriginFormat(ge::FORMAT_NHWC);
  input_tensor1.SetFormat(ge::FORMAT_NHWC);
  auto ret1 = op.UpdateInputDesc("images", input_tensor1);
  if ((ret0 != ge::GRAPH_SUCCESS) || (ret1 != ge::GRAPH_SUCCESS)) {
    OP_LOGE(op.GetName().c_str(), "update input format failed.");
    return FAILED;
  }
  return SUCCESS;
}
// register CropAndResizeGradBoxes op to GE
REGISTER_CUSTOM_OP("CropAndResizeGradBoxes")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CropAndResizeGradBoxes")
    .ParseParamsFn(ParseCropAndResizeGradBoxes)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
