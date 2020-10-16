/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "register/register.h"
#include "tensor.h"
#include "op_log.h"
using namespace ge;
namespace domi
{
  Status ParseCropAndResizeGradBoxes(const Message* op_src, ge::Operator& op)
  {
      AutoMappingFn(op_src, op);

      ge::TensorDesc input_tensor0 = op.GetInputDesc("grads");
      input_tensor0.SetOriginFormat(ge::FORMAT_NHWC);
      input_tensor0.SetFormat(ge::FORMAT_NHWC);
      auto ret0 = op.UpdateInputDesc("grads", input_tensor0);
      ge::TensorDesc input_tensor1 = op.GetInputDesc("images");
      input_tensor1.SetOriginFormat(ge::FORMAT_NHWC);
      input_tensor1.SetFormat(ge::FORMAT_NHWC);
      auto ret1 = op.UpdateInputDesc("images", input_tensor1);
      if((ret0 != ge::GRAPH_SUCCESS) || (ret1 != ge::GRAPH_SUCCESS))
      {
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