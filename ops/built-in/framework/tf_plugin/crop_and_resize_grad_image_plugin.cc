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
  Status ParseCropAndResizeGradImage(const Message* op_src, ge::Operator& op)
  {
    AutoMappingFn(op_src, op);

    ge::TensorDesc input_tensor = op.GetInputDesc("grads");
    input_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    input_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret = op.UpdateInputDesc("grads", input_tensor);
    if(ret != ge::GRAPH_SUCCESS){
        OP_LOGE(op.GetName().c_str(), "update input format failed.");
        return FAILED;
    }
    ge::TensorDesc output_tensor = op.GetOutputDesc("y");
    output_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    output_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret_output = op.UpdateOutputDesc("y", output_tensor);
    if(ret_output != ge::GRAPH_SUCCESS){
        OP_LOGE(op.GetName().c_str(), "update output format failed.");
        return FAILED;
    }
    return SUCCESS;
  }
// register CropAndResizeGradImage op to GE
REGISTER_CUSTOM_OP("CropAndResizeGradImage")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("CropAndResizeGradImage")
  .ParseParamsFn(ParseCropAndResizeGradImage)
  .ImplyType(ImplyType::AI_CPU);
}  // namespace domi