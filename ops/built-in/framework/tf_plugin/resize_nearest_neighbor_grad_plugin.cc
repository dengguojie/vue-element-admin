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
#include "graph/utils/op_desc_utils.h"
#include "operator.h"
#include "op_log.h"
using namespace ge;
namespace domi
{
  const int POS_0 = 0;
  const int POS_1 = 1;

  Status ParseResizeNearestNeighborV2Grad(const Message* op_src, ge::Operator& op)
  {
    AutoMappingFn(op_src, op);

    auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDesc input_tensor = op_dsc->GetInputDesc(POS_0);
    input_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    input_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret = op_dsc->UpdateInputDesc(POS_0, input_tensor);
    if(ret != ge::GRAPH_SUCCESS){
        OP_LOGE(op.GetName().c_str(), "update input format failed.");
        return FAILED;
    }
    ge::GeTensorDesc output_tensor = op_dsc->GetOutputDesc(POS_0);
    output_tensor.SetOriginFormat(ge::FORMAT_NHWC);
    output_tensor.SetFormat(ge::FORMAT_NHWC);
    auto ret_output = op_dsc->UpdateOutputDesc(POS_0, output_tensor);
    if(ret_output != ge::GRAPH_SUCCESS){
        OP_LOGE(op.GetName().c_str(), "update output format failed.");
        return FAILED;
    }
    return SUCCESS;
  }
// register ResizeNearestNeighborV2Grad op to GE
REGISTER_CUSTOM_OP("ResizeNearestNeighborV2Grad")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("ResizeNearestNeighborGrad")
  .ParseParamsFn(ParseResizeNearestNeighborV2Grad)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
