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

#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

using namespace ge;
namespace domi
{
Status ParseParamsElu(const Message* op_origin,  ge::Operator&  op_dest)
{
  // trans op_src to op_dest
  if (op_origin == nullptr){
    OP_LOGI("Elu", "op_origin get failed");
    return FAILED;
  }
  const caffe::LayerParameter* layer = \
  dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGE("Elu",
            "Dynamic cast op_src to LayerParameter failed");
    return FAILED;
  }

  // get layer
  const caffe::ELUParameter& param = layer->elu_param();
  if (param.has_alpha()) {
    op_dest.SetAttr("alpha", param.alpha());
  }
  return SUCCESS;
}


// elu is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
REGISTER_CUSTOM_OP("Elu")
  .FrameworkType(CAFFE)   // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
  .OriginOpType({"Elu", "ELU"})         // Elu indicates the type name of the operator in the caffe framework
  .ParseParamsFn(ParseParamsElu)  // AutoMappingFn indicates automatic mapping the parameters of op.
  .ImplyType(ImplyType::TVM); // Implementation type. Enumerated type, The options are as follows: TVM, AI_CPU.
}  // namespace domi
