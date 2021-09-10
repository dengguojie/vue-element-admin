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
 * \file elu_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status ParseParamsElu(const Message* op_origin, ge::Operator& op_dest) {
  // trans op_src to op_dest
  if (op_origin == nullptr) {
    OP_LOGI("Elu", "op_origin get failed");
    return FAILED;
  }
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGE("Elu", "Dynamic cast op_src to LayerParameter failed");
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
    .FrameworkType(CAFFE)           // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType({"Elu", "ELU"})   // Elu indicates the type name of the operator in the caffe framework
    .ParseParamsFn(ParseParamsElu)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);     // Implementation type. Enumerated type, The options are as follows: TVM, AI_CPU.
}  // namespace domi
