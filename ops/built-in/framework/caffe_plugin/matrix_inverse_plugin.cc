/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018. All rights reserved.
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
 * \file matrix_inverse_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsMatrixInverse(const Message* op_origin, ge::Operator& op_dest) {
  // trans op_src to op_dest
  OP_LOGI("MatrixInverse", "Start the ParseParamsMatrixInverse!");
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGE("MatrixInverse", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }
  // need to change
  const caffe::MatrixInverseParameter& param = layer->matrix_inverse_param();
  if (param.has_adjoint()) {
    op_dest.SetAttr("adjoint", static_cast<bool>(param.adjoint()));
  }

  OP_LOGI("MatrixInverse", "End of the ParseParamsMatrixInverse!");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("MatrixInverse")
    .FrameworkType(CAFFE)           // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("MatrixInverse")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParamsMatrixInverse)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
