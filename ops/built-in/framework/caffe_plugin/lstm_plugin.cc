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
 * \file lstm_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsLstm(const Message* op_origin, ge::Operator& op_dest) {
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("LSTM", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::RecurrentParameter& param = layer->recurrent_param();
  if (param.has_num_output()) {
    op_dest.SetAttr("num_output", static_cast<int>(param.num_output()));
  }

  if (param.has_expose_hidden()) {
    op_dest.SetAttr("expose_hidden", static_cast<bool>(param.expose_hidden()));
  } else {
    op_dest.SetAttr("expose_hidden", false);
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("LSTM")
    .FrameworkType(CAFFE)            // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("LSTM")            // // LSTM indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParamsLstm)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
