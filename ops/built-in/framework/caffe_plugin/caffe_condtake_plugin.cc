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
 * \file caffe_condtake_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParams_CondTake(const Message* op_origin, ge::Operator& op_dest) {
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("CondTake", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::CondTakeParameter& param = layer->condtake_param();
  if (param.has_mode()) {
    op_dest.SetAttr("mode", (string)param.mode());
  }
  if (param.has_val()) {
    op_dest.SetAttr("val", static_cast<float>(param.val()));
  }
  if (param.has_eps()) {
    op_dest.SetAttr("eps", static_cast<float>(param.eps()));
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("CondTake")
    .FrameworkType(CAFFE)      // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("CondTake")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_CondTake)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);

}  // namespace domi
