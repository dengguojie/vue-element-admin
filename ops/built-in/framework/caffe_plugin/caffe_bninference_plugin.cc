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
 * \file caffe_bninference_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParams_BNInference(const Message* op_origin, ge::Operator& op_dest) {
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGE("BNInference", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::BatchNormParameter& param = layer->batch_norm_param();

  if (param.has_eps()) {
    op_dest.SetAttr("epsilon", static_cast<float>(param.eps()));
  }
  if (param.has_use_global_stats()) {
    op_dest.SetAttr("use_global_stats", static_cast<bool>(param.use_global_stats()));
  }

  return SUCCESS;
}
// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
REGISTER_CUSTOM_OP("BNInference")
    .FrameworkType(CAFFE)       // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("BatchNorm")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_BNInference)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
