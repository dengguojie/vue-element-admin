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
 * \file mvn_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
/*
 * parse mvn parameters
 * param[in] op_src  source op description
 * param[out] op_dst destination op description
 * return SUCCESS: parse success
 *        FAILED: parse failed
 */
Status ParseParamsMVN(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("MVN", "------Parse Params for MVN begin------");
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (nullptr == layer) {
    OP_LOGE("MVN", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::MVNParameter& param = layer->mvn_param();

  if (param.has_normalize_variance()) {
    op_dest.SetAttr("normalize_variance", static_cast<bool>(param.normalize_variance()));
  }
  if (param.has_across_channels()) {
    op_dest.SetAttr("across_channels", static_cast<bool>(param.across_channels()));
  }
  if (param.has_eps()) {
    op_dest.SetAttr("eps", static_cast<float>(param.eps()));
  }

  OP_LOGI("MVN", "------Parse Params for MVN end------");
  return SUCCESS;
}

// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
REGISTER_CUSTOM_OP("MVN")
    .FrameworkType(CAFFE)           // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("MVN")            // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParamsMVN)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
