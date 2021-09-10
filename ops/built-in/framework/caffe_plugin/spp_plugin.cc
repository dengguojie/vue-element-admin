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
 * \file spp_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsSPP(const Message* op_origin, ge::Operator& op_dest) {
  OP_LOGI("SPP", "enter into ParseParams SPP ------begin!!\n");
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("SPP", "Dynamic cast op_src to LayerParameter failed.\n");
    return FAILED;
  }

  // get layer
  const caffe::SPPParameter& param = layer->spp_param();
  if (param.has_pyramid_height()) {
    op_dest.SetAttr("pyramid_height", static_cast<int64_t>(param.pyramid_height()));
  }
  if (param.has_pool()) {
    op_dest.SetAttr("pool_method", param.pool());
  }

  OP_LOGI("SPP", "ParseParamsSPP ------end!!\n");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("SPP")
    .FrameworkType(CAFFE)
    .OriginOpType("SPP")
    .ParseParamsFn(ParseParamsSPP)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
