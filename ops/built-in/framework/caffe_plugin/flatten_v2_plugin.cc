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
 * \file flatten_v2_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsFlattenV2(const Message* op_origin, ge::Operator& op_dest) {
  OP_LOGI("FlattenV2", "enter into ParseParams FlattenV2 ------begin!!");
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("FlattenV2", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  // get layer
  const caffe::FlattenParameter& param = layer->flatten_param();
  if (param.has_axis()) {
    op_dest.SetAttr("axis", param.axis());
  }
  if (param.has_end_axis()) {
    op_dest.SetAttr("end_axis", param.end_axis());
  }

  OP_LOGI("FlattenV2", "ParseParamsFlattenV2 ------end!!");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("FlattenV2")
    .FrameworkType(CAFFE)
    .OriginOpType("Flatten")
    .ParseParamsFn(ParseParamsFlattenV2)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
