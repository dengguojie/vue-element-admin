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
 * \file pass_through_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsPassThrough(const Message* op_origin, ge::Operator& op_dest) {
  OP_LOGI("PassThrough", "enter into ParseParams PassThrough ------begin!!");
  // trans op_src to op_dest
  const caffe::LayerParameter* layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGE("PassThrough", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }
  // get layer
  const caffe::ReorgParameter& param = layer->reorg_param();
  if (param.has_stride()) {
    op_dest.SetAttr("stride", static_cast<int32_t>(param.stride()));
  }
  if (param.has_reverse()) {
    op_dest.SetAttr("reverse", param.reverse());
  }

  OP_LOGI("PassThrough", "ParseParamsPassThrough ------end!!");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("PassThrough")
    .FrameworkType(CAFFE)
    .OriginOpType("Reorg")
    .ParseParamsFn(ParseParamsPassThrough)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
