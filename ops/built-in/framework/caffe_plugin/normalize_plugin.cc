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
 * \file normalize_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {

// #### Set param in attr for transfer
Status ParseParamsNormalize(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Normalize", "Start into the ParseParamsNormalize!");
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);

  if (nullptr == layer) {
    OP_LOGE("Normalize", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::NormalizeParameter& normalize_param = layer->norm_param();

  // Parse across_spatial
  if (!normalize_param.has_across_spatial()) {
    op_dest.SetAttr("across_spatial", true);
  } else {
    bool across_spatial = static_cast<bool>(normalize_param.across_spatial());
    op_dest.SetAttr("across_spatial", across_spatial);
  }

  // Parse channel_shared
  if (!normalize_param.has_channel_shared()) {
    op_dest.SetAttr("channel_shared", true);
  } else {
    bool channel_shared = normalize_param.channel_shared();
    op_dest.SetAttr("channel_shared", channel_shared);
  }

  // Parse eps
  if (!normalize_param.has_eps()) {
    op_dest.SetAttr("eps", static_cast<float>(1e-10));
  } else {
    float eps = static_cast<float>(normalize_param.eps());
    op_dest.SetAttr("eps", eps);
  }

  OP_LOGI("Normalize", "End of the ParseParamsNormalize!");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Normalize")
    .FrameworkType(CAFFE)                 // type: CAFFE, TENSORFLOW
    .OriginOpType("Normalize")            // name in caffe module
    .ParseParamsFn(ParseParamsNormalize)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);

}  // namespace domi
