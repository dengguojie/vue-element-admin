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
 * \file arg_max_with_k_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {

// #### Set param in attr for transfer
Status ParseParamsArgMaxWithK(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("ArgMaxWithK", "Start into the ParseParamsArgMaxWithK!");
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);

  if (nullptr == layer) {
    OP_LOGE("ArgMaxWithK", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::ArgMaxParameter& argmax_param = layer->argmax_param();

  // Parse axis
  if (argmax_param.has_axis()) {
    int axis = static_cast<int>(argmax_param.axis());
    op_dest.SetAttr("axis", axis);
  } else {
    op_dest.SetAttr("axis", 10000);
  }

  // Parse out_max_val
  if (argmax_param.has_out_max_val()) {
    bool out_max_val = static_cast<bool>(argmax_param.out_max_val());
    op_dest.SetAttr("out_max_val", out_max_val);
  } else {
    op_dest.SetAttr("out_max_val", false);
  }

  // Parse top_k
  if (argmax_param.has_top_k()) {
    int top_k = static_cast<int>(argmax_param.top_k());
    op_dest.SetAttr("topk", top_k);
  } else {
    op_dest.SetAttr("topk", 1);
  }

  OP_LOGI("ArgMaxWithK", "End of the ParseParamsArgMaxWithK!");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("ArgMaxWithK")
    .FrameworkType(CAFFE)                   // type: CAFFE, TENSORFLOW
    .OriginOpType("ArgMax")                 // name in caffe module
    .ParseParamsFn(ParseParamsArgMaxWithK)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);

}  // namespace domi
