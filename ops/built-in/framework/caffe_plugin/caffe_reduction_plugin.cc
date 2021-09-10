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
 * \file caffe_reduction_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsReduction(const Message* op_src, ge::Operator& op_dst) {
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);

  if (nullptr == layer) {
    OP_LOGE("Reduction", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::ReductionParameter& param = layer->reduction_param();

  if (param.has_operation()) {
    op_dst.SetAttr("operation", static_cast<int64_t>(param.operation()));
  }

  if (param.has_axis()) {
    op_dst.SetAttr("axis", static_cast<int64_t>(param.axis()));
  }

  if (param.has_coeff()) {
    op_dst.SetAttr("coeff", param.coeff());
  }

  return SUCCESS;
}

// register Reduction operation
REGISTER_CUSTOM_OP("Reduction")
    .FrameworkType(CAFFE)                 // type: CAFFE, TENSORFLOW
    .OriginOpType("Reduction")            // name in caffe module
    .ParseParamsFn(ParseParamsReduction)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);

}  // namespace domi
