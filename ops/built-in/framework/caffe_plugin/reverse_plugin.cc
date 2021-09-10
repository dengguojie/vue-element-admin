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
 * \file reverse_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsReverse(const Message* op_src, ge::Operator& op_dest) {
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (nullptr == layer) {
    return FAILED;
  }

  const caffe::ReverseParameter& reverse_param = layer->reverse_param();

  vector<int> v_axis;
  if (reverse_param.axis_size() == 0) {
    v_axis.push_back(0);
  } else {
    for (int i = 0; i < reverse_param.axis_size(); ++i) {
      v_axis.push_back(reverse_param.axis(i));
    }
  }
  
  op_dest.SetAttr("axis", v_axis);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("ReverseV2D")
    .FrameworkType(CAFFE)               // type: CAFFE, TENSORFLOW
    .OriginOpType("Reverse")            // name in caffe module
    .ParseParamsFn(ParseParamsReverse)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
