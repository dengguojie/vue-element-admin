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
 * \file upsample_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParams_Upsample(const Message* op_origin, ge::Operator& op_dest) {
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("Upsample", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::UpsampleParameter& param = layer->upsample_param();
  if (param.has_scale() && !param.has_stride() && !param.has_stride_h()) {
    OP_LOGW("Upsample", "scale means each pixel's scale factor, stride means upsample's shape magnification");
  }
  if (param.has_scale()) {
    op_dest.SetAttr("scale", static_cast<float>(param.scale()));
  }

  if (param.has_stride()) {
    op_dest.SetAttr("stride_h", static_cast<int>(param.stride()));
    op_dest.SetAttr("stride_w", static_cast<int>(param.stride()));
  } else {
    op_dest.SetAttr("stride_h", static_cast<int>(param.stride_h()));
    op_dest.SetAttr("stride_w", static_cast<int>(param.stride_w()));
  }

  return SUCCESS;
}

// test_reduction is the type name of the operator in the OM model.
// It can be specified randomly and cannot be the same as an existing type name. It is case sensitive.
REGISTER_CUSTOM_OP("Upsample")
    .FrameworkType(CAFFE)      // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("Upsample")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_Upsample)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
