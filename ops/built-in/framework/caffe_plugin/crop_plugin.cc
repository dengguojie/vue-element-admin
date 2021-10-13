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
 * \file crop_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsCrop(const Message* op_origin, ge::Operator& op_dest) {
  // trans op_src to op_dest
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (layer == nullptr) {
    OP_LOGE("Crop", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::CropParameter& param = layer->crop_param();
  int axis = 2;
  if (param.has_axis()) {
    axis = static_cast<int>(param.axis());
  }
  op_dest.SetAttr("axis", axis);

  std::vector<int64_t> v_offsets;
  int offsetSize = param.offset_size();
  if (offsetSize == 0) {
    v_offsets.push_back(0);

  } else {
    for (int32_t i = 0; i < param.offset_size(); i++) {
      v_offsets.push_back(param.offset(i));
    }
  }
  op_dest.SetAttr("offsets", v_offsets);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Crop")
    .FrameworkType(CAFFE)            // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("Crop")            // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParamsCrop)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);
}  // namespace domi
