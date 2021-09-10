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
 * \file softmax_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
/*
 * parse softmax parameters
 * param[in] op_src  source op description
 * param[out] op_dst destination op description
 * return SUCCESS:parse success
 *        FAILED: parse failed
 */
Status ParseParamsSoftmax(const Message* op_src, ge::Operator& op_dst) {
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (layer == nullptr) {
    OP_LOGI("Softmax", "[Softmax_Plugin] cast op_src to LayerParameter failed.");
    return FAILED;
  }
  const caffe::SoftmaxParameter& param = layer->softmax_param();
  if (param.has_axis()) {
    std::vector<int64_t> vec = {};
    vec.push_back(param.axis());
    op_dst.SetAttr("axes", vec);
  } else {
    std::vector<int64_t> vec = {1};
    op_dst.SetAttr("axes", vec);
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("SoftmaxV2")
    .FrameworkType(CAFFE)
    .OriginOpType("Softmax")
    .ParseParamsFn(ParseParamsSoftmax)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
