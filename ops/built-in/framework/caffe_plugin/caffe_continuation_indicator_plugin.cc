/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file caffe_continuation_indicator_plugin.cpp
 * \brief
 */

#include "graph/utils/op_desc_utils.h"
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"

using namespace ge;
namespace domi {
// Caffe ParseParams
Status ParseParamsContinuationIndicator(const Message* op_origin, ge::Operator& op_dest)
{
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);
  if (layer == nullptr) {
    OP_LOGE("ContinuationIndicator", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }
  const caffe::ContinuationIndicatorParameter& param = layer->continuation_indicator_param();
  if (!param.has_time_step() || !param.has_batch_size()) {
    OP_LOGE("ContinuationIndicator", "op must have attr time_step and batch_size.");
    return FAILED;
  }
  op_dest.SetAttr("time_step",  (int64_t)param.time_step());
  op_dest.SetAttr("batch_size",  (int64_t)param.batch_size());
  return SUCCESS;
}

/**
 * FrameworkType:  Enumerated type. The options are as follows: CAFFE, TENSORFLOW
 * OriginOpType:   ContinuationIndicator indicates the type name of the operator in the caffe framework.
 * ParseParamsFn:  AutoMappingFn indicates automatic mapping the parameters of op.
 * ImplyType:      Instantiation type, TVM
 */
REGISTER_CUSTOM_OP("ContinuationIndicator")
    .FrameworkType(CAFFE)
    .OriginOpType("ContinuationIndicator")
    .ParseParamsFn(ParseParamsContinuationIndicator)
    .ImplyType(ImplyType::TVM);
} // namespace domi
