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
 * \file concat_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace domi {
static const int DEFAULT_CONCAT_DIM = 1;

Status ParseParamsConcat(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Concat", "[PLUGIN_CONCAT]------------ParseParams Concatstart---------------");
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);

  if (layer == nullptr) {
    OP_LOGE("Concat", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::ConcatParameter& param = layer->concat_param();
  int concatDim = 0;
  if (param.has_axis()) {
    concatDim = param.axis();
  } else if (param.has_concat_dim()) {
    concatDim = param.concat_dim();
  } else {
    OP_LOGI("Concat", "Caffe Concat has no axis nor concat_dim.");
    concatDim = DEFAULT_CONCAT_DIM;
  }
  op_dest.SetAttr("concat_dim", concatDim);

  int n = layer->bottom_size();
  OP_LOGI("Concat", "[PLUGIN_CONCAT]--------------bottom_size=%d---------------", n);
  op_dest.SetAttr("N", n);
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", n);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("ConcatD")
    .FrameworkType(CAFFE)
    .OriginOpType("Concat")
    .ParseParamsFn(ParseParamsConcat)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
