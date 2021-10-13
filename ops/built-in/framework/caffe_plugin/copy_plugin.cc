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
 * \file copy_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace domi {
// Caffe ParseParams
Status ParseParamsCopy(const Message* op_src, ge::Operator& op_dst)
{
  OP_LOGI("Copy", "enter into ParseParamsCopy  ------begin!!");

  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (layer == nullptr) {
    OP_LOGE("Copy", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  int64_t n = layer->top_size();
  op_dst.SetAttr("N", n);
  OP_LOGI(op_dst.GetName().c_str(), "[PLUGIN_Copy]--------------top_size=%d---------------", n);
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  op_desc->AddDynamicOutputDesc("y", n);

  OP_LOGI("Copy", "ParseParamsCopy ------end!!");

  return SUCCESS;
}

// register Copy operation
REGISTER_CUSTOM_OP("Copy")
    .FrameworkType(CAFFE)            // type: CAFFE, TENSORFLOW
    .OriginOpType("Split")           // name in caffe module
    .ParseParamsFn(ParseParamsCopy)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
