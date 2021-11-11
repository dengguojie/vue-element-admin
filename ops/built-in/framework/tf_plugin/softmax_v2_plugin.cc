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
 * \file softmax_v2_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {
Status ParseParamsSoftmaxMappingFn(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);
  std::vector<int64_t> reduce_dims = {-1};
  if (op.GetAttr("axis", reduce_dims) != ge::GRAPH_SUCCESS) {
    OP_LOGW("SoftmaxV2", "GetAttr axis failed");
  }
  op.SetAttr("axes", reduce_dims);
  return SUCCESS;
}
REGISTER_CUSTOM_OP("SoftmaxV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Softmax")
    .ParseParamsFn(ParseParamsSoftmaxMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
