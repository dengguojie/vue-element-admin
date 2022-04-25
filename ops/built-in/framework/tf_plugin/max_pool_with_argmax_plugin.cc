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
 * \file max_pool_with_argmax_plugin.cpp
 * \brief
 */
#include "graph/utils/op_desc_utils.h"
#include "register/register.h"
#include "op_log.h"

namespace domi {
Status MaxPoolWithArgMappingFn(const Message* op_src, ge::Operator& op) {
  if (AutoMappingFn(op_src, op) != SUCCESS) {
    OP_LOGE("max_pool_with_argmax",
            "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc tensor_descw = op_dsc->GetInputDesc(0);
  ge::GeTensorDesc tensor_descw1 = op_dsc->GetOutputDesc(0);
  ge::GeTensorDesc tensor_descw2 = op_dsc->GetOutputDesc(1);
  tensor_descw.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_descw1.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_descw2.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_descw.SetFormat(ge::FORMAT_NHWC);
  tensor_descw1.SetFormat(ge::FORMAT_NHWC);
  tensor_descw2.SetFormat(ge::FORMAT_NHWC);
  auto ret = op_dsc->UpdateInputDesc(0, tensor_descw);
  auto ret1 = op_dsc->UpdateOutputDesc(0, tensor_descw1);
  auto ret2 = op_dsc->UpdateOutputDesc(1, tensor_descw2);
  if (ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS || ret2 != ge::GRAPH_SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("MaxPoolWithArgmax")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MaxPoolWithArgmax")
    .ParseParamsFn(MaxPoolWithArgMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
