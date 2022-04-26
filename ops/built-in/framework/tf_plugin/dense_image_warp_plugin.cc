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
 * \file dense_image_warp_plugin.cc
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {
Status DenseImageWarpMappingFn(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_dsc == nullptr) {
    OP_LOGE(TbeGetName(op).c_str(), "Get op_dsc failed");
    return FAILED;
  }
  ge::GeTensorDesc tensor_desc_image = op_dsc->GetInputDesc(0);
  ge::GeTensorDesc tensor_desc_flow = op_dsc->GetInputDesc(1);
  ge::GeTensorDesc tensor_desc_y = op_dsc->GetOutputDesc(0);
  tensor_desc_image.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_desc_flow.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_desc_y.SetOriginFormat(ge::FORMAT_NHWC);
  tensor_desc_image.SetFormat(ge::FORMAT_NHWC);
  tensor_desc_flow.SetFormat(ge::FORMAT_NHWC);
  tensor_desc_y.SetFormat(ge::FORMAT_NHWC);

  auto ret = op_dsc->UpdateInputDesc(0, tensor_desc_image);
  auto ret1 = op_dsc->UpdateInputDesc(1, tensor_desc_flow);
  auto ret2 = op_dsc->UpdateOutputDesc(0, tensor_desc_y);
  if (ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS || ret2 != ge::GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update input_desc or output_desc failed");
    return FAILED;
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("DenseImageWarp")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DenseImageWarp")
    .ParseParamsFn(DenseImageWarpMappingFn)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
