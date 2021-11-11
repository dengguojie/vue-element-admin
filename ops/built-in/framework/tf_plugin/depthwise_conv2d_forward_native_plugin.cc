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
 * \file depthwise_conv2d_forward_native_plugin.cpp
 * \brief
 */
#include "graph/utils/op_desc_utils.h"
#include "register/register.h"
#include "../../op_proto/util/error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "op_log.h"

namespace domi {
Status DepthwiseConv2DMappingFn(const Message* op_src, ge::Operator& op) {
  if (AutoMappingFn(op_src, op) != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "AutoMappingFn failed.");
    return FAILED;
  }
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_dsc == nullptr) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "GetOpDescFromOperator got nullptr failed.");
    return FAILED;
  }
  ge::GeTensorDesc tensorDescW = op_dsc->GetInputDesc(1);
  tensorDescW.SetOriginFormat(ge::FORMAT_HWCN);
  tensorDescW.SetFormat(ge::FORMAT_HWCN);
  auto ret = op_dsc->UpdateInputDesc(1, tensorDescW);
  if (ret != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "updating filter's format failed.");
    return FAILED;
  }
  std::vector<int32_t> padList = {0, 0, 0, 0};
  op.SetAttr("pads", padList);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("DepthwiseConv2D")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DepthwiseConv2dNative")
    .ParseParamsFn(DepthwiseConv2DMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
