/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file depthwise_conv2d_backprop_filter_plugin.cpp
 * \brief
 */
#include "graph/utils/op_desc_utils.h"
#include "register/register.h"
#include "common/util/error_manager/error_manager.h"
#include "../../op_proto/util/error_util.h"
#include "op_log.h"

namespace domi {
Status DepthwiseConv2DBackpropFilterMappingFn(const Message* op_src, ge::Operator& op) {
  if (AutoMappingFn(op_src, op) != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "AutoMappingFn failed.");
    return FAILED;
  }
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_dsc == nullptr) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "GetOpDescFromOperator got nullptr failed.");
    return FAILED;
  }
  auto tensorDescW = op_dsc->MutableOutputDesc("filter_grad");
  tensorDescW->SetOriginFormat(ge::FORMAT_HWCN);
  tensorDescW->SetFormat(ge::FORMAT_HWCN);
  OP_LOGD(op.GetName().c_str(), "update output format success.");
  std::vector<int32_t> padList{0, 0, 0, 0};
  op.SetAttr("pads", padList);
  OP_LOGD(op.GetName().c_str(), "update pads success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("DepthwiseConv2DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DepthwiseConv2dNativeBackpropFilter")
    .ParseParamsFn(DepthwiseConv2DBackpropFilterMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
