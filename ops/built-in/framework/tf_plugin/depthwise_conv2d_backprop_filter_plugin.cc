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
#include "../../op_proto/util/axis_util.h"
#include "error_util.h"
#include "op_log.h"

namespace domi {
Status DepthwiseConv2DBackpropFilterMappingFn(const ge::Operator& op_src, ge::Operator& op) {
  if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(TbeGetName(op), "AutoMappingFn failed.");
    return FAILED;
  }
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_dsc == nullptr) {
    CUBE_INNER_ERR_REPORT_PLUGIN(TbeGetName(op), "GetOpDescFromOperator got nullptr failed.");
    return FAILED;
  }
  auto tensor_desc_w = op_dsc->MutableOutputDesc("filter_grad");
  tensor_desc_w->SetOriginFormat(ge::FORMAT_HWCN);
  tensor_desc_w->SetFormat(ge::FORMAT_HWCN);
  OP_LOGD(TbeGetName(op), "Update output format success.");
  std::vector<int32_t> pad_list{0, 0, 0, 0};
  op.SetAttr("pads", pad_list);
  OP_LOGD(TbeGetName(op), "Update pads success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("DepthwiseConv2DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DepthwiseConv2dNativeBackpropFilter")
    .ParseParamsByOperatorFn(DepthwiseConv2DBackpropFilterMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
