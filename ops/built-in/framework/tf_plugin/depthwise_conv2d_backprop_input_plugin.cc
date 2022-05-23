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
 * \file depthwise_conv2d_backprop_input_plugin.cpp
 * \brief
 */
#include "graph/utils/op_desc_utils.h"
#include "register/register.h"
#include "common/util/error_manager/error_manager.h"
#include "../../op_proto/util/axis_util.h"
#include "error_util.h"
#include "op_log.h"

namespace domi {
Status DepthwiseConv2DBackpropInputMappingFn(const ge::Operator& op_src, ge::Operator& op) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != ge::GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return FAILED);

  if (AutoMappingByOpFn(op_src, op) != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "AutoMappingFn failed.");
    return FAILED;
  }
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_dsc == nullptr, CUBE_CALL_ERR_REPORT(op_name.GetString(), "GetOpDescFromOperator got nullptr failed."),
        return FAILED);
  auto tensorDescW = op_dsc->MutableInputDesc("filter");
  tensorDescW->SetOriginFormat(ge::FORMAT_HWCN);
  tensorDescW->SetFormat(ge::FORMAT_HWCN);
  OP_LOGD(op_name.GetString(), "update filter format success.");
  std::vector<int32_t> padList = {0, 0, 0, 0};
  op.SetAttr("pads", padList);
  OP_LOGD(op_name.GetString(), "update pads success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("DepthwiseConv2DBackpropInput")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DepthwiseConv2dNativeBackpropInput")
    .ParseParamsByOperatorFn(DepthwiseConv2DBackpropInputMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
