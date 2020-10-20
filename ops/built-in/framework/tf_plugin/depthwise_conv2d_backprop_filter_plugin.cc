/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "op_log.h"

namespace domi {

Status DepthwiseConv2DBackpropFilterMappingFn(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc tensorDescW = op_dsc->GetOutputDesc(0);
  tensorDescW.SetOriginFormat(ge::FORMAT_HWCN);
  tensorDescW.SetFormat(ge::FORMAT_HWCN);
  auto ret = op_dsc->UpdateOutputDesc(0, tensorDescW);
  if (ret != ge::GRAPH_SUCCESS) {
    map<string, string> err_map;
    err_map["op_name"] = op.GetName().c_str();
    err_map["param_name"] = "updating filter's format";
    err_map["rule_desc"] = "update filter's format";
    err_map["format"] = "failed";
    std::string report_error_code = "E50012";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    OP_LOGE("[Plugin][ERROR]update filter_grad format failed\n");
    return FAILED;
  }
  std::vector<int32_t> padList{0, 0, 0, 0};
  op.SetAttr("pads", padList);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("DepthwiseConv2DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DepthwiseConv2dNativeBackpropFilter")
    .ParseParamsFn(DepthwiseConv2DBackpropFilterMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
