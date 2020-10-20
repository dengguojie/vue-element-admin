/**
 * Copyright 2018 Huawei Technologies Co., Ltd
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
 * \file conv2d_backprop_filter.cpp
 * \brief
 */
#include <map>
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "register/register.h"

namespace domi {

namespace {
  const int32_t CV_NUM_0 = 0;
}
Status ParseParamsConv2DBackpropFilter(const Message* op_src, ge::Operator& op) {
  OP_LOGI(op.GetName().c_str(), "Enter ParseParamsConv2DBackpropFilter.");

  AutoMappingFn(op_src, op);

  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc org_tensor_w = op_dsc->GetOutputDesc(CV_NUM_0);
  org_tensor_w.SetOriginFormat(ge::FORMAT_HWCN);
  org_tensor_w.SetFormat(ge::FORMAT_HWCN);
  auto ret = op_dsc->UpdateOutputDesc(CV_NUM_0, org_tensor_w);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update filter format failed!");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2dBackpropFilter";
    err_map["param_name"] = "updating output_desc's format";
    err_map["rule_desc"] = "updata output_desc format ";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return FAILED;
  } else {
    OP_LOGI(op.GetName().c_str(), "Update filter format success, now is %d", op.GetInputDesc(CV_NUM_0).GetFormat());
  }

  // Escape GE require attr [pads] check here
  std::vector<int32_t> pad_list = {0, 0, 0, 0};
  op.SetAttr("pads", pad_list);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv2DBackpropFilter")
    .ParseParamsFn(ParseParamsConv2DBackpropFilter)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
