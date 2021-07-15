/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file conv3d_backprop_filter_plugin.cpp
 * \brief
 */
#include <map>

#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "../../op_proto/util/util.h"

#include "op_log.h"

namespace domi {

namespace {
  const int32_t kInputIdx0 = 0;
  const int32_t kInputIdx1 = 1;
  const int32_t kOutputIdx0 = 0;
}

Status ParseParamsConv3DBackpropFilter(const Message* op_src, ge::Operator& op) {
  OP_LOGI(op.GetName().c_str(), "Enter ParseParamsConv3DBackpropFilter.");
  AutoMappingFn(op_src, op);

  ge::Format data_format = ge::FORMAT_NDHWC;
  std::string data_format_attr;
  if (op.GetAttr("data_format", data_format_attr) == ge::GRAPH_SUCCESS) {
    if (data_format_attr == "NCDHW") {
      data_format = ge::FORMAT_NCDHW;
    }
  }

  auto opDsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(opDsc, "op desc", return FAILED);
  ge::GeTensorDesc org_tensor_x = opDsc->GetInputDesc(kInputIdx0);
  org_tensor_x.SetOriginFormat(data_format);
  org_tensor_x.SetFormat(data_format);
  auto ret = opDsc->UpdateInputDesc(kInputIdx0, org_tensor_x);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update input_x format failed.");
    return FAILED;
  }

  ge::GeTensorDesc org_tensor_y = opDsc->GetInputDesc(kInputIdx1);
  org_tensor_y.SetOriginFormat(data_format);
  org_tensor_y.SetFormat(data_format);
  ret = opDsc->UpdateInputDesc(kInputIdx1, org_tensor_y);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update out_backprop format failed.");
    return FAILED;
  }

  ge::GeTensorDesc org_tensor_w = opDsc->GetOutputDesc(kOutputIdx0);
  org_tensor_w.SetOriginFormat(ge::FORMAT_DHWCN);
  org_tensor_w.SetFormat(ge::FORMAT_DHWCN);
  ret = opDsc->UpdateOutputDesc(kInputIdx0, org_tensor_w);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update output dw format failed.");
    return FAILED;
  }
  std::vector<int32_t> pad_list = {0, 0, 0, 0, 0, 0};
  op.SetAttr("pads", pad_list);

  OP_LOGI(op.GetName().c_str(), "update output dw format success.");

  OP_LOGI(op.GetName().c_str(), "Exit ParseParamsConv3DBackpropFilter.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv3DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv3DBackpropFilterV2")
    .ParseParamsFn(ParseParamsConv3DBackpropFilter)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
