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
 * \file conv3d_backprop_input_plugin.cpp
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
  const int32_t kIndex0 = 0;
  const int32_t kIndex1 = 1;
}

Status ParseParamsConv3DBackpropInput(const Message* op_src, ge::Operator& op) {
  OP_LOGI(op.GetName().c_str(), "Enter ParseParamsConv3DBackpropInput.");

  AutoMappingFn(op_src, op);
  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_dsc, "op desc", return FAILED);
  ge::GeTensorDesc org_tensor_w = op_dsc->GetInputDesc(kIndex1);
  org_tensor_w.SetOriginFormat(ge::FORMAT_DHWCN);
  org_tensor_w.SetFormat(ge::FORMAT_DHWCN);
  auto ret = op_dsc->UpdateInputDesc(kIndex1, org_tensor_w);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update filter format failed.");
    return FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "Update filter format success");

  ge::Format data_format = ge::FORMAT_NDHWC;
  std::string data_format_attr;
  if (op.GetAttr("data_format", data_format_attr) == ge::GRAPH_SUCCESS) {
    if (data_format_attr == "NCDHW") {
      data_format = ge::FORMAT_NCDHW;
    }
  }

  ge::GeTensorDesc org_tensor_y = op_dsc->GetInputDesc(kIndex0);
  org_tensor_y.SetOriginFormat(data_format);
  org_tensor_y.SetFormat(data_format);
  ret = op_dsc->UpdateInputDesc(kIndex0, org_tensor_y);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update inout out_backprop format failed.");
    return FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "update inout out_backprop format success.");

  ge::GeTensorDesc org_tensor_x = op_dsc->GetOutputDesc(kIndex0);
  org_tensor_x.SetOriginFormat(data_format);
  org_tensor_x.SetFormat(data_format);
  ret = op_dsc->UpdateOutputDesc(kIndex0, org_tensor_x);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output dx format failed.");
    return FAILED;
  }

  std::vector<int32_t> pad_list = {0, 0, 0, 0, 0, 0};
  op.SetAttr("pads", pad_list);

  OP_LOGI(op.GetName().c_str(), "update output dx format success.");

  OP_LOGI(op.GetName().c_str(), "Exit ParseParamsConv3DBackpropInput.");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv3DBackpropInput")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv3DBackpropInputV2")
    .ParseParamsFn(ParseParamsConv3DBackpropInput)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
