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
 * \file avg_pool3d_grad_plugin.cpp
 * \brief
 */
#include <map>
#include <iostream>

#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "../../op_proto/util/axis_util.h"
#include "../../op_proto/util/error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "op_log.h"

namespace domi {

namespace {
  const int32_t kIndex0 = 0;
  const int32_t kIndex1 = 1;
  const size_t kKsizeLength = 5;
  const size_t kStridesLength = 5;
}

// Replace ge ParseParams fuction to process graph avgpool3dgrad node attrs
Status ParseParamsAvgPool3dGrad(const Message* op_src, ge::Operator& op) {
  OP_LOGD(op.GetName().c_str(), "Enter Parse Params AvgPool3dGrad.");
  auto res = AutoMappingFn(op_src, op);
  CHECK(res != SUCCESS,
        CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "plugin parser failed. auto mapping failed."),
        return FAILED);

  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_dsc == nullptr,
        CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "op_dsc is null. auto mapping failed."),
        return FAILED);
  ge::Format data_format = ge::FORMAT_NDHWC;
  std::string data_format_attr;
  if (op.GetAttr("data_format", data_format_attr) == ge::GRAPH_SUCCESS) {
    CHECK(data_format_attr != "NCDHW" && data_format_attr != "NDHWC",
           CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "data_format only support NCDHW and NDHWC."),
           return FAILED);
    data_format = data_format_attr == "NCDHW" ? ge::FORMAT_NCDHW: ge::FORMAT_NDHWC;
  }
  ge::GeTensorDesc org_tensor_grads = op_dsc->GetInputDesc(kIndex1);
  org_tensor_grads.SetOriginFormat(data_format);
  org_tensor_grads.SetFormat(data_format);

  auto ret = op_dsc->UpdateInputDesc(kIndex0, org_tensor_grads);
  CHECK(ret != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "update input:grads desc failed."),
        return FAILED);

  ge::GeTensorDesc org_tensor_out = op_dsc->GetOutputDesc(kIndex0);
  org_tensor_out.SetOriginFormat(data_format);
  org_tensor_out.SetFormat(data_format);
  ret = op_dsc->UpdateOutputDesc(kIndex0, org_tensor_out);
  CHECK(ret != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "update output desc failed."),
        return FAILED);

  std::vector<int64_t> ksize;
  CHECK(op.GetAttr("ksize", ksize) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "Get ksize attr failed."),
        return FAILED);
  CHECK(ksize.size() != kKsizeLength,
        CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "Ksize has an incorrected length."),
        return FAILED);

  vector<int64_t> strides;
  CHECK(op.GetAttr("strides", strides) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "Get strides attr failed."),
        return FAILED);
  CHECK(strides.size() != kStridesLength,
        CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "Strides has an incorrected length."),
        return FAILED);

  std::string padding = "";
  CHECK(op.GetAttr("padding", padding) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "Get padding attr failed."),
        return FAILED);
  CHECK(padding != "SAME" && padding != "VALID",
        CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "TF padding pattern is incorrected."),
        return FAILED);

  std::vector<int64_t> pads = {0,0,0,0,0,0};
  op.SetAttr("pads", pads);
  op.SetAttr("ceil_mode", false);
  op.SetAttr("count_include_pad", false);
  op.SetAttr("divisor_override", 0);

  OP_LOGI(op.GetName().c_str(), "Exit Parse Params AvgPool3dGrad.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("AvgPool3DGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AvgPool3DGrad")
    .ParseParamsFn(ParseParamsAvgPool3dGrad)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
