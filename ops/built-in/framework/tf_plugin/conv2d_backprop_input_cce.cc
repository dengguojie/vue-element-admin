/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file conv2d_backprop_input_cce.cpp
 * \brief
 */
#include <map>
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "register/register.h"
#include "../../op_proto/util/error_util.h"
#include "../../op_proto/util/axis_util.h"
#include "../../op_proto/util/util.h"

namespace domi {
namespace {
  const int32_t CV_NUM_1 = 1;
}
Status ParseParamsConv2DBackpropInput(const ge::Operator& op_src, ge::Operator& op) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != ge::GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return FAILED);

  OP_LOGI(op_name.GetString(), "Enter ParseParamsConv2DBackpropInput.");

  AutoMappingByOpFn(op_src, op);

  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_dsc, "op desc", return FAILED);
  ge::GeTensorDesc org_tensor_w = op_dsc->GetInputDesc(CV_NUM_1);
  org_tensor_w.SetOriginFormat(ge::FORMAT_HWCN);
  org_tensor_w.SetFormat(ge::FORMAT_HWCN);
  auto ret = op_dsc->UpdateInputDesc(CV_NUM_1, org_tensor_w);
  if (ret != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "updating filter's format failed.");
    return FAILED;
  } else {
    OP_LOGI(op_name.GetString(), "Update filter format success, now is %d", op.GetInputDesc(CV_NUM_1).GetFormat());
  }

  // Escape GE require attr [pads] check here
  std::vector<int32_t> pad_list = {0, 0, 0, 0};
  op.SetAttr("pads", pad_list);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2DBackpropInput")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv2DBackpropInput")
    .ParseParamsByOperatorFn(ParseParamsConv2DBackpropInput)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
