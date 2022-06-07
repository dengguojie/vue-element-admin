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
 * \file conv2d_backprop_filter.cpp
 * \brief
 */
#include <map>
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "register/register.h"
#include "../../op_proto/util/axis_util.h"
#include "error_util.h"

namespace domi {
namespace {
  const int32_t CV_NUM_0 = 0;
}
Status ParseParamsConv2DBackpropFilter(const ge::Operator& op_src, ge::Operator& op) {
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != ge::GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return FAILED);

  OP_LOGD(op_name.GetString(), "Enter ParseParamsConv2DBackpropFilter.");

  AutoMappingByOpFn(op_src, op);

  auto op_dsc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDesc org_tensor_w = op_dsc->GetOutputDesc(CV_NUM_0);
  org_tensor_w.SetOriginFormat(ge::FORMAT_HWCN);
  org_tensor_w.SetFormat(ge::FORMAT_HWCN);
  auto ret = op_dsc->UpdateOutputDesc(CV_NUM_0, org_tensor_w);

  CHECK(ret != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "failed to update output_desc format!"),
        return FAILED);
  OP_LOGD(op_name.GetString(), "update output_desc format succeeded, now is %d", op.GetInputDesc(CV_NUM_0).GetFormat());
  // Escape GE require attr [pads] check here
  std::vector<int32_t> pad_list = {0, 0, 0, 0};
  op.SetAttr("pads", pad_list);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2DBackpropFilter")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Conv2DBackpropFilter")
    .ParseParamsByOperatorFn(ParseParamsConv2DBackpropFilter)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
