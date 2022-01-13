/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file non_zero_with_value_shape_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"
#include "common/util/error_manager/error_manager.h"
#include "../../op_proto/util/error_util.h"
#include "graph/utils/tensor_utils.h"

#include "op_log.h"

namespace domi {
Status ParseParamsNonZeroWithValueShape(const Message* op_src, ge::Operator& op) {
  if (AutoMappingFn(op_src, op) != SUCCESS) {
      CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "AutoMappingFn failed.");
      return FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
      CUBE_INNER_ERR_REPORT_PLUGIN(op.GetName().c_str(), "GetOpDescFromOperator got nullptr failed.");
      return FAILED;
  }

  const auto output_desc = op_desc->MutableOutputDesc("out_value");
  if (output_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "get output value failed.");
    return FAILED;
  }
  ge::TensorUtils::SetReuseInput(*output_desc, true);
  ge::TensorUtils::SetReuseInputIndex(*output_desc, 0);

  const auto output_desc1 = op_desc->MutableOutputDesc("out_index");
  if (output_desc1 == nullptr) {
    OP_LOGE(op.GetName().c_str(), "get output index failed.");
    return FAILED;
  }
  ge::TensorUtils::SetReuseInput(*output_desc1, true);
  ge::TensorUtils::SetReuseInputIndex(*output_desc1, 1);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("NonZeroWithValueShape")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonZeroWithValueShape")
    .ParseParamsFn(ParseParamsNonZeroWithValueShape)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
