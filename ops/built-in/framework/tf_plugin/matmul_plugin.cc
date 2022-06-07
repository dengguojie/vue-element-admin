/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2021. All rights reserved.
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
 * \file matmul_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "common/util/error_manager/error_manager.h"
#include "error_util.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnMatMulV2(const ge::Operator& op_src, ge::Operator& op)
{
  Status ret = AutoMappingByOpFn(op_src, op);
  if (ret != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(TbeGetName(op), "Tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  bool transpose_a = false;
  if (op.GetAttr("transpose_a", transpose_a) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(TbeGetName(op), "GetAttr transpose_a failed");
    return FAILED;
  }
  bool transpose_b = false;
  if (op.GetAttr("transpose_b", transpose_b) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(TbeGetName(op), "GetAttr transpose_b failed");
    return FAILED;
  }

  op.SetAttr("transpose_x1", transpose_a);
  op.SetAttr("transpose_x2", transpose_b);
  OP_LOGD(TbeGetName(op), "Tensorflow plugin parser[AutoMapping] success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("MatMulV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatMul")
    .ParseParamsByOperatorFn(AutoMappingFnMatMulV2)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
