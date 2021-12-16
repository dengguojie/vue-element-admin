/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018. All rights reserved.
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
#include "../../op_proto/util/error_util.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnMatMulV2(const ge::Operator& op_src, ge::Operator& op)
{
  Status ret = AutoMappingByOpFn(op_src, op);
  if (ret != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN("MatMul", "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  bool transposeA = false;
  if (op.GetAttr("transpose_a", transposeA) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN("MatMul", "GetAttr transpose_a failed");
    return FAILED;
  }
  bool transposeB = false;
  if (op.GetAttr("transpose_b", transposeB) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN("MatMul", "GetAttr transpose_b failed");
    return FAILED;
  }

  op.SetAttr("transpose_x1", transposeA);
  op.SetAttr("transpose_x2", transposeB);
  OP_LOGI("MatMul", "op[MatMul] tensorflow plugin parser[AutoMapping] success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("MatMulV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MatMul")
    .ParseParamsByOperatorFn(AutoMappingFnMatMulV2)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
