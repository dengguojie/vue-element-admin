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
 * \file batch_matmul_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "common/util/error_manager/error_manager.h"
#include "op_log.h"
#include "../../op_proto/util/axis_util.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
Status AutoMappingFnBatchMatMul(const ge::Operator& op_src, ge::Operator& op)
{
  ge::AscendString op_name;
  CHECK(op.GetName(op_name) != ge::GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return FAILED);

  Status ret = AutoMappingByOpFn(op_src, op);
  if (ret != SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "tensorflow plugin parser failed.");
    return FAILED;
  }
  bool transposeA = false;
  if (op.GetAttr("adj_x", transposeA) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "GetAttr adj_x failed.");
    return FAILED;
  }
  bool transposeB = false;
  if (op.GetAttr("adj_y", transposeB) != ge::GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op_name.GetString(), "GetAttr adj_y failed.");
    return FAILED;
  }
  op.SetAttr("adj_x1", transposeA);
  op.SetAttr("adj_x2", transposeB);
  OP_LOGI(op_name.GetString(), "op[BatchMatMul] tensorflow plugin parser[AutoMapping] success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("BatchMatMul")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"BatchMatMul", "BatchMatMulV2"})
    .ParseParamsByOperatorFn(AutoMappingFnBatchMatMul)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
