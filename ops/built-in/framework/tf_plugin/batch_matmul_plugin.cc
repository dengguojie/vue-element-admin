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
 * \file batch_matmul_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "common/util/error_manager/error_manager.h"
#include "../../op_proto/util/error_util.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnBatchMatMul(const google::protobuf::Message* op_src, ge::Operator& op) {
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("BatchMatMul", "tensorflow plugin parser failed. auto mapping failed.");
    ErrorManager::GetInstance().ATCReportErrMessage("E50058",
                                                    {"op_name", "description"},
                                                    {"BatchMatMul", "tensorflow plugin parser failed"});
    return FAILED;
  }
  bool transposeA = false;
  if (op.GetAttr("adj_x", transposeA) != ge::GRAPH_SUCCESS) {
    OP_LOGE("MatMul", "GetAttr adj_x failed");
    ErrorManager::GetInstance().ATCReportErrMessage("E50058",
                                                    {"op_name", "description"},
                                                    {"BatchMatMul", "GetAttr adj_x failed"});
    return FAILED;
  }
  bool transposeB = false;
  if (op.GetAttr("adj_y", transposeB) != ge::GRAPH_SUCCESS) {
    OP_LOGE("MatMul", "GetAttr adj_y failed");
    ErrorManager::GetInstance().ATCReportErrMessage("E50058",
                                                    {"op_name", "description"},
                                                    {"BatchMatMul", "GetAttr adj_y failed"});
    return FAILED;
  }

  op.SetAttr("adj_x1", transposeA);
  op.SetAttr("adj_x2", transposeB);
  OP_LOGI("BatchMatMul", "op[BatchMatMul] tensorflow plugin parser[AutoMapping] success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("BatchMatMul")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"BatchMatMul", "BatchMatMulV2"})
    .ParseParamsFn(AutoMappingFnBatchMatMul)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
