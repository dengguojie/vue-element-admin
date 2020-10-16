/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "register/register.h"
#include "op_log.h"

namespace domi {
Status AutoMappingFnBatchMatMul(const google::protobuf::Message* op_src,
                                ge::Operator& op) {
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("BatchMatMul", "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  bool transposeA = false;
  if (op.GetAttr("adj_x", transposeA) != ge::GRAPH_SUCCESS) {
    OP_LOGE("MatMul", "GetAttr adj_x failed");
    return FAILED;
  }
  bool transposeB = false;
  if (op.GetAttr("adj_y", transposeB) != ge::GRAPH_SUCCESS) {
    OP_LOGE("MatMul", "GetAttr adj_y failed");
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
