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

using namespace ge;
namespace domi {
Status AutoMappingFnMatMulV2(const google::protobuf::Message* op_src,
                           ge::Operator& op) {
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE("MatMul", "tensorflow plugin parser failed. auto mapping failed.");
    return FAILED;
  }
  bool transposeA = false;
  if (op.GetAttr("transpose_a", transposeA) != ge::GRAPH_SUCCESS) {
    OP_LOGE("MatMul", "GetAttr transpose_a failed");
    return FAILED;
  }
  bool transposeB = false;
  if (op.GetAttr("transpose_b", transposeB) != ge::GRAPH_SUCCESS) {
    OP_LOGE("MatMul", "GetAttr transpose_b failed");
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
    .ParseParamsFn(AutoMappingFnMatMulV2)
    .ImplyType(ImplyType::TVM);
}
