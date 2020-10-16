/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file  functional_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/functional_ops.h"
#include "op_log.h"
#include "common_shape_fns.h"

namespace ge {
namespace {
  graphStatus VerifyInt32Scalar(Operator &op, const std::vector<std::string> &input_names) {
    for (const std::string &name : input_names) {
      auto dims = op.GetInputDesc(name).GetShape().GetDims();
      if (dims.size() != 0) {
        OP_LOGE(op.GetName().c_str(), "input %s should be a scalar, actually size=%lu", name.c_str(), dims.size());
        return GRAPH_FAILED;
      }
      DataType type = op.GetInputDesc(name).GetDataType();
      if (type != DT_INT32) {
        OP_LOGE(op.GetName().c_str(), "input %s should be int32 type, actually type=%u", name.c_str(), type);
        return GRAPH_FAILED;
      }
    }
    return GRAPH_SUCCESS;
  }
}

IMPLEMT_INFERFUNC(SymbolicGradient, SymbolicGradientInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(SymbolicGradient, SymbolicGradientInfer);
IMPLEMT_VERIFIER(SymbolicGradient, SymbolicGradientVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(SymbolicGradient, SymbolicGradientVerify);

IMPLEMT_INFERFUNC(RemoteCall, RemoteCallInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(RemoteCall, RemoteCallInfer);
IMPLEMT_VERIFIER(RemoteCall, RemoteCallVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(RemoteCall, RemoteCallVerify);

IMPLEMT_INFERFUNC(_If, _IfInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(_If, _IfInfer);
IMPLEMT_VERIFIER(_If, _IfVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(_If, _IfVerify);

IMPLEMT_INFERFUNC(StatelessIf, StatelessIfInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(StatelessIf, StatelessIfInfer);
IMPLEMT_VERIFIER(StatelessIf, StatelessIfVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(StatelessIf, StatelessIfVerify);

IMPLEMT_INFERFUNC(If, IfInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(If, IfInfer);
IMPLEMT_VERIFIER(If, IfVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(If, IfVerify);

IMPLEMT_INFERFUNC(Case, CaseInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(Case, CaseInfer);
IMPLEMT_VERIFIER(Case, CaseVerify) {
  return VerifyInt32Scalar(op, {"branch_index"});
}
VERIFY_FUNC_REG(Case, CaseVerify);

IMPLEMT_INFERFUNC(_While, _WhileInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(_While, _WhileInfer);
IMPLEMT_VERIFIER(_While, _WhileVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(_While, _WhileVerify);

IMPLEMT_INFERFUNC(While, WhileInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(While, WhileInfer);
IMPLEMT_VERIFIER(While, WhileVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(While, WhileVerify);

IMPLEMT_INFERFUNC(StatelessWhile, StatelessWhileInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(StatelessWhile, StatelessWhileInfer);
IMPLEMT_VERIFIER(StatelessWhile, StatelessWhileVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(StatelessWhile, StatelessWhileVerify);

IMPLEMT_INFERFUNC(For, ForInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(For, ForInfer);
IMPLEMT_VERIFIER(For, ForVerify) {
  return VerifyInt32Scalar(op, {"start", "limit", "delta"});
}
VERIFY_FUNC_REG(For, ForVerify);

IMPLEMT_INFERFUNC(PartitionedCall, PartitionedCallInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(PartitionedCall, PartitionedCallInfer);
IMPLEMT_VERIFIER(PartitionedCall, PartitionedCallVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(PartitionedCall, PartitionedCallVerify);

IMPLEMT_INFERFUNC(StatefulPartitionedCall, StatefulPartitionedCallInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(StatefulPartitionedCall, StatefulPartitionedCallInfer);
IMPLEMT_VERIFIER(StatefulPartitionedCall, StatefulPartitionedCallVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(StatefulPartitionedCall, StatefulPartitionedCallVerify);

IMPLEMT_INFERFUNC(FakeParam, FakeParamInfer) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(FakeParam, FakeParamInfer);
IMPLEMT_VERIFIER(FakeParam, FakeParamVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(FakeParam, FakeParamVerify);
}  // namespace ge

