/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file functional_ops.cpp
 * \brief
 */
#include "inc/functional_ops.h"
#include "common/inc/op_log.h"
#include "common_shape_fns.h"
#include "util/error_util.h"
#include "util/util.h"

namespace ge {
namespace {
graphStatus VerifyInt32Scalar(Operator& op, const std::vector<std::string>& input_names) {
  for (const std::string& name : input_names) {
    auto dims = op.GetInputDesc(name).GetShape().GetDims();
    if (dims.size() != 0) {
      string reason = "input " + name + " should be a scalar, actually rank=" + std::to_string(dims.size());
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), name + "dims", reason);
      REPORT_INNER_ERROR("E19999", "[Node:%s] Check shape rank failed, as %s", op.GetName().c_str(), reason.c_str());
      GE_OP_LOGE(op.GetName().c_str(), "[Verify][Check] Check shape rank failed, as %s", reason.c_str());
      return GRAPH_FAILED;
    }
    DataType type = op.GetInputDesc(name).GetDataType();
    if (type != DT_INT32) {
      string reason = "input " + name + " should be DT_INT32, actually is " + DataTypeToStringDesc(type);
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", reason);
      REPORT_INNER_ERROR("E19999", "[Node:%s] Check dtype failed, as %s", op.GetName().c_str(), reason.c_str());
      GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Check] Check dtype failed, as %s", reason.c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace

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
}  // namespace ge
