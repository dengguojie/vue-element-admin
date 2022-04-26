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
const std::vector<int64_t> DUMMY_SHAPE = {-3};
const graphStatus GRAPH_NODE_NEED_REPASS = 50331647; // current can not update submodule, so define here
graphStatus VerifyInt32Scalar(Operator& op, const std::vector<std::string>& input_names) {
  for (const std::string& name : input_names) {
    auto dims = op.GetInputDesc(name).GetShape().GetDims();
    if (dims.size() != 0) {
      string reason = "input " + name + " should be a scalar, actually rank=" + std::to_string(dims.size());
      REPORT_INNER_ERROR("E19999", "[Node:%s] Check shape rank failed, as %s", TbeGetName(op).c_str(), reason.c_str());
      GE_OP_LOGE(TbeGetName(op).c_str(), "[Verify][Check] Check shape rank failed, as %s", reason.c_str());
      return GRAPH_FAILED;
    }
    DataType type = op.GetInputDesc(name).GetDataType();
    if (type != DT_INT32) {
      string reason = "input " + name + " should be DT_INT32, actually is " + DataTypeToStringDesc(type);
      REPORT_INNER_ERROR("E19999", "[Node:%s] Check dtype failed, as %s", TbeGetName(op).c_str(), reason.c_str());
      GE_OP_LOGE(TbeGetName(op).c_str(), "[InferShape][Check] Check dtype failed, as %s", reason.c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}
graphStatus WhileInferImpl(Operator &op) {
  size_t in_num = op.GetInputsSize();
  size_t out_num = op.GetOutputsSize();
  GE_OP_LOGD(TbeGetName(op).c_str(), "Begin to infer while node shape, input size %zu, output size %zu", in_num, out_num);
  if (in_num != out_num) {
    string reason = "input num not equal with out num.";
    REPORT_INNER_ERROR("E19999",
                       "[Node:%s] Check input num and output num failed, as %s",
                       TbeGetName(op).c_str(),
                       reason.c_str());
    GE_OP_LOGE(TbeGetName(op).c_str(), "[InferShape][Check] Check input num failed, as %s", reason.c_str());
    return GRAPH_FAILED;
  }
  bool need_infer_again = false;
  for (size_t i = 0; i < in_num; ++i) {
    auto in_desc = op.GetDynamicInputDesc("input", i);
    auto out_desc = op.GetDynamicOutputDesc("output", i);
    auto data_shape = in_desc.GetShape();
    auto out_shape = out_desc.GetShape();
    if(out_shape.GetDims() == DUMMY_SHAPE){
      GE_OP_LOGI(TbeGetName(op).c_str(), "First time to infer while node shape, no need update from output to input.");
      return GRAPH_SUCCESS;
    }
    // check datatype between output and input
    if (in_desc.GetDataType() != out_desc.GetDataType()) {
      REPORT_INNER_ERROR("E19999",
                         "node[%s] does not support diff dtype or format among all ref output. src datatype :%d, "
                         "dst datatype: %d",
                         TbeGetName(op).c_str(), in_desc.GetDataType(), out_desc.GetDataType());
      GE_OP_LOGE(TbeGetName(op).c_str(),
                 "[Check][Param] node does not support diff dtype or format output. src datatype :%d,"
                 "dst datatype: %d", in_desc.GetDataType(), out_desc.GetDataType());
      return GRAPH_FAILED;
    }

    if (data_shape.GetDims() != out_shape.GetDims()) {
      GE_OP_LOGI(TbeGetName(op).c_str(), "While %zu output shape is not match with input shape.Need infer again.", i);
      if (data_shape.GetDimNum() != out_shape.GetDimNum()) {
        in_desc.SetUnknownDimNumShape();
      } else {
        size_t data_dim_num = data_shape.GetDimNum();
        std::vector<std::pair<int64_t, int64_t>> data_shape_range = {data_dim_num, std::make_pair(1, UNKNOWN_DIM)};
        for (size_t j = 0; j < data_dim_num; ++j) {
          if (data_shape.GetDim(j) != out_shape.GetDim(j)) {
            if (data_shape.GetDim(j) != UNKNOWN_DIM) {
               // if input data is fix shape, output is different, need_infer_again
               need_infer_again = true;
            }
            data_shape.SetDim(j, UNKNOWN_DIM);
          }
          if (data_shape.GetDim(j) != UNKNOWN_DIM) {
            data_shape_range[j] = std::make_pair(data_shape.GetDim(j), data_shape.GetDim(j));
          }
        }
        in_desc.SetShape(data_shape);
        in_desc.SetShapeRange(data_shape_range);
      }
      op.UpdateDynamicOutputDesc("output", i, in_desc);
      op.UpdateDynamicInputDesc("input", i, in_desc);
    }
  }
  return need_infer_again ? GRAPH_NODE_NEED_REPASS : GRAPH_SUCCESS;
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
  return WhileInferImpl(op);
}
INFER_FUNC_REG(_While, _WhileInfer);
IMPLEMT_VERIFIER(_While, _WhileVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(_While, _WhileVerify);

IMPLEMT_INFERFUNC(While, WhileInfer) {
  return WhileInferImpl(op);
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
