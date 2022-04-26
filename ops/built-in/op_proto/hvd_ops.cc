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
 * \file hvd_ops.cpp
 * \brief
 */
#include "inc/hvd_ops.h"

#include <string>
#include <vector>
#include <algorithm>

#include "op_log.h"

namespace ge {

bool CheckSupportDateTpye(ge::DataType type) {
  const std::vector<ge::DataType> SUPPORTED_TYPE = {DT_INT32, DT_FLOAT16, DT_FLOAT, DT_INT8};
  auto it = std::find(SUPPORTED_TYPE.begin(), SUPPORTED_TYPE.end(), type);
  if (it == SUPPORTED_TYPE.end()) {
    return false;
  } else {
    return true;
  }
}
// HorovodAllgather op
IMPLEMT_INFERFUNC(HorovodAllgather, HorovodAllgatherInferShape) {
  auto inTensorDesc = op.get_input_desc_x();
  auto outTensorDesc = inTensorDesc;
  auto inShape = inTensorDesc.GetShape();
  std::vector<int64_t> inDims = inShape.GetDims();
  int64_t rankSize = op.get_attr_rank_size();
  std::vector<int64_t> outDims;
  if (rankSize <= 0) {
    OP_LOGE(TbeGetName(op).c_str(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
    return GRAPH_FAILED;
  }
  if (!(inDims.size() > 0)) {
    OP_LOGE(TbeGetName(op).c_str(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  outDims = inDims;
  if (!(outDims.size() > 0)) {
    OP_LOGE(TbeGetName(op).c_str(), "out tensor is empty, expected: > 0, actual: %zu.", outDims.size());
    return GRAPH_FAILED;
  }
  outDims[0] = inDims[0] * rankSize;
  ge::Shape outputShape = ge::Shape(outDims);
  ge::DataType outputDtype = inTensorDesc.GetDataType();
  outTensorDesc.SetShape(outputShape);
  outTensorDesc.SetDataType(outputDtype);
  op.update_output_desc_y(outTensorDesc);
  OP_LOGI(TbeGetName(op).c_str(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HorovodAllgather, HorovodAllgatherVerify) {
  std::vector<int64_t> inDims = op.get_input_desc_x().GetShape().GetDims();
  int64_t rankSize = op.get_attr_rank_size();
  if (rankSize <= 0) {
    OP_LOGE(TbeGetName(op).c_str(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
    return GRAPH_FAILED;
  }
  if (inDims.size() == 0) {
    OP_LOGE(TbeGetName(op).c_str(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  // check supported data type in HCCL
  ge::DataType inputDtype = op.get_input_desc_x().GetDataType();
  if (!CheckSupportDateTpye(inputDtype)) {
    OP_LOGE(TbeGetName(op).c_str(), "dataType [%d] is not supported in HCCL.", inputDtype);
  }
  OP_LOGI(TbeGetName(op).c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HorovodAllgather, HorovodAllgatherInferShape);
VERIFY_FUNC_REG(HorovodAllgather, HorovodAllgatherVerify);

// HorovodAllreduce op
IMPLEMT_VERIFIER(HorovodAllreduce, HorovodAllreduceVerify) {
  // check supported reduce op in HCCL
  int reduction = op.get_attr_reduce_op();
  OP_LOGI(TbeGetName(op).c_str(), "reduce type is [%d]", reduction);
  const std::vector<int> SUPPORTED_REDUCTION = {
      1, 3, 4, 5  // corresponding sum, min, max, prod
  };
  auto it = std::find(SUPPORTED_REDUCTION.begin(), SUPPORTED_REDUCTION.end(), reduction);
  if (it == SUPPORTED_REDUCTION.end()) {
    OP_LOGE(TbeGetName(op).c_str(), "Attr reduction [%d] is not supported. expecttd: min, max, prod, sum", reduction);
    return GRAPH_FAILED;
  }
  // check supported data type in HCCL
  ge::DataType inputDtype = op.get_input_desc_x().GetDataType();
  if (!CheckSupportDateTpye(inputDtype)) {
    OP_LOGE(TbeGetName(op).c_str(), "dataType [%d] is not supported in HCCL.", inputDtype);
  }
  OP_LOGI(TbeGetName(op).c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(HorovodAllreduce, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
VERIFY_FUNC_REG(HorovodAllreduce, HorovodAllreduceVerify);

IMPLEMT_VERIFIER(HorovodBroadcast, HorovodBroadcastVerify) {
  OP_LOGI(TbeGetName(op).c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(HorovodBroadcast, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
VERIFY_FUNC_REG(HorovodBroadcast, HorovodBroadcastVerify);
}  // namespace ge
