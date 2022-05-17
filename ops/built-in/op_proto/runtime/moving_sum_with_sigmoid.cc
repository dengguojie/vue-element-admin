/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "runtime_util.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_INDEX_OFFSET = 2;
constexpr int64_t TWICE = 2;

ge::graphStatus InferShapeForMovingSumWithSigmoid(gert::InferShapeContext *context) {
  auto offset_tensor = context->GetInputTensor(INPUT_INDEX_OFFSET);
  OPS_CHECK_NULL_WITH_CONTEXT(context, offset_tensor);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  auto offset_size = static_cast<int32_t>(offset_tensor->GetShapeSize());
  if (offset_size < 1) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "offset_size is invalic!");
    return ge::GRAPH_FAILED;
  }
  if (offset_tensor->GetDataType() != ge::DT_INT32) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "offset_tensor datatype must be int32!");
    return ge::GRAPH_FAILED;
  }
  int64_t batch_size = offset_size / TWICE;
  int64_t beam_sum = 0;
  int64_t frame_sum = 0;
  const int32_t* offset_data = offset_tensor->GetData<int32_t>();
  for (int64_t i = 0; i < batch_size; i++) {
    beam_sum += offset_data[i];
    frame_sum += offset_data[i + batch_size];
  }
  out_shape->AppendDim(beam_sum);
  out_shape->AppendDim(frame_sum);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(MovingSumWithSigmoid)
    .InferShape(InferShapeForMovingSumWithSigmoid)
    .InputsDataDependency({2});
}  // namespace ops
