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
namespace {
constexpr size_t INPUT_INDEX_OFFSET = 2;
constexpr size_t OFFSET_NUM_DIMS = 2;
constexpr size_t INDEX_BEAM_NUM = 0;
constexpr size_t INDEX_FRAME_NUM = 1;
}  // namespace

namespace ops {
ge::graphStatus InferShapeForMovingSumWithSigmoid(gert::InferShapeContext *context) {
  auto offset_tensor = context->GetInputTensor(INPUT_INDEX_OFFSET);
  OPS_CHECK_NULL_WITH_CONTEXT(context, offset_tensor);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  if (offset_tensor->GetDataType() != ge::DT_INT32) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "offset_tensor datatype must be int32!");
    return ge::GRAPH_FAILED;
  }
  const int32_t* offset_data = offset_tensor->GetData<int32_t>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, offset_data);
  int64_t beam_sum = 0;
  int64_t frame_sum = 0;
  int64_t batch_size = offset_tensor->GetShapeSize() / OFFSET_NUM_DIMS;
  for (int64_t i = 0; i < batch_size; i++) {
    beam_sum += offset_data[i];
    frame_sum += offset_data[i + batch_size];
  }

  out_shape->SetDimNum(OFFSET_NUM_DIMS);
  out_shape->SetDim(INDEX_BEAM_NUM, beam_sum);
  out_shape->SetDim(INDEX_FRAME_NUM, frame_sum);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(MovingSumWithSigmoid)
    .InferShape(InferShapeForMovingSumWithSigmoid)
    .InputsDataDependency({2});
}  // namespace ops
