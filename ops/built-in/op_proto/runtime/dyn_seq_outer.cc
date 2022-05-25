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
/*!
 * \file dyn_seq_outer.cc
 * \brief
 */

#include "op_log.h"
#include "runtime_util.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace ge;
namespace ops {
ge::graphStatus DynSeqOuterInferShapeFunc(gert::InferShapeContext *context) {
  auto output_shape = context->GetOutputShape(0);
  auto seq_len1_tensor = context->GetInputTensor(2);
  auto seq_len2_tensor = context->GetInputTensor(3);

  int64_t batch_size = seq_len1_tensor->GetShapeSize();  

  if (seq_len1_tensor->GetDataType() != ge::DT_INT32 || seq_len2_tensor->GetDataType() != ge::DT_INT32) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "offset_tensor dtype must be int32!");
    return ge::GRAPH_FAILED;
  }

  const int32_t* seq_len1 = seq_len1_tensor->GetData<int32_t>();
  const int32_t* seq_len2 = seq_len2_tensor->GetData<int32_t>();

  int64_t bst = 0;
  for (int64_t i = 0; i < batch_size; i++) {
    bst += seq_len1[i] * seq_len2[i];
  }

  output_shape->AppendDim(bst);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(DynSeqOuter).InputsDataDependency({2, 3}).InferShape(DynSeqOuterInferShapeFunc);
}  // namespace ops