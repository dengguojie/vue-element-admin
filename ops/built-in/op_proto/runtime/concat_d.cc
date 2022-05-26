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
#include "op_util.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_CONCAT_DIM = 0;
constexpr size_t INDEX_N = 1;

ge::graphStatus ConcatInferShapeCommon(gert::InferShapeContext* context, const int64_t dynamic_input_idx,
                                       int64_t num_concat, int64_t axis) {
  auto in_shape = context->GetDynamicInputShape(dynamic_input_idx, 0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  *out_shape = *in_shape;

  if (num_concat == 1) {
    // dynamic case or the input only one will use dynamic infer func
    return ge::GRAPH_SUCCESS;
  }

  if (out_shape->IsScalar()) {
    // scalar to shape [1]
    out_shape->AppendDim(1);
  }

  size_t output_dim = out_shape->GetDimNum();
  OP_CHECK(
      !IsDimValid(output_dim, axis),
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), GenInvalidDimMsg("concat_dim", output_dim, axis)),
      return ge::GRAPH_FAILED);
  if (axis < 0) {
    axis += output_dim;
  }

  int64_t concat_dim_size = out_shape->GetDim(axis);
  for (int64_t relative_index = 1; relative_index < num_concat; relative_index++) {
    const gert::Shape* input_i_shape = context->GetDynamicInputShape(dynamic_input_idx, relative_index);
    if (input_i_shape->IsScalar() && output_dim == 1) {
      concat_dim_size += 1;
      continue;
    }
    if (input_i_shape->GetDimNum() != output_dim) {
      // input shape size is not equal output
      std::string msg = ConcatString("shape[", relative_index, "].GetDimNum ", input_i_shape->GetDimNum(),
                                     ", must be equal to ", output_dim, "!");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), msg);
      return ge::GRAPH_FAILED;
    }
    // check whether the non concat dim is equal
    for (int64_t check_dim = 0; check_dim < static_cast<int64_t>(output_dim); check_dim++) {
      if (check_dim != axis && input_i_shape->GetDim(check_dim) != out_shape->GetDim(check_dim)) {
        std::string msg =
            ConcatString("shape[", relative_index, "][", check_dim, "] is ", input_i_shape->GetDim(check_dim),
                         ", must be equal to ", out_shape->GetDim(check_dim), "!");
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), msg);
        return ge::GRAPH_FAILED;
      }
    }
    concat_dim_size += input_i_shape->GetDim(axis);
  }
  out_shape->SetDim(axis, concat_dim_size);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForConcatD(gert::InferShapeContext* context) {
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t* concat_dim = attrs->GetAttrPointer<int64_t>(INDEX_CONCAT_DIM);
  OPS_CHECK_NULL_WITH_CONTEXT(context, concat_dim);
  const int64_t* N = attrs->GetAttrPointer<int64_t>(INDEX_N);
  OPS_CHECK_NULL_WITH_CONTEXT(context, N);
  return ConcatInferShapeCommon(context, 0, *N, *concat_dim);
}

IMPL_OP(ConcatD).InferShape(InferShapeForConcatD);
}  // namespace ops
