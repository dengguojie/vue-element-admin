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
#include <graph/utils/type_utils.h>
#include "runtime_util.h"
#include "op_util.h"

using namespace ge;
namespace ops {
constexpr size_t TRANSPOSE_IDX_IN_X = 0;
constexpr size_t TRANSPOSE_IDX_IN_PERM = 1;
constexpr size_t TRANSPOSE_IDX_OUT_Y = 0;

#define TRANSPOSE_INFERSHAPE_WITH_PERM_TENSOR(DATATYPE, C_TYPE)                                      \
  case (DATATYPE): {                                                                                 \
    const C_TYPE* perm_value = perm_tensor->GetData<C_TYPE>();                                       \
    if (!TransposeInferCommon(context, x_shape, perm_value, y_shape)) {                              \
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "do transpose infershape failed"); \
      return ge::GRAPH_FAILED;                                                                       \
    }                                                                                                \
    break;                                                                                           \
  }

template <typename T>
static bool TransposeInferCommon(const gert::InferShapeContext* context, const gert::Shape* x_shape,
                                 const T* perm_value, gert::Shape* y_shape) {
  OP_LOGD(context->GetNodeName(), "start to do TransposeInferCommon");
  size_t input_dim_size = x_shape->GetDimNum();
  y_shape->SetDimNum(input_dim_size);
  for (size_t i = 0; i < input_dim_size; ++i) {
    OP_CHECK(!IsDimValid(input_dim_size, perm_value[i]),
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                 GenInvalidDimMsg("perm", i, input_dim_size, perm_value[i])),
             return false);
    T perm_v = perm_value[i] < 0 ? perm_value[i] + input_dim_size : perm_value[i];
    y_shape->SetDim(i, x_shape->GetDim(perm_v));
  }

  OP_LOGD(context->GetNodeName(), "end to do TransposeInferCommon");
  return true;
}

ge::graphStatus TransposeInferShape(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do TransposeInferShape");
  const gert::Shape* x_shape = context->GetInputShape(TRANSPOSE_IDX_IN_X);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  gert::Shape* y_shape = context->GetOutputShape(TRANSPOSE_IDX_OUT_Y);
  OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  const gert::Tensor* perm_tensor = context->GetInputTensor(TRANSPOSE_IDX_IN_PERM);
  OPS_CHECK_NULL_WITH_CONTEXT(context, perm_tensor);

  int64_t perm_size = perm_tensor->GetShapeSize();
  size_t input_dim_size = x_shape->GetDimNum();
  OP_CHECK(
      perm_size != static_cast<int64_t>(input_dim_size),
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), GetAttrSizeErrMsg("perm", ge::ConcatString(perm_size),
                                                                                    ge::ConcatString(input_dim_size))),
      return ge::GRAPH_FAILED);

  ge::DataType perm_dtype = perm_tensor->GetDataType();
  switch (perm_dtype) {
    TRANSPOSE_INFERSHAPE_WITH_PERM_TENSOR(ge::DT_INT32, int32_t)
    TRANSPOSE_INFERSHAPE_WITH_PERM_TENSOR(ge::DT_INT64, int64_t)
    default:
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
          context->GetNodeName(),
          GetInputDtypeNotSupportErrMsg("perm", "[int32, int64]", ToString(perm_dtype).c_str()));
      return ge::GRAPH_FAILED;
  }

  OP_LOGD(context->GetNodeName(), "End to do TransposeInferShape");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Transpose).InferShape(TransposeInferShape).InputsDataDependency({1});
}  // namespace ops
