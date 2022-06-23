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
constexpr size_t INPUT_NUM_THREE = 3;

static std::string ShapeCannotBroadcastMsg(const gert::Shape* shape1, const gert::Shape* shape2) {
  std::string res = "shape ";
  res += ToString(*shape1);
  res += " and ";
  res += ToString(*shape2);
  res += " cannot broadcast!";
  return res;
}

static bool BroadcastDim(int64_t& dim1, const int64_t dim2) {
  if (dim1 == dim2) return true;
  /* column is dim1, row is dim2, matrix value is broadcast(dim1, dim2)
  dim   0     1    d2
  0     0     0    E
  1     0     1    d2
  d1    E     d1   E
  */
  if ((dim1 != 1) && (dim2 != 1)) {
    string msg = ConcatString(dim1, " and ", dim2, " cannot broadcast!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastDim", msg);
    return false;
  }
  dim1 = (dim1 == 1) ? dim2 : dim1;

  return true;
}

/*
 * @brief: broadcast new shape to output shape
 * @param [in] shape: const gert::Shape*, new shape to broadcast
 * @param [in/out] shape_output: gert::Shape*, output shape
 * @return succeed or not
 */
static bool BroadcastShapeToOutShape(const gert::Shape* shape, gert::Shape* shape_output) {
  OP_LOGD("BroadcastShapeToOutShape",
          "start broadcast %s and %s!", ToString(*shape).c_str(), ToString(*shape_output).c_str());
  size_t shape_len = shape->GetDimNum();
  size_t shape_y_len = shape_output->GetDimNum();
  if (shape_len > shape_y_len) {
    shape_output->SetDimNum(shape_len);
    size_t len_sub = shape_len - shape_y_len;
    for (size_t i = shape_y_len; i > 0; i--) {
      int64_t dim1 = shape->GetDim(len_sub + i - 1);
      int64_t dim2 = shape_output->GetDim(i - 1);
      if (!BroadcastDim(dim1, dim2)) {
        string msg = ConcatString(dim1, " and ", dim2, " cannot broadcast!");
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShapeToOutShape", msg);
        return false;
      }
      shape_output->SetDim(len_sub + i - 1, dim1);
    }
    for (size_t i = 0; i < len_sub; i++) {
      shape_output->SetDim(i, shape->GetDim(i));
    }
  } else {
    auto len_sub = shape_y_len - shape_len;
    for (size_t i = 0; i < shape_len; i++) {
      int64_t dim1 = shape_output->GetDim(len_sub + i);
      int64_t dim2 = shape->GetDim(i);
      if (!BroadcastDim(dim1, dim2)) {
        string msg = ConcatString(dim1, " and ", dim2, " cannot broadcast!");
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShapeToOutShape", msg);
        return false;
      }
      shape_output->SetDim(len_sub + i, dim1);
    }
  }
  return true;
}

ge::graphStatus InferShapeForTwoInOneOut(gert::InferShapeContext *context) {
  auto in_shape1 = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape1);
  auto in_shape2 = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape2);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  *out_shape = *in_shape1;

  OP_CHECK(!BroadcastShapeToOutShape(in_shape2, out_shape),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                               ShapeCannotBroadcastMsg(in_shape2, in_shape1)),
           return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForCompare(gert::InferShapeContext *context) {
  return InferShapeForTwoInOneOut(context);
}

ge::graphStatus InferShapeForMultiInput(gert::InferShapeContext *context, const size_t num) {
  auto in_shape1 = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape1);
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  *out_shape = *in_shape1;

  for (size_t i = 1; i < num; i++) {
    auto in_shape2 = context->GetInputShape(i);
    OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape2);
    OP_CHECK(!BroadcastShapeToOutShape(in_shape2, out_shape),
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                 ShapeCannotBroadcastMsg(in_shape2, out_shape)),
             return ge::GRAPH_FAILED);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForThreeInOneOut(gert::InferShapeContext *context) {
  return InferShapeForMultiInput(context, INPUT_NUM_THREE);
}

IMPL_OP(Add)
    .InferShape(InferShapeForTwoInOneOut);

IMPL_OP(Mul)
    .InferShape(InferShapeForTwoInOneOut);

IMPL_OP(RealDiv)
    .InferShape(InferShapeForTwoInOneOut);

IMPL_OP(ClipByValue)
    .InferShape(InferShapeForThreeInOneOut);

IMPL_OP(SoftmaxGrad)
    .InferShape(InferShapeForTwoInOneOut);

IMPL_OP(Sub)
    .InferShape(InferShapeForTwoInOneOut);

IMPL_OP(MaskedFill)
    .InferShape(InferShapeForTwoInOneOut);

IMPL_OP(ReluGrad)
    .InferShape(InferShapeForTwoInOneOut);

IMPL_OP(Equal)
    .InferShape(InferShapeForCompare);

IMPL_OP(NotEqual)
    .InferShape(InferShapeForCompare);

IMPL_OP(Greater)
    .InferShape(InferShapeForCompare);

IMPL_OP(GreaterEqual)
    .InferShape(InferShapeForCompare);

IMPL_OP(Less)
    .InferShape(InferShapeForCompare);

IMPL_OP(LessEqual)
    .InferShape(InferShapeForCompare);
}  // namespace ops
