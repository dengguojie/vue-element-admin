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
 * \file elewise_calculation_ops.cpp
 * \brief
 */
#include "inc/elewise_calculation_ops.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"
#include "graph/utils/node_utils.h"

namespace ge {
bool BroadCastTwoShape(const Operator& op, const ge::Shape& shape_x, const ge::Shape& shape_y,
                       std::vector<int64_t>& dim_out) {
  std::vector<int64_t> dim_x = shape_x.GetDims();
  std::vector<int64_t> dim_y = shape_y.GetDims();
  // exchange them
  if (dim_x.size() < dim_y.size()) {
    std::vector<int64_t> dim_tmp = dim_x;
    dim_x = dim_y;
    dim_y = dim_tmp;
  }

  // expand smalll shape
  if (dim_x.size() != dim_x.size()) {
    int dec = dim_x.size() - dim_y.size();
    for (int i = 0; i < dec; i++) {
      dim_y.insert(dim_y.begin(), (int64_t)1);
    }
  }

  // set out dims
  for (size_t i = 0; i < dim_x.size(); i++) {
    if ((dim_x[i] != dim_y[i]) && (dim_x[i] != 1) && (dim_y[i] != 1)) {
      OP_LOGE(op.GetName().c_str(), "The %s's dimensions does not match the broadcast rule(%lu %lu).",
              op.GetName().c_str(), dim_x[i], dim_y[i]);
      return false;
    }

    int64_t dim = dim_x[i] > dim_y[i] ? dim_x[i] : dim_y[i];
    dim_out.push_back(dim);
  }
  return true;
}

bool InferShapeForMaximumAndMinimum(Operator& op) {
  auto attr_grad_x = false;
  auto attr_grad_y = false;
  if (op.GetAttr("grad_x", attr_grad_x) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr grad_x failed");
  }
  if (op.GetAttr("grad_y", attr_grad_y) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "get attr grad_y failed");
  }
  if (attr_grad_x == false && attr_grad_y == false) {
    OP_LOGE(op.GetName().c_str(), "the grad_x and grad_y is not support all false");
    return false;
  }
  if (attr_grad_x) {
    TensorDesc input_tensor_desc_x1 = op.GetInputDesc("x1");
    TensorDesc y_desc1 = op.GetOutputDesc("y1");
    y_desc1.SetShape(input_tensor_desc_x1.GetShape());
    y_desc1.SetDataType(input_tensor_desc_x1.GetDataType());
    (void)op.UpdateOutputDesc("y1", y_desc1);
  }
  if (attr_grad_y) {
    TensorDesc input_tensor_desc_x2 = op.GetInputDesc("x2");
    TensorDesc y_desc2 = op.GetOutputDesc("y2");
    y_desc2.SetShape(input_tensor_desc_x2.GetShape());
    y_desc2.SetDataType(input_tensor_desc_x2.GetDataType());
    (void)op.UpdateOutputDesc("y2", y_desc2);
  }

  return true;
}

// ----------------MaximumGrad-------------------
IMPLEMT_COMMON_INFERFUNC(MaximumGradInferShape) {
  if (InferShapeForMaximumAndMinimum(op)) {
    return GRAPH_SUCCESS;
  }

  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(MaximumGrad, MaximumGradInferShape);
// ----------------MaximumGrad End-------------------

// ----------------MinimumGrad-------------------
IMPLEMT_COMMON_INFERFUNC(MinimumGradInferShape) {
  if (InferShapeForMaximumAndMinimum(op)) {
    return GRAPH_SUCCESS;
  }

  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(MinimumGrad, MinimumGradInferShape);
// ----------------MinimumGrad End-------------------

// ----------------------Add--------------------------
IMPLEMT_VERIFIER(Add, AddVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AddInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Add, AddInferShape);
VERIFY_FUNC_REG(Add, AddVerify);
// ---------------------Add END------------------------

// ----------------------FusedMulAdd--------------------------
IMPLEMT_VERIFIER(FusedMulAdd, FusedMulAddVerify) {
  DataType input_type_x1 = op.GetInputDesc("x1").GetDataType();
  DataType input_type_x2 = op.GetInputDesc("x2").GetDataType();
  DataType input_type_x3 = op.GetInputDesc("x3").GetDataType();
  if (input_type_x1 != input_type_x2) {
    OpsTwoInputDtypeErrReport(op.GetName(), "x1", "x2", ConcatString(input_type_x1), ConcatString(input_type_x2));
    OP_LOGE(op.GetName().c_str(), "The %s op dtype is not same, type1:%d, type2:%d", op.GetName().c_str(),
            input_type_x1, input_type_x2);
    return false;
  }

  if (input_type_x2 != input_type_x3) {
    OpsTwoInputDtypeErrReport(op.GetName(), "x2", "x3", ConcatString(input_type_x2), ConcatString(input_type_x3));
    OP_LOGE(op.GetName().c_str(), "The %s op dtype is not same, type2:%d, type3:%d", op.GetName().c_str(),
            input_type_x2, input_type_x3);
    return false;
  }

  return true;
}

IMPLEMT_COMMON_INFERFUNC(FusedMulAddInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }

  ge::Shape shape1 = op.GetInputDesc("x1").GetShape();
  ge::Shape shape2 = op.GetInputDesc("x2").GetShape();
  std::vector<int64_t> vec_mul_out;
  if (!BroadCastTwoShape(op, shape1, shape2, vec_mul_out)) {
    return GRAPH_FAILED;
  }

  ge::Shape shape_mul_out = ge::Shape(vec_mul_out);
  ge::Shape shape3 = op.GetInputDesc("x3").GetShape();
  std::vector<int64_t> vec_add_out;
  if (!BroadCastTwoShape(op, shape_mul_out, shape3, vec_add_out)) {
    return GRAPH_FAILED;
  }

  ge::Shape shape_add_out = ge::Shape(vec_add_out);
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(shape_add_out);
  DataType dtype_input = op.GetInputDesc("x1").GetDataType();
  y_desc.SetDataType(dtype_input);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FusedMulAdd, FusedMulAddInferShape);
VERIFY_FUNC_REG(FusedMulAdd, FusedMulAddVerify);
// ---------------------FusedMulAdd END------------------------

// ---------------------AddV2--------------------------
IMPLEMT_VERIFIER(AddV2, AddV2Verify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AddV2, AddInferShape);
VERIFY_FUNC_REG(AddV2, AddV2Verify);
// -------------------AddV2 END----------------------

// ----------------Cast-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(CastInferShape)
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  Shape x_shape = op.GetInputDesc("x").GetShape();
  tensordesc_output.SetShape(x_shape);

  std::vector<std::pair<int64_t, int64_t>> range;
  auto status = op.GetInputDesc("x").GetShapeRange(range);
  if (status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  tensordesc_output.SetShapeRange(range);

  int type;
  if (op.GetAttr("dst_type", type) == GRAPH_SUCCESS) {
    tensordesc_output.SetDataType((ge::DataType)type);
  }

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Cast, CastInferShape);
// --------------Cast END-----------------

// ---------------------GreaterEqual-----------------------
IMPLEMT_VERIFIER(GreaterEqual, GreaterEqualVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(GreaterEqualInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(GreaterEqual, GreaterEqualInferShape);
VERIFY_FUNC_REG(GreaterEqual, GreaterEqualVerify);
// ------------------GreaterEqual END-------------------

// --------------------Less--------------------
IMPLEMT_VERIFIER(Less, LessVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(LessInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Less, LessInferShape);
VERIFY_FUNC_REG(Less, LessVerify);
// -----------------Less END-----------------------

// ------------------RealDiv---------------------
IMPLEMT_VERIFIER(RealDiv, RealDivVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(RealDivInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(RealDiv, RealDivInferShape);
VERIFY_FUNC_REG(RealDiv, RealDivVerify);
// ----------------RealDiv END------------------

// ----------------Sqrt Op Begin------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(SqrtInferShape)
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op.GetInputDesc("x").GetShapeRange(shape_range_x);
  tensordesc_output.SetShapeRange(shape_range_x);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Sqrt, SqrtInferShape);
// ----------------Sqrt Op End---------------

// ----------------Maximum--------------------
IMPLEMT_VERIFIER(Maximum, MaximumVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(MaximumInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Maximum, MaximumInferShape);
VERIFY_FUNC_REG(Maximum, MaximumVerify);
// --------------Maximum END------------------

// ----------------Minimum--------------------
IMPLEMT_VERIFIER(Minimum, MinimumVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(MinimumInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Minimum, MinimumInferShape);
VERIFY_FUNC_REG(Minimum, MinimumVerify);
// -----------------Minimum END-----------------

// ----------------Reciprocal-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ReciprocalInferShape)
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op.GetInputDesc("x").GetShapeRange(shape_range_x);

  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetShapeRange(shape_range_x);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Reciprocal, ReciprocalInferShape);
// ---------------Reciprocal END-----------------

// -------------------Sub----------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(SubInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Sub, SubInferShape);
// -----------------Sub END-----------------

// ----------------Abs-------------------
IMPLEMT_COMMON_INFERFUNC(AbsInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Abs, AbsInferShape);
// --------------Abs END-----------------

// ----------------Sign-------------------
IMPLEMT_COMMON_INFERFUNC(SignInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Sign, SignInferShape);
// ---------------Sign END-----------------

// ----------------SquaredDifference-------------------
IMPLEMT_COMMON_INFERFUNC(SquaredDifferenceInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SquaredDifference, SquaredDifferenceInferShape);
// ----------------SquaredDifference END------------------

// ----------------Cos-------------------
IMPLEMT_COMMON_INFERFUNC(CosInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cos, CosInferShape);
// --------------Cos END-----------------

// ------------------Div---------------------
IMPLEMT_VERIFIER(Div, DivVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(DivInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Div, DivInferShape);
VERIFY_FUNC_REG(Div, DivVerify);
// -----------------Div END------------------

// -------------------Equal--------------------
IMPLEMT_VERIFIER(Equal, EqualVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(EqualInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Equal, EqualInferShape);
VERIFY_FUNC_REG(Equal, EqualVerify);
// ------------------Equal END--------------------

// ----------------Exp-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ExpInferShape)
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  auto status = op.GetInputDesc("x").GetShapeRange(shape_range_x);
  if (status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetShapeRange(shape_range_x);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Exp, ExpInferShape);
// ----------------Exp END-------------------

// ----------------Expm1-------------------
IMPLEMT_COMMON_INFERFUNC(Expm1InferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Expm1, Expm1InferShape);
// --------------Expm1 END-----------------

// ----------------------Inv----------------------
IMPLEMT_COMMON_INFERFUNC(InvInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(x_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Inv, InvInferShape);
// ----------------------Inv END----------------------

// ----------------------InvGrad----------------------
IMPLEMT_VERIFIER(InvGrad, InvGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "grad")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InvGradInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x", "grad", "y")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(InvGrad, InvGradInferShape);
VERIFY_FUNC_REG(InvGrad, InvGradVerify);
// ----------------------InvGrad END----------------------

// -------------------LessEqual---------------------
IMPLEMT_VERIFIER(LessEqual, LessEqualVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(LessEqualInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(LessEqual, LessEqualInferShape);
VERIFY_FUNC_REG(LessEqual, LessEqualVerify);
// --------------------LessEqual END-----------------------

// ----------------Log1p-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(Log1pInferShape)
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  auto status = op.GetInputDesc("x").GetShapeRange(shape_range_x);
  if (status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetShapeRange(shape_range_x);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Log1p, Log1pInferShape);
// --------------Log1p END-----------------

// ------------------Mod--------------------
IMPLEMT_VERIFIER(Mod, ModVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ModInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Mod, ModInferShape);
VERIFY_FUNC_REG(Mod, ModVerify);
// ----------------Mod END---------------

// -------------------NotEqual----------------------
IMPLEMT_VERIFIER(NotEqual, NotEqualVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(NotEqualInferShape) {
  Shape x_shape = op.GetInputDesc("x1").GetShape();
  Shape y_shape = op.GetInputDesc("x2").GetShape();
  TensorDesc td = op.GetOutputDesc("y");
  std::vector<int64_t> dims_x = x_shape.GetDims();
  std::vector<int64_t> dims_y = y_shape.GetDims();
  if (dims_x.size() < dims_y.size()) {
    std::vector<int64_t> dims_tmp = dims_x;
    dims_x = dims_y;
    dims_y = dims_tmp;
  }

  if (dims_x.size() != dims_y.size()) {
    int dec = dims_x.size() - dims_y.size();
    for (int i = 0; i < dec; i++) {
      dims_y.insert(dims_y.begin(), (int64_t)1);
    }
  }

  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1) && (dims_x[i] != -1) && (dims_y[i] != -1)) {
      OpsInputShapeBroadcastErrReport(op.GetName(), "x1", "x2", ConcatString(dims_x[i]), ConcatString(dims_y[i]));
      OP_LOGE(op.GetName().c_str(),
              "The %s op dimensions does not "
              "match the broadcast rule(%lu %lu).",
              op.GetName().c_str(), dims_x[i], dims_y[i]);
      return GRAPH_FAILED;
    }
    if ((dims_x[i] == -1) && (dims_y[i] != -1)) {
      if (dims_y[i] > 1) {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
      } else if (dims_y[i] == 1) {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
        dim_vec[i] = -1;
      }
    } else if ((dims_x[i] != -1) && (dims_y[i] == -1)) {
      if (dims_x[i] > 1) {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
      } else if (dims_x[i] == 1) {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
        dim_vec[i] = -1;
      }
    } else {
      if ((dims_x[i] == -1) && (dims_y[i] == -1)) {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
        dim_vec[i] = -1;
      } else {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
      }
    }
  }

  td.SetShape(Shape(dim_vec));
  td.SetDataType(DT_BOOL);
  (void)op.UpdateOutputDesc("y", td);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(NotEqual, NotEqualInferShape);
VERIFY_FUNC_REG(NotEqual, NotEqualVerify);
// -------------------NotEqual END---------------------

// ----------------Neg-------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(NegInferShape)
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op.GetInputDesc("x").GetShapeRange(shape_range_x);

  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetShapeRange(shape_range_x);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Neg, NegInferShape);
// ---------------Neg EDN-----------------

// -----------------TruncateDiv------------------
IMPLEMT_VERIFIER(TruncateDiv, TruncateDivVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TruncateDivInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TruncateDiv, TruncateDivInferShape);
VERIFY_FUNC_REG(TruncateDiv, TruncateDivVerify);
// -----------------TruncateDiv END----------------

// --------------Xdivy----------------
IMPLEMT_VERIFIER(Xdivy, XdivyVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(XdivyInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Xdivy, XdivyInferShape);
VERIFY_FUNC_REG(Xdivy, XdivyVerify);
// ------------Xdivy END----------------

// ------------Xlogy-------------------
IMPLEMT_VERIFIER(Xlogy, XlogyVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(XlogyInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Xlogy, XlogyInferShape);
VERIFY_FUNC_REG(Xlogy, XlogyVerify);
// ------------Xlogy END----------------

// ----------------Cosh-------------------
IMPLEMT_COMMON_INFERFUNC(CoshInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cosh, CoshInferShape);
// ---------------Cosh END------------------

// ------------------DivNoNan-----------------------
IMPLEMT_VERIFIER(DivNoNan, DivNoNanVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DivNoNanInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DivNoNan, DivNoNanInferShape);
VERIFY_FUNC_REG(DivNoNan, DivNoNanVerify);
// --------------DivNoNan END----------------------

// ----------------Invert-------------------
IMPLEMT_COMMON_INFERFUNC(InvertInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(x_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Invert, InvertInferShape);
// ----------------Invert END-------------------

// ---------------OnesLike-----------------
IMPLEMT_COMMON_INFERFUNC(OnesLikeInferShape) {
  Shape y_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(y_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(OnesLike, OnesLikeInferShape);
// ----------------OnesLike END-----------------

// ----------------ReciprocalGrad-------------------
IMPLEMT_VERIFIER(ReciprocalGrad, ReciprocalGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ReciprocalGradInferShape) {
  Shape y_shape = op.GetInputDesc("y").GetShape();
  DataType input_dtype = op.GetInputDesc("y").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("z");
  tensordesc_output.SetShape(y_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("z", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReciprocalGrad, ReciprocalGradInferShape);
VERIFY_FUNC_REG(ReciprocalGrad, ReciprocalGradVerify);
// --------------ReciprocalGrad END-----------------

// ----------------Square Op Begin-----------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(SquareInferShape)
  (void)op.UpdateOutputDesc("y", op.GetInputDesc("x"));
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Square, SquareInferShape);
// ----------------Square Op End-------------------

// ----------------RsqrtGrad----------------------
IMPLEMT_VERIFIER(RsqrtGrad, RsqrtGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(RsqrtGradInferShape) {
  Shape y_shape = op.GetInputDesc("y").GetShape();
  DataType input_dtype = op.GetInputDesc("y").GetDataType();
  std::vector<std::pair<int64_t, int64_t>> shape_range_y;
  auto status = op.GetInputDesc("y").GetShapeRange(shape_range_y);
  if (status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  TensorDesc td = op.GetOutputDesc("z");
  td.SetShape(ge::Shape(y_shape));
  td.SetDataType(input_dtype);
  td.SetShapeRange(shape_range_y);
  (void)op.UpdateOutputDesc("z", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(RsqrtGrad, RsqrtGradInferShape);
VERIFY_FUNC_REG(RsqrtGrad, RsqrtGradVerify);
// ----------------RsqrtGrad END----------------------

// ----------------Sinh-------------------
IMPLEMT_COMMON_INFERFUNC(SinhInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Sinh, SinhInferShape);
// ---------------Sinh END----------------

// --------------------ClipByValue-----------------------
IMPLEMT_VERIFIER(ClipByValue, ClipByValueVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "clip_value_min") || !CheckTwoInputDtypeSame(op, "x", "clip_value_max")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ClipByValueInferShape) {
  Shape input_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(Shape(input_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ClipByValue, ClipByValueInferShape);
VERIFY_FUNC_REG(ClipByValue, ClipByValueVerify);
// -------------------ClipByValue END-------------------

// -------------------LogicalOr--------------------
IMPLEMT_VERIFIER(LogicalOr, LogicalOrVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(LogicalOrInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(LogicalOr, LogicalOrInferShape);
VERIFY_FUNC_REG(LogicalOr, LogicalOrVerify);
// ----------------LogicalOr END--------------------

// ----------------Rsqrt-------------------
COMMON_INFER_FUNC_REG(Rsqrt, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// ----------------Rsqrt-------------------

// ----------------Asin-------------------
COMMON_INFER_FUNC_REG(Asin, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------Asin END-----------------

// ----------------Acos-------------------
COMMON_INFER_FUNC_REG(Acos, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------Acos END-----------------

// ----------------BesselI0e-------------------
COMMON_INFER_FUNC_REG(BesselI0e, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------BesselI0e END-----------------

// ----------------BesselI1e-------------------
COMMON_INFER_FUNC_REG(BesselI1e, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------BesselI1e END-----------------

// ------------------Mul --------------------
IMPLEMT_VERIFIER(Mul, MulVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(MulInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Mul, MulInferShape);
VERIFY_FUNC_REG(Mul, MulVerify);
// ----------------Mul END--------------------

// ----------------SqrtGrad Op Begin-----------------
IMPLEMT_VERIFIER(SqrtGrad, SqrtGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(SqrtGradInferShape)
  Shape shape_x = op.GetInputDesc("y").GetShape();
  DataType input_dtype = op.GetInputDesc("y").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("z");
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op.GetInputDesc("y").GetShapeRange(shape_range_x);
  tensordesc_output.SetShape(shape_x);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetShapeRange(shape_range_x);
  if (op.UpdateOutputDesc("z", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(SqrtGrad, SqrtGradInferShape);
VERIFY_FUNC_REG(SqrtGrad, SqrtGradVerify);
// ----------------SqrtGrad Op End-------------------

// ----------------Log-------------------
IMPLEMT_COMMON_INFERFUNC(LogInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Log, LogInferShape);
// ----------------Log END-------------------

// ----------------Assign-------------------
IMPLEMT_VERIFIER(Assign, AssignVerify) {
  if (!CheckTwoInputDtypeSame(op, "ref", "value")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AssignInferShape) {
  if (OneInOneOutDynamicInfer(op, "value", {"ref"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;

}

COMMON_INFER_FUNC_REG(Assign, AssignInferShape);
VERIFY_FUNC_REG(Assign, AssignVerify);
// ----------------Assign END-------------------

// ----------------AddN-------------------
int64_t GetAddNConstValue(const ge::Operator& op) {
  int64_t tensor_num;
  if (ge::GRAPH_SUCCESS != op.GetAttr("N", tensor_num)) {
    OpsGetAttrErrReport(op.GetName(), "N");
    OP_LOGE(op.GetName().c_str(), "The add_n op GetOpAttr failed!");
  }
  return tensor_num;
}

int64_t AddNInferClassify(ge::Operator& op, int64_t &tensor_num) {
  const int64_t infer_condition_one_one = 11;
  const int64_t infer_condition_one_two = 12;
  const int64_t infer_condition_two = 2;
  const int64_t infer_condition_three = 3;

  int64_t empty_num = 0;
  int64_t static_num = 0;
  int64_t dynamic_shape_num = 0;
  int64_t dynamic_dim_num = 0;

  for (int64_t i = 0; i < tensor_num; i++) {
    vector<int64_t> tempVector = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
    if (tempVector.empty()) {
      empty_num++;
    } else if (std::find(tempVector.begin(), tempVector.end(), -1) != tempVector.end()) {
      dynamic_shape_num++;
    } else if (std::find(tempVector.begin(), tempVector.end(), -2) != tempVector.end()) {
      dynamic_dim_num++;
    } else {
      static_num++;
    }
  }
  if (tensor_num == empty_num + dynamic_dim_num) {
    if (tensor_num == empty_num) {
      return infer_condition_one_one;
    } else {
      return infer_condition_one_two;
    }
  } else if (tensor_num == static_num || tensor_num == empty_num + static_num || tensor_num == static_num +
             dynamic_dim_num || tensor_num == empty_num + static_num + dynamic_dim_num) {
    return infer_condition_two;
  } else {
    return infer_condition_three;
  }
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(AddNInferShape)
  /*
  add_n has four type inputs:
  1.empty 2.static shape 3.-1 4.-2
  The combinations bring 15 scenes, and the 15 scenes can be classify into 4 categories:
  1.input with no range and output no need range, and it can be divided half:
    1.1 all input is empty
    1.2 input only contains empty and -2 shape
  2.input contains static shape and with no -1 shape
  3.input contains -1 shape
  */
  int64_t tensor_num = GetAddNConstValue(op);
  int64_t infer_classify = AddNInferClassify(op, tensor_num);
  // condition 1: all input shape is empty
  if (infer_classify == 11) {
    std::vector<int64_t> shape_vector = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(Shape(shape_vector));
    y_desc.SetDataType(x_dtype);
    (void)op.UpdateOutputDesc("y", y_desc);
  // condition 2: all input is -2 or only empty and -2
  } else if (infer_classify == 12) {
    std::vector<int64_t> shape_vector = {-2};
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(Shape(shape_vector));
    y_desc.SetDataType(x_dtype);
    (void)op.UpdateOutputDesc("y", y_desc);
  // condition 3: contains static shape and no -1 shape
  } else if (infer_classify == 2) {
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    std::vector<int64_t> shape_vector = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
    for (int64_t i = 0; i < tensor_num; i++) {
      std::vector<int64_t> temp_vector = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
      if (!shape_vector.empty() && !IsUnknownRankShape(shape_vector)) {
        shape_vector = temp_vector;
        break;
      }
    }
    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(ge::Shape(shape_vector));
    y_desc.SetDataType(x_dtype);
    std::vector<std::pair<int64_t,int64_t>> out_range;
    MakeUpShapeRange(shape_vector, out_range);
    y_desc.SetShapeRange(out_range);
    (void)op.UpdateOutputDesc("y", y_desc);
  // condition 4: contains -1 shape, range need to choose the intersection
  } else {
    Shape out_shape = op.GetDynamicInputDesc("x", 0).GetShape();
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    std::vector<int64_t> out_vector;
    std::vector<std::pair<int64_t, int64_t>> out_range;
    // Init the output shape and range
    for (int64_t i = 0; i < tensor_num; i++) {
      std::vector<int64_t> temp_vector = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
      if (!temp_vector.empty() && !IsUnknownRankShape(temp_vector)) {
        out_vector = temp_vector;
        op.GetDynamicInputDesc("x", i).GetShapeRange(out_range);
        MakeUpShapeRange(out_vector, out_range);
        break;
      }
    }
    // compute the shape dims and range intersection
    for (int64_t i = 0; i < tensor_num; i++) {
      std::vector<int64_t> temp_vector = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
      if (temp_vector.empty() || IsUnknownRankShape(temp_vector)) {
        continue;
      }
      std::vector<std::pair<int64_t, int64_t>> temp_range;
      op.GetDynamicInputDesc("x", i).GetShapeRange(temp_range);
      MakeUpShapeRange(temp_vector, temp_range);
      for (size_t j = 0; j < temp_vector.size(); j++) {
        // two condition: const == const; const > -1
        if (temp_vector[j] >= out_vector[j]) {
          out_vector[j] = temp_vector[j];
          // update range: left choose the max value
          if (temp_range[j].first >= out_range[j].first) {
            out_range[j].first = temp_range[j].first;
          }
          // update range: right choose the miner value but when it was > 0
          if ((temp_range[j].second <= out_range[j].second && temp_range[j].second > 0) ||
              (out_range[j].second == -1 && temp_range[j].second != -1)) {
            out_range[j].second = temp_range[j].second;
          }
        }
      }
    }
    TensorDesc y_desc = op.GetOutputDesc("y");
    out_shape = Shape(out_vector);
    y_desc.SetShape(out_shape);
    y_desc.SetDataType(x_dtype);
    y_desc.SetShapeRange(out_range);
    (void)op.UpdateOutputDesc("y", y_desc);
  }

IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(AddN, AddNInferShape);
// ----------------AddN END-------------------

// ----------------AssignAdd-------------------
IMPLEMT_VERIFIER(AssignAdd, AssignAddVerify) {
  if (!CheckTwoInputDtypeSame(op, "ref", "value")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AssignAddInferShape) {
  if (TwoInOneOutDynamicInferNoBroadcast(op, "ref", "value", {"ref"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(AssignAdd, AssignAddInferShape);
VERIFY_FUNC_REG(AssignAdd, AssignAddVerify);
// ----------------AssignAdd END-------------------

// ----------------AssignSub-------------------
IMPLEMT_VERIFIER(AssignSub, AssignSubVerify) {
  OP_LOGI(op.GetName().c_str(), "the op begin verify");
  if (!CheckTwoInputDtypeSame(op, "var", "value")) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "the op verify success");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(AssignSub, AssignSubVerify);
COMMON_INFER_FUNC_REG(AssignSub, ELMTWISE_INFER_SHAPEANDTYPE("var", "var"));
// ----------------AssignSub END-------------------

// ----------------Atanh-------------------
COMMON_INFER_FUNC_REG(Atanh, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------Atanh END-----------------

// ----------------Asinh-------------------
COMMON_INFER_FUNC_REG(Asinh, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------Asinh END-----------------

// ----------------Acosh-------------------
COMMON_INFER_FUNC_REG(Acosh, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------Acosh END-----------------

// ----------------Atan-------------------
COMMON_INFER_FUNC_REG(Atan, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------Atan END-----------------

// ----------------Atan2-------------------
IMPLEMT_VERIFIER(Atan2, Atan2Verify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(Atan2InferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Atan2, Atan2InferShape);
VERIFY_FUNC_REG(Atan2, Atan2Verify);
// --------------Atan2 END-----------------

// ----------------AbsGrad-------------------
IMPLEMT_VERIFIER(AbsGrad, AbsGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AbsGrad, ELMTWISE_INFER_SHAPEANDTYPE("y", "z"));
VERIFY_FUNC_REG(AbsGrad, AbsGradVerify);
// --------------AbsGrad END-----------------

// ----------------AsinGrad-------------------
IMPLEMT_VERIFIER(AsinGrad, AsinGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AsinGrad, ELMTWISE_INFER_SHAPEANDTYPE("y", "z"));
VERIFY_FUNC_REG(AsinGrad, AsinGradVerify);
// --------------AsinGrad END-----------------

// ----------------AcosGrad-------------------
IMPLEMT_VERIFIER(AcosGrad, AcosGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AcosGrad, ELMTWISE_INFER_SHAPEANDTYPE("y", "z"));
VERIFY_FUNC_REG(AcosGrad, AcosGradVerify);
// --------------AcosGrad END-----------------

// ----------------AcoshGrad-------------------
IMPLEMT_VERIFIER(AcoshGrad, AcoshGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AcoshGrad, ELMTWISE_INFER_SHAPEANDTYPE("y", "z"));
VERIFY_FUNC_REG(AcoshGrad, AcoshGradVerify);
// --------------AcoshGrad END-----------------

// ----------------AsinhGrad-------------------
IMPLEMT_VERIFIER(AsinhGrad, AsinhGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AsinhGrad, ELMTWISE_INFER_SHAPEANDTYPE("y", "z"));
VERIFY_FUNC_REG(AsinhGrad, AsinhGradVerify);
// --------------AsinhGrad END-----------------

// ----------------AtanGrad-------------------
IMPLEMT_VERIFIER(AtanGrad, AtanGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AtanGrad, ELMTWISE_INFER_SHAPEANDTYPE("y", "z"));
VERIFY_FUNC_REG(AtanGrad, AtanGradVerify);
// --------------AtanGrad END-----------------

// -------------------ApproximateEqual----------------------
IMPLEMT_VERIFIER(ApproximateEqual, ApproximateEqualVerify) {
  float tolerance_data;
  if (ge::GRAPH_SUCCESS != op.GetAttr("tolerance", tolerance_data)) {
    OpsGetAttrErrReport(op.GetName(), "tolerance");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of ApproximateEqual!");
    return GRAPH_FAILED;
  }
  if (tolerance_data < 0) {
    OpsAttrValueErrReport(op.GetName(), "tolerance", ">= 0", ConcatString(tolerance_data));
    OP_LOGE(op.GetName().c_str(), "tolerance should >= 0!");
    return GRAPH_FAILED;
  }

  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ApproximateEqualInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(op.GetInputDesc("x1").GetShape());
  tensordesc_output.SetDataType(DT_BOOL);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApproximateEqual, ApproximateEqualInferShape);
VERIFY_FUNC_REG(ApproximateEqual, ApproximateEqualVerify);
// -------------------ApproximateEqual-------------------------

// --------------------AccumulateNV2--------------------------
bool CheckInputSize(const Operator& op) {
  OP_LOGI(op.GetName().c_str(), "The op begin verify");
  auto input_size = op.GetInputsSize();
  if (input_size == 0) {
    OpsMissInputErrReport(op.GetName(), "x");
    OP_LOGE(op.GetName().c_str(), "The op input size is zero");
    return false;
  }
  return true;
}

bool CheckDynamicInputDtype(const Operator& op, const string& input_name1) {
  DataType first_input_dtype = op.GetDynamicInputDesc(input_name1, 0).GetDataType();
  auto input_dynamic_size = op.GetInputsSize();
  for (size_t i = 0; i < input_dynamic_size; ++i) {
    DataType input_dtype = op.GetDynamicInputDesc(input_name1, i).GetDataType();
    if (first_input_dtype != input_dtype) {
      OpsInputDtypeErrReport(op.GetName(), "x", ConcatString(first_input_dtype), ConcatString(input_dtype));
      OP_LOGE(op.GetName().c_str(),
              "the op type is not same,"
              "type1:%d,type2:%d",
              input_dtype, first_input_dtype);
      return false;
    }
  }
  return true;
}

IMPLEMT_VERIFIER(AccumulateNV2, AccumulateNV2Verify) {
  if (CheckInputSize(op) == false) {
    return GRAPH_FAILED;
  }
  if (CheckDynamicInputDtype(op, "x") == false) {
    return GRAPH_FAILED;
  }
  int64_t num;
  if (GRAPH_SUCCESS != op.GetAttr("N", num)) {
    OpsGetAttrErrReport(op.GetName(), "N");
    OP_LOGE(op.GetName().c_str(), "GetAttr of N failed.");
    return GRAPH_FAILED;
  } else {
    if (op.GetInputsSize() != static_cast<uint64_t>(num)) {
      OpsInputShapeErrReport(op.GetName(), "input size and N must be same.", "N",
                             ConcatString(static_cast<uint64_t>(num)));
      OP_LOGE(op.GetName().c_str(), "input size and N must be same.");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AccumulateNV2InferShape) {
  uint32_t first_input_index = 0;
  (void)op.UpdateOutputDesc("y", op.GetDynamicInputDesc("x", first_input_index));
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AccumulateNV2, AccumulateNV2InferShape);
VERIFY_FUNC_REG(AccumulateNV2, AccumulateNV2Verify);
// --------------------AccumulateNV2 END-----------------------

// -------------------Greater-------------------
IMPLEMT_VERIFIER(Greater, GreaterVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(GreaterInferShape) {
  Shape shape_x = op.GetInputDesc("x1").GetShape();
  Shape shape_y = op.GetInputDesc("x2").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  if (dims_x.size() < dims_y.size()) {
    std::vector<int64_t> dims_tmp = dims_x;
    dims_x = dims_y;
    dims_y = dims_tmp;
  }

  if (dims_x.size() != dims_y.size()) {
    int dec = dims_x.size() - dims_y.size();
    for (int i = 0; i < dec; i++) {
      dims_y.insert(dims_y.begin(), (int64_t)1);
    }
  }

  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1) && (dims_y[i] != -1) && (dims_x[i] != -1)) {
      OpsInputShapeBroadcastErrReport(op.GetName(), "x1", "x2", ConcatString(dims_x[i]), ConcatString(dims_y[i]));
      OP_LOGE(op.GetName().c_str(),
              "The %s op dimensions does not match the broadcast"
              "rule(%lu %lu).",
              op.GetName().c_str(), dims_x[i], dims_y[i]);
      return GRAPH_FAILED;
    }
    if ((dims_x[i] == -1) && (dims_y[i] != -1)) {
      if (dims_y[i] > 1) {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
      } else if (dims_y[i] == 1) {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
        dim_vec[i] = -1;
      }
    } else if ((dims_x[i] != -1) && (dims_y[i] == -1)) {
      if (dims_x[i] > 1) {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
      } else if (dims_x[i] == 1) {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
        dim_vec[i] = -1;
      }
    } else {
      if ((dims_x[i] == -1) && (dims_y[i] == -1)) {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
        dim_vec[i] = -1;
      } else {
        int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
        dim_vec.push_back(dims);
      }
    }
  }

  Shape output_shape = Shape(dim_vec);
  DataType output_dtype = DT_BOOL;
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("y", td);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Greater, GreaterInferShape);
VERIFY_FUNC_REG(Greater, GreaterVerify);
// --------------------Greater END---------------------

// --------------------ZerosLike----------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ZerosLikeInferShape)
  Shape y_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(y_shape));
  td.SetDataType(input_dtype);
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  auto status = op.GetInputDesc("x").GetShapeRange(shape_range_x);
  if (status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  td.SetShapeRange(shape_range_x);
  (void)op.UpdateOutputDesc("y", td);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ZerosLike, ZerosLikeInferShape);
// ----------------ZerosLike END-----------------

// ----------------LogicalNot-------------------
IMPLEMT_COMMON_INFERFUNC(LogicalNotInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }

  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(LogicalNot, LogicalNotInferShape);
// --------------LogicalNot END-----------------

// ----------------------LogicalAnd--------------------------
IMPLEMT_COMMON_INFERFUNC(LogicalAndInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LogicalAnd, LogicalAndInferShape);
// ---------------------LogicalAnd END---------------------

// ----------------FakeQuantWithMinMaxVarsPerChannel----------------------------
IMPLEMT_VERIFIER(FakeQuantWithMinMaxVarsPerChannel, FakeQuantWithMinMaxVarsPerChannelVerify) {
  int64_t num_bits;
  if (ge::GRAPH_SUCCESS != op.GetAttr("num_bits", num_bits)) {
    OpsGetAttrErrReport(op.GetName(), "num_bits");
    LOG_ERROR("[ERROR]op [%s] Attr num_bits is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (ge::GRAPH_SUCCESS != op.GetAttr("narrow_range", narrow_range)) {
    OpsGetAttrErrReport(op.GetName(), "narrow_range");
    LOG_ERROR("[ERROR]op [%s] Attr narrow_range is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  if (num_bits < 2 || num_bits > 16) {
    OpsAttrValueErrReport(op.GetName(), "num_bits", "between 2 and 16", ConcatString(num_bits));
    LOG_ERROR("[ERROR]op [%s] num_bits is between 2 and 16\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  Shape shape_x = op.GetInputDesc("x").GetShape();
  Shape shape_min = op.GetInputDesc("min").GetShape();
  Shape shape_max = op.GetInputDesc("max").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_min = shape_min.GetDims();
  std::vector<int64_t> dims_max = shape_max.GetDims();
  if (dims_x.size() < 1) {
    OpsAttrValueErrReport(op.GetName(), "x'shape", "equal or greater than 1", ConcatString(dims_x.size()));
    OP_LOGE(op.GetName().c_str(), "shape of x must greater 1");
    return GRAPH_FAILED;
  }
  if ((dims_min.size() != 1) || (dims_max.size() != 1)) {
    string input_value = ConcatString("[", dims_min.size(), "] and [", dims_max.size(), "]");
    OpsAttrValueErrReport(op.GetName(), "min's and max's shape", "rank 1", input_value);
    OP_LOGE(op.GetName().c_str(), "shape of min and max must be rank 1");
    return GRAPH_FAILED;
  }
  if (dims_min[0] != dims_max[0]) {
    string excepted_value = ConcatString("same as max[", dims_max[0], "]");
    OpsAttrValueErrReport(op.GetName(), "min'shape", excepted_value, ConcatString(dims_min[0]));
    OP_LOGE(op.GetName().c_str(), "shape of min and max must be same");
    return GRAPH_FAILED;
  }
  if (dims_x[dims_x.size() - 1] != dims_min[0]) {
    string excepted_value = ConcatString("same as min[", dims_min[0], "]");
    OpsAttrValueErrReport(op.GetName(), "x'last dimension", excepted_value, ConcatString(dims_x[dims_x.size() - 1]));
    OP_LOGE(op.GetName().c_str(),
            "The last dimension of x must"
            " be the same as min");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FakeQuantWithMinMaxVarsPerChannelInferShape) {
  Shape shape_input = op.GetInputDesc("x").GetShape();
  DataType dtype_input = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape_input);
  tensordesc_output.SetDataType(dtype_input);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FakeQuantWithMinMaxVarsPerChannel, FakeQuantWithMinMaxVarsPerChannelInferShape);
VERIFY_FUNC_REG(FakeQuantWithMinMaxVarsPerChannel, FakeQuantWithMinMaxVarsPerChannelVerify);
// ----------------FakeQuantWithMinMaxVarsPerChannel----------------------------

// ----------------Rint-----------------------------
COMMON_INFER_FUNC_REG(Rint, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// ----------------Rint END-------------------------

// --------------------------------BiasAdd-------------------------------------
IMPLEMT_VERIFIER(BiasAdd, BiasAddVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(BiasAddInferShape) {
  if (!OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(), "get attr N failed");
  }
  if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NDHWC" && data_format != "NCDHW") {
    string expected_format_list = ConcatString("NHWC, NCHW, NDHWC, NCDHW");
    OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
    OP_LOGE(op.GetName().c_str(),
            "data_format only "
            "support 'NHWC', 'NCHW', 'NDHWC' and 'NCDHW'.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BiasAdd, BiasAddInferShape);
VERIFY_FUNC_REG(BiasAdd, BiasAddVerify);
// ----------------------------------BiasAdd END-----------------------------

// -------------------BitwiseAnd----------------------------
IMPLEMT_COMMON_INFERFUNC(BitwiseAndInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y") == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BitwiseAnd, BitwiseAndInferShape);
// ----------------BitwiseAnd END--------------------------

// ---------------------BitwiseOr----------------------------
IMPLEMT_COMMON_INFERFUNC(BitwiseOrInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y") == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BitwiseOr, BitwiseOrInferShape);
// --------------------BitwiseOr END------------------------

// -----------------------BitwiseXor-------------------------
IMPLEMT_COMMON_INFERFUNC(BitwiseXorInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y") == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BitwiseXor, BitwiseXorInferShape);
// ------------------BitwiseXor END-------------------------

// ----------------Ceil-------------------
COMMON_INFER_FUNC_REG(Ceil, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------Ceil END-----------------

// ----------------FakeQuantWithMinMaxArgs-----------------------
IMPLEMT_VERIFIER(FakeQuantWithMinMaxArgs, FakeQuantWithMinMaxArgsVerify) {
  float min;
  if (GetConstValue(op, "min", min) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr min is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  float max;
  if (GetConstValue(op, "max", max) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr max is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  int64_t numBits;
  if (GetConstValue(op, "num_bits", numBits) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr num_bits is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (GetConstValue(op, "narrow_range", narrow_range) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr narrow_range is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  if (min >= max) {
    string excepted_value = ConcatString("less than max[", max, "]");
    OpsAttrValueErrReport(op.GetName(), "min", excepted_value, ConcatString(min));
    LOG_ERROR("[ERROR]op [%s] min must be less than max !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  if (numBits < 2 || numBits > 16) {
    OpsAttrValueErrReport(op.GetName(), "numBits", "between 2 and 16", ConcatString(numBits));
    LOG_ERROR("[ERROR]op [%s] numBits is between 2 and 16\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FakeQuantWithMinMaxArgsInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FakeQuantWithMinMaxArgs, FakeQuantWithMinMaxArgsInferShape);
VERIFY_FUNC_REG(FakeQuantWithMinMaxArgs, FakeQuantWithMinMaxArgsVerify);
// ----------------FakeQuantWithMinMaxArgs END----------------------

// ----------------FakeQuantWithMinMaxArgsGradient-----------------
IMPLEMT_VERIFIER(FakeQuantWithMinMaxArgsGradient, FakeQuantWithMinMaxArgsGradientVerify) {
  float min;
  if (GetConstValue(op, "min", min) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr min is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  float max;
  if (GetConstValue(op, "max", max) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr max is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  int64_t num_bits;
  if (GetConstValue(op, "num_bits", num_bits) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr num_bits is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (GetConstValue(op, "narrow_range", narrow_range) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr narrow_range is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  if (min >= max) {
    string excepted_value = ConcatString("less than max[", max, "]");
    OpsAttrValueErrReport(op.GetName(), "min", excepted_value, ConcatString(min));
    LOG_ERROR("[ERROR]op [%s] min must be less than max !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  if (num_bits < 2 || num_bits > 16) {
    OpsAttrValueErrReport(op.GetName(), "num_bits", "between 2 and 16", ConcatString(num_bits));
    LOG_ERROR("[ERROR]op [%s] num_bits is between 2 and 16\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "x", "gradients")) {
    return GRAPH_FAILED;
  }
  Shape shape_x = op.GetInputDesc("x").GetShape();
  Shape shape_y = op.GetInputDesc("gradients").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  if (dims_x.size() != dims_y.size()) {
    string excepted_value = ConcatString("same as gradients[", dims_y.size(), "]");
    OpsAttrValueErrReport(op.GetName(), "x'shape", excepted_value, ConcatString(dims_x.size()));
    OP_LOGE(op.GetName().c_str(), "two input shape not same");
    return GRAPH_FAILED;
  } else {
    for (size_t i = 0; i < dims_x.size(); i++) {
      if (dims_x[i] != dims_y[i]) {
        string excepted_value = ConcatString("same as gradients[", dims_y[i], "]");
        OpsAttrValueErrReport(op.GetName(), "x'shape", excepted_value, ConcatString(dims_x[i]));
        OP_LOGE(op.GetName().c_str(), "two input shape not same");
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FakeQuantWithMinMaxArgsGradientInferShape) {
  Shape shape_x = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape_x);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FakeQuantWithMinMaxArgsGradient, FakeQuantWithMinMaxArgsGradientInferShape);
VERIFY_FUNC_REG(FakeQuantWithMinMaxArgsGradient, FakeQuantWithMinMaxArgsGradientVerify);
// ----------------FakeQuantWithMinMaxArgsGradient-------------------

// ----------------FakeQuantWithMinMaxVars---------------------------
IMPLEMT_VERIFIER(FakeQuantWithMinMaxVars, FakeQuantWithMinMaxVarsVerify) {
  int64_t num_bits;
  if (GetConstValue(op, "num_bits", num_bits) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr num_bits is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (GetConstValue(op, "narrow_range", narrow_range) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr narrow_range is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "x", "min")) {
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "min", "max")) {
    return GRAPH_FAILED;
  }
  if (num_bits < 2 || num_bits > 16) {
    OpsAttrValueErrReport(op.GetName(), "num_bits", "between 2 and 16", ConcatString(num_bits));
    LOG_ERROR("[ERROR]op [%s] num_bits is between 2 and 16\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FakeQuantWithMinMaxVarsInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FakeQuantWithMinMaxVars, FakeQuantWithMinMaxVarsInferShape);
VERIFY_FUNC_REG(FakeQuantWithMinMaxVars, FakeQuantWithMinMaxVarsVerify);
// ----------------FakeQuantWithMinMaxVars--------------------------------------

// ----------------FakeQuantWithMinMaxVarsGradient------------------------------
IMPLEMT_VERIFIER(FakeQuantWithMinMaxVarsGradient, FakeQuantWithMinMaxVarsGradientVerify) {
  int64_t num_bits;
  if (GetConstValue(op, "num_bits", num_bits) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr num_bits is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (GetConstValue(op, "narrow_range", narrow_range) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr narrow_range is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "x", "gradients")) {
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "min", "max")) {
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "max", "gradients")) {
    return GRAPH_FAILED;
  }
  if (num_bits < 2 || num_bits > 16) {
    OpsAttrValueErrReport(op.GetName(), "num_bits", "between 2 and 16", ConcatString(num_bits));
    LOG_ERROR("[ERROR]op [%s] num_bits is between 2 and 16\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  Shape shape_x = op.GetInputDesc("x").GetShape();
  Shape shape_y = op.GetInputDesc("gradients").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  if (dims_x.size() != dims_y.size()) {
    string excepted_value = ConcatString("same as gradients[", dims_y.size(), "]");
    OpsAttrValueErrReport(op.GetName(), "x'shape", excepted_value, ConcatString(dims_x.size()));
    OP_LOGE(op.GetName().c_str(), "two input shape not same");
    return GRAPH_FAILED;
  } else {
    for (size_t i = 0; i < dims_x.size(); i++) {
      if (dims_x[i] != dims_y[i]) {
        string excepted_value = ConcatString("same as gradients[", dims_y[i], "]");
        OpsAttrValueErrReport(op.GetName(), "x'shape", excepted_value, ConcatString(dims_x[i]));
        OP_LOGE(op.GetName().c_str(), "two input shape not same");
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FakeQuantWithMinMaxVarsGradientInferShape) {
  Shape shape_input_gradients = op.GetInputDesc("gradients").GetShape();
  Shape shape_input_min = op.GetInputDesc("min").GetShape();
  Shape shape_input_max = op.GetInputDesc("max").GetShape();
  DataType dtype_input_gradients = op.GetInputDesc("gradients").GetDataType();
  DataType dtype_input_min = op.GetInputDesc("min").GetDataType();
  DataType dtype_input_max = op.GetInputDesc("max").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("backprops_wrt_x");
  TensorDesc tensordesc_output_min = op.GetOutputDesc("backprops_wrt_min");
  TensorDesc tensordesc_output_max = op.GetOutputDesc("backprops_wrt_max");
  tensordesc_output.SetShape(shape_input_gradients);
  tensordesc_output_min.SetShape(shape_input_min);
  tensordesc_output_max.SetShape(shape_input_max);
  tensordesc_output.SetDataType(dtype_input_gradients);
  tensordesc_output_min.SetDataType(dtype_input_min);
  tensordesc_output_max.SetDataType(dtype_input_max);
  (void)op.UpdateOutputDesc("backprops_wrt_x", tensordesc_output);
  (void)op.UpdateOutputDesc("backprops_wrt_min", tensordesc_output_min);
  (void)op.UpdateOutputDesc("backprops_wrt_max", tensordesc_output_max);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FakeQuantWithMinMaxVarsGradient, FakeQuantWithMinMaxVarsGradientInferShape);
VERIFY_FUNC_REG(FakeQuantWithMinMaxVarsGradient, FakeQuantWithMinMaxVarsGradientVerify);
// ----------------FakeQuantWithMinMaxVarsGradient END---------------------

// ----------------FakeQuantWithMinMaxVarsPerChannelGradient---------------
IMPLEMT_VERIFIER(FakeQuantWithMinMaxVarsPerChannelGradient, FakeQuantWithMinMaxVarsPerChannelGradientVerify) {
  int64_t num_bits;
  if (GetConstValue(op, "num_bits", num_bits) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr num_bits is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (GetConstValue(op, "narrow_range", narrow_range) == false) {
    LOG_ERROR("[ERROR]op [%s] Attr narrow_range is empty !\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "x", "gradients")) {
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "min", "max")) {
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "max", "gradients")) {
    return GRAPH_FAILED;
  }
  if (num_bits < 2 || num_bits > 16) {
    OpsAttrValueErrReport(op.GetName(), "num_bits", "between 2 and 16", ConcatString(num_bits));
    LOG_ERROR("[ERROR]op [%s] num_bits is between 2 and 16\n", op.GetName().c_str());
    return GRAPH_FAILED;
  }
  Shape shape_x = op.GetInputDesc("x").GetShape();
  Shape shape_min = op.GetInputDesc("min").GetShape();
  Shape shape_max = op.GetInputDesc("max").GetShape();
  Shape shape_y = op.GetInputDesc("gradients").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_min = shape_min.GetDims();
  std::vector<int64_t> dims_max = shape_max.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  if (dims_x.size() != dims_y.size()) {
    string excepted_value = ConcatString("same as gradients[", dims_y.size(), "]");
    OpsAttrValueErrReport(op.GetName(), "x'shape", excepted_value, ConcatString(dims_x.size()));
    OP_LOGE(op.GetName().c_str(), "two input shape not same");
    return GRAPH_FAILED;
  } else {
    for (size_t i = 0; i < dims_x.size(); i++) {
      if (dims_x[i] != dims_y[i]) {
        string excepted_value = ConcatString("same as gradients[", dims_y[i], "]");
        OpsAttrValueErrReport(op.GetName(), "x'shape", excepted_value, ConcatString(dims_x[i]));
        OP_LOGE(op.GetName().c_str(), "two input shape not same");
        return GRAPH_FAILED;
      }
    }
  }
  if ((dims_min.size() != 1) || (dims_max.size() != 1)) {
    string input_value = ConcatString("[", dims_min.size(), "] and [", dims_max.size(), "]");
    OpsAttrValueErrReport(op.GetName(), "min's and max's shape", "rank 1", input_value);
    OP_LOGE(op.GetName().c_str(), "shape of min and max must be rank 1");
    return GRAPH_FAILED;
  }
  if (dims_min[0] != dims_max[0]) {
    string excepted_value = ConcatString("same as max[", dims_max[0], "]");
    OpsAttrValueErrReport(op.GetName(), "min'shape", excepted_value, ConcatString(dims_min[0]));
    OP_LOGE(op.GetName().c_str(), "shape of min and max must be same");
    return GRAPH_FAILED;
  }
  if (dims_x[dims_x.size() - 1] != dims_min[0]) {
    string excepted_value = ConcatString("same as min[", dims_min[0], "]");
    OpsAttrValueErrReport(op.GetName(), "x'last dimension", excepted_value, ConcatString(dims_x[dims_x.size() - 1]));
    OP_LOGE(op.GetName().c_str(),
            "The last dimension of x "
            "must be the same as min");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FakeQuantWithMinMaxVarsPerChannelGradientInferShape) {
  Shape shape_input_gradients = op.GetInputDesc("gradients").GetShape();
  Shape shape_input_min = op.GetInputDesc("min").GetShape();
  Shape shape_input_max = op.GetInputDesc("max").GetShape();
  DataType dtype_input_gradients = op.GetInputDesc("gradients").GetDataType();
  DataType dtype_input_min = op.GetInputDesc("min").GetDataType();
  DataType dtype_input_max = op.GetInputDesc("max").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("backprops_wrt_x");
  TensorDesc tensordesc_output_min = op.GetOutputDesc("backprops_wrt_min");
  TensorDesc tensordesc_output_max = op.GetOutputDesc("backprops_wrt_max");
  tensordesc_output.SetShape(shape_input_gradients);
  tensordesc_output_min.SetShape(shape_input_min);
  tensordesc_output_max.SetShape(shape_input_max);
  tensordesc_output.SetDataType(dtype_input_gradients);
  tensordesc_output_min.SetDataType(dtype_input_min);
  tensordesc_output_max.SetDataType(dtype_input_max);
  (void)op.UpdateOutputDesc("backprops_wrt_x", tensordesc_output);
  (void)op.UpdateOutputDesc("backprops_wrt_min", tensordesc_output_min);
  (void)op.UpdateOutputDesc("backprops_wrt_max", tensordesc_output_max);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FakeQuantWithMinMaxVarsPerChannelGradient, FakeQuantWithMinMaxVarsPerChannelGradientInferShape);
VERIFY_FUNC_REG(FakeQuantWithMinMaxVarsPerChannelGradient, FakeQuantWithMinMaxVarsPerChannelGradientVerify);
// ----------------FakeQuantWithMinMaxVarsPerChannelGradient--------------------

// ----------------Floor-------------------
IMPLEMT_COMMON_INFERFUNC(FloorInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Floor, FloorInferShape);
// ---------------Floor END-----------------

// -------------------FloorDiv-----------------------
IMPLEMT_VERIFIER(FloorDiv, FloorDivVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(FloorDivInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(FloorDiv, FloorDivInferShape);
VERIFY_FUNC_REG(FloorDiv, FloorDivVerify);
// ----------------FloorDiv END------------------------

// ------------------FloorMod--------------------------
IMPLEMT_VERIFIER(FloorMod, FloorModVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(FloorModInferShape)
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(FloorMod, FloorModInferShape);
VERIFY_FUNC_REG(FloorMod, FloorModVerify);
// ----------------FloorMod END---------------------

// ---------------------Pow-------------------------
IMPLEMT_VERIFIER(Pow, PowVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(PowInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Pow, PowInferShape);
VERIFY_FUNC_REG(Pow, PowVerify);
// -------------------Pow END------------------------

// ----------------Round---------------------------------------------
COMMON_INFER_FUNC_REG(Round, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// ----------------Round END-----------------------------------------

// --------------------Tan Op Begin-------------------
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(TanInferShape) {
  (void)op.UpdateOutputDesc("y", op.GetInputDesc("x"));
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Tan, TanInferShape);
// --------------------Tan Op End-------------------

// ----------------TruncateMod-------------------------------
IMPLEMT_COMMON_INFERFUNC(TruncateModInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y") == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TruncateMod, TruncateModInferShape);

// ----------------TruncateMod END---------------------------------

// ----------------Sin-------------------
COMMON_INFER_FUNC_REG(Sin, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// ---------------Sin END----------------

// ---------------------------------ArgMin--------------------------------------
IMPLEMT_COMMON_INFERFUNC(ArgMinInferShape) {
  auto input_shape = op.GetInputDesc("x").GetShape().GetDims();
  const std::string axis_name = "dimension";
  ge::TensorDesc dimension_desc;
  dimension_desc = op.GetInputDesc("dimension");
  auto dimension_shape = dimension_desc.GetShape();
  int64_t dimension_dimnum = dimension_shape.GetDimNum();
  if (dimension_dimnum >= 1) {
    OP_LOGE(op.GetName().c_str(), "dimension_dimnum must be 0");
    return GRAPH_FAILED;
  }
  Tensor dimension;
  if (GRAPH_SUCCESS != op.GetInputConstData(axis_name, dimension)) {
    OP_LOGI(op.GetName().c_str(), "GetInputConstData %s failed.", axis_name.c_str());
    TensorDesc result_desc = op.GetInputDesc("x");
    auto shape = result_desc.GetShape();
    std::vector<int64_t> shape_vector = shape.GetDims();
    int64_t dim_num = shape.GetDimNum();
    std::vector<int64_t> oshape_vector;
    Shape oShape(oshape_vector);
    TensorDesc td = op.GetOutputDesc("y");
    if (dim_num > 1) {
      for (int64_t item = 0; item < (dim_num - 1); ++item) {
        oshape_vector.push_back(-1);
      }
      Shape oShape(oshape_vector);
      td.SetShape(oShape);
    } else {
      td.SetShape(ge::Shape(oShape));
    }
    ge::DataType dtype;
    if (op.GetAttr("dtype", dtype) == GRAPH_SUCCESS) {
      td.SetDataType(dtype);
    } else {
      OP_LOGE(op.GetName().c_str(), "get attr dtype failed.");
      return GRAPH_FAILED;
    }
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
  }

  auto data_type = op.GetInputDesc(axis_name).GetDataType();
  std::vector<int64_t> const_data;
  if (!GetConstIntData(dimension, data_type, const_data)) {
    OP_LOGE(op.GetName().c_str(), "invalid data type of dimension, data_type is %d.", (int)data_type);
    return GRAPH_FAILED;
  }

  int64_t axis_value = const_data[0];
  if (axis_value < 0) {
    axis_value += op.GetInputDesc("x").GetShape().GetDimNum();
  }

  std::vector<int64_t> output_shape(input_shape);
  int64_t max_size = input_shape.size();
  axis_value = axis_value % max_size;
  output_shape.erase(output_shape.begin() + axis_value);
  TensorDesc td = op.GetOutputDesc("y");

  td.SetShape(ge::Shape(output_shape));
  ge::DataType dtype;
  if (op.GetAttr("dtype", dtype) == GRAPH_SUCCESS) {
    td.SetDataType(dtype);
  } else {
    OP_LOGE(op.GetName().c_str(), "get attr dtype failed.");
    return GRAPH_FAILED;
  }
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ArgMin, ArgMinInferShape);
// -------------------------------ArgMin----------------------------------------

// --------------------------------ArgMinD--------------------------------------

IMPLEMT_COMMON_INFERFUNC(ArgMinDInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto shapeX = tensordesc.GetShape();
  int64_t dimension;
  if (GRAPH_SUCCESS != op.GetAttr("dimension", dimension)) {
    OpsGetAttrErrReport(op.GetName(), "dimension");
    OP_LOGE(op.GetName().c_str(), "GetAttr dimension failed.");
    return GRAPH_FAILED;
  }
  int64_t dim_num = shapeX.GetDimNum();
  if (dimension < -dim_num || dimension >= dim_num) {
    OpsInputShapeDimErrReport(op.GetName(), "dimension", ConcatString(dim_num), ConcatString(-dim_num),
                              ConcatString(dimension));
    OP_LOGE(op.GetName().c_str(), "dimension value out of range");
    return GRAPH_FAILED;
  }
  if (dimension < 0) {
    dimension += shapeX.GetDimNum();
  }
  auto dimNum = shapeX.GetDimNum();
  vector<int64_t> y_shape;
  for (size_t i = 0; i < dimNum; ++i) {
    y_shape.push_back(shapeX.GetDim(i));
  }

  int64_t max_size = y_shape.size();
  dimension = dimension % max_size;
  OP_LOGI(op.GetName().c_str(), "the dimension is %d.", (int)dimension);
  y_shape.erase(y_shape.begin() + dimension);
  Shape outputShape(y_shape);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(outputShape);
  td.SetDataType(DT_INT32);

  (void)op.UpdateOutputDesc("y", td);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ArgMinD, ArgMinDInferShape);
// ------------------------------ArgMinD----------------------------------------

// -----------------------------ArgMax------------------------------------------
IMPLEMT_COMMON_INFERFUNC(ArgMaxInferShape) {
  // get all input desc
  const vector<string> depend_names = {"dimension"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto const_desc = op_info->MutableInputDesc("dimension");
  auto y_desc = op_info->MutableOutputDesc("y");
  // get x shape
  auto x_shape = input_desc->MutableShape().GetDims();

  // get and set output dtype
  ge::DataType dtype;
  if (op.GetAttr("dtype", dtype) == GRAPH_SUCCESS) {
    y_desc->SetDataType(dtype);
  } else {
    OP_LOGE(op.GetName().c_str(), "get attr dtype failed.");
    return GRAPH_FAILED;
  }

  // if x_shape == -2, set output -2
  if (IsUnknownRankShape(x_shape)) {
      y_desc->SetShape(GeShape(x_shape));
      return GRAPH_SUCCESS;
  }

  // if x_shape.size() < 2, set output scalar
  if (x_shape.size() < 2) {
      vector<int64_t> output_shape;
      y_desc->SetShape(GeShape(output_shape));
      return GRAPH_SUCCESS;
  }

  // read dimension const value
  GeTensorPtr dimension_tensor = nullptr;
  vector<int64_t> dimension_value;
  if (GRAPH_SUCCESS == NodeUtils::GetInputConstData(node, "dimension", dimension_tensor)) {
    auto const_dtype = const_desc->GetDataType();
    GetConstValue(op, dimension_tensor, const_dtype, dimension_value);
    if (dimension_value[0] < 0) {
    dimension_value[0] += x_shape.size();
    }
    vector<int64_t> output_shape(x_shape);
    output_shape.erase(output_shape.begin() + dimension_value[0]);
    y_desc->SetShape(GeShape(output_shape));

    // when output is dynamic will update range
    if (IsUnknown(output_shape)) {
      std::vector<std::pair<int64_t, int64_t>> input_range;
      input_desc->GetShapeRange(input_range);
      MakeUpShapeRange(x_shape, input_range);
      input_range.erase(input_range.begin() + dimension_value[0]);
      y_desc->SetShapeRange(input_range);
    }
    return GRAPH_SUCCESS;
  }

  // dimension is not const, set all output is -1， range is [1, -1]
  vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  for (int64_t item = 0; item < (x_shape.size() - 1); ++item) {
    output_shape.push_back(-1);
  }
  MakeUpShapeRange(output_shape, output_range);
  y_desc->SetShape(GeShape(output_shape));
  y_desc->SetShapeRange(output_range);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ArgMaxV2, ArgMaxInferShape);
// --------------------------ArgMax---------------------------------------------

// --------------------------------ArgMaxD--------------------------------------
IMPLEMT_COMMON_INFERFUNC(ArgMaxDInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto shape_x = tensordesc.GetShape();
  int64_t dimension;
  if (GRAPH_SUCCESS != op.GetAttr("dimension", dimension)) {
    OpsGetAttrErrReport(op.GetName(), "dimension");
    OP_LOGE(op.GetName().c_str(), "GetAttr dimension failed.");
    return GRAPH_FAILED;
  }
  int64_t dim_num = shape_x.GetDimNum();
  if (dimension < -dim_num || dimension >= dim_num) {
    OpsInputShapeDimErrReport(op.GetName(), "dimension", ConcatString(dim_num), ConcatString(-dim_num),
                              ConcatString(dimension));
    OP_LOGE(op.GetName().c_str(), "Axis value out of range");
    return GRAPH_FAILED;
  }
  if (dimension < 0) {
    dimension += shape_x.GetDimNum();
  }
  vector<int64_t> y_shape;
  for (size_t i = 0; i < dim_num; ++i) {
    y_shape.push_back(shape_x.GetDim(i));
  }

  int64_t max_size = y_shape.size();
  dimension = dimension % max_size;
  y_shape.erase(y_shape.begin() + dimension);
  Shape outputShape(y_shape);

  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(outputShape);
  td.SetDataType(DT_INT32);

  (void)op.UpdateOutputDesc("y", td);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ArgMaxD, ArgMaxDInferShape);
// ------------------------------ArgMaxD----------------------------------------

// ----------------------------ArgMaxWithValue----------------------------------
IMPLEMT_COMMON_INFERFUNC(ArgMaxWithValueInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto shape_x = tensordesc.GetShape();
  int64_t dimension;
  if (GRAPH_SUCCESS != op.GetAttr("dimension", dimension)) {
    OpsGetAttrErrReport(op.GetName(), "dimension");
    OP_LOGE(op.GetName().c_str(), "GetAttr dimension failed.");
    return GRAPH_FAILED;
  }
  if (dimension < 0) {
    dimension += shape_x.GetDimNum();
  }
  auto dim_num = shape_x.GetDimNum();
  vector<int64_t> y_shape;
  for (size_t i = 0; i < dim_num; ++i) {
    y_shape.push_back(shape_x.GetDim(i));
  }
  int64_t max_size = y_shape.size();
  dimension = dimension % max_size;
  OP_LOGI(op.GetName().c_str(), "the dimension is %d.", (int)dimension);

  bool keep_dims;
  if (GRAPH_SUCCESS != op.GetAttr("keep_dims", keep_dims)) {
    OpsGetAttrErrReport(op.GetName(), "keep_dims");
    OP_LOGE(op.GetName().c_str(), "GetAttr of keep_dims failed.");
    return GRAPH_FAILED;
  }
  if (keep_dims) {
    // If keepDims is true, current dimesion set to 1
    y_shape[dimension] = 1;
  } else {
    y_shape.erase(y_shape.begin() + dimension);
  }

  Shape outputShape(y_shape);
  DataType input_dtype = tensordesc.GetDataType();
  TensorDesc td = op.GetOutputDesc("indice");
  TensorDesc td2 = op.GetOutputDesc("values");
  td.SetShape(outputShape);
  td2.SetShape(outputShape);
  td.SetDataType(DT_INT32);
  td2.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("indice", td);
  (void)op.UpdateOutputDesc("values", td2);

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ArgMaxWithValue, ArgMaxWithValueInferShape);
// -----------------------------ArgMaxWithValue---------------------------------

// ---------------------------ArgMinWithValue-----------------------------------
IMPLEMT_COMMON_INFERFUNC(ArgMinWithValueInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto shape_x = tensordesc.GetShape();
  int64_t dimension;
  if (GRAPH_SUCCESS != op.GetAttr("dimension", dimension)) {
    OpsGetAttrErrReport(op.GetName(), "dimension");
    OP_LOGE(op.GetName().c_str(), "GetAttr dimension failed.");
    return GRAPH_FAILED;
  }
  if (dimension < 0) {
    dimension += shape_x.GetDimNum();
  }
  auto dim_num = shape_x.GetDimNum();
  vector<int64_t> y_shape;
  for (size_t i = 0; i < dim_num; ++i) {
    y_shape.push_back(shape_x.GetDim(i));
  }
  int64_t max_size = y_shape.size();
  dimension = dimension % max_size;
  OP_LOGI(op.GetName().c_str(), "the dimension is %d.", (int)dimension);

  bool keep_dims;
  if (GRAPH_SUCCESS != op.GetAttr("keep_dims", keep_dims)) {
    OpsGetAttrErrReport(op.GetName(), "keep_dims");
    OP_LOGE(op.GetName().c_str(), "GetAttr of keep_dims failed.");
    return GRAPH_FAILED;
  }
  if (keep_dims) {
    // If keepDims is true, current dimesion set to 1
    y_shape[dimension] = 1;
  } else {
    y_shape.erase(y_shape.begin() + dimension);
  }

  Shape outputShape(y_shape);
  DataType input_dtype = tensordesc.GetDataType();
  TensorDesc td = op.GetOutputDesc("indice");
  TensorDesc td2 = op.GetOutputDesc("values");
  td.SetShape(outputShape);
  td2.SetShape(outputShape);
  td.SetDataType(DT_INT32);
  td2.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("indice", td);
  (void)op.UpdateOutputDesc("values", td2);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ArgMinWithValue, ArgMinWithValueInferShape);
// ---------------------------ArgMinWithValue-----------------------------------

// ----------------Eltwise-------------------
IMPLEMT_COMMON_INFERFUNC(EltwiseInferShape) {
  uint32_t first_input_index = 0;
  TensorDesc td = op.GetDynamicInputDesc("x", first_input_index);
  auto x_shape = td.GetShape().GetDims();
  auto x_dtype = td.GetDataType();
  TensorDesc td1 = op.GetOutputDesc("y");
  td1.SetShape(ge::Shape(x_shape));
  td1.SetDataType((DataType)x_dtype);
  (void)op.UpdateOutputDesc("y", td1);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Eltwise, EltwiseInferShape);
// ----------------Eltwise END-------------------

// ------------PopulationCount----------------
IMPLEMT_COMMON_INFERFUNC(PopulationCountInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape);
  tensordesc_output.SetDataType(DT_UINT8);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(PopulationCount, PopulationCountInferShape);
// ------------PopulationCount END-----------------

// ------------LambNextMVWithDecay----------------
IMPLEMT_COMMON_INFERFUNC(LambNextMVWithDecayInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_mul3", "input_mul2", "y1")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_mul2", "input_realdiv1", "y3")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_mul2", "input_mul1", "y2")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_mul3", "input_mul0", "y4")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LambNextMVWithDecay, LambNextMVWithDecayInferShape);
// ------------LambNextMVWithDecay END----------------

// ------------LambNextMV----------------
IMPLEMT_COMMON_INFERFUNC(LambNextMVInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_mul3", "input_mul2", "y1")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_mul2", "input_realdiv1", "y3")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_mul2", "input_mul1", "y2")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_mul3", "input_mul0", "y4")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LambNextMV, LambNextMVInferShape);
// ------------LambNextMV END----------------

// ------------LambNextRight----------------
IMPLEMT_COMMON_INFERFUNC(LambNextRightInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_square", "input_mul2", "y1")) {
    return GRAPH_FAILED;
  }

  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_square", "input_mul2", "y2")) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(LambNextRight, LambNextRightVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(LambNextRight, LambNextRightInferShape);
VERIFY_FUNC_REG(LambNextRight, LambNextRightVerify);
// ------------LambNextRight----------------

// ------------LambUpdateWithLr----------------
IMPLEMT_COMMON_INFERFUNC(LambUpdateWithLrInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input_greater1", "input_sub", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LambUpdateWithLr, LambUpdateWithLrInferShape);
// ------------LambUpdateWithLr END----------------

// ------------LambUpdateWithLrV2----------------
IMPLEMT_COMMON_INFERFUNC(LambUpdateWithLrV2InferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x4", "output_y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(LambUpdateWithLrV2, LambUpdateWithLrV2Verify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(LambUpdateWithLrV2, LambUpdateWithLrV2InferShape);
VERIFY_FUNC_REG(LambUpdateWithLrV2, LambUpdateWithLrV2Verify);
// ------------LambUpdateWithLrV2----------------

// ----------------AdamApplyOneWithDecay-------------------
IMPLEMT_COMMON_INFERFUNC(AdamApplyOneWithDecayInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input1", "output0")) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input2", "output1")) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input2", "input3", "output2")) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AdamApplyOneWithDecay, AdamApplyOneWithDecayVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdamApplyOneWithDecay, AdamApplyOneWithDecayInferShape);
VERIFY_FUNC_REG(AdamApplyOneWithDecay, AdamApplyOneWithDecayVerify);
// ----------------AdamApplyOneWithDecay-------------------

// ----------------AdamApplyOne-------------------
IMPLEMT_COMMON_INFERFUNC(AdamApplyOneInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input1", "output0")) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input2", "output1")) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input2", "input3", "output2")) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AdamApplyOne, AdamApplyOneVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdamApplyOne, AdamApplyOneInferShape);
VERIFY_FUNC_REG(AdamApplyOne, AdamApplyOneVerify);
// ----------------AdamApplyOne-------------------

// ----------------AdamApplyOneWithDecayAssign-------------------
IMPLEMT_COMMON_INFERFUNC(AdamApplyOneWithDecayAssignInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AdamApplyOneWithDecayAssign, AdamApplyOneWithDecayAssignVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdamApplyOneWithDecayAssign, AdamApplyOneWithDecayAssignInferShape);
VERIFY_FUNC_REG(AdamApplyOneWithDecayAssign, AdamApplyOneWithDecayAssignVerify);
// ----------------AdamApplyOneWithDecayAssign-------------------

// ----------------AdamApplyOneAssign-------------------
IMPLEMT_COMMON_INFERFUNC(AdamApplyOneAssignInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AdamApplyOneAssign, AdamApplyOneAssignVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdamApplyOneAssign, AdamApplyOneAssignInferShape);
VERIFY_FUNC_REG(AdamApplyOneAssign, AdamApplyOneAssignVerify);
// ----------------AdamApplyOneAssign-------------------

// ----------------LambApplyOptimizerAssign-------------------
IMPLEMT_COMMON_INFERFUNC(LambApplyOptimizerAssignInferShape) {
  Shape x_shape = op.GetInputDesc("grad").GetShape();
  DataType input_dtype = op.GetInputDesc("grad").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("output0");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("output0", tensordesc_output);

  Shape v_shape = op.GetInputDesc("inputv").GetShape();
  DataType inputv_dtype = op.GetInputDesc("inputv").GetDataType();
  TensorDesc tensordesc_outputv = op.GetOutputDesc("inputv");
  tensordesc_outputv.SetShape(v_shape);
  tensordesc_outputv.SetDataType(inputv_dtype);
  (void)op.UpdateOutputDesc("inputv", tensordesc_outputv);

  Shape m_shape = op.GetInputDesc("inputm").GetShape();
  DataType inputm_dtype = op.GetInputDesc("inputm").GetDataType();
  TensorDesc tensordesc_outputm = op.GetOutputDesc("inputm");
  tensordesc_outputm.SetShape(m_shape);
  tensordesc_outputm.SetDataType(inputm_dtype);
  (void)op.UpdateOutputDesc("inputm", tensordesc_outputm);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(LambApplyOptimizerAssign, LambApplyOptimizerAssignVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(LambApplyOptimizerAssign, LambApplyOptimizerAssignInferShape);
VERIFY_FUNC_REG(LambApplyOptimizerAssign, LambApplyOptimizerAssignVerify);
// ----------------LambApplyOptimizerAssign-------------------

// ----------------LambApplyWeightAssign-------------------
IMPLEMT_COMMON_INFERFUNC(LambApplyWeightAssignInferShape) {
  Shape x_shape = op.GetInputDesc("input_param").GetShape();
  DataType input_dtype = op.GetInputDesc("input_param").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("input_param");
  tensordesc_output.SetShape(x_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("input_param", tensordesc_output);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(LambApplyWeightAssign, LambApplyWeightAssignVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(LambApplyWeightAssign, LambApplyWeightAssignInferShape);
VERIFY_FUNC_REG(LambApplyWeightAssign, LambApplyWeightAssignVerify);
// ----------------LambApplyWeightAssign-------------------

// ------------SquareSumV2 Op Begin----------------
IMPLEMT_COMMON_INFERFUNC(SquareSumV2InferShape) {
  auto shape = op.GetInputDesc("input_x").GetShape();
  DataType input_dtype = op.GetInputDesc("input_x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("output1");
  std::vector<int64_t> shapeVector = shape.GetDims();
  int64_t dimNum = shape.GetDimNum();
  std::vector<int64_t> axis;
  if (ge::GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OpsGetAttrErrReport(op.GetName(), "axis");
    OP_LOGE(op.GetName().c_str(),
            "The input_size op GetOpAttr"
            "ConstValue failed!");
    return GRAPH_FAILED;
  }

  bool keep_dims;
  if (ge::GRAPH_SUCCESS != op.GetAttr("keep_dims", keep_dims)) {
    OpsGetAttrErrReport(op.GetName(), "keep_dims");
    OP_LOGE(op.GetName().c_str(), "get keep_dims op GetOpAttr failed!");
    return GRAPH_FAILED;
  }

  if (axis.size() == 0) {
    for (size_t i = 0; i < shapeVector.size(); ++i) {
      axis.push_back(i);
    }
  }

  for (size_t i = 0; i < axis.size(); ++i) {
    if (axis[i] < 0) {
      axis[i] = dimNum + axis[i];
    }
  }

  std::vector<int64_t> oShapeVector;
  std::vector<int64_t>::iterator tmp;
  for (int64_t item = 0; item < dimNum; ++item) {
    tmp = std::find(axis.begin(), axis.end(), item);
    if (tmp != axis.end()) {
      // item in axis
      // If keepDims is true, current dimesion set to 1
      if (keep_dims == true) {
        oShapeVector.push_back(1);
      }
    } else {
      // item is not in ConstValueAxis
      oShapeVector.push_back(shapeVector[item]);
    }
  }

  Shape oShape(oShapeVector);
  tensordesc_output.SetShape(oShape);
  tensordesc_output.SetDataType(input_dtype);
  TensorDesc tensordesc_output1 = op.GetOutputDesc("output2");
  tensordesc_output1.SetShape(shape);
  tensordesc_output1.SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SquareSumV2, SquareSumV2InferShape);
// ------------SquareSumV2 Op End----------------

// ------------ClipByNormNoDivSum----------------
IMPLEMT_COMMON_INFERFUNC(ClipByNormNoDivSumInferShape) {
  auto shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(shape);
  tensordesc_output.SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ClipByNormNoDivSum, ClipByNormNoDivSumInferShape);
// ------------ClipByNormNoDivSum----------------

// ------------SquareSumV1 Op Begin----------------
IMPLEMT_COMMON_INFERFUNC(SquareSumV1InferShape) {
  auto shape = op.GetInputDesc("input_x").GetShape();
  DataType input_dtype = op.GetInputDesc("input_x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("output1");
  std::vector<int64_t> shapeVector = shape.GetDims();
  int64_t dimNum = shape.GetDimNum();
  std::vector<int64_t> axis;
  if (ge::GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OpsGetAttrErrReport(op.GetName(), "axis");
    OP_LOGE(op.GetName().c_str(),
            "The input_size op GetOpAttr"
            "ConstValue failed!");
    return GRAPH_FAILED;
  }

  bool keep_dims;
  if (ge::GRAPH_SUCCESS != op.GetAttr("keep_dims", keep_dims)) {
    OpsGetAttrErrReport(op.GetName(), "keep_dims");
    OP_LOGE(op.GetName().c_str(), "get keep_dims op GetOpAttr failed!");
    return GRAPH_FAILED;
  }

  if (axis.size() == 0) {
    for (size_t i = 0; i < shapeVector.size(); ++i) {
      axis.push_back(i);
    }
  }

  for (size_t i = 0; i < axis.size(); ++i) {
    if (axis[i] < 0) {
      axis[i] = dimNum + axis[i];
    }
  }

  std::vector<int64_t> oShapeVector;
  std::vector<int64_t>::iterator tmp;
  for (int64_t item = 0; item < dimNum; ++item) {
    tmp = std::find(axis.begin(), axis.end(), item);
    if (tmp != axis.end()) {
      // item in axis
      // If keepDims is true, current dimesion set to 1
      if (keep_dims == true) {
        oShapeVector.push_back(1);
      }
    } else {
      // item is not in ConstValueAxis
      oShapeVector.push_back(shapeVector[item]);
    }
  }

  Shape oShape(oShapeVector);
  tensordesc_output.SetShape(oShape);
  tensordesc_output.SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SquareSumV1, SquareSumV1InferShape);
// ------------SquareSumV1 Op End----------------

// ----------------SquareSumAll Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(SquareSumALlInferShape) {
  std::vector<int64_t> o_shape_vector;
  Shape o_shape(o_shape_vector);
  DataType input_x1_dtype = op.GetInputDesc("x1").GetDataType();
  DataType input_x2_dtype = op.GetInputDesc("x2").GetDataType();
  TensorDesc tensor_desc_y1 = op.GetOutputDesc("y1");
  TensorDesc tensor_desc_y2 = op.GetOutputDesc("y2");
  tensor_desc_y1.SetShape(o_shape);
  tensor_desc_y1.SetDataType(input_x1_dtype);
  tensor_desc_y2.SetShape(Shape(o_shape));
  tensor_desc_y2.SetDataType(input_x2_dtype);
  if (op.UpdateOutputDesc("y1", tensor_desc_y1) != GRAPH_SUCCESS ||
      op.UpdateOutputDesc("y2", tensor_desc_y2) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SquareSumAll, SquareSumALlInferShape);
// ----------------SquareSumAll Op End-------------------

// ----------------FusedMulAddN-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(FusedMulAddN, FusedMulAddNVerify) {
  const std::map<std::string, std::vector<DataType>> kInputTensorMap = {
      {"x1", {DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT16}}, {"x2", {DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT16}}};
  const std::vector<DataType> kSupportList = {DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT16};
  if (!CheckInputDataType(op, "x3", kSupportList)) {
    return GRAPH_FAILED;
  }

  // input tensor params, must have same shape and dtype
  if (!CheckInputDtypeAndShape(op, kInputTensorMap)) {
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "The op verify end");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FusedMulAddN, ELMTWISE_INFER_SHAPEANDTYPE("x1", "y"));
VERIFY_FUNC_REG(FusedMulAddN, FusedMulAddNVerify);
// ----------------FusedMulAddN END------------------
// ----------------FusedMulAddNL2loss-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(FusedMulAddNL2loss, FusedMulAddNL2lossVerify) {
  const std::map<std::string, std::vector<DataType>> kInputTensorMap = {{"x1", {DT_FLOAT}}, {"x2", {DT_FLOAT}}};
  const std::vector<DataType> kSupportList = {DT_FLOAT};
  if (!CheckInputDataType(op, "x3", kSupportList)) {
    return GRAPH_FAILED;
  }

  // input tensor params, must have same shape and dtype
  if (!CheckInputDtypeAndShape(op, kInputTensorMap)) {
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "The op verify end");
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FusedMulAddNL2lossInferShape) {
  std::vector<int64_t> o_shape_vector;
  Shape o_shape(o_shape_vector);
  auto shape_x = op.GetInputDesc("x1").GetShape();
  DataType input_dtype = op.GetInputDesc("x1").GetDataType();
  TensorDesc tensordesc_output1 = op.GetOutputDesc("y1");
  TensorDesc tensordesc_output2 = op.GetOutputDesc("y2");
  tensordesc_output1.SetShape(shape_x);
  tensordesc_output1.SetDataType(input_dtype);
  tensordesc_output2.SetShape(ge::Shape(o_shape));
  tensordesc_output2.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y1", tensordesc_output1) != GRAPH_SUCCESS ||
      op.UpdateOutputDesc("y2", tensordesc_output2) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FusedMulAddNL2loss, FusedMulAddNL2lossInferShape);
VERIFY_FUNC_REG(FusedMulAddNL2loss, FusedMulAddNL2lossVerify);
// ----------------FusedMulAddNL2loss end-------------------
// ---------------------------------Bias----------------------------------
IMPLEMT_INFERFUNC(Bias, BiasInferShape) {
  OP_LOGI("Bias", "infer shape begin---");
  DataType dtype_x = op.GetInputDesc("x").GetDataType();
  ge::Shape shape_x = op.GetInputDesc("x").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  // set output
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape_x);
  output_desc.SetDataType(dtype_x);
  (void)op.UpdateOutputDesc("y", output_desc);

  int64_t axis;
  int64_t num_axes;
  bool bias_from_blob;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OP_LOGE("[ERROR] GetOpAttr axis failed!");
    OpsGetAttrErrReport(op.GetName(), "axis");
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("num_axes", num_axes)) {
    OP_LOGE("[ERROR] GetOpAttr num_axes failed!");
    OpsGetAttrErrReport(op.GetName(), "num_axes");
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("bias_from_blob", bias_from_blob)) {
    OP_LOGE("[ERROR] GetOpAttr bias_from_blob failed!");
    OpsGetAttrErrReport(op.GetName(), "bias_from_blob");
    return GRAPH_FAILED;
  }

  ge::Shape shape_bias = op.GetInputDesc("bias").GetShape();
  int64_t bias_dim_num = shape_bias.GetDimNum();

  if (dims_x.size() == 4 && bias_dim_num != 0) {
    int64_t length_x = dims_x.size();
    std::vector<int64_t> dims_bias = shape_bias.GetDims();
    int64_t length_bias = dims_bias.size();
    int64_t axis_;
    if (axis < 0) {
      axis_ = length_x + axis;
    } else {
      axis_ = axis;
    }

    std::vector<int64_t> dims_bias_tmp = shape_bias.GetDims();
    if (bias_from_blob) {
      if (num_axes == -1) {
        for (int64_t i = 0; i < axis_; i++) {
          dims_bias_tmp.insert(dims_bias_tmp.begin(), (int64_t)1);
        }
      } else if (num_axes > 0) {
        int64_t left_length = length_x - num_axes - axis_;
        for (int64_t i = 0; i < axis_; i++) {
          dims_bias_tmp.insert(dims_bias_tmp.begin(), (int64_t)1);
        }
        for (int64_t i = 0; i < left_length; i++) {
          dims_bias_tmp.push_back((int64_t)1);
        }
      }
    } else {
      int64_t left_length = length_x - length_bias - axis_;
      for (int64_t i = 0; i < axis_; i++) {
        dims_bias_tmp.insert(dims_bias_tmp.begin(), (int64_t)1);
      }
      for (int64_t i = 0; i < left_length; i++) {
        dims_bias_tmp.push_back((int64_t)1);
      }
    }

    // update bias shape
    ge::Shape output_bias_shape = ge::Shape(dims_bias_tmp);
    TensorDesc bias_desc = op.GetInputDesc("bias");
    bias_desc.SetShape(output_bias_shape);
    bias_desc.SetOriginShape(output_bias_shape);
    (void)op.UpdateInputDesc("bias", bias_desc);
  }

  OP_LOGI("Bias", "infer shape end---");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Bias, BiasVerify) {
  ge::Shape shape_x = op.GetInputDesc("x").GetShape();
  ge::Shape shape_bias = op.GetInputDesc("bias").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_bias = shape_bias.GetDims();
  int64_t bias_dim_num = shape_bias.GetDimNum();

  int64_t axis;
  int64_t num_axes;
  bool bias_from_blob;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    OP_LOGE("[ERROR] GetOpAttr axis failed!");
    OpsGetAttrErrReport(op.GetName(), "axis");
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("num_axes", num_axes)) {
    OP_LOGE("[ERROR] GetOpAttr num_axes failed!");
    OpsGetAttrErrReport(op.GetName(), "num_axes");
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("bias_from_blob", bias_from_blob)) {
    OP_LOGE("[ERROR] GetOpAttr bias_from_blob failed!");
    OpsGetAttrErrReport(op.GetName(), "bias_from_blob");
    return GRAPH_FAILED;
  }

  int64_t length_x = dims_x.size();
  int64_t length_bias = dims_bias.size();

  if ((axis >= length_x) || (axis < (-length_x))) {
    OP_LOGE("[ERROR] axis out of range index");
    string minvalue = ConcatString(-length_x);
    string maxvalue = ConcatString(length_x - 1);
    string excepted_value = ConcatString("in the range of [", minvalue,",", maxvalue,"]");
    OpsAttrValueErrReport(op.GetName(), "axis", excepted_value, ConcatString(axis));
    return GRAPH_FAILED;
  }
  if (num_axes < -1) {
    OP_LOGE("[ERROR] num_axes must be non-negative or -1");
    OpsAttrValueErrReport(op.GetName(), "num_axes", "non-negative or -1", ConcatString(num_axes));
    return GRAPH_FAILED;
  }

  int64_t axis_;
  if (axis < 0) {
    axis_ = length_x + axis;
  } else {
    axis_ = axis;
  }

  if (bias_from_blob) {
    if (num_axes == -1) {
      int64_t bias_num = length_x - axis_;
      if (length_bias != bias_num) {
        OP_LOGE("[ERROR] length_bias and bias_num must be equal");
        OpsInputShapeErrReport(op.GetName(), "length_bias and bias_num must be equal",
                              "length_bias", ConcatString(length_bias));
        return GRAPH_FAILED;
      }
      for (int64_t i = 0; i < bias_num; i++) {
        if (dims_x[axis_ + i] != dims_bias[i]) {
          OP_LOGE("[ERROR] dimensions shape_x and shape_bias must be equal");
          OpsInputShapeErrReport(op.GetName(), "The dimensions of shape_x and shape_bias must be equal",
                              "shape_bias's dimension", ConcatString(dims_bias[i]));
          return GRAPH_FAILED;
        }
      }
    } else if (num_axes == 0) {
      if (bias_dim_num != 0) {
        OP_LOGE("[ERROR] bias must be a scalar ");
        OpsAttrValueErrReport(op.GetName(), "bias", "scalar", ConcatString(bias_dim_num));
        return GRAPH_FAILED;
      }
    } else if (num_axes > 0) {
      int64_t num_axis = axis_ + num_axes;
      if (num_axis > length_x) {
        OP_LOGE("[ERROR] bias shape extends x shape when applied");
        OpsOneInputShapeErrReport(op.GetName(), "bias", "Bias shape extends x_shape when applied.");
        return GRAPH_FAILED;
      }
      if (length_bias != num_axes) {
        OP_LOGE("[ERROR] length_bias and num_axes must be equal");
        OpsInputShapeErrReport(op.GetName(), "length_bias and bias_num must be equal",
                              "length_bias", ConcatString(length_bias));
        return GRAPH_FAILED;
      }
      for (int64_t i = 0; i < num_axes; i++) {
        if (dims_x[axis_ + i] != dims_bias[i]) {
          OP_LOGE("[ERROR] dimensions shape_x and shape_bias must be equal");
          OpsInputShapeErrReport(op.GetName(), "The dimensions of shape_x and shape_bias must be equal",
                              "shape_bias's dimension", ConcatString(dims_bias[i]));
          return GRAPH_FAILED;
        }
      }
    }
  } else {
    if (bias_dim_num != 0) {
      int64_t bias_num = axis_ + length_bias;
      if (bias_num > length_x) {
        OP_LOGE("[ERROR] bias shape extends x shape when applied");
        OpsOneInputShapeErrReport(op.GetName(), "bias", "Bias shape extends x_shape when applied");
        return GRAPH_FAILED;
      }
      for (int64_t i = 0; i < length_bias; i++) {
        if (dims_x[axis_ + i] != dims_bias[i]) {
          OP_LOGE("[ERROR] dimensions shape_x and shape_bias must be equal");
          OpsInputShapeErrReport(op.GetName(), "The dimensions of shape_x and shape_bias must be equal",
                              "shape_bias's dimension", ConcatString(dims_bias[i]));
          return GRAPH_FAILED;
        }
      }
    }
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Bias, BiasInferShape);
VERIFY_FUNC_REG(Bias, BiasVerify);
// ---------------------------------------Bias-----------------------------------------------

// ----------------------Threshold-------------------------
IMPLEMT_INFERFUNC(Threshold, ThresholdInferShape) {
  TensorDesc tensordesc_input = op.GetInputDesc("x");
  Shape input_shape = tensordesc_input.GetShape();
  DataType input_dtype = tensordesc_input.GetDataType();
  Format input_format = tensordesc_input.GetFormat();

  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(input_shape);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetFormat(input_format);

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Threshold, ThresholdVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Threshold, ThresholdInferShape);
VERIFY_FUNC_REG(Threshold, ThresholdVerify);
// ---------------------Threshold--------------------------

// ------------ConfusionMulGrad Op Begin----------------
IMPLEMT_COMMON_INFERFUNC(ConfusionMulGradInferShape) {
  auto shape = op.GetInputDesc("input0").GetShape();
  auto shape1 = op.GetInputDesc("input1").GetShape();
  DataType input_dtype = op.GetInputDesc("input0").GetDataType();
  DataType input_dtype1 = op.GetInputDesc("input1").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("output0");
  TensorDesc tensordesc_output1 = op.GetOutputDesc("output1");
  std::vector<int64_t> shapeVector = shape1.GetDims();
  int64_t dimNum = shape1.GetDimNum();
  std::vector<int64_t> axis;
  if (ge::GRAPH_SUCCESS != op.GetAttr("axes", axis)) {
    OpsGetAttrErrReport(op.GetName(), "axes");
    OP_LOGE(op.GetName().c_str(),
            "The input_size op GetOpAttr"
            "ConstValue failed!");
    return GRAPH_FAILED;
  }

  bool keep_dims;
  if (ge::GRAPH_SUCCESS != op.GetAttr("keep_dims", keep_dims)) {
    OpsGetAttrErrReport(op.GetName(), "keep_dims");
    OP_LOGE(op.GetName().c_str(), "get keep_dims op GetOpAttr failed!");
    return GRAPH_FAILED;
  }

  if (axis.size() == 0) {
    for (size_t i = 0; i < shapeVector.size(); ++i) {
      axis.push_back(i);
    }
  }

  for (size_t i = 0; i < axis.size(); ++i) {
    if (axis[i] < 0) {
      axis[i] = dimNum + axis[i];
    }
  }

  std::vector<int64_t> oShapeVector;
  std::vector<int64_t>::iterator tmp;
  for (int64_t item = 0; item < dimNum; ++item) {
    tmp = std::find(axis.begin(), axis.end(), item);
    if (tmp != axis.end()) {
      // item in axis
      // If keepDims is true, current dimesion set to 1
      if (keep_dims == true) {
        oShapeVector.push_back(1);
      }
    } else {
      // item is not in ConstValueAxis
      oShapeVector.push_back(shapeVector[item]);
    }
  }

  Shape oShape(oShapeVector);
  tensordesc_output1.SetShape(oShape);
  tensordesc_output1.SetDataType(input_dtype);
  tensordesc_output.SetShape(shape);
  tensordesc_output.SetDataType(input_dtype1);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ConfusionMulGrad, ConfusionMulGradInferShape);
// ------------ConfusionMulGrad Op End----------------

// ------------ArgMaxWithK Op Begin----------------
IMPLEMT_INFERFUNC(ArgMaxWithK, ArgMaxWithKInfer) {
  auto input_dtype = op.get_input_desc_x().GetDataType();
  Shape input_shape = op.get_input_desc_x().GetShape();
  Shape origin_shape = op.get_input_desc_x().GetOriginShape();
  Format input_format = op.get_input_desc_x().GetFormat();
  Format origin_format = op.get_input_desc_x().GetOriginFormat();

  int axis = op.get_attr_axis();
  int topk = op.get_attr_topk();
  bool out_max_val = op.get_attr_out_max_val();
  bool out_max_index = true;
  if (out_max_val && axis != 10000) {
    out_max_index = false;
  }

  auto output_dtype = input_dtype;
  auto output_shape = input_shape;
  Format output_format = input_format;

  if (input_format == FORMAT_NC1HWC0) {
    if (origin_shape.GetDimNum() == 4) {
      if (origin_format == FORMAT_NCHW) {
        if (axis < 0) {
          axis = axis - 1;
        }
      } else if (origin_format == FORMAT_NHWC) {
        if (axis == -4) {
          axis = -5;
        } else if (axis == -1) {
          axis = -4;
        } else if (axis == 1) {
          axis = 2;
        } else if (axis == 2) {
          axis = 3;
        } else if (axis == 3) {
          axis = 1;
        }
      } else {
        OP_LOGE(op.GetName().c_str(), "5D tensor's origin format should in [NCHW, NHWC]");
        return GRAPH_FAILED;
      }
    } else {
      OP_LOGE(op.GetName().c_str(), "5D tensor's origin shape should be 4D tensor");
      return GRAPH_FAILED;
    }

    if (axis < 0) {
      axis = axis + 5;
    }
    if (axis == 10000 || axis == 1 || axis == 4) {
      OP_LOGE(op.GetName().c_str(), "5D tensor's axis is invalid");
      return GRAPH_FAILED;
    }
  } else if (axis < 0) {
    axis = axis + input_shape.GetDimNum();
  }

  if (axis == 10000) {
    std::vector<int64_t> output_shape_vector;
    output_shape_vector.push_back(input_shape.GetDim(0));
    output_shape_vector.push_back(topk);
    output_shape = Shape(output_shape_vector);
  } else {
    output_shape.SetDim(axis, topk);
  }

  TensorDesc indicesTensorDesc = TensorDesc(output_shape, output_format, DT_INT32);
  indicesTensorDesc.SetRealDimCnt(output_shape.GetDimNum());
  indicesTensorDesc.SetOriginShape(output_shape);
  if (!out_max_index) {
    indicesTensorDesc.SetDataType(output_dtype);
  }

  TensorDesc valuesTensorDesc = TensorDesc(output_shape, output_format, output_dtype);
  valuesTensorDesc.SetRealDimCnt(output_shape.GetDimNum());
  valuesTensorDesc.SetOriginShape(output_shape);

  op.update_output_desc_indices(indicesTensorDesc);
  op.update_output_desc_values(valuesTensorDesc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ArgMaxWithK, ArgMaxWithKVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ArgMaxWithK, ArgMaxWithKInfer);
VERIFY_FUNC_REG(ArgMaxWithK, ArgMaxWithKVerify);
// ------------ArgMaxWithK Op End----------------

// ------------Muls Op Begin----------------
IMPLEMT_VERIFIER(Muls, MulsVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(Muls, MulsInferShape) {
  auto x_shape = op.GetInputDesc("x").GetShape().GetDims();
  DataType x_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(ge::Shape(x_shape));
  y_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(Muls, MulsInferShape);
VERIFY_FUNC_REG(Muls, MulsVerify);
// ------------Muls Op End----------------
// ------------adds Op Start----------------
bool InferShapeAndTypeAdds(Operator& op, const string& x, const string& y, const string& value) {
  float value_num;
  if (op.GetAttr(value, value_num) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  TensorDesc vOutputDesc = op.GetOutputDesc(y);

  DataType input_dtype = op.GetInputDesc(x).GetDataType();
  Format input_format = op.GetInputDesc(x).GetFormat();
  ge::Shape shapeX = op.GetInputDesc(x).GetShape();

  vOutputDesc.SetShape(shapeX);
  vOutputDesc.SetDataType(input_dtype);
  vOutputDesc.SetFormat(input_format);
  op.UpdateOutputDesc(y, vOutputDesc);

  return true;
}

// ----------------Add-------------------
IMPLEMT_VERIFIER(Adds, AddsVerify) {
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(AddsInferShape) {
  if (InferShapeAndTypeAdds(op, "x", "y", "value")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(Adds, AddsInferShape);

// Registered verify function
VERIFY_FUNC_REG(Adds, AddsVerify);

// ------------adds Op End----------------
// ------------fills Op Start----------------
bool InferShapeAndTypeFills(Operator& op, const string& x, const string& y, const string& value) {
  float value_num;
  if (op.GetAttr(value, value_num) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  TensorDesc vOutputDesc = op.GetOutputDesc(y);

  DataType input_dtype = op.GetInputDesc(x).GetDataType();
  Format input_format = op.GetInputDesc(x).GetFormat();
  ge::Shape shapeX = op.GetInputDesc(x).GetShape();

  vOutputDesc.SetShape(shapeX);
  vOutputDesc.SetDataType(input_dtype);
  vOutputDesc.SetFormat(input_format);
  op.UpdateOutputDesc(y, vOutputDesc);

  return true;
}
// ----------------Add-------------------
IMPLEMT_VERIFIER(Fills, FillsVerify) {
  return GRAPH_SUCCESS;
}
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(FillsInferShape) {
  if (InferShapeAndTypeFills(op, "x", "y", "value")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(Fills, FillsInferShape);
// Registered verify function
VERIFY_FUNC_REG(Fills, FillsVerify);
// -----------fills Op End----------------
// -----------mul_no_nan start----------------
bool InferShapeAndTypeMulNoNan(Operator& op, const string& input_name1, const string& input_name2,
                               const string& output_name) {
  // vOutputDesc.push_back(op.GetInputDesc(0));
  TensorDesc vOutputDesc = op.GetOutputDesc(output_name);

  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();
  ge::Shape shapeX = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shapeY = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dimsX = shapeX.GetDims();
  std::vector<int64_t> dimsY = shapeY.GetDims();
  if (dimsX.size() < dimsY.size()) {
    std::vector<int64_t> dimsTmp = dimsX;
    dimsX = dimsY;
    dimsY = dimsTmp;
  }

  // pad 1 for small shape
  if (dimsX.size() != dimsY.size()) {
    int dec = dimsX.size() - dimsY.size();
    for (int i = 0; i < dec; i++) {
      dimsY.insert(dimsY.begin(), (int64_t)1);
    }
  }

  std::vector<int64_t> dimVec;
  for (size_t i = 0; i < dimsX.size(); i++) {
    if ((dimsX[i] != dimsY[i]) && (dimsX[i] != 1) && (dimsY[i] != 1)) {
      OP_LOGE(op.GetName().c_str(), "The %s's dimensions does not match the broadcast rule(%lu %lu).",
              op.GetName().c_str(), dimsX[i], dimsY[i]);
      return false;
    }

    int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
    dimVec.push_back(dims);
  }
  ge::Shape outputShape = ge::Shape(dimVec);

  vOutputDesc.SetShape(outputShape);
  vOutputDesc.SetDataType(input_dtype);
  vOutputDesc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, vOutputDesc);

  return true;
}

IMPLEMT_VERIFIER(MulNoNan, MulNoNanVerify) {
  DataType input_type_x1 = op.GetInputDesc("x1").GetDataType();
  DataType input_type_x2 = op.GetInputDesc("x2").GetDataType();
  if (input_type_x1 != input_type_x2) {
    OP_LOGE(op.GetName().c_str(), "The %s op dtype is not same, type1:%d, type2:%d", op.GetName().c_str(),
            input_type_x1, input_type_x2);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MulNoNanInferShape) {
  if (InferShapeAndTypeMulNoNan(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(MulNoNan, MulNoNanInferShape);
// Registered verify function
VERIFY_FUNC_REG(MulNoNan, MulNoNanVerify);
// -----------mul_no_nan Op End----------------

// ----------------------Axpy--------------------------
IMPLEMT_VERIFIER(Axpy, AxpyVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AxpyInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Axpy, AxpyInferShape);
VERIFY_FUNC_REG(Axpy, AxpyVerify);
// ---------------------Axpy END------------------------

// ------------CosineEmbeddingLoss Op Begin----------------
IMPLEMT_VERIFIER(CosineEmbeddingLoss, CosineEmbeddingLossVerify) {
  Shape shape_x1 = op.GetInputDesc("x1").GetShape();
  Shape shape_x2 = op.GetInputDesc("x2").GetShape();
  if ((shape_x1.GetDimNum() < 2) && (shape_x2.GetDimNum() < 2)) {
    OP_LOGE(op.GetName().c_str(), "input x1 or x2 dims must bigger than 1");
    return GRAPH_FAILED;
  }

  std::string reduction;
  op.GetAttr("reduction", reduction);
  if ((reduction != "mean") && (reduction != "sum") && (reduction != "none")) {
    OP_LOGE(op.GetName().c_str(), "reduction only support \"mean\", \"sum\" and \"none\"");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(CosineEmbeddingLossInferShape) {
  Shape shape_x1 = op.GetInputDesc("x1").GetShape();
  Shape shape_x2 = op.GetInputDesc("x2").GetShape();
  Shape shape_tgt = op.GetInputDesc("target").GetShape();

  vector<int64_t> x_dims_broadcast;
  vector<int64_t> tgt_dims_broadcast;

  if (!BroadCastTwoShape(op, shape_x1, shape_x2, x_dims_broadcast)) {
    OP_LOGE(op.GetName().c_str(), "input x1 and x2 shape can't broadcast");
    return GRAPH_FAILED;
  }

  // reduce aixs = 1
  x_dims_broadcast.erase(x_dims_broadcast.begin() + 1);

  Shape shape_x_broadcast(x_dims_broadcast);
  if (!BroadCastTwoShape(op, shape_x_broadcast, shape_tgt, tgt_dims_broadcast)) {
    OP_LOGE(op.GetName().c_str(), "input target shape can't broadcast to x shape");
    return GRAPH_FAILED;
  }

  float margin = 0.0;
  std::string reduction;
  (void)op.GetAttr("margin", margin);
  (void)op.GetAttr("reduction", reduction);
  OP_LOGI(op.GetName().c_str(), "setting margin:%f, reduction:%s\n", margin, reduction.c_str());

  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  Shape y_shape = Shape(tgt_dims_broadcast);
  if ((reduction == "mean") || (reduction == "sum")) {
    tensordesc_output.SetShape(Shape({1}));
  } else if (reduction == "none") {
    tensordesc_output.SetShape(y_shape);
  }

  tensordesc_output.SetDataType(DT_FLOAT);
  tensordesc_output.SetFormat(FORMAT_ND);
  (void)op.UpdateOutputDesc("y", tensordesc_output);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CosineEmbeddingLoss, CosineEmbeddingLossInferShape);
VERIFY_FUNC_REG(CosineEmbeddingLoss, CosineEmbeddingLossVerify);
// ------------CosineEmbeddingLoss Op End----------------

// ----------------------KLDiv Begin--------------------------
IMPLEMT_VERIFIER(KLDiv, KLDivVerify) {
  if (!CheckInputsShapeDtypeSame(op, {"x", "target"})) {
    return GRAPH_FAILED;
  }

  std::vector<std::string> const_attr;
  if (!GetConstAttr(op, {"reduction"}, const_attr)) {
    OP_LOGE(op.GetName().c_str(), "The GetOpAttr ConstValue failed!");
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(KLDivInferShape) {
  std::vector<int64_t> o_shape_vector;
  Shape o_shape(o_shape_vector);

  DataType dtype_x = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");

  tensordesc_output.SetShape(ge::Shape(o_shape));
  tensordesc_output.SetDataType(dtype_x);
  if (op.UpdateOutputDesc("y", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(KLDiv, KLDivInferShape);
VERIFY_FUNC_REG(KLDiv, KLDivVerify);
// ---------------------KLDiv End------------------------

// ----------------TensorMove Begin-------------------
COMMON_INFER_FUNC_REG(TensorMove, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// ----------------TensorMove END---------------------

// ----------------TensorRedirect Begin-------------------
COMMON_INFER_FUNC_REG(TensorRedirect, ELMTWISE_INFER_SHAPEANDTYPE("x", "output_x"));
// --------------TensorRedirect END-----------------------

}  // namespace ge
