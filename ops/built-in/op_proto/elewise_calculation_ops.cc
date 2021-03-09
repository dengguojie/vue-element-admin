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
#include "register/infer_data_slice_registry.h"
#include "graph/debug/ge_attr_define.h"

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
  if (dim_x.size() != dim_y.size()) {
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
    if(!OneInOneOutDynamicInfer(op,"x1",{"y1"})){
        return false;
    }
  }
  if (attr_grad_y) {
    if(!OneInOneOutDynamicInfer(op,"x2",{"y2"})){
        return false;
    }
  }

  return true;
}

IMPLEMT_COMMON_INFERFUNC(TwoInOneOutCommonInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(OneInOneOutCommonInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// --------------------------elewise data slice begin--------------------------
static void InferElewiseTwoInput(vector<vector<int64_t>>& in_data_slice, const vector<vector<int64_t>> out_data_slice,
                                 const vector<int64_t> in_dims, const vector<int64_t> out_dims) {
  if (in_dims.size() == out_dims.size()) {
    for (size_t i = 0; i < in_dims.size(); i++) {
      if (in_dims[i] == 1) {
        in_data_slice.push_back({0, 1});
      } else {
        in_data_slice.push_back(out_data_slice[i]);
      }
    }
  } else {
    for (size_t i = 0; i < in_dims.size(); i++) {
      if (in_dims[i] == 1) {
        in_data_slice.push_back({0, 1});
      } else {
        in_data_slice.push_back(out_data_slice[out_dims.size() - in_dims.size() + i]);
      }
    }
  }
}

static void InferElewiseTwoInputdif(vector<vector<int64_t>>& in_data_slice, const vector<vector<int64_t>> out_data_slice,
                                    const vector<int64_t> in_dims, const vector<int64_t> out_dims, const int64_t aixs) {
  if (in_dims.size() == out_dims.size()) {
    for (size_t i = 0; i < in_dims.size(); i++) {
      if (in_dims[i] == 1) {
        in_data_slice.push_back({0, 1});
      } else {
        in_data_slice.push_back(out_data_slice[i]);
      }
    }
  } else if (in_dims.size() == 1) {
    in_data_slice.push_back({out_data_slice[aixs][0] * 16, out_data_slice[aixs][1] * 16});
  }
}

IMPLEMT_COMMON_INFER_DATA_SLICE(ElewiseTwoInputInferDataSlice) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (!op_desc) {
    OP_LOGW(op.GetName().c_str(), "GetOpDescFromOperator failed.");
    return GRAPH_FAILED;
  }

  auto tensor_desc_in_x1 = op_desc->MutableInputDesc("x1");
  if (!tensor_desc_in_x1) {
    OP_LOGW(op.GetName().c_str(), "Get input desc x1 failed.");
    return GRAPH_FAILED;
  }
  auto x1_shape = tensor_desc_in_x1->MutableShape();
  auto x1_format = tensor_desc_in_x1->GetFormat();
  std::vector<int64_t> x1_dims = x1_shape.GetDims();

  auto tensor_desc_in_x2 = op_desc->MutableInputDesc("x2");
  if (!tensor_desc_in_x2) {
    OP_LOGW(op.GetName().c_str(), "Get input desc x2 failed.");
    return GRAPH_FAILED;
  }
  auto x2_shape = tensor_desc_in_x2->MutableShape();
  auto x2_format = tensor_desc_in_x2->GetFormat();
  std::vector<int64_t> x2_dims = x2_shape.GetDims();

  auto tensor_desc_out_y = op_desc->MutableOutputDesc("y");
  if (!tensor_desc_out_y) {
    OP_LOGW(op.GetName().c_str(), "Get input desc y failed.");
    return GRAPH_FAILED;
  }
  auto y_shape = tensor_desc_out_y->MutableShape();
  auto y_format = tensor_desc_out_y->GetFormat();
  std::vector<int64_t> y_dims = y_shape.GetDims();

  vector<vector<int64_t>> y_data_slice = {};
  vector<vector<int64_t>> x1_data_slice = {};
  vector<vector<int64_t>> x2_data_slice = {};
  if (!ge::AttrUtils::GetListListInt(tensor_desc_out_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGW(op.GetName().c_str(), "no data slice, use default as {}");
    return GRAPH_FAILED;
  }

  if (x1_format == x2_format) {
    InferElewiseTwoInput(x1_data_slice, y_data_slice, x1_dims, y_dims);
    InferElewiseTwoInput(x2_data_slice, y_data_slice, x2_dims, y_dims);
  } else {
    if ((x1_format == FORMAT_NC1HWC0 && (x2_dims.size() == 0 || x2_dims.size() == 1)) ||
        ((x1_dims.size() == 0 || x1_dims.size() == 1) && x2_format == FORMAT_NC1HWC0)) {
      // 5HD+ND
      InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, 1);
      InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, 1);
    } else if ((x1_format == FORMAT_FRACTAL_NZ && (x2_dims.size() == 0 || x2_dims.size() == 1)) ||
               ((x1_dims.size() == 0 || x1_dims.size() == 1) && x2_format == FORMAT_FRACTAL_NZ)) {
      // NZ+ND
      InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, y_dims.size() - 3);
      InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, y_dims.size() - 3);
    } else if ((x1_format == FORMAT_FRACTAL_Z && (x2_dims.size() == 0 || x2_dims.size() == 1)) ||
               ((x1_dims.size() == 0 || x1_dims.size() == 1) && x2_format == FORMAT_FRACTAL_Z)) {
      // F_Z+ND
      InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, 0);
      InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, 0);
    } else {
      for (size_t i = 0; i < x1_dims.size(); i++) {
        x1_data_slice.push_back({});
      }
      for (size_t i = 0; i < x2_dims.size(); i++) {
        x2_data_slice.push_back({});
      }
    }
  }

  if (!ge::AttrUtils::SetListListInt(tensor_desc_in_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice)) {
    OP_LOGW(op.GetName().c_str(), "data slice set failed");
    return GRAPH_FAILED;
  }
  if (!ge::AttrUtils::SetListListInt(tensor_desc_in_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice)) {
    OP_LOGW(op.GetName().c_str(), "data slice set failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
// --------------------------elewise data slice end--------------------------

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
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Add, ElewiseTwoInputInferDataSlice);
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
IMPLEMT_COMMON_INFERFUNC(CastInferShape) {
  // get input desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  vector<int64_t> input_shape = input_desc->MutableShape().GetDims();

  auto output_desc = op_info->MutableOutputDesc("y");
  if (IsUnknown(input_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);

    output_desc->SetShape(GeShape(input_shape));
    output_desc->SetOriginShape(GeShape(input_shape));
    output_desc->SetShapeRange(input_range);
  } else {
    output_desc->SetShape(GeShape(input_shape));
  }
  int type;
  if (op.GetAttr("dst_type", type) == GRAPH_SUCCESS) {
    output_desc->SetDataType((ge::DataType)type);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cast, CastInferShape);
// --------------Cast END-----------------

// ---------------------GreaterEqual-----------------------
IMPLEMT_VERIFIER(GreaterEqual, GreaterEqualVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(GreaterEqualInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
  return GRAPH_SUCCESS;
}

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

IMPLEMT_COMMON_INFERFUNC(LessInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
  return GRAPH_SUCCESS;
}

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


COMMON_INFER_FUNC_REG(RealDiv, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(RealDiv, RealDivVerify);
// ----------------RealDiv END------------------

// ----------------Sqrt Op Begin------------
COMMON_INFER_FUNC_REG(Sqrt, OneInOneOutCommonInferShape);
// ----------------Sqrt Op End---------------

// ----------------Maximum--------------------
IMPLEMT_VERIFIER(Maximum, MaximumVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Maximum, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(Maximum, MaximumVerify);
// --------------Maximum END------------------

// ----------------Minimum--------------------
IMPLEMT_VERIFIER(Minimum, MinimumVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Minimum, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(Minimum, MinimumVerify);
// -----------------Minimum END-----------------

// ----------------Reciprocal-------------------
COMMON_INFER_FUNC_REG(Reciprocal, OneInOneOutCommonInferShape);
// ---------------Reciprocal END-----------------

// -------------------Sub----------------------
IMPLEMT_COMMON_INFERFUNC(SubInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Sub, ElewiseTwoInputInferDataSlice);
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
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SquaredDifference, SquaredDifferenceInferShape);
// ----------------SquaredDifference END---------------

// ------------------Div---------------------
IMPLEMT_VERIFIER(Div, DivVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Div, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(Div, DivVerify);
// -----------------Div END------------------

// -------------------Equal--------------------
IMPLEMT_VERIFIER(Equal, EqualVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(EqualInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Equal, EqualInferShape);
VERIFY_FUNC_REG(Equal, EqualVerify);
// ------------------Equal END--------------------

// ----------------Exp-------------------
COMMON_INFER_FUNC_REG(Exp, OneInOneOutCommonInferShape);
// ----------------Exp END-------------------

// ----------------------Inv----------------------
IMPLEMT_COMMON_INFERFUNC(InvInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
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
    bool is_dynamic_output = true;
    if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x", "grad", "y", is_dynamic_output)) {
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

IMPLEMT_COMMON_INFERFUNC(LessEqualInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
  return GRAPH_SUCCESS;
}


COMMON_INFER_FUNC_REG(LessEqual, LessEqualInferShape);
VERIFY_FUNC_REG(LessEqual, LessEqualVerify);
// --------------------LessEqual END-----------------------

// ----------------Log1p-------------------
COMMON_INFER_FUNC_REG(Log1p, OneInOneOutCommonInferShape);
// --------------Log1p END-----------------

// -------------------NotEqual--------------------
IMPLEMT_VERIFIER(NotEqual, NotEqualVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(NotEqualInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(NotEqual, NotEqualInferShape);
VERIFY_FUNC_REG(NotEqual, NotEqualVerify);
// ------------------NotEqual END--------------------

// ----------------Neg-------------------
COMMON_INFER_FUNC_REG(Neg, OneInOneOutCommonInferShape);
// ---------------Neg EDN-----------------

// ------------------DivNoNan-----------------------
IMPLEMT_VERIFIER(DivNoNan, DivNoNanVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DivNoNanInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y",is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DivNoNan, DivNoNanInferShape);
VERIFY_FUNC_REG(DivNoNan, DivNoNanVerify);
// --------------DivNoNan END----------------------

// ----------------Invert-------------------
IMPLEMT_COMMON_INFERFUNC(InvertInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Invert, InvertInferShape);
// ----------------Invert END-------------------

// ---------------OnesLike-----------------
IMPLEMT_COMMON_INFERFUNC(OnesLikeInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})){
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
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
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "y", "dy", "z")) {
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto vec_y = op_desc->MutableOutputDesc("z")->MutableShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "y", "dy", "z")) {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReciprocalGrad, ReciprocalGradInferShape);
VERIFY_FUNC_REG(ReciprocalGrad, ReciprocalGradVerify);
// --------------ReciprocalGrad END-----------------

// ----------------Square Op Begin-----------------
COMMON_INFER_FUNC_REG(Square, OneInOneOutCommonInferShape);
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

// --------------------ClipByValue-----------------------
IMPLEMT_VERIFIER(ClipByValue, ClipByValueVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "clip_value_min") || !CheckTwoInputDtypeSame(op, "x", "clip_value_max")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ClipByValueInferShape) {
  Shape input_shape = op.GetInputDesc("x").GetShape();
  Shape input_ori_shape = op.GetInputDesc("x").GetOriginShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(Shape(input_shape));
  td.SetOriginShape(Shape(input_ori_shape));
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

COMMON_INFER_FUNC_REG(LogicalOr, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(LogicalOr, LogicalOrVerify);
// ----------------LogicalOr END--------------------

// ----------------Rsqrt-------------------
IMPLEMT_COMMON_INFERFUNC(RsqrtInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Rsqrt, RsqrtInferShape);
// ----------------Rsqrt-------------------

// ----------------Acos-------------------
IMPLEMT_COMMON_INFERFUNC(AcosInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Acos, AcosInferShape);
// --------------Acos END-----------------

// ----------------BesselI0e-------------------
IMPLEMT_COMMON_INFERFUNC(BesselI0eInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(BesselI0e, BesselI0eInferShape);
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

INFER_DATA_SLICE_FUNC_REG(Mul, ElewiseTwoInputInferDataSlice);
COMMON_INFER_FUNC_REG(Mul, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(Mul, MulVerify);
// ----------------Mul END--------------------

// ----------------SqrtGrad Op Begin-----------------
IMPLEMT_VERIFIER(SqrtGrad, SqrtGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SqrtGradInferShape) {
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
   return GRAPH_SUCCESS;
}

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

IMPLEMT_COMMON_INFERFUNC(AddNInferShape) {
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
  return GRAPH_SUCCESS;
}

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
    if (!CheckTwoInputDtypeSame(op, "var", "value")) {
    return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AssignSubInferShape) {
    if (TwoInOneOutDynamicInferNoBroadcast(op, "var", "value", {"var"})) {
    return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(AssignSub, AssignSubInferShape);
VERIFY_FUNC_REG(AssignSub, AssignSubVerify);
// ----------------AssignSub END-------------------

// ----------------Atanh-------------------
IMPLEMT_COMMON_INFERFUNC(AtanhInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Atanh, AtanhInferShape);
// --------------Atanh END-----------------

// ----------------Atan--------------------
IMPLEMT_COMMON_INFERFUNC(AtanInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Atan, AtanInferShape);
// --------------Atan END-----------------

// ----------------Atan2-------------------
IMPLEMT_VERIFIER(Atan2, Atan2Verify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(Atan2InferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}


COMMON_INFER_FUNC_REG(Atan2, Atan2InferShape);
VERIFY_FUNC_REG(Atan2, Atan2Verify);
// --------------Atan2 END-----------------

// --------------AcosGrad----------------
IMPLEMT_VERIFIER(AcosGrad, AcosGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(AcosGrad, AcosGradVerify);

IMPLEMT_COMMON_INFERFUNC(AcosGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "y", "dy", "z", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AcosGrad, AcosGradInferShape);
// ------------AcosGrad END----------------

// ----------------AcoshGrad-------------------
IMPLEMT_VERIFIER(AcoshGrad, AcoshGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(AcoshGrad, AcoshGradVerify);

IMPLEMT_COMMON_INFERFUNC(AcoshGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "y", "dy", "z", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AcoshGrad, AcoshGradInferShape);
// --------------AcoshGrad END-----------------

// ----------------AtanGrad-------------------
IMPLEMT_COMMON_INFERFUNC(AtanGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "y", "dy", "z", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AtanGrad, AtanGradInferShape);
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

int64_t GetAccumulateNV2ConstValue(const ge::Operator& op) {
  int64_t tensor_num;
  if (ge::GRAPH_SUCCESS != op.GetAttr("N", tensor_num)) {
    OpsGetAttrErrReport(op.GetName(), "N");
    OP_LOGE(op.GetName().c_str(), "The add_n op GetOpAttr failed!");
  }
  return tensor_num;
}

int64_t GenerateOutshape(std::vector<int64_t> shape1, std::vector<int64_t> shape2,int64_t index) {
    if (shape1[index] < shape2[index]) {
        if (shape2[index] % shape1[index] == 0) {
            return shape2[index];
        } else {
            return -1;
        }
    } else {
        if (shape1[index] % shape2[index] == 0){
            return shape1[index];
        } else {
            return -1;
        }
    }
}

std::vector<int64_t> Broadcast(std::vector<int64_t> shape1, std::vector<int64_t> shape2) {
    std::vector<int64_t> shorter;
    std::vector<int64_t> longer;
    if (shape1.size() < shape2.size()) {
        shorter = shape1;
        longer = shape2;
    } else {
        shorter = shape2;
        longer = shape1;
    }
    int64_t l_size = longer.size();
    int64_t s_size = shorter.size();
    std::vector<int64_t> temp(l_size - s_size, 1);
    for (int64_t i  = 0; i < s_size; i++) {
        temp.push_back(shorter[i]);
    }
    std::vector<int64_t> out_shape;
    for (int64_t i  = 0; i < l_size; i++) {
        if (temp[i] == 1) {
            out_shape.push_back(longer[i]);
        } else {
            if (longer[i] == 1) {
                out_shape.push_back(temp[i]);
            } else {
                out_shape.push_back(GenerateOutshape(temp, longer, i));
            }
        }
    }
    return out_shape;
}

IMPLEMT_COMMON_INFERFUNC(AccumulateNV2InferShape) {
  /*
  Accumulate_nv2 has four type inputs:
  1.empty 2.static shape 3.-1 4.-2
  The combinations bring 15 scenes, and the 15 scenes can be classify into 4 categories:
  1.input with no range and output no need range, and it can be divided half:
    1.1 all input is empty
    1.2 input only contains empty and -2 shape
  2.input contains static shape and with no -1 shape
  3.input contains -1 shape
  */  
  const int64_t infer_condition_one_one = 11;
  const int64_t infer_condition_one_two = 12;
  const int64_t infer_condition_two = 2;
  const int64_t infer_condition_three = 3;

  int64_t empty_num = 0;
  int64_t static_num = 0;
  int64_t dynamic_shape_num = 0;
  int64_t dynamic_dim_num = 0;
  int64_t infer_classify = 0;
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  int64_t tensor_num = GetAccumulateNV2ConstValue(op);

  for (uint32_t i = 0; i < tensor_num; i++) {
    auto input_desc = op_info->MutableInputDesc(i);
    vector<int64_t> tempVector = input_desc->MutableShape().GetDims();
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
      infer_classify = infer_condition_one_one;
    } else {
      infer_classify = infer_condition_one_two;
    }
  } else if (tensor_num == static_num || tensor_num == empty_num + static_num || tensor_num == static_num +
             dynamic_dim_num || tensor_num == empty_num + static_num + dynamic_dim_num) {
    infer_classify = infer_condition_two;
  } else {
    infer_classify = infer_condition_three;
  }

  // condition 1: all input shape is empty
  if (infer_classify == 11) {
    auto input_desc = op_info->MutableInputDesc(0);
    std::vector<int64_t> shape_vector = input_desc->MutableShape().GetDims();
    DataType x_dtype = input_desc->GetDataType();
    auto y_desc = op_info->MutableOutputDesc("y");
    y_desc->SetShape(GeShape(shape_vector));
    y_desc->SetDataType(x_dtype);
  } else if (infer_classify == 12) {
    auto input_desc = op_info->MutableInputDesc( 0);
    std::vector<int64_t> shape_vector = {-2};
    DataType x_dtype = input_desc->GetDataType();
    auto y_desc = op_info->MutableOutputDesc("y");
    y_desc->SetShape(GeShape(shape_vector));
    y_desc->SetDataType(x_dtype);
  } else if (infer_classify == 2) {
    auto input_desc = op_info->MutableInputDesc(0);
    std::vector<int64_t> shape_vector = input_desc->MutableShape().GetDims();
    DataType x_dtype = input_desc->GetDataType();
    std::vector<int64_t> temp_vector;
    for (int64_t i = 1; i < tensor_num; i++) {
        auto input_desc = op_info->MutableInputDesc(i);
        temp_vector = input_desc->MutableShape().GetDims();
        if (!shape_vector.empty() && !IsUnknownRankShape(shape_vector)) {
            if (!IsUnknownRankShape(temp_vector)) {
                shape_vector = Broadcast(shape_vector, temp_vector);
                for (int64_t j = 0; j < shape_vector.size(); j++) {
                    if (shape_vector[j] == -1) {
                        OP_LOGE(op.GetName().c_str(),
                            "Operands could not be broadcast together with these shapes."); 
                        return GRAPH_FAILED;
                    }
                }
            } else {
                shape_vector = temp_vector;
                break;
            }
        }
    }
    auto y_desc = op_info->MutableOutputDesc("y");
    y_desc->SetShape(GeShape(shape_vector));
    y_desc->SetDataType(x_dtype);
    std::vector<std::pair<int64_t, int64_t>> out_range;
    MakeUpShapeRange(shape_vector, out_range);
    y_desc->SetShapeRange(out_range);
    } else {
    auto input_desc = op_info->MutableInputDesc(0);
    std::vector<int64_t> out_shape = input_desc->MutableShape().GetDims();
    DataType x_dtype = input_desc->GetDataType();
    std::vector<int64_t> out_vector;
    std::vector<std::pair<int64_t, int64_t>> out_range;
    // Init the output shape and range
    for (int64_t i = 0; i < tensor_num; i++) {
      auto input_desc = op_info->MutableInputDesc(i);

      std::vector<int64_t> temp_vector = input_desc->MutableShape().GetDims();
      if (!temp_vector.empty() && !IsUnknownRankShape(temp_vector)) {
        out_vector = temp_vector;
        input_desc->GetShapeRange(out_range);
        MakeUpShapeRange(out_vector, out_range);
        break;
      }
    }
    // compute the shape dims and range intersection
    for (int64_t i = 0; i < tensor_num; i++) {
      auto input_desc = op_info->MutableInputDesc(i);
      std::vector<int64_t> temp_vector = input_desc->MutableShape().GetDims();
      if (temp_vector.empty() || IsUnknownRankShape(temp_vector)) {
        continue;
      }
      std::vector<std::pair<int64_t, int64_t>> temp_range;
      input_desc->GetShapeRange(temp_range);
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
    auto y_desc = op_info->MutableOutputDesc("y");
    y_desc->SetShape(GeShape(out_shape));
    y_desc->SetShapeRange(out_range);
    y_desc->SetDataType(x_dtype);
  }
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
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto vec_y = op_desc->MutableOutputDesc("y")->MutableShape().GetDims();
  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcase(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Greater, GreaterInferShape);
VERIFY_FUNC_REG(Greater, GreaterVerify);
// --------------------Greater END---------------------

// --------------------ZerosLike----------------
COMMON_INFER_FUNC_REG(ZerosLike, OneInOneOutCommonInferShape);
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

// ----------------FakeQuantWithMinMaxArgs------------------
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

// -------------------FloorDiv-----------------------
IMPLEMT_VERIFIER(FloorDiv, FloorDivVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FloorDiv, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(FloorDiv, FloorDivVerify);
// ----------------FloorDiv END------------------------

// ------------------FloorMod--------------------------
IMPLEMT_VERIFIER(FloorMod, FloorModVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FloorMod, TwoInOneOutCommonInferShape);
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
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Pow, ElewiseTwoInputInferDataSlice);
COMMON_INFER_FUNC_REG(Pow, PowInferShape);
VERIFY_FUNC_REG(Pow, PowVerify);
// -------------------Pow END------------------------

// ----------------Round-------------------------------------
COMMON_INFER_FUNC_REG(Round, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// ----------------Round END---------------------------------

// ---------------------------------ArgMin--------------------------------------
IMPLEMT_COMMON_INFERFUNC(ArgMinInferShape) {
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
    // verify dimension_value
    if (dimension_value.size() != 1) {
      OP_LOGE(op.GetName().c_str(), "The length of dimension value must be equal to 1, but got %d.",
              dimension_value.size());
      return GRAPH_FAILED;
    }
    int64_t dimension = dimension_value[0] < 0 ? dimension_value[0] + x_shape.size() : dimension_value[0];
    if (dimension >= x_shape.size()) {
      OP_LOGE(op.GetName().c_str(),
              "The dimension value must be range at input shape size, but got dimension value %d, input shape size %d.",
              dimension_value[0], x_shape.size());
      return GRAPH_FAILED;
    }

    vector<int64_t> output_shape(x_shape);
    output_shape.erase(output_shape.begin() + dimension);
    y_desc->SetShape(GeShape(output_shape));

    // when output is dynamic will update range
    if (IsUnknown(output_shape)) {
      std::vector<std::pair<int64_t, int64_t>> input_range;
      input_desc->GetShapeRange(input_range);
      MakeUpShapeRange(x_shape, input_range);
      input_range.erase(input_range.begin() + dimension);
      y_desc->SetShapeRange(input_range);
    }
    return GRAPH_SUCCESS;
  }

  // dimension is not const, set all output is -1, range is [1, -1]
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

COMMON_INFER_FUNC_REG(ArgMin, ArgMinInferShape);
// -------------------------------ArgMin----------------------------------------

// --------------------------------ArgMinD--------------------------------------

IMPLEMT_COMMON_INFERFUNC(ArgMinDInferShape) {
  // get all input desc
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto y_desc = op_info->MutableOutputDesc("y");
  // get x shape
  auto x_shape = input_desc->MutableShape().GetDims();

  // set output dtype
  y_desc->SetDataType(DT_INT32);

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

  int64_t dimension;
  if (GRAPH_SUCCESS != op.GetAttr("dimension", dimension)) {
    OpsGetAttrErrReport(op.GetName(), "dimension");
    OP_LOGE(op.GetName().c_str(), "GetAttr dimension failed.");
    return GRAPH_FAILED;
  }
  if (dimension < 0) {
    dimension += x_shape.size();
  }

  vector<int64_t> output_shape(x_shape);
  output_shape.erase(output_shape.begin() + dimension);
  y_desc->SetShape(GeShape(output_shape));

  // when output is dynamic will update range
  if (IsUnknown(output_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(x_shape, input_range);
    input_range.erase(input_range.begin() + dimension);
    y_desc->SetShapeRange(input_range);
  }

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
    // verify dimension_value
    if (dimension_value.size() != 1) {
      OP_LOGE(op.GetName().c_str(), "The length of dimension value must be equal to 1, but got %d.",
              dimension_value.size());
      return GRAPH_FAILED;
    }
    int64_t dimension = dimension_value[0] < 0 ? dimension_value[0] + x_shape.size() : dimension_value[0];
    if (dimension >= x_shape.size()) {
      OP_LOGE(op.GetName().c_str(),
              "The dimension value must be range at input shape size, but got dimension value %d, input shape size %d.",
              dimension_value[0], x_shape.size());
      return GRAPH_FAILED;
    }

    vector<int64_t> output_shape(x_shape);
    output_shape.erase(output_shape.begin() + dimension);
    y_desc->SetShape(GeShape(output_shape));

    // when output is dynamic will update range
    if (IsUnknown(output_shape)) {
      std::vector<std::pair<int64_t, int64_t>> input_range;
      input_desc->GetShapeRange(input_range);
      MakeUpShapeRange(x_shape, input_range);
      input_range.erase(input_range.begin() + dimension);
      y_desc->SetShapeRange(input_range);
    }
    return GRAPH_SUCCESS;
  }

  // dimension is not const, set all output is -1, range is [1, -1]
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
  // get all input desc
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto y_desc = op_info->MutableOutputDesc("y");
  // get x shape
  auto x_shape = input_desc->MutableShape().GetDims();

  // set output dtype
  y_desc->SetDataType(DT_INT32);

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

  int64_t dimension;
  if (GRAPH_SUCCESS != op.GetAttr("dimension", dimension)) {
    OpsGetAttrErrReport(op.GetName(), "dimension");
    OP_LOGE(op.GetName().c_str(), "GetAttr dimension failed.");
    return GRAPH_FAILED;
  }
  if (dimension < 0) {
    dimension += x_shape.size();
  }

  vector<int64_t> output_shape(x_shape);
  output_shape.erase(output_shape.begin() + dimension);
  y_desc->SetShape(GeShape(output_shape));

  // when output is dynamic will update range
  if (IsUnknown(output_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(x_shape, input_range);
    input_range.erase(input_range.begin() + dimension);
    y_desc->SetShapeRange(input_range);
  }

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
  if (max_size != 0) {
    dimension = dimension % max_size;
  }
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

// ----------------Erfinv-------------------
bool InferShapeAndTypeErfinv(Operator& op, const string& input_name, const string& output_name) {
    TensorDesc v_output_desc = op.GetOutputDesc(output_name);

    DataType input_dtype = op.GetInputDesc(input_name).GetDataType();
    Format input_format = op.GetInputDesc(input_name).GetFormat();
    ge::Shape shape = op.GetInputDesc(input_name).GetShape();

    v_output_desc.SetShape(shape);
    v_output_desc.SetDataType(input_dtype);
    v_output_desc.SetFormat(input_format);
    op.UpdateOutputDesc(output_name, v_output_desc);

    return true;
}

IMPLEMT_VERIFIER(Erfinv, ErfinvVerify) {
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ErfinvInferShape) {
    if (InferShapeAndTypeErfinv(op, "input_x", "output_y")) {
        return GRAPH_SUCCESS;
    }
    OP_LOGE("The InferShapeAndTypeErfinv is one input and one output.");
    return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(Erfinv, ErfinvInferShape);

// Registered verify function
VERIFY_FUNC_REG(Erfinv, ErfinvVerify);
// ----------------Erfinv END-------------------

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
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input1", "output0", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input2", "output1", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input2", "input3", "output2", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AdamApplyOneWithDecay, AdamApplyOneWithDecayVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AdamApplyOneWithDecay, AdamApplyOneWithDecayInferShape);
VERIFY_FUNC_REG(AdamApplyOneWithDecay, AdamApplyOneWithDecayVerify);
// ----------------AdamApplyOneWithDecay-------------------

// ----------------AdamApplyOne-------------------
IMPLEMT_COMMON_INFERFUNC(AdamApplyOneInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input1", "output0", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input2", "output1", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input2", "input3", "output2", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(AdamApplyOne, AdamApplyOneVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AdamApplyOne, AdamApplyOneInferShape);
VERIFY_FUNC_REG(AdamApplyOne, AdamApplyOneVerify);
// ----------------AdamApplyOne-------------------

// ----------------AdamApplyOneWithDecayAssign-------------------
IMPLEMT_COMMON_INFERFUNC(AdamApplyOneWithDecayAssignInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input1", "output0", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input2", "output1", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input2", "input3", "output2", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

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
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input1", "output0", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input0", "input2", "output1", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "input2", "input3", "output2", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

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
  auto shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
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

  if(op.UpdateOutputDesc("y", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update output failed");
    return GRAPH_FAILED;
  }
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
  OP_LOGI("Bias", "bias infer shape begin---%d", op.GetInputDesc("bias").GetShape().GetDims().size());
  DataType dtype_x = op.GetInputDesc("x").GetDataType();
  ge::Shape shape_x = op.GetInputDesc("x").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<std::pair<int64_t, int64_t>> input_range;
  op.GetInputDesc("x").GetShapeRange(input_range);
  // set output
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape_x);
  output_desc.SetDataType(dtype_x);
  output_desc.SetShapeRange(input_range);
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
    std::vector<std::pair<int64_t, int64_t>> range_bias_new;
    op.GetInputDesc("bias").GetShapeRange(range_bias_new);
    if (bias_from_blob) {
      if (num_axes == -1) {
        for (int64_t i = 0; i < axis_; i++) {
          dims_bias_tmp.insert(dims_bias_tmp.begin(), (int64_t)1);
          range_bias_new.insert(range_bias_new.begin(), {1, 1});
        }
      } else if (num_axes > 0) {
        int64_t left_length = length_x - num_axes - axis_;
        for (int64_t i = 0; i < axis_; i++) {
          dims_bias_tmp.insert(dims_bias_tmp.begin(), (int64_t)1);
          range_bias_new.insert(range_bias_new.begin(), {1, 1});
        }
        for (int64_t i = 0; i < left_length; i++) {
          dims_bias_tmp.push_back((int64_t)1);
          range_bias_new.push_back({1, 1});
        }
      }
    } else {
      int64_t left_length = length_x - length_bias - axis_;
      for (int64_t i = 0; i < axis_; i++) {
        dims_bias_tmp.insert(dims_bias_tmp.begin(), (int64_t)1);
        range_bias_new.insert(range_bias_new.begin(), {1, 1});
      }
      for (int64_t i = 0; i < left_length; i++) {
        dims_bias_tmp.push_back((int64_t)1);
        range_bias_new.push_back({1, 1});
      }
    }

    // update bias shape
    ge::Shape output_bias_shape = ge::Shape(dims_bias_tmp);
    TensorDesc bias_desc = op.GetInputDesc("bias");
    
    bias_desc.SetShape(output_bias_shape);
    bias_desc.SetOriginShape(output_bias_shape);
    bias_desc.SetShapeRange(range_bias_new);
    (void)op.UpdateInputDesc("bias", bias_desc);
  }

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
    }
  } else {
    if (bias_dim_num != 0) {
      int64_t bias_num = axis_ + length_bias;
      if (bias_num > length_x) {
        OP_LOGE("[ERROR] bias shape extends x shape when applied");
        OpsOneInputShapeErrReport(op.GetName(), "bias", "Bias shape extends x_shape when applied");
        return GRAPH_FAILED;
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
COMMON_INFER_FUNC_REG(Muls, OneInOneOutCommonInferShape);
VERIFY_FUNC_REG(Muls, MulsVerify);
// ------------Muls Op End----------------

// ------------Fills Op Start----------------
IMPLEMT_VERIFIER(Fills, FillsVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(Fills, FillsVerify);

IMPLEMT_COMMON_INFERFUNC(FillsInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter FillsInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Fills, FillsInferShape);
// -----------Fills Op End----------------

// --------------MulNoNan
IMPLEMT_VERIFIER(MulNoNan, MulNoNanVerify) {
	DataType input_type_x1 = op.GetInputDesc("x1").GetDataType();
	DataType input_type_x2 = op.GetInputDesc("x2").GetDataType();
	if (input_type_x1 != input_type_x2) {
		OP_LOGE(op.GetName().c_str(),
			"The %s op dtype is not same, type1:%d, type2:%d",
			op.GetName().c_str(), input_type_x1, input_type_x2);
		return GRAPH_FAILED;
	}
	return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(MulNoNan, MulNoNanVerify);

IMPLEMT_COMMON_INFERFUNC(MulNoNanInferShape) {
	bool is_dynamic_output = true;
	if(InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", 
											is_dynamic_output)) {
		return GRAPH_SUCCESS;
	}
	return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(MulNoNan, MulNoNanInferShape);
// ------------MulNoNan END

// ----------------------Axpy--------------------------
IMPLEMT_VERIFIER(Axpy, AxpyVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AxpyInferShape) {
  bool is_dynamic_output = true;
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
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

// ----------------------KLDiv--------------------------
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

  // get input desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);

  auto x_desc = op_info->MutableInputDesc("x");
  auto x_dtype = x_desc->GetDataType();
  std::vector<int64_t> x_dims;

  auto y_desc = op_info->MutableOutputDesc("y");

  y_desc->SetShape(GeShape(x_dims));
  y_desc->SetDataType(x_dtype);
  
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

// ----------------MaxN Begin-------------------
static bool MaxNCheckDtype(const ge::Operator& op) {
  int32_t input_num = op.GetInputsSize();
  if (input_num <= 0) {
    OP_LOGE("MaxNInferShape", "DynamicInputNum is le 0");
    return false;
  }
  ge::TensorDesc input_desc0 = op.GetDynamicInputDesc("x", 0);
  DataType data_ty0 = input_desc0.GetDataType();
  for (int i = 1; i < input_num; ++i) {
    ge::TensorDesc input_desc = op.GetDynamicInputDesc("x", i);
    DataType data_ty = input_desc.GetDataType();
    if (data_ty0 != data_ty) {
      OP_LOGE("MaxNInferShape", "DynamicInput DataType is not equal");
      return false;
    }
  }
  return true;
}

static void MaxNUpdateInferShape(std::vector<int64_t>& dims,
                             const ge::Shape input_shape) {
  int32_t dims_size = dims.size();
  std::vector<int64_t> input_dims = input_shape.GetDims();
  int32_t input_dims_size = input_dims.size();
  if (input_dims_size > dims_size) {
    for (int i = 0; i < input_dims_size - dims_size; ++i) {
      dims.insert(dims.begin(), 0);
    }
    dims_size = dims.size();
  }
  int32_t i = dims_size - input_dims_size;
  int32_t j = 0;
  while (i < dims_size && j < input_dims_size) {
    if (dims[i] < input_dims[j]) {
      dims[i] = input_dims[j];
    }
    i++;
    j++;
  }
}
IMPLEMT_COMMON_INFERFUNC(MaxNInferShape) {
  std::vector<int64_t> dims(1, 0);
  int32_t input_num = op.GetInputsSize();
  if (input_num <= 0) {
    OP_LOGE("MaxNInferShape", "DynamicInputNum is le 0");
    return GRAPH_FAILED;
  }
  for (int i = 0; i < input_num; ++i) {
    ge::TensorDesc input_desc = op.GetDynamicInputDesc("x", i);
    ge::Shape input_shape = input_desc.GetShape();
    MaxNUpdateInferShape(dims, input_shape);
  }
  ge::TensorDesc input_desc0 = op.GetDynamicInputDesc("x", 0);
  ge::TensorDesc output_desc = op.GetOutputDesc("y");
  ge::Shape inferShape(dims);
  output_desc.SetShape(inferShape);
  output_desc.SetDataType(input_desc0.GetDataType());
  output_desc.SetFormat(input_desc0.GetFormat());
  op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MaxN, MaxNVerify) {
  if (!MaxNCheckDtype(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(MaxN, MaxNInferShape);
VERIFY_FUNC_REG(MaxN, MaxNVerify);
// ----------------MaxN END---------------------

// ----------------TensorEqual Begin-------------------
bool InferShapeAndTypeTensorEqual(Operator &op, const string &input_name1,
                                  const string &input_name2,
                                  const string &output_name) {
  TensorDesc v_output_desc = op.GetOutputDesc(output_name);

  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();

  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();

  if (shape_x.GetShapeSize() != shape_y.GetShapeSize()) {
    OP_LOGE("The ShapeSize of input_x does not match input_y.");
    return false;
  }
  return true;

  std::vector<int64_t> dim_vec = {1};
  ge::Shape output_shape = ge::Shape(dim_vec);
  v_output_desc.SetShape(output_shape);
  v_output_desc.SetDataType(DT_BOOL);
  v_output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, v_output_desc);

  return true;
}

IMPLEMT_COMMON_INFERFUNC(TensorEqualInferShape) {
  if (InferShapeAndTypeTensorEqual(op, "input_x", "input_y", "output_z")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

IMPLEMT_VERIFIER(TensorEqual, TensorEqualVerify) {
  // Check whether the data types of two input tensors are the same.
  if (op.GetInputDesc("input_x").GetDataType() !=
      op.GetInputDesc("input_y").GetDataType()) {
    OP_LOGE("input_x input_y tensor dtype does not match.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TensorEqual, TensorEqualInferShape);
VERIFY_FUNC_REG(TensorEqual, TensorEqualVerify);
// ----------------TensorEqual END---------------------

void CompareBothShape(std::vector<int64_t>& dims_fst,
                      std::vector<int64_t>& dims_sec) {
  if (dims_fst.size() < dims_sec.size()) {
    std::vector<int64_t> dims_tmp = dims_fst;
    dims_fst = dims_sec;
    dims_sec = dims_tmp;
  }

  if (dims_fst.size() > dims_sec.size()) {
    int dec = dims_fst.size() - dims_sec.size();
    dims_sec.insert(dims_sec.begin(), dec, (int64_t)1);
  }
}

graphStatus ChangeShape(std::vector<int64_t>& dims_fst,
                        std::vector<int64_t>& dims_sec,
                        std::vector<int64_t>& dims_vec) {
  CompareBothShape(dims_fst, dims_sec);
  // calculate shape of output: shape[i] = max(dims_fst[i], dims_sec[i])
  for (size_t i = 0; i < dims_fst.size(); i++) {
    if ((dims_fst[i] != dims_sec[i]) && (dims_fst[i] != 1) &&
        (dims_sec[i] != 1)) {
      OP_LOGE("[ERROR] dims_fst and dims_sec can not be broadcast");
      return GRAPH_FAILED;
    }

    int64_t dims = (dims_fst[i] > dims_sec[i]) ? dims_fst[i] : dims_sec[i];
    dims_vec.push_back(dims);
  }
  return GRAPH_SUCCESS;
}

graphStatus ReplenishShape(std::vector<int64_t>& dims_x,
                           std::vector<int64_t>& dims_y,
                           std::vector<int64_t>& dims_z,
                           std::vector<int64_t>& dims_vec) {
  std::vector<int64_t> dims_vec1;
  if (ChangeShape(dims_x, dims_y, dims_vec1) == GRAPH_FAILED) {
    return GRAPH_FAILED;
  }

  if (ChangeShape(dims_vec1, dims_z, dims_vec) == GRAPH_FAILED) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus InferShapeAndTypeAddcdivAndAddcmul(Operator& op,
                                               const string& input_name1,
                                               const string& input_name2,
                                               const string& input_name3,
                                               const string& output_name) {
  TensorDesc v_output_desc = op.GetOutputDesc(output_name);

  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();
  ge::Shape shape_x = op.GetInputDesc(input_name3).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  ge::Shape shape_z = op.GetInputDesc(input_name1).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  std::vector<int64_t> dims_z = shape_z.GetDims();
  if (dims_x.size() < dims_y.size()) {
    std::vector<int64_t> dims_tmp = dims_x;
    dims_x = dims_y;
    dims_y = dims_tmp;
  }

  std::vector<int64_t> dims_vec;
  if (ReplenishShape(dims_x, dims_y, dims_z, dims_vec) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "ReplenishShape run failed");
    return GRAPH_FAILED;
  }

  ge::Shape output_shape = ge::Shape(dims_vec);

  v_output_desc.SetShape(output_shape);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, v_output_desc);

  return GRAPH_SUCCESS;
}

// ----------------Addcdiv begin-------------------
IMPLEMT_VERIFIER(Addcdiv, AddcdivVerify) {
  // the data type of input_data, x1 and x2 should be same.
  if (op.GetInputDesc("x2").GetDataType() !=
          op.GetInputDesc("input_data").GetDataType() ||
      op.GetInputDesc("x1").GetDataType() !=
          op.GetInputDesc("input_data").GetDataType()) {
    OP_LOGE(op.GetName().c_str(),
            "input_data data type and x1, x2 match failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(AddcdivInferShape) {
  if (InferShapeAndTypeAddcdivAndAddcmul(op, "input_data", "x1", "x2", "y") ==
      GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(Addcdiv, AddcdivInferShape);
// Registered verify function
VERIFY_FUNC_REG(Addcdiv, AddcdivVerify);
// ----------------Addcdiv end-------------------

// ----------------Addcmul begin-------------------
IMPLEMT_VERIFIER(Addcmul, AddcmulVerify) {
  // the data type of input_data,x1 and x2 should be same.
  if (op.GetInputDesc("input_data").GetDataType() !=
          op.GetInputDesc("x1").GetDataType() ||
      op.GetInputDesc("input_data").GetDataType() !=
          op.GetInputDesc("x2").GetDataType()) {
    OP_LOGE(op.GetName().c_str(),
            "input_data data type and x1,x2 match failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(AddcmulInferShape) {
  if (InferShapeAndTypeAddcdivAndAddcmul(op, "input_data", "x1", "x2", "y") ==
      GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
// Registered inferfunction
COMMON_INFER_FUNC_REG(Addcmul, AddcmulInferShape);
// Registered verify function
VERIFY_FUNC_REG(Addcmul, AddcmulVerify);
// ----------------Addcmul end-------------------

// ----------------AxpyV2 Begin-------------------
IMPLEMT_VERIFIER(AxpyV2, AxpyV2Verify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AxpyV2InferShape) {
  bool is_dynamic_output = true;
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(AxpyV2, AxpyV2InferShape);
VERIFY_FUNC_REG(AxpyV2, AxpyV2Verify);
// ----------------AxpyV2 END---------------------

// ----------------PtAdd Begin-------------------
bool InferShapeAndTypePtAdd(Operator& op, const string& input_name1,
                            const string& input_name2,
                            const string& output_name) {
  TensorDesc output_desc = op.GetOutputDesc(output_name);
  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();
  // The size of the shape dimension is exchanged.
  // Each dimension of dims_x uses the larger value of the corresponding
  // dimension in two tensors.
  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  if (dims_x.size() < dims_y.size()) {
    dims_x.swap(dims_y);
  }

  // The small shape is padded with 1.
  if (dims_x.size() != dims_y.size()) {
    int dec = dims_x.size() - dims_y.size();
    for (int i = 0; i < dec; i++) {
      dims_y.insert(dims_y.begin(), (int64_t)1);
    }
  }

  // The value of each dimension in the shape of the output tensor is the
  // larger value of the corresponding dimension in the two inputs.
  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1)) {
      OP_LOGE("The shape of x1 and x2 can not broadcast.");
      return false;
    }

    int64_t dims = (dims_x[i] > dims_y[i]) ? dims_x[i] : dims_y[i];
    dim_vec.push_back(dims);
  }
  ge::Shape output_shape = ge::Shape(dim_vec);

  output_desc.SetShape(output_shape);
  output_desc.SetDataType(input_dtype);
  output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, output_desc);

  return true;
}

IMPLEMT_VERIFIER(PtAdd, PtAddVerify) {
  // Check whether the data types of two input tensors are the same.
  if (op.GetInputDesc("x1").GetDataType() !=
      op.GetInputDesc("x2").GetDataType()) {
    OP_LOGE("x1 x2 tensor dtype does not match.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(PtAddInferShape) {
  // Check whether the data shape of two input tensors are the same.
  if (InferShapeAndTypePtAdd(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE("The shape of output y does not match that of x1 x2.");
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(PtAdd, PtAddInferShape);
VERIFY_FUNC_REG(PtAdd, PtAddVerify);
// ----------------PtAdd END---------------------

// ----------------PtMuls Begin-------------------
bool InferShapeAndTypePtMuls(Operator &op, const string &input_name1,
                             const string &input_name2,
                             const string &output_name) {
  TensorDesc v_output_desc = op.GetOutputDesc(output_name);

  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();
  // The size of the shape dimension is exchanged.
  // Each dimension of dims_x uses the larger value of the corresponding
  // dimension in two tensors.
  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  if (dims_x.size() < dims_y.size()) {
    dims_x.swap(dims_y);
  }

  // The small shape is padded with 1.
  if (dims_x.size() != dims_y.size()) {
    int dec = dims_x.size() - dims_y.size();
    for (int i = 0; i < dec; i++) {
      dims_y.insert(dims_y.begin(), (int64_t)1);
    }
  }

  // The value of each dimension in the shape of the output tensor is the
  // larger value of the corresponding dimension in the two inputs.
  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1)) {
      return false;
    }

    int64_t dims = (dims_x[i] > dims_y[i]) ? dims_x[i] : dims_y[i];
    dim_vec.push_back(dims);
  }
  ge::Shape output_shape = ge::Shape(dim_vec);

  v_output_desc.SetShape(output_shape);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, v_output_desc);

  return true;
}

IMPLEMT_COMMON_INFERFUNC(PtMulsInferShape) {
  if (InferShapeAndTypePtMuls(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE("The shape of output y does not match that of x1 x2.");
  return GRAPH_FAILED;
}

IMPLEMT_VERIFIER(PtMuls, PtMulsVerify) {
  // Check whether the data types of two input tensors are the same.
  if (op.GetInputDesc("x1").GetDataType() !=
      op.GetInputDesc("x2").GetDataType()) {
    OP_LOGE("x1 x2 tensor dtype does not match.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(PtMuls, PtMulsInferShape);
VERIFY_FUNC_REG(PtMuls, PtMulsVerify);
// ----------------PtMuls END---------------------

// ----------------PtSub Begin-------------------
bool InferShapeAndTypePtSub(Operator& op, const string& input_name1,
                            const string& input_name2,
                            const string& output_name) {
  TensorDesc output_desc = op.GetOutputDesc(output_name);
  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();
  // The size of the shape dimension is exchanged.
  // Each dimension of dims_x uses the larger value of the corresponding
  // dimension in two tensors.
  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  if (dims_x.size() < dims_y.size()) {
    dims_x.swap(dims_y);
  }

  // The small shape is padded with 1.
  if (dims_x.size() != dims_y.size()) {
    int dec = dims_x.size() - dims_y.size();
    for (int i = 0; i < dec; i++) {
      dims_y.insert(dims_y.begin(), (int64_t)1);
    }
  }

  // The value of each dimension in the shape of the output tensor is the
  // larger value of the corresponding dimension in the two inputs.
  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1)) {
      OP_LOGE("The shape of x1 and x2 can not broadcast.");
      return false;
    }

    int64_t dims = (dims_x[i] > dims_y[i]) ? dims_x[i] : dims_y[i];
    dim_vec.push_back(dims);
  }
  ge::Shape output_shape = ge::Shape(dim_vec);

  output_desc.SetShape(output_shape);
  output_desc.SetDataType(input_dtype);
  output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, output_desc);

  return true;
}

IMPLEMT_VERIFIER(PtSub, PtSubVerify) {
  // Check whether the data types of two input tensors are the same.
  if (op.GetInputDesc("x1").GetDataType() !=
      op.GetInputDesc("x2").GetDataType()) {
    OP_LOGE("x1 x2 tensor dtype does not match.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(PtSubInferShape) {
  // Check whether the data shape of two input tensors are the same.
  if (InferShapeAndTypePtSub(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE("The shape of output y does not match that of x1 x2.");
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(PtSub, PtSubInferShape);
VERIFY_FUNC_REG(PtSub, PtSubVerify);
// ----------------PtSub END---------------------

// ----------------StrideAdd Begin-------------------
bool InferShapeAndTypeStrideAdd(Operator &op, const string &input_name1,
                                const string &input_name2,
                                const string &outputName) {
  TensorDesc output_desc = op.GetOutputDesc(outputName);
  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();

  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();  // (N, x1_C1, H, W, C0)

  int64_t c1_len = 0;

  op.GetAttr("c1_len", c1_len);  // (N, c1_len, H, W, C0)
  dims_x[1] = c1_len;
  ge::Shape output_shape = ge::Shape(dims_x);

  output_desc.SetShape(output_shape);
  output_desc.SetDataType(input_dtype);
  output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(outputName, output_desc);

  return true;
}

IMPLEMT_VERIFIER(StrideAdd, StrideAddVerify) { return GRAPH_SUCCESS; }

// Obtains the processing function of the output tensor description
IMPLEMT_COMMON_INFERFUNC(StrideAddInferShape) {
  if (InferShapeAndTypeStrideAdd(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(op.GetName().c_str(), "IMPLEMT_COMMON_INFERFUNC FAILED.");
  return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(StrideAdd, StrideAddInferShape);
// Registered verify function
VERIFY_FUNC_REG(StrideAdd, StrideAddVerify);
// ----------------StrideAdd END---------------------

// ----------------MaskedScale Begin-------------------
bool VerifyMaskedScaleShapeAndType(Operator &op, DataType x_dtype, DataType mask_dtype)
{
    if ((x_dtype != DT_FLOAT) && (x_dtype != DT_FLOAT16)) {
        OP_LOGE(op.GetName().c_str(), "The input dtype of x is invalid, please check!");
        return false;
    }

    if ((mask_dtype != DT_INT8) && (mask_dtype != DT_FLOAT) && (mask_dtype != DT_FLOAT16)) {
        OP_LOGE(op.GetName().c_str(), "The input dtype of mask is invalid, please check!");
        return false;
    }

    return true;
}

IMPLEMT_VERIFIER(MaskedScale, MaskedScaleVerify) {
    TensorDesc x_tensordesc = op.GetInputDesc("x");
    DataType x_dtype = x_tensordesc.GetDataType();
    TensorDesc mask_tensordesc = op.GetInputDesc("mask");
    DataType mask_dtype = mask_tensordesc.GetDataType();

    if (false == VerifyMaskedScaleShapeAndType(op, x_dtype, mask_dtype)) {
        return GRAPH_FAILED;	
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(MaskedScaleInferShape) {
    OP_LOGI("MaskedScale", "infer shape begin");
    TensorDesc tensordesc_input = op.GetInputDesc("x");
    ge::Shape input_shape = tensordesc_input.GetShape();
    DataType input_dtype = tensordesc_input.GetDataType();
	
    TensorDesc mask_tensordesc = op.GetInputDesc("mask");
    ge::Shape mask_shape = mask_tensordesc.GetShape();
    DataType mask_dtype = mask_tensordesc.GetDataType();
	
    if (input_shape.GetShapeSize() != mask_shape.GetShapeSize()) {
        OP_LOGE(op.GetName().c_str(), "shapesize of x not match mask");
        return GRAPH_FAILED;
    }

    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetShape(input_shape);
    tensordesc_output.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("y", tensordesc_output);

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaskedScale, MaskedScaleInferShape);
VERIFY_FUNC_REG(MaskedScale, MaskedScaleVerify);
// ----------------MaskedScale END-----------

// ----------------AbsGrad-------------------
IMPLEMT_VERIFIER(AbsGrad, AbsGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(AbsGrad, AbsGradVerify);

IMPLEMT_COMMON_INFERFUNC(AbsGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "y", "dy", "z", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(AbsGrad, AbsGradInferShape);
// --------------AbsGrad END----------------

// ----------------Acosh--------------------
IMPLEMT_COMMON_INFERFUNC(AcoshInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Acosh, AcoshInferShape);
// --------------Acosh END-----------------

// ------------Adds------------------------
IMPLEMT_VERIFIER(Adds, AddsVerify) {
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(Adds, AddsVerify);

IMPLEMT_COMMON_INFERFUNC(AddsInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter AddsInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Adds, AddsInferShape);
// ------------Adds Op End-----------------

// ----------------Asin--------------------
IMPLEMT_COMMON_INFERFUNC(AsinInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Asin, AsinInferShape);
// --------------Asin END-----------------

// ----------------AsinGrad---------------
IMPLEMT_VERIFIER(AsinGrad, AsinGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(AsinGrad, AsinGradVerify);

IMPLEMT_COMMON_INFERFUNC(AsinGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "y", "dy", "z", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(AsinGrad, AsinGradInferShape);
// --------------AsinGrad END-------------

// ----------------Ceil-------------------
IMPLEMT_COMMON_INFERFUNC(CeilInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter CeilInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Ceil, CeilInferShape);
// --------------Ceil END----------------

// ----------------Cos-------------------
IMPLEMT_COMMON_INFERFUNC(CosInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter CosInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Cos, CosInferShape);
// --------------Cos END------------------

// ----------------Cosh-------------------
IMPLEMT_COMMON_INFERFUNC(CoshInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter CoshInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Cosh, CoshInferShape);
// ---------------Cosh END----------------

// ----------------Sin--------------------
IMPLEMT_COMMON_INFERFUNC(SinInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter SinInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Sin, SinInferShape);
// ---------------Sin END-----------------

// ----------------Sinh-------------------
IMPLEMT_COMMON_INFERFUNC(SinhInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter SinhInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Sinh, SinhInferShape);
// ---------------Sinh END----------------

// ---------------Tan---------------------
IMPLEMT_COMMON_INFERFUNC(TanInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter TanInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Tan, TanInferShape);
// --------------------Tan Op End----------

// ----------------Lerp Begin-------------------
bool InferShapeAndTypeLerp(Operator& op, 
                           const string& input_name1, const string& input_name2, 
                           const string& input_name3, const string& output_name) {
  TensorDesc v_output_desc = op.GetOutputDesc(output_name);

  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();

  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  ge::Shape shape_z = op.GetInputDesc(input_name3).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  std::vector<int64_t> dims_z = shape_z.GetDims();
  if (dims_x.size() < dims_y.size()) {
    std::vector<int64_t> dims_tmp = dims_x;
    dims_x = dims_y;
    dims_y = dims_tmp;
  }
  if (dims_x.size() < dims_z.size()) {
    std::vector<int64_t> dims_tmp = dims_x;
    dims_x = dims_z;
    dims_z = dims_tmp;
  }

  if (dims_x.size() != dims_y.size()) {
    int dec = dims_x.size() - dims_y.size();
    for (int i = 0; i < dec; i++) {
      dims_y.insert(dims_y.begin(), (int64_t)1);
    }
  }
  if (dims_x.size() != dims_z.size()) {
    int dec = dims_x.size() - dims_z.size();
    for (int i = 0; i < dec; i++) {
      dims_z.insert(dims_z.begin(), (int64_t)1);
    }
  }

  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1)) {
      OP_LOGE(op.GetName().c_str(), "Input shapes are not compatible.");
      return false;
    }
    if ((dims_x[i] != dims_z[i]) && (dims_x[i] != 1) && (dims_z[i] != 1)) {
      OP_LOGE(op.GetName().c_str(), "Input shapes are not compatible.");
      return false;
    }
    int64_t dims_tmp = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
    int64_t dims = dims_tmp > dims_z[i] ? dims_tmp : dims_z[i];
    dim_vec.push_back(dims);
  }
  ge::Shape output_shape = ge::Shape(dim_vec);

  v_output_desc.SetShape(output_shape);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, v_output_desc);

  return true;
}

IMPLEMT_VERIFIER(Lerp, LerpVerify) {
  DataType start_type = op.GetInputDesc("start").GetDataType();
  DataType end_type = op.GetInputDesc("end").GetDataType();
  DataType weight_type = op.GetInputDesc("weight").GetDataType();
  if (start_type != end_type) {
    OP_LOGE(op.GetName().c_str(), "Input dtypes are not the same.");
    return GRAPH_FAILED;
  }
  if (start_type != weight_type) {
    OP_LOGE(op.GetName().c_str(), "Input dtypes are not the same.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(LerpInferShape) {
    if (InferShapeAndTypeLerp(op, "start", "end", "weight", "y")) {
      return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Lerp, LerpInferShape);
VERIFY_FUNC_REG(Lerp, LerpVerify);
// ----------------Lerp END---------------------
// ----------------Asinh-------------------
IMPLEMT_COMMON_INFERFUNC(AsinhInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Asinh, AsinhInferShape);
// --------------Asinh END-----------------

// ------------------Mod--------------------
IMPLEMT_VERIFIER(Mod, ModVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(Mod, ModVerify);

IMPLEMT_COMMON_INFERFUNC(ModInferShape) {
  bool is_dynamic_output = true; 
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
	}
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Mod, ModInferShape);
// ----------------Mod END---------------

// --------------Xdivy-------------------
IMPLEMT_VERIFIER(Xdivy, XdivyVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(XdivyInferShape) {
  bool is_dynamic_output = true; 
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Xdivy, XdivyInferShape);
VERIFY_FUNC_REG(Xdivy, XdivyVerify);
// ------------Xdivy END-----------------

// ------------Xlogy---------------------
IMPLEMT_VERIFIER(Xlogy, XlogyVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(XlogyInferShape) {
  bool is_dynamic_output = true; 
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Xlogy, XlogyInferShape);
VERIFY_FUNC_REG(Xlogy, XlogyVerify);
// ------------Xlogy END------------------

// ----------------AsinhGrad-------------------
IMPLEMT_VERIFIER(AsinhGrad, AsinhGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(AsinhGrad, AsinhGradVerify);
IMPLEMT_COMMON_INFERFUNC(AsinhGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "y", "dy", "z", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AsinhGrad, AsinhGradInferShape);
// --------------AsinhGrad END-----------------

// -----------------TruncateDiv--------------------
IMPLEMT_COMMON_INFERFUNC(TruncateDivInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TruncateDiv, TruncateDivInferShape);
// -----------------TruncateDiv END----------------

// ----------------TruncateMod---------------------
IMPLEMT_COMMON_INFERFUNC(TruncateModInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(TruncateMod, TruncateModInferShape);
// ----------------TruncateMod END-----------------

// ----------------Floor---------------------
IMPLEMT_COMMON_INFERFUNC(FloorInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter FloorInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Floor, FloorInferShape);
// ----------------Floor END-----------------

// ----------------Expm1---------------------
IMPLEMT_COMMON_INFERFUNC(Expm1InferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter Expm1InferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Expm1, Expm1InferShape);
// ----------------Expm1 END-----------------


// -------------------DataCompare----------------------
IMPLEMT_VERIFIER(DataCompare, DataCompareVerify) {
  float atol_data;
  if (ge::GRAPH_SUCCESS != op.GetAttr("atol", atol_data)) {
    OpsGetAttrErrReport(op.GetName(), "atol");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of DataCompare!");
    return GRAPH_FAILED;
  }
  if (atol_data < 0) {
    OpsAttrValueErrReport(op.GetName(), "atol", ">= 0", ConcatString(atol_data));
    OP_LOGE(op.GetName().c_str(), "atol should >= 0!");
    return GRAPH_FAILED;
  }

  float rtol_data;
  if (ge::GRAPH_SUCCESS != op.GetAttr("rtol", rtol_data)) {
    OpsGetAttrErrReport(op.GetName(), "rtol");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of DataCompare!");
    return GRAPH_FAILED;
  }
  if (rtol_data < 0) {
    OpsAttrValueErrReport(op.GetName(), "rtol", ">= 0", ConcatString(rtol_data));
    OP_LOGE(op.GetName().c_str(), "rtol should >= 0!");
    return GRAPH_FAILED;
  }

  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DataCompareInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto output_desc = op_info->MutableOutputDesc("num");

  std::vector<std::pair<int64_t, int64_t>> input_range;
  std::vector<int64_t> shape_vector;
  output_desc->SetShape(GeShape(shape_vector));
  output_desc->SetOriginShape(GeShape(shape_vector));
  output_desc->SetShapeRange(input_range);
  output_desc->SetDataType(DT_FLOAT);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DataCompare, DataCompareInferShape);
VERIFY_FUNC_REG(DataCompare, DataCompareVerify);
// -------------------DataCompare-------------------------

// ---------------HardMax Begin-----------------
IMPLEMT_COMMON_INFERFUNC(HardMaxInferShape)
{
    ge::TensorDesc input_desc = op.GetInputDesc(0);
    ge::TensorDesc output_desc = op.GetOutputDesc(0);
    output_desc.SetShape(input_desc.GetShape());
    output_desc.SetFormat(input_desc.GetFormat());
    output_desc.SetDataType(input_desc.GetDataType());
    op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HardMax, HardMaxVerify)
{
    int dimension = -1;
    auto ret = op.GetAttr("axis", dimension);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE("HardMaxVerify", "OP GetAttr axis fail.");
        return GRAPH_FAILED;
        }
    ge::TensorDesc input_desc = op.GetInputDesc(0);
    ge::DataType data_type = input_desc.GetDataType();
    if (data_type != DT_FLOAT16 && data_type != DT_FLOAT) {
        OP_LOGE("HardMaxVerify", "Input DataType is not fp16 or fp32");
        return GRAPH_FAILED;
        }
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(HardMax, HardMaxInferShape);
VERIFY_FUNC_REG(HardMax, HardMaxVerify);
// ---------------HardMax END-------------------

// ---------------Dot Begin-----------------
bool InferShapeDot(Operator& op,
                   const string& input1_name,
                   const string input2_name,
                   const string output_name) {
  TensorDesc output_desc = op.GetOutputDesc(output_name);
  TensorDesc input1_desc = op.GetInputDesc(input1_name);
  TensorDesc input2_desc = op.GetInputDesc(input2_name);

  //input dim
  ge::Shape shape_input1 = input1_desc.GetShape();
  ge::Shape shape_input2 = input2_desc.GetShape();

  std::vector<int64_t> dims_input1 = shape_input1.GetDims();
  std::vector<int64_t> dims_input2 = shape_input2.GetDims();

  if(dims_input1.size() != dims_input2.size()) {
      OP_LOGE("The dim of input_x and input_y not match.");
      return false;
  }

  if(dims_input1.size() != 1) {
      OP_LOGE("The dim of input must be 1");
      return false;
  }

  if(dims_input1[0] != dims_input2[0]) {
      OP_LOGE("The 0-dim of input_x and input_y not match.");
      return false;
  }
  
  std::vector<int64_t> dim_output;
  dim_output.push_back(1);

  ge::Shape output_shape = ge::Shape(dim_output);

  output_desc.SetShape(output_shape);
  output_desc.SetDataType(input1_desc.GetDataType());
  output_desc.SetFormat(Format::FORMAT_ND);
  op.UpdateOutputDesc(output_name, output_desc);
  return true;
}


IMPLEMT_COMMON_INFERFUNC(DotInferShape) {
  if(InferShapeDot(op, "input_x", "input_y", "output")) {
      return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}


IMPLEMT_VERIFIER(Dot, DotVerify) {
  if (op.GetInputDesc("input_x").GetDataType() != op.GetInputDesc("input_y").GetDataType()) {
      OP_LOGE("The dataType of input_x and input_y not match.");
      return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Dot, DotInferShape);
VERIFY_FUNC_REG(Dot, DotVerify);
// ---------------Dot END-------------------

// ---------------IsClose Begin-----------------
IMPLEMT_VERIFIER(IsClose, IsCloseVerify)
{
    if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(IsCloseInferShape)
{
    Format input_format = op.GetInputDesc("x1").GetFormat();
    Shape x1_shape = op.GetInputDesc("x1").GetShape();
    TensorDesc td = op.GetOutputDesc("y");
    td.SetShape(ge::Shape(x1_shape));
    td.SetDataType(DT_BOOL);
    td.SetFormat(input_format);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(IsClose, IsCloseInferShape);
VERIFY_FUNC_REG(IsClose, IsCloseVerify);
// ---------------IsClose END-----------------

// ----------------ArgMaxGrad--------------------
IMPLEMT_COMMON_INFERFUNC(ArgMaxGradInferShape) {
    Shape shape = op.GetInputDesc("var").GetShape();
    DataType input_dtype = op.GetInputDesc("var").GetDataType();
    Format input_format = op.GetInputDesc("var").GetFormat();
    TensorDesc td = op.GetOutputDesc("y");

    td.SetShape(shape);
    td.SetDataType(input_dtype);
    td.SetFormat(input_format);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

bool IsArgMaxGradCheckPass(Operator& op,
                           const string& var_name,
                           const string& indices_name,
                           const string& updates_name,
                           const string& dimmension_name) {
    TensorDesc input_var_desc = op.GetInputDesc(var_name);
    TensorDesc input_indices_desc = op.GetInputDesc(indices_name);
    TensorDesc input_updates_desc = op.GetInputDesc(updates_name);

    ge::Shape shape_indices = input_indices_desc.GetShape();
    ge::Shape shape_updates = input_updates_desc.GetShape();
    ge::Shape shape_var = input_var_desc.GetShape();

    std::vector<int64_t> shape_indices_list = shape_indices.GetDims();
    std::vector<int64_t> shape_updates_list = shape_updates.GetDims();
    std::vector<int64_t> shape_var_list = shape_var.GetDims();

    auto dim = 0;
    if (op.GetAttr(dimmension_name, dim) == GRAPH_FAILED) {
        OP_LOGE(op.GetName().c_str(), "get attr dimension failed");
        return false;
    }

    int32_t max_shape_len = shape_var.GetDimNum();
    int32_t dims = dim;
    if (dims < 0) {
        if (dims < (0 - max_shape_len)) {
            OP_LOGE(op.GetName().c_str(), "attr dimension invalid.should bigger than -max_shape_len");
            return false;
        }
        dims = dims + max_shape_len;
    } else if (dims >= max_shape_len) {
        OP_LOGE(op.GetName().c_str(), "attr dimension invalid. should less than max_shape_len");
        return false;
    }

    if ((shape_var_list.size() > 1) && 
        (shape_var_list.size() != shape_updates_list.size() + 1)) {
        OP_LOGE("The dim size of var should biger than updates(indices) 1.");
        return false;
    }
  
    if ((1 == shape_var_list.size()) && (1 != shape_updates_list.size())) {
        OP_LOGE("The dim size of var should equal updates(indices) when size=1.");
        return false;
    }
  
    if (shape_indices_list.size() != shape_updates_list.size()) {
        OP_LOGE("The dim size of indices and updates not match.");
        return false;
    }

    for (size_t i = 0; i < shape_indices.GetDimNum(); i++) {
        if (shape_indices.GetDim(i) != shape_updates.GetDim(i)) {
            OP_LOGE("The dim value of indices and updates not match.");
            return false;
        }
    }

    if (shape_var_list.size() > 1) {
        for (size_t i = 0; i < shape_indices.GetDimNum(); i++) {
            if (((i < dims) && (shape_indices.GetDim(i) != shape_var.GetDim(i))) ||
                ((i >= dims) && (shape_indices.GetDim(i) != shape_var.GetDim(i + 1)))) {
                OP_LOGE("The dim value of var and updates not match.");
                return false;
            }
        }
    }

    DataType var_dtype = input_var_desc.GetDataType();
    DataType updates_dtype = input_updates_desc.GetDataType();
    if (var_dtype != updates_dtype) {
        OP_LOGE("The dtype of var and updates not match.");
        return false;
    }

    return true;
}

IMPLEMT_VERIFIER(ArgMaxGrad, ArgMaxGradVerify) {
    if (true != IsArgMaxGradCheckPass(op, "var", "indices", "updates", "dimension")) {
        OP_LOGE(op.GetName().c_str(),"the ArgMaxGrad op inputs check fail!\n");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ArgMaxGrad, ArgMaxGradInferShape);
VERIFY_FUNC_REG(ArgMaxGrad, ArgMaxGradVerify);
// ------------------ArgMaxGrad END---------------------

// ----------------ArgMaxGradD--------------------
IMPLEMT_COMMON_INFERFUNC(ArgMaxGradDInferShape) {
    Shape shape = op.GetInputDesc("var").GetShape();
    DataType input_dtype = op.GetInputDesc("var").GetDataType();
    Format input_format = op.GetInputDesc("var").GetFormat();
    TensorDesc td = op.GetOutputDesc("y");

    td.SetShape(shape);
    td.SetDataType(input_dtype);
    td.SetFormat(input_format);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
}

bool IsArgMaxGradDCheckPass(Operator& op,
                            const string& var_name,
                            const string& indices_name,
                            const string& updates_name,
                            const string& dimmension_name,
                            const string& assist_name) {
    TensorDesc input_var_desc = op.GetInputDesc(var_name);
    TensorDesc input_indices_desc = op.GetInputDesc(indices_name);
    TensorDesc input_updates_desc = op.GetInputDesc(updates_name);
    TensorDesc input_assist_desc = op.GetInputDesc(assist_name);

    ge::Shape shape_indices = input_indices_desc.GetShape();
    ge::Shape shape_updates = input_updates_desc.GetShape();
    ge::Shape shape_var = input_var_desc.GetShape();
    ge::Shape shape_assist = input_assist_desc.GetShape();

    std::vector<int64_t> shape_indices_list = shape_indices.GetDims();
    std::vector<int64_t> shape_updates_list = shape_updates.GetDims();
    std::vector<int64_t> shape_var_list = shape_var.GetDims();
    std::vector<int64_t> shape_assist_list = shape_assist.GetDims();

    if (shape_var_list.size() != shape_assist_list.size()) {
        OP_LOGE(op.GetName().c_str(), "shape of var and assist mot match.");
        return false;
    }

    auto dim = 0;
    if (op.GetAttr(dimmension_name, dim) == GRAPH_FAILED) {
        OP_LOGE(op.GetName().c_str(), "get attr dimension failed");
        return false;
    }

    int32_t max_shape_len = shape_var.GetDimNum();
    int32_t dims = dim;
    if (dims < 0) {
        if (dims < (0 - max_shape_len)) {
            OP_LOGE(op.GetName().c_str(), "attr dimension invalid.should bigger than -max_shape_len");
            return false;
        }
        dims = dims + max_shape_len;
    } else if (dims >= max_shape_len) {
        OP_LOGE(op.GetName().c_str(), "attr dimension invalid. should less than max_shape_len");
        return false;
    }

    if ((shape_var_list.size() > 1) && 
        (shape_var_list.size() != shape_updates_list.size() + 1)) {
        OP_LOGE("The dim size of var should biger than updates(indices) 1.");
        return false;
    }

    if ((1 == shape_var_list.size()) && (1 != shape_updates_list.size())) {
        OP_LOGE("The dim size of var should equal updates(indices) when size=1.");
        return false;
    }
  
    if (shape_indices_list.size() != shape_updates_list.size()) {
        OP_LOGE("The dim size of indices and updates not match.");
        return false;
    }

    for (size_t i = 0; i < shape_indices.GetDimNum(); i++) {
        if (shape_indices.GetDim(i) != shape_updates.GetDim(i)) {
            OP_LOGE("The dim value of indices and updates not match.");
            return false;
        }
    }

    for (size_t i = 0; i < shape_var.GetDimNum(); i++) {
        if (shape_var.GetDim(i) != shape_assist.GetDim(i)) {
            OP_LOGE("The dim value of var and assist not match.");
            return false;
        }
    }

    if (shape_var_list.size() > 1) {
        for (size_t i = 0; i < shape_indices.GetDimNum(); i++) {
            if (((i < dims) && (shape_indices.GetDim(i) != shape_var.GetDim(i))) ||
                ((i >= dims) && (shape_indices.GetDim(i) != shape_var.GetDim(i + 1)))) {
                OP_LOGE("The dim value of var and updates not match.");
                return false;
            }
        }
    }

    DataType var_dtype = input_var_desc.GetDataType();
    DataType updates_dtype = input_updates_desc.GetDataType();
    if (var_dtype != updates_dtype) {
        OP_LOGE("The dtype of var and updates not match.");
        return false;
    }

    return true;
}

IMPLEMT_VERIFIER(ArgMaxGradD, ArgMaxGradDVerify) {
    if (true != IsArgMaxGradDCheckPass(op, "var", "indices", "updates", "dimension", "assist")) {
        OP_LOGE(op.GetName().c_str(),"the ArgMaxGradD op inputs check fail!\n");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ArgMaxGradD, ArgMaxGradDInferShape);
VERIFY_FUNC_REG(ArgMaxGradD, ArgMaxGradDVerify);
// ------------------ArgMaxGradD END---------------------

// ------------------CosineSimilarity---------------------
bool InferShapeAndTypeCosineSimilarity(Operator &op,
                                       const string &input_name,
                                       const string &attr_name,
                                       const string &output_name)
{
    TensorDesc v_output_desc = op.GetOutputDesc(output_name);
    DataType input_dtype = op.GetInputDesc(input_name).GetDataType();
    Format input_format = op.GetInputDesc(input_name).GetFormat();
    ge::Shape shape_x = op.GetInputDesc(input_name).GetShape();
    std::vector<int64_t> dims_x = shape_x.GetDims();
    std::vector<int64_t> dim_vec;
    int64_t attr_dim;
    op.GetAttr(attr_name, attr_dim);
    //Valid dim value [-shape_x, shape_x - 1] in Python,
    //which needs to be converted to [0, shape_x - 1] here in C++.
    if (attr_dim < 0)
    {
        attr_dim += dims_x.size();
    }
    //Shape of output Tensor is the same as input except for
    //the axis that dim points to.
    //Example.
    //Shape of input [2,3,4,5], dim = 0, then shape of output is [3,4,5]
    for (size_t i = 0; i < dims_x.size(); i++)
    {
        if (i != attr_dim)
        {
            dim_vec.push_back(dims_x[i]);
        }
    }
    ge::Shape output_shape = ge::Shape(dim_vec);
    v_output_desc.SetShape(output_shape);
    v_output_desc.SetDataType(input_dtype);
    v_output_desc.SetFormat(input_format);
    op.UpdateOutputDesc(output_name, v_output_desc);
    return true;
}

IMPLEMT_VERIFIER(CosineSimilarity, CosineSimilarityVerify)
{

    if (op.GetInputDesc("input_x1").GetDataType() != op.GetInputDesc("input_x2").GetDataType())
    {
        OP_LOGE(op.GetName().c_str(), "the op two inputs dtype need equal!\n");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(CosineSimilarityInferShape)
{
    if (InferShapeAndTypeCosineSimilarity(op, "input_x1", "dim", "output_y"))
    {
        return GRAPH_SUCCESS;
    }
    OP_LOGE(op.GetName().c_str(), "Obtains the processing function of the output tensor fail!\n");
    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(CosineSimilarity, CosineSimilarityInferShape);
VERIFY_FUNC_REG(CosineSimilarity, CosineSimilarityVerify);
// ------------------CosineSimilarity END---------------------

}  // namespace ge
