/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2021. All rights reserved.
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
#include "op_attr.h"
#include "op_log.h"
#include "op_const.h"
#include "util/util.h"
#include "util/error_util.h"
#include "util/vector_proto_profiling.h"
#include "util/reduce_infer_util.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/infer_data_slice_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/axis_type_info.h"
#include "register/infer_axis_slice_registry.h"

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
  for (size_t i = 0UL; i < dim_x.size(); i++) {
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
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, 0, 1, 0, is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(OneInOneOutCommonInferShape) {
  static const int64_t input_x_idx = 0;
  static const int64_t output_y_idx = 0;
  if (OneInOneOutDynamicInfer(op, input_x_idx, {output_y_idx})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// --------------------------elewise data slice begin--------------------------
static void InferElewiseTwoInput(vector<vector<int64_t>>& in_data_slice, const vector<vector<int64_t>> out_data_slice,
                                 const vector<int64_t> in_dims, const vector<int64_t> out_dims) {
  if (in_dims.size() == out_dims.size()) {
    for (size_t i = 0UL; i < in_dims.size(); i++) {
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
    for (size_t i = 0UL; i < in_dims.size(); i++) {
      if (in_dims[i] == 1) {
        in_data_slice.push_back({0, 1});
      } else {
        in_data_slice.push_back(out_data_slice[i]);
      }
    }
  } else if (in_dims.size() == 1 && in_dims[0] != 1) {
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
  std::vector<int64_t> y_dims = y_shape.GetDims();

  vector<vector<int64_t>> y_data_slice = {};
  vector<vector<int64_t>> x1_data_slice = {};
  vector<vector<int64_t>> x2_data_slice = {};
  if (!ge::AttrUtils::GetListListInt(tensor_desc_out_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGW(op.GetName().c_str(), "no data slice, use default as {}");
    return GRAPH_FAILED;
  }

if ((x1_format == FORMAT_NHWC and x2_format == FORMAT_ND) or (x1_format == FORMAT_ND and x2_format == FORMAT_NHWC) or
    (x1_format == x2_format)) {
    InferElewiseTwoInput(x1_data_slice, y_data_slice, x1_dims, y_dims);
    InferElewiseTwoInput(x2_data_slice, y_data_slice, x2_dims, y_dims);
  } else {
    if ((x1_format == FORMAT_NC1HWC0 && x2_dims.size() <= 1) ||
        (x1_dims.size() <= 1 && x2_format == FORMAT_NC1HWC0)) {
      // 5HD+ND
      InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, 1);
      InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, 1);
    } else if ((x1_format == FORMAT_FRACTAL_NZ && x2_dims.size() <= 1) ||
               (x1_dims.size() <= 1 && x2_format == FORMAT_FRACTAL_NZ)) {
      // NZ+ND
      InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, y_dims.size() - 3);
      InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, y_dims.size() - 3);
    } else if ((x1_format == FORMAT_FRACTAL_Z && x2_dims.size() <= 1) ||
               (x1_dims.size() <= 1 && x2_format == FORMAT_FRACTAL_Z)) {
      // F_Z+ND
      InferElewiseTwoInputdif(x1_data_slice, y_data_slice, x1_dims, y_dims, 0);
      InferElewiseTwoInputdif(x2_data_slice, y_data_slice, x2_dims, y_dims, 0);
    } else {
      x1_data_slice.assign(x1_dims.size(), {});
      x2_data_slice.assign(x2_dims.size(), {});
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

// --------------------------infer elewise axis type begin--------------------------
IMPLEMT_COMMON_INFER_AXIS_TYPE_INFO(OneInOneOutElewiseAxisType) {
  if (OneInOneOutElewiseDynamicAxisType(op, axis_type)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
// --------------------------infer elewise axis type end--------------------------

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

INFER_DATA_SLICE_FUNC_REG(Add, ElewiseTwoInputInferDataSlice);
COMMON_INFER_FUNC_REG(Add, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(Add, AddVerify);
INFER_VALUE_RANGE_DEFAULT_REG(Add);
// ---------------------Add END------------------------

// ---------------------FusedMulAdd--------------------
IMPLEMT_VERIFIER(FusedMulAdd, FusedMulAddVerify) {
  DataType input_type_x1 = op.GetInputDesc("x1").GetDataType();
  DataType input_type_x2 = op.GetInputDesc("x2").GetDataType();
  DataType input_type_x3 = op.GetInputDesc("x3").GetDataType();
  if (input_type_x1 != input_type_x2) {
    string err_msg1 = ConcatString("The ",op.GetName().c_str()," op dtype is not same, type1:",input_type_x1, ", type2:",input_type_x2);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return false;
  }

  if (input_type_x2 != input_type_x3) {
    string err_msg1 = ConcatString("The ",op.GetName().c_str()," op dtype is not same, type2:",input_type_x2, ", type3:",input_type_x3);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return false;
  }

  return true;
}

IMPLEMT_COMMON_INFERFUNC(FusedMulAddInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  
  string input_name1 = "x1";
  string input_name2 = "x2";
  string input_name3 = "x3";
  string output_name = "y";
  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc(output_name);
  GeTensorDescPtr tensordesc_input1 = op_desc->MutableInputDesc(input_name1);
  GeTensorDescPtr tensordesc_input2 = op_desc->MutableInputDesc(input_name2);
  GeTensorDescPtr tensordesc_input3 = op_desc->MutableInputDesc(input_name3);
  CHECK(op_desc == nullptr ||
        tensordesc_output == nullptr ||
        tensordesc_input1 == nullptr ||
        tensordesc_input2 == nullptr ||
        tensordesc_input3 == nullptr,
        OP_LOGE(op.GetName().c_str(), "invalid OpDesc."), return GRAPH_FAILED);
  DataType input_dtype = tensordesc_input1->GetDataType();

  // output Desc
  tensordesc_output->SetDataType(input_dtype);

  // shape
  ge::GeShape shapeX = tensordesc_input1->GetShape();
  ge::GeShape shapeY = tensordesc_input2->GetShape();
  ge::GeShape shapeZ = tensordesc_input3->GetShape();
  OP_LOGI(op.GetName().c_str(), "shape %s: %s, shape %s: %s, shape %s: %s.",
                  input_name1.c_str(), to_string(shapeX).c_str(),
                  input_name2.c_str(), to_string(shapeY).c_str(),
                  input_name3.c_str(), to_string(shapeZ).c_str());
  std::vector<int64_t> dimsX = shapeX.GetDims();
  std::vector<int64_t> dimsY = shapeY.GetDims();
  std::vector<int64_t> dimsZ = shapeZ.GetDims();

  // unknown rank
  if (IsUnknownRankShape(dimsX) || IsUnknownRankShape(dimsY) || IsUnknownRankShape(dimsZ)) {
    tensordesc_output->SetShape(ge::GeShape(UNKNOWN_RANK));
    OP_LOGI(op.GetName().c_str(), "output shape is: %s, output dtype is:%d.",
            to_string(ge::Shape(UNKNOWN_RANK)).c_str(),
            input_dtype);
    return GRAPH_SUCCESS;
  }

  // range
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  tensordesc_input1->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_y;
  tensordesc_input2->GetShapeRange(shape_range_y);
  std::vector<std::pair<int64_t, int64_t>> shape_range_z;
  tensordesc_input3->GetShapeRange(shape_range_z);

  std::vector<int64_t> dimVec;
  std::vector<std::pair<int64_t, int64_t>> Vec_range;
  dimVec = dimsX;
  Vec_range = shape_range_x;
  MakeUpShapeRange(dimsX, shape_range_x);
  if (!TwoShapeAndRangeBroadcastIntegration(op, dimVec, Vec_range, dimsY, shape_range_y, "x1", "x2")){
    return GRAPH_FAILED;
  }
  if (!TwoShapeAndRangeBroadcastIntegration(op, dimVec, Vec_range, dimsZ, shape_range_z,
                                           "x1_broadcast", "x3")){
    return GRAPH_FAILED;
  }
  ge::GeShape outputShape = ge::GeShape(dimVec);
  tensordesc_output->SetShape(outputShape);
  tensordesc_output->SetShapeRange(Vec_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FusedMulAdd, FusedMulAddInferShape);
VERIFY_FUNC_REG(FusedMulAdd, FusedMulAddVerify);
// ---------------------FusedMulAdd END-----------------

// ---------------------FusedMulAddAdd--------------------
IMPLEMT_VERIFIER(FusedMulAddAdd, FusedMulAddAddVerify) {
  DataType input_type_x1 = op.GetInputDesc("x1").GetDataType();
  DataType input_type_x2 = op.GetInputDesc("x2").GetDataType();
  DataType input_type_x3 = op.GetInputDesc("x3").GetDataType();
  DataType input_type_x4 = op.GetInputDesc("x4").GetDataType();
  if (input_type_x1 != input_type_x2 || input_type_x2 != input_type_x3 || input_type_x3 != input_type_x4) {
    string err_msg1 = ConcatString("The ",op.GetName().c_str()," op dtype is not same, type1:", input_type_x1,
                                    ", type2:", input_type_x2, ", type3:", input_type_x3, ", type4:", input_type_x4);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FusedMulAddAddInferShape) {
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr x1Desc = opDesc->MutableInputDesc("x1");
  std::vector<int64_t> x1Dims = x1Desc->MutableShape().GetDims();
  GeTensorDescPtr x2Desc = opDesc->MutableInputDesc("x2");
  std::vector<int64_t> x2Dims = x2Desc->MutableShape().GetDims();
  GeTensorDescPtr x3Desc = opDesc->MutableInputDesc("x3");
  std::vector<int64_t> x3Dims = x3Desc->MutableShape().GetDims();
  GeTensorDescPtr x4Desc = opDesc->MutableInputDesc("x4");
  std::vector<int64_t> x4Dims = x4Desc->MutableShape().GetDims();

  GeTensorDescPtr yDesc = opDesc->MutableOutputDesc("y");
  CHECK(yDesc == nullptr, OP_LOGE(op.GetName().c_str(), "Failed to get y desc"), return GRAPH_FAILED);
  
  yDesc->SetDataType(x1Desc->GetDataType());

  if (IsUnknownVec(x1Dims) || IsUnknownVec(x2Dims) || IsUnknownVec(x3Dims) || IsUnknownVec(x4Dims)) {
    OP_LOGW(op.GetName().c_str(), "Inputs do not support dynamic shape!");
    return GRAPH_FAILED;
  }
  
  yDesc->SetShape(ge::GeShape(x1Dims));
  
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FusedMulAddAdd, FusedMulAddAddInferShape);
VERIFY_FUNC_REG(FusedMulAddAdd, FusedMulAddAddVerify);
// ---------------------FusedMulAddAdd END-----------------

// ---------------------AddV2--------------------------
IMPLEMT_VERIFIER(AddV2, AddV2Verify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AddV2, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(AddV2, AddV2Verify);
// -------------------AddV2 END----------------------

// ----------------Cast-------------------
IMPLEMT_COMMON_INFERFUNC(CastInferShape) {
  // get input desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);

  // get x input desc use idx = 0
  auto input_desc = op_info->MutableInputDesc(0);
  const GeShape &input_shape = input_desc->MutableShape();

  // get y output desc use idx = 0
  auto output_desc = op_info->MutableOutputDesc(0);

  if (input_shape.IsUnknownShape()) {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);

    output_desc->SetShape(input_shape);
    output_desc->SetOriginShape(input_shape);
    output_desc->SetShapeRange(input_range);
  } else {
    output_desc->SetShape(input_shape);
  }

  // set the output dtype base on attr: dst_type
  int type;
  if (op.GetAttr("dst_type", type) == GRAPH_SUCCESS) {
    output_desc->SetDataType((ge::DataType)type);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cast, CastInferShape);
INFER_AXIS_TYPE_INFO_REG(Cast, OneInOneOutElewiseAxisType);
INFER_VALUE_RANGE_DEFAULT_REG(Cast);
// --------------Cast END-----------------

// ---------------------GreaterEqual-----------------------
IMPLEMT_VERIFIER(GreaterEqual, GreaterEqualVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(CompareTwoOutBoolInferShape) {
  bool is_dynamic_output = true;
  static const int64_t input_x1_idx = 0;
  static const int64_t input_x2_idx = 1;
  static const int64_t output_y_idx = 0;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, input_x1_idx, input_x2_idx, output_y_idx, is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->MutableOutputDesc(output_y_idx)->SetDataType(DT_BOOL);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GreaterEqual, CompareTwoOutBoolInferShape);
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
INFER_DATA_SLICE_FUNC_REG(Sub, ElewiseTwoInputInferDataSlice);
COMMON_INFER_FUNC_REG(Sub, TwoInOneOutCommonInferShape);
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
COMMON_INFER_FUNC_REG(SquaredDifference, TwoInOneOutCommonInferShape);
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

COMMON_INFER_FUNC_REG(Equal, CompareTwoOutBoolInferShape);
VERIFY_FUNC_REG(Equal, EqualVerify);
// ------------------Equal END--------------------

// ----------------Exp-------------------
COMMON_INFER_FUNC_REG(Exp, OneInOneOutCommonInferShape);
// ----------------Exp END-------------------

// ----------------------Inv----------------------
IMPLEMT_COMMON_INFERFUNC(InvInferShape) {
  const int64_t input_x_idx = 0;
  const int64_t output_y_idx = 0;
  if (OneInOneOutDynamicInfer(op, input_x_idx, {output_y_idx})) {
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

COMMON_INFER_FUNC_REG(InvGrad, TwoInOneOutCommonInferShape);
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

COMMON_INFER_FUNC_REG(DivNoNan, TwoInOneOutCommonInferShape);
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
  static const int64_t x_input_idx = 0;
  static const int64_t clip_value_min_input_idx = 1;
  static const int64_t clip_value_max_input_idx = 2;
  static const int64_t y_output_idx = 0;
  static const std::vector<int64_t> input_idxs{x_input_idx, clip_value_min_input_idx, clip_value_max_input_idx};
  bool is_dynamic = true;
  if (!InferShapeAndTypeBroadcast(op, input_idxs, y_output_idx, is_dynamic)) {
    std::string err_msg = "InferShapeAndTypeBroadcast failed.";
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (!is_dynamic) {
    return GRAPH_SUCCESS;
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")),
        return GRAPH_FAILED);
  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc(y_output_idx);
  GeTensorDescPtr tensordesc_input1 = op_desc->MutableInputDesc(x_input_idx);
  GeTensorDescPtr tensordesc_input2 = op_desc->MutableInputDesc(clip_value_min_input_idx);
  GeTensorDescPtr tensordesc_input3 = op_desc->MutableInputDesc(clip_value_max_input_idx);
  CHECK(tensordesc_output == nullptr ||
        tensordesc_input1 == nullptr ||
        tensordesc_input2 == nullptr ||
        tensordesc_input3 == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")),
        return GRAPH_FAILED);
  DataType input_dtype = tensordesc_input1->GetDataType();
  tensordesc_output->SetDataType(input_dtype);
  // shape
  ge::GeShape shapeX = tensordesc_input1->GetShape();
  ge::GeShape shapeY = tensordesc_input2->GetShape();
  ge::GeShape shapeZ = tensordesc_input3->GetShape();
  OP_LOGI(TbeGetName(op), "shape x: %s, shape clip_value_min: %s, shape clip_value_max: %s.",
          shapeX.ToString().c_str(),
          shapeY.ToString().c_str(),
          shapeZ.ToString().c_str());
  std::vector<int64_t> dimsX = shapeX.GetDims();
  std::vector<int64_t> dimsY = shapeY.GetDims();
  std::vector<int64_t> dimsZ = shapeZ.GetDims();
  // range
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  tensordesc_input1->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_y;
  tensordesc_input2->GetShapeRange(shape_range_y);
  std::vector<std::pair<int64_t, int64_t>> shape_range_z;
  tensordesc_input3->GetShapeRange(shape_range_z);

  std::vector<int64_t> dimVec;
  std::vector<std::pair<int64_t, int64_t>> Vec_range;
  dimVec = dimsX;
  Vec_range = shape_range_x;
  MakeUpShapeRange(dimsX, shape_range_x);
  if (!TwoShapeAndRangeBroadcastIntegration(op, dimVec, Vec_range, dimsY, shape_range_y, "x", "clip_value_min")){
    std::string err_msg = "TwoShapeAndRangeBroadcastIntegration failed.";
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (!TwoShapeAndRangeBroadcastIntegration(op, dimVec, Vec_range, dimsZ, shape_range_z,
                                           "x_min_broadcast", "clip_value_max")){
    std::string err_msg = "TwoShapeAndRangeBroadcastIntegration failed.";
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  tensordesc_output->SetShapeRange(Vec_range);
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
IMPLEMT_COMMON_INFERFUNC(BesselI1eInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(BesselI1e, BesselI1eInferShape);
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
INFER_VALUE_RANGE_DEFAULT_REG(Mul);
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
    std::string err_msg = UpdateParamErrMsg("z");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
   return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SqrtGrad, SqrtGradInferShape);
VERIFY_FUNC_REG(SqrtGrad, SqrtGradVerify);
// ----------------SqrtGrad Op End-------------------

// ----------------Log-------------------
COMMON_INFER_FUNC_REG(Log, OneInOneOutCommonInferShape);
// ----------------Log END-------------------

// ----------------Assign-------------------
IMPLEMT_VERIFIER(Assign, AssignVerify) {
  if (!CheckTwoInputDtypeSame(op, "ref", "value")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AssignInferShape) {
  const int64_t input_value_idx = 1;
  const int64_t output_ref_idx = 0;
  if (OneInOneOutDynamicInfer(op, input_value_idx, {output_ref_idx})) {
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
    std::string err_msg = GetInputInvalidErrMsg("N");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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

COMMON_INFER_FUNC_REG(Atan2, TwoInOneOutCommonInferShape);
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

COMMON_INFER_FUNC_REG(AcosGrad, TwoInOneOutCommonInferShape);
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
    std::string err_msg = GetInputInvalidErrMsg("tolerance");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (tolerance_data < 0) {
    std::string err_msg = GetAttrValueErrMsg("tolerance_data", std::to_string(tolerance_data), ConcatString("tolerance_data >= 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ApproximateEqualInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->MutableOutputDesc("y")->SetDataType(DT_BOOL);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApproximateEqual, ApproximateEqualInferShape);
VERIFY_FUNC_REG(ApproximateEqual, ApproximateEqualVerify);
// -------------------ApproximateEqual-------------------------

// --------------------AccumulateNV2--------------------------
bool CheckInputSize(const Operator& op) {
  const char *op_name = "AccumulateNV2";
  auto input_size = op.GetInputsSize();
  if (input_size == 0) {
    std::string err_msg  = string("The op input size is zero");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg);
    return false;
  }
  return true;
}

bool CheckDynamicInputDtype(const Operator& op, const string& input_name1) {
  const char *op_name = "AccumulateNV2";
  DataType first_input_dtype = op.GetDynamicInputDesc(input_name1, 0).GetDataType();
  auto input_dynamic_size = op.GetInputsSize();
  for (size_t i = 0; i < input_dynamic_size; ++i) {
    DataType input_dtype = op.GetDynamicInputDesc(input_name1, i).GetDataType();
    if (first_input_dtype != input_dtype) {
      std::string err_msg = OtherErrMsg(ConcatString("the op type is not same,type1:",input_dtype,",type2:",first_input_dtype));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg);
      return false;
    }
  }
  return true;
}

IMPLEMT_VERIFIER(AccumulateNV2, AccumulateNV2Verify) {
  const char *op_name = "AccumulateNV2";
  if (CheckInputSize(op) == false) {
    return GRAPH_FAILED;
  }
  if (CheckDynamicInputDtype(op, "x") == false) {
    return GRAPH_FAILED;
  }
  int64_t num;
  if (GRAPH_SUCCESS != op.GetAttr("N", num)) {
    std::string err_msg = GetInputInvalidErrMsg("N");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg);
    return GRAPH_FAILED;
  } else {
    if (op.GetInputsSize() != static_cast<uint64_t>(num)) {
      string err_msg1 = ConcatString("The ",op.GetName().c_str()," op size is not same, op.GetInputsSize():",op.GetInputsSize(), ", num:",static_cast<uint64_t>(num));
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

int64_t GetAccumulateNV2ConstValue(const ge::Operator& op) {
  int64_t tensor_num;
  const char *op_name = "AccumulateNV2";
  if (ge::GRAPH_SUCCESS != op.GetAttr("N", tensor_num)) {
    std::string err_msg = GetInputInvalidErrMsg("N");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg);
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
  const char *op_name = "AccumulateNV2";
  OP_LOGD(op_name, "AccumulateNV2InferShape begin.");
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

  for (uint32_t i = 0U; i < tensor_num; i++) {
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
                for (size_t j = 0UL; j < shape_vector.size(); j++) {
                    if (shape_vector[j] == -1) {
                        std::string err_msg = OtherErrMsg("Operands could not be broadcast together with these shapes.");
                        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg);
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
    y_desc->SetDataType(x_dtype);
    y_desc->SetShape(GeShape(out_shape));
    y_desc->SetShapeRange(out_range);
  }
  OP_LOGD(op_name, "AccumulateNV2InferShape end.");
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

COMMON_INFER_FUNC_REG(Greater, CompareTwoOutBoolInferShape);
VERIFY_FUNC_REG(Greater, GreaterVerify);
// --------------------Greater END---------------------

// --------------------ZerosLike----------------
COMMON_INFER_FUNC_REG(ZerosLike, OneInOneOutCommonInferShape);
// ----------------ZerosLike END-----------------

// ----------------LogicalNot-------------------
COMMON_INFER_FUNC_REG(LogicalNot, OneInOneOutCommonInferShape);
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
  int64_t num_bits = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("num_bits", num_bits)) {
    std::string err_msg = GetInputInvalidErrMsg("num_bits");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (ge::GRAPH_SUCCESS != op.GetAttr("narrow_range", narrow_range)) {
    std::string err_msg = GetInputInvalidErrMsg("narrow_range");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (num_bits < 2 || num_bits > 16) {
    string num_bits_range = ConcatString("2,16");
    std::string err_msg = GetParamOutRangeErrMsg("num_bits", num_bits_range, std::to_string(num_bits));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  Shape shape_x = op.GetInputDesc("x").GetShape();
  Shape shape_min = op.GetInputDesc("min").GetShape();
  Shape shape_max = op.GetInputDesc("max").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_min = shape_min.GetDims();
  std::vector<int64_t> dims_max = shape_max.GetDims();
  if (dims_x.size() < 1) {
    std::string err_msg = GetAttrValueErrMsg("dims_x", std::to_string(dims_x.size()), ConcatString("dims_x.size()",">=",1));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if ((dims_min.size() != 1) || (dims_max.size() != 1)) {
    string input = ConcatString("dims_min.size(),","dims_max.size()");
    string expected_list = ConcatString("shape of min and max must be rank 1");
    string input_list = ConcatString("dims_min.size():",dims_min.size(),",","dims_max.size():",dims_max.size());
    std::string err_msg = GetAttrValueErrMsg(input, input_list, expected_list);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (dims_min[0] != dims_max[0]) {
    std::string err_msg = GetAttrValueErrMsg("dims_min[0]", std::to_string(dims_min[0]), std::to_string(dims_max[0]));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (dims_x[dims_x.size() - 1] != dims_min[0]) {
    string excepted_value = ConcatString("same as min[", dims_min[0], "]");
    std::string err_msg = GetAttrSizeErrMsg("dims_x", std::to_string(dims_x[dims_x.size() - 1]), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
IMPLEMT_COMMON_INFERFUNC(RintInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Rint, RintInferShape);
// ----------------Rint END-------------------------

// --------------------------------BiasAdd-------------------------------------
IMPLEMT_VERIFIER(BiasAdd, BiasAddVerify) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }
  if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NDHWC" && data_format != "NCDHW") {
    string expected_format_list = ConcatString("NHWC, NCHW, NDHWC, NCDHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg(op.GetName().c_str(), expected_format_list, data_format);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(BiasAddInferShape) {
  const int64_t input_x_idx = 0;
  const int64_t output_y_idx = 0;
  if (!OneInOneOutDynamicInfer(op, input_x_idx, {output_y_idx})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BiasAdd, BiasAddInferShape);
VERIFY_FUNC_REG(BiasAdd, BiasAddVerify);
// ----------------------------------BiasAdd END-----------------------------

// -------------------BitwiseAnd----------------------------
IMPLEMT_COMMON_INFERFUNC(BitwiseAndInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BitwiseAnd, BitwiseAndInferShape);
// ----------------BitwiseAnd END--------------------------

// ---------------------BitwiseOr----------------------------
IMPLEMT_COMMON_INFERFUNC(BitwiseOrInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BitwiseOr, BitwiseOrInferShape);
// --------------------BitwiseOr END------------------------

// -----------------------BitwiseXor-------------------------
IMPLEMT_COMMON_INFERFUNC(BitwiseXorInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BitwiseXor, BitwiseXorInferShape);
// ------------------BitwiseXor END-------------------------

// ----------------FakeQuantWithMinMaxArgs------------------
IMPLEMT_VERIFIER(FakeQuantWithMinMaxArgs, FakeQuantWithMinMaxArgsVerify) {
  float min = 0.0;
  if (GetConstValue(op, "min", min) == false) {
    std::string err_msg = GetInputInvalidErrMsg("min");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  float max = 0.0;
  if (GetConstValue(op, "max", max) == false) {
    std::string err_msg = GetInputInvalidErrMsg("max");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  int64_t numBits = 0;
  if (GetConstValue(op, "num_bits", numBits) == false) {
    std::string err_msg = GetInputInvalidErrMsg("mub_bits");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (GetConstValue(op, "narrow_range", narrow_range) == false) {
    std::string err_msg = GetInputInvalidErrMsg("narrow_range");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (min >= max) {
    string excepted_value = ConcatString("less than max[", max, "]");
    std::string err_msg = GetAttrValueErrMsg("min", std::to_string(min), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (numBits < 2 || numBits > 16) {
    string numBits_range = ConcatString("2,16");
    std::string err_msg = GetParamOutRangeErrMsg("numBits", numBits_range, std::to_string(numBits));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  float min = 0.0;
  if (GetConstValue(op, "min", min) == false) {
    std::string err_msg = GetInputInvalidErrMsg("min");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  float max = 0.0;
  if (GetConstValue(op, "max", max) == false) {
    std::string err_msg = GetInputInvalidErrMsg("max");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  int64_t num_bits = 0;
  if (GetConstValue(op, "num_bits", num_bits) == false) {
    std::string err_msg = GetInputInvalidErrMsg("num_bits");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (GetConstValue(op, "narrow_range", narrow_range) == false) {
    std::string err_msg = GetInputInvalidErrMsg("narrow_range");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (min >= max) {
    string excepted_value = ConcatString("less than max[", max, "]");
    std::string err_msg = GetAttrValueErrMsg("min", std::to_string(min), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (num_bits < 2 || num_bits > 16) {
    string num_bits_range = ConcatString("2, 16");
    std::string err_msg = GetParamOutRangeErrMsg("num_bits", num_bits_range, std::to_string(num_bits));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetAttrSizeErrMsg("x'shape", std::to_string(dims_x.size()), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);

    return GRAPH_FAILED;
  } else {
    for (size_t i = 0; i < dims_x.size(); i++) {
      if (dims_x[i] != dims_y[i]) {
        string excepted_value = ConcatString("same as gradients[", dims_y[i], "]");
        std::string err_msg = GetAttrSizeErrMsg("x'shape", std::to_string(dims_x[i]), excepted_value);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  int64_t num_bits = 0;
  if (GetConstValue(op, "num_bits", num_bits) == false) {
    std::string err_msg = GetInputInvalidErrMsg("num_bits");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (GetConstValue(op, "narrow_range", narrow_range) == false) {
    std::string err_msg = GetInputInvalidErrMsg("narrow_range");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "x", "min")) {
    return GRAPH_FAILED;
  }
  if (!CheckTwoInputDtypeSame(op, "min", "max")) {
    return GRAPH_FAILED;
  }
  if (num_bits < 2 || num_bits > 16) {
    string num_bits_range = ConcatString("2, 16");
    std::string err_msg = GetParamOutRangeErrMsg("num_bits", num_bits_range, std::to_string(num_bits));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FakeQuantWithMinMaxVarsInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter FakeQuantWithMinMaxVarsInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})){
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(FakeQuantWithMinMaxVars, FakeQuantWithMinMaxVarsInferShape);
VERIFY_FUNC_REG(FakeQuantWithMinMaxVars, FakeQuantWithMinMaxVarsVerify);
// ----------------FakeQuantWithMinMaxVars--------------------------------------

// ----------------FakeQuantWithMinMaxVarsGradient------------------------------
IMPLEMT_VERIFIER(FakeQuantWithMinMaxVarsGradient, FakeQuantWithMinMaxVarsGradientVerify) {
  int64_t num_bits = 0;
  if (GetConstValue(op, "num_bits", num_bits) == false) {
    std::string err_msg = GetInputInvalidErrMsg("num_bits");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (GetConstValue(op, "narrow_range", narrow_range) == false) {
    std::string err_msg = GetInputInvalidErrMsg("narrow_range");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    string num_bits_range = ConcatString("2, 16");
    std::string err_msg = GetParamOutRangeErrMsg("num_bits", num_bits_range, std::to_string(num_bits));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  Shape shape_x = op.GetInputDesc("x").GetShape();
  Shape shape_y = op.GetInputDesc("gradients").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  if (dims_x.size() != dims_y.size()) {
    string err_msg1 = ConcatString("The dim size is not same, dims_x.size():",dims_x.size(), ", dims_y.size():",dims_y.size());
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  } else {
    for (size_t i = 0; i < dims_x.size(); i++) {
      if (dims_x[i] != dims_y[i]) {
        string err_msg1 = ConcatString("The dim size is not same, dims_x:",dims_x[i], ", dims_y:",dims_y[i]);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  int64_t num_bits = 0;
  if (GetConstValue(op, "num_bits", num_bits) == false) {
    std::string err_msg = GetInputInvalidErrMsg("num_bits");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  bool narrow_range;
  if (GetConstValue(op, "narrow_range", narrow_range) == false) {
    std::string err_msg = GetInputInvalidErrMsg("narrow_range");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    string num_bits_range = ConcatString("2, 16");
    std::string err_msg = GetParamOutRangeErrMsg("num_bits", num_bits_range, std::to_string(num_bits));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
      string err_msg1 = ConcatString("The dim size is not same, dims_x.size():",dims_x.size(), ", dims_y.size():",dims_y.size()); 
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  } else {
    for (size_t i = 0; i < dims_x.size(); i++) {
      if (dims_x[i] != dims_y[i]) {
        string err_msg1 = ConcatString("The dim size is not same, dims_x:",dims_x[i], ", dims_y:",dims_y[i]); 
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    }
  }
  if ((dims_min.size() != 1) || (dims_max.size() != 1)) {
    string err_msg1 = ConcatString("shape of min and max must be rank 1 ,dims_min.size():",dims_min.size(), ", dims_max.size():",dims_max.size()); 
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (dims_min[0] != dims_max[0]) {
    string err_msg1 = ConcatString("shape of min and max must be same ,dims_min[0]:",dims_min[0], ", dims_max[0]:",dims_max[0]); 
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (dims_x[dims_x.size() - 1] != dims_min[0]) {
    string err_msg1 = ConcatString("The last dimension of x must be the same as min ,dims_min[0]:",dims_min[0], ", dims_x[dims_x.size() - 1]:",dims_x[dims_x.size() - 1]); 
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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

INFER_DATA_SLICE_FUNC_REG(Pow, ElewiseTwoInputInferDataSlice);
COMMON_INFER_FUNC_REG(Pow, TwoInOneOutCommonInferShape);
VERIFY_FUNC_REG(Pow, PowVerify);
// -------------------Pow END------------------------

// ----------------Round-------------------------------------
IMPLEMT_COMMON_INFERFUNC(RoundInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Round, RoundInferShape);
// ----------------Round END---------------------------------

// ---------------------------------ArgMin--------------------------------------
IMPLEMT_COMMON_INFERFUNC(ArgMinInferShape) {
  // get all input desc
  const vector<string> depend_names = {"dimension"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto op_info_arg = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info_arg->MutableInputDesc("x");
  auto const_desc = op_info_arg->MutableInputDesc("dimension");
  auto y_desc = op_info_arg->MutableOutputDesc("y");

  // get and set output dtype
  ge::DataType dtype;
  if (op.GetAttr("dtype", dtype) == GRAPH_SUCCESS) {
    y_desc->SetDataType(dtype);
  } else {
    OP_LOGW(op.GetName().c_str(), "get attr dtype failed.");
    y_desc->SetDataType(DT_INT32);
  }

  // get x shape
  auto x_shape = input_desc->MutableShape().GetDims();
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
  vector<int64_t> dimension_value;
  auto dimension_idx = static_cast<uint32_t>(op_desc->GetInputIndexByName("dimension"));
  const GeTensor *dimension_tensor = OpDescUtils::GetInputConstData(op, dimension_idx);
  if (dimension_tensor != nullptr) {
    auto const_dtype = const_desc->GetDataType();
    GetConstValue(op, dimension_tensor, const_dtype, dimension_value);
    // verify dimension_value
    if (dimension_value.size() != 1) {
      string error_msg = ConcatString(
          "the element size of input[dimension] should be equal to 1, but get ",
          dimension_value.size(), ".");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), error_msg);
      return GRAPH_FAILED;
    }
    int64_t dimension = dimension_value[0] < 0 ? dimension_value[0] + x_shape.size() : dimension_value[0];
    if (dimension >= static_cast<int64_t>(x_shape.size())) {
      string error_msg = ConcatString(
          "the value of input[dimension] must be range at input shape size,",
          " but get input[dimension] value ", dimension_value[0],
          ", input[x] shape size ", x_shape.size(), ".");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), error_msg);
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
  std::vector<std::pair<int64_t, int64_t>> output_range;
  vector<int64_t> output_shape;
  for (size_t item = 0; item < (x_shape.size() - 1); ++item) {
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
  auto inputx_desc = op_info->MutableInputDesc("x");
  auto y_desc = op_info->MutableOutputDesc("y");
  // get x shape
  auto x_shape = inputx_desc->MutableShape().GetDims();

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
    std::string err_msg = GetInputInvalidErrMsg("dimension");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    inputx_desc->GetShapeRange(input_range);
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
  auto op_info_arg = OpDescUtils::GetOpDescFromOperator(op);
  static const int64_t x_input_idx = 0;
  static const int64_t y_output_idx = 0;
  auto input_desc = op_info_arg->MutableInputDesc(x_input_idx);
  auto y_desc = op_info_arg->MutableOutputDesc(y_output_idx);
  // get x shape
  const ge::GeShape& x_shape = input_desc->MutableShape();

  // get and set output dtype
  ge::DataType dtype;
  if (op.GetAttr("dtype", dtype) == GRAPH_SUCCESS) {
    y_desc->SetDataType(dtype);
  } else {
    OP_LOGW(op.GetName().c_str(), "get attr dtype failed.");
    y_desc->SetDataType(DT_INT32);
  }

  // if x_shape == -2, set output -2
  if (IsUnknownRankShape(x_shape)) {
    y_desc->SetShape(x_shape);
    return GRAPH_SUCCESS;
  }

  // if x_shape.size() < 2, set output scalar
  if (x_shape.GetDimNum() <= 1) {
    vector<int64_t> output_dims;
    y_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // read dimension const value
  int64_t dimension = 0;
  static const int64_t dimension_input_idx = 1;
  if (ops::GetConstInt(op, dimension_input_idx, dimension)) {
    dimension = dimension < 0 ? dimension + x_shape.GetDimNum() : dimension;
    if ((dimension < 0) || (dimension >= static_cast<int64_t>(x_shape.GetDimNum()))) {
      OP_LOGE(TbeGetName(op), "The dimension value %ld must in range of input shape size %ld.",
              dimension, x_shape.GetDimNum());
      return GRAPH_FAILED;
    }

    ge::GeShape& output_shape = y_desc->MutableShape();
    output_shape.SetDimNum(x_shape.GetDimNum() - 1);
    for (int64_t i = 0; i < dimension; i++) {
      output_shape.SetDim(i, x_shape.GetDim(i));
    }
    for (int64_t i = dimension + 1; i < x_shape.GetDimNum(); i++) {
      output_shape.SetDim(i - 1, x_shape.GetDim(i));
    }

    // when output is dynamic will update range
    if (output_shape.IsUnknownShape()) {
      std::vector<std::pair<int64_t, int64_t>> input_range;
      input_desc->GetShapeRange(input_range);
      MakeUpShapeRange(x_shape, input_range);
      input_range.erase(input_range.begin() + dimension);
      y_desc->SetShapeRange(input_range);
    }
    return GRAPH_SUCCESS;
  }

  // dimension is not const, set all output is -1, range is [0, -1]
  vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  for (size_t item = 0; item < (x_shape.GetDimNum() - 1); ++item) {
    output_dims.push_back(-1);
  }
  MakeUpShapeRange(output_dims, output_range);
  y_desc->SetShape(GeShape(output_dims));
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
    std::string err_msg = GetInputInvalidErrMsg("dimension");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  PROFILING_PROTO_INIT(op.GetName().c_str());
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_info == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), OtherErrMsg("invalid OpDesc.")),
        return GRAPH_FAILED);
  auto input_desc = op_info->MutableInputDesc(0);
  CHECK(input_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), OtherErrMsg("invalid InputDesc.")),
        return GRAPH_FAILED);
  const GeShape& input_shape = input_desc->MutableShape();
  auto input_dtype = input_desc->GetDataType();

  // get output desc
  const int64_t indice_output_idx = 0;
  const int64_t values_output_idx = 1;
  auto indice_desc = op_info->MutableOutputDesc(indice_output_idx);
  CHECK(indice_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), OtherErrMsg("invalid OutputDesc.")),
        return GRAPH_FAILED);
  auto values_desc = op_info->MutableOutputDesc(values_output_idx);
  CHECK(values_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), OtherErrMsg("invalid OutputDesc.")),
        return GRAPH_FAILED);
  indice_desc->SetDataType(DT_INT32);
  values_desc->SetDataType(input_dtype);
  // get attr dimension
  int64_t dimension;
  CHECK(GRAPH_SUCCESS != op.GetAttr("dimension", dimension),
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), GetInputInvalidErrMsg("dimension")), return GRAPH_FAILED);
  // get attr keep_dims
  bool keep_dims;
  CHECK(GRAPH_SUCCESS != op.GetAttr("keep_dims", keep_dims),
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), GetInputInvalidErrMsg("keep_dims")), return GRAPH_FAILED);

  PROFILING_PROTO_AFTER_GET_SHAPE_REG();
  // if input_shape == -2, set output -2
  if (input_shape.IsUnknownDimNum() || input_shape.IsScalar()) {
    OP_LOGD(op.GetName().c_str(), "input is UnknownDimNum, set the output is UnknownDimNum");
    indice_desc->SetShape(input_shape);
    values_desc->SetShape(input_shape);
    return GRAPH_SUCCESS;
  }

  if (dimension < 0) {
    dimension += input_shape.GetDimNum();
  }
  if (dimension >= static_cast<int64_t>(input_shape.GetDimNum())) {
    std::string err_msg = GetInputInvalidErrMsg(std::to_string(input_shape.GetDimNum()));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape& output_shape = indice_desc->MutableShape();
  if (keep_dims) {
    // If keepDims is true, current dimesion set to 1
    output_shape = input_shape;
    output_shape.SetDim(dimension, 1);
  } else {
    output_shape.SetDimNum(input_shape.GetDimNum() - 1);
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.GetDimNum()); ++i) {
      if (i < dimension) {
        output_shape.SetDim(i, input_shape.GetDim(i));
      } else if (i > dimension) {
        output_shape.SetDim(i - 1, input_shape.GetDim(i));
      }
    }
  }
  values_desc->SetShape(output_shape);

  // when output is dynamic will update range
  if (output_shape.IsUnknownShape()) {
    std::vector<int64_t> input_shape_vec = input_shape.GetDims();
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape_vec, input_range);
    if (keep_dims) {
      input_range[dimension] = {1, 1};
    } else {
      input_range.erase(input_range.begin() + dimension);
    }
    indice_desc->SetShapeRange(input_range);
    values_desc->SetShapeRange(input_range);
    return GRAPH_SUCCESS;
  }
  PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
  PROFILING_PROTO_END();
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ArgMaxWithValue, ArgMaxWithValueInferShape);
// -----------------------------ArgMaxWithValue---------------------------------
COMMON_INFER_FUNC_REG(ArgMinWithValue, ArgMaxWithValueInferShape);
// ---------------------------ArgMinWithValue-----------------------------------

// ----------------Eltwise-------------------
static const int64_t ALL_EMPTY_TENSOR = 11;
static const int64_t ONLY_EMPTY_AND_UNKNOWN_RANK_TENSOR = 12;
static const int64_t HAS_STATIC_WITHOUT_UNKNOWN_SHAPE_TENSOR = 2;
static const int64_t HAS_UNKNOWN_SHAPE_TENSOR = 3;
int64_t GetEltwiseConstValue(ge::Operator& op) {
  int64_t tensor_num;
  if (ge::GRAPH_SUCCESS != op.GetAttr("N", tensor_num)) {
    OP_LOGE(op.GetName().c_str(), "The eltwise op GetOpAttr failed!");
  }
  return tensor_num;
}

int64_t EltwiseInferClassify(ge::Operator& op, int64_t &tensor_num) {
  int64_t empty_num = 0;
  int64_t static_num = 0;
  int64_t unknown_shape_num = 0;
  int64_t unknown_rank_num = 0;

  for (int64_t i = 0; i < tensor_num; i++) {
    vector<int64_t> tempVector = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
    if (tempVector.empty()) {
      empty_num++;
      continue;
    } 
    if (IsUnKnownShape(tempVector)) {
      unknown_shape_num++;
      continue;
    } 
    if (IsUnknownRankShape(tempVector)) {
      unknown_rank_num++;
      continue;
    } 
    static_num++;
  }
  
  if (tensor_num == empty_num + unknown_rank_num) {
    if (tensor_num == empty_num) {
      return ALL_EMPTY_TENSOR;
    } 
    return ONLY_EMPTY_AND_UNKNOWN_RANK_TENSOR;
  } 
  if (unknown_shape_num == 0) {
    return HAS_STATIC_WITHOUT_UNKNOWN_SHAPE_TENSOR;
  } 
  return HAS_UNKNOWN_SHAPE_TENSOR;
}

IMPLEMT_COMMON_INFERFUNC(EltwiseInferShape) {
  /*
  eltwise has four type inputs:
  1.empty 2.static shape 3.-1 4.-2
  The combinations bring 15 scenes, and the 15 scenes can be classify into 4 categories:
  1.input with no range and output no need range, and it can be divided half:
    1.1 all input is empty
    1.2 input only contains empty and -2 shape
  2.input contains static shape and with no -1 shape
  3.input contains -1 shape
  */
  int64_t tensor_num = GetEltwiseConstValue(op);
  int64_t infer_classify = EltwiseInferClassify(op, tensor_num);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto output_desc = op_info->MutableOutputDesc("y");
  // condition 1: all input shape is empty
  if (infer_classify == ALL_EMPTY_TENSOR) {
    std::vector<int64_t> shape_vector = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    output_desc->SetShape(GeShape(shape_vector));
    output_desc->SetDataType(x_dtype);
  // condition 2: all input is -2 or only empty and -2
  } else if (infer_classify == ONLY_EMPTY_AND_UNKNOWN_RANK_TENSOR) {
    std::vector<int64_t> shape_vector = {-2};
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    output_desc->SetShape(GeShape(shape_vector));
    output_desc->SetDataType(x_dtype);
  // condition 3: contains static shape and no -1 shape
  } else if (infer_classify == HAS_STATIC_WITHOUT_UNKNOWN_SHAPE_TENSOR) {
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    std::vector<int64_t> shape_vector = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
    for (int64_t i = 0; i < tensor_num; i++) {
      std::vector<int64_t> temp_vector = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
      if (!shape_vector.empty() && !IsUnknownRankShape(shape_vector)) {
        shape_vector = temp_vector;
        break;
      }
    }
    std::vector<std::pair<int64_t,int64_t>> out_range;
    MakeUpShapeRange(shape_vector, out_range);
    output_desc->SetShape(GeShape(shape_vector));
    output_desc->SetShapeRange(out_range);
    output_desc->SetDataType(x_dtype);
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
      for (size_t j = 0UL; j < temp_vector.size(); j++) {
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
    output_desc->SetShape(GeShape(out_vector));
    output_desc->SetShapeRange(out_range);
    output_desc->SetDataType(x_dtype);
  }
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
    OP_LOGE(op.GetName().c_str(), "The InferShapeAndTypeErfinv is one input and one output.");
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
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  bool keep_dims;
  if (ge::GRAPH_SUCCESS != op.GetAttr("keep_dims", keep_dims)) {
    std::string err_msg = GetInputInvalidErrMsg("keep_dims");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  string input_name1 = "x";
  string input_name2 = "greater_zeros";
  string input_name3 = "select_ones";
  string input_name4 = "maximum_ones";
  string output_name = "y";
  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc(output_name);
  GeTensorDescPtr tensordesc_input1 = op_desc->MutableInputDesc(input_name1);
  GeTensorDescPtr tensordesc_input2 = op_desc->MutableInputDesc(input_name2);
  GeTensorDescPtr tensordesc_input3 = op_desc->MutableInputDesc(input_name3);
  GeTensorDescPtr tensordesc_input4 = op_desc->MutableInputDesc(input_name4);
  CHECK(op_desc == nullptr ||
        tensordesc_output == nullptr ||
        tensordesc_input1 == nullptr ||
        tensordesc_input2 == nullptr ||
        tensordesc_input3 == nullptr ||
        tensordesc_input4 == nullptr,
        OP_LOGE(op.GetName().c_str(), "invalid OpDesc."), return GRAPH_FAILED);
  DataType input_dtype = tensordesc_input1->GetDataType();
  // output Des
  tensordesc_output->SetDataType(input_dtype);
  // shape
  ge::GeShape shapeX = tensordesc_input1->GetShape();
  ge::GeShape shapeY = tensordesc_input2->GetShape();
  ge::GeShape shapeZ = tensordesc_input3->GetShape();
  ge::GeShape shapeH = tensordesc_input4->GetShape();
  OP_LOGI(op.GetName().c_str(), "shape %s: %s, shape %s: %s, shape %s: %s.",
                  input_name1.c_str(), to_string(shapeX).c_str(),
                  input_name2.c_str(), to_string(shapeY).c_str(),
                  input_name3.c_str(), to_string(shapeZ).c_str(),
                  input_name4.c_str(), to_string(shapeH).c_str());
  std::vector<int64_t> dimsX = shapeX.GetDims();
  std::vector<int64_t> dimsY = shapeY.GetDims();
  std::vector<int64_t> dimsZ = shapeZ.GetDims();
  std::vector<int64_t> dimsH = shapeH.GetDims();
  // unknown rank
  if (IsUnknownRankShape(dimsX) || IsUnknownRankShape(dimsY) ||
      IsUnknownRankShape(dimsZ) || IsUnknownRankShape(dimsH)) {
    tensordesc_output->SetShape(ge::GeShape(UNKNOWN_RANK));
    OP_LOGI(op.GetName().c_str(), "output shape is: %s, output dtype is:%d.",
            to_string(ge::Shape(UNKNOWN_RANK)).c_str(),
            input_dtype);
    return GRAPH_SUCCESS;
  }
  // range
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  tensordesc_input1->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_y;
  tensordesc_input2->GetShapeRange(shape_range_y);
  std::vector<std::pair<int64_t, int64_t>> shape_range_z;
  tensordesc_input3->GetShapeRange(shape_range_z);
  std::vector<std::pair<int64_t, int64_t>> shape_range_h;
  tensordesc_input4->GetShapeRange(shape_range_h);
  std::vector<int64_t> dimVec;
  std::vector<std::pair<int64_t, int64_t>> Vec_range;
  dimVec = dimsX;
  Vec_range = shape_range_x;
  MakeUpShapeRange(dimsX, shape_range_x);
  // Broadcast x and y
  if (!TwoShapeAndRangeBroadcastIntegration(op, dimVec, Vec_range, dimsY, shape_range_y, "x", "greater_zeros")) {
    return GRAPH_FAILED;
  }
  // Broadcast dimVec and z
  if (!TwoShapeAndRangeBroadcastIntegration(op, dimVec, Vec_range, dimsZ, shape_range_z,
                                           "dimVec", "select_ones")) {
    return GRAPH_FAILED;
  }
  // Broadcast dimVec and h
  if (!TwoShapeAndRangeBroadcastIntegration(op, dimVec, Vec_range, dimsH, shape_range_h,
                                           "dimVec", "maximum_ones")) {
    return GRAPH_FAILED;
  }
  ge::GeShape outputShape = ge::GeShape(dimVec);
  tensordesc_output->SetShape(outputShape);
  tensordesc_output->SetShapeRange(Vec_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ClipByNormNoDivSum, ClipByNormNoDivSumInferShape);
// ------------ClipByNormNoDivSum----------------

// ------------SquareSumV1 Op Begin----------------
IMPLEMT_COMMON_INFERFUNC(SquareSumV1InferShape) {
  const int64_t input_x_idx = 0;
  const int64_t output_y_idx = 0;
  vector<int64_t> attr_axis;
  static const std::pair<int64_t, std::string> axis_attr_info{0, "axis"};
  if (!(ops::GetAttrValue(op, axis_attr_info, attr_axis))) {
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  bool keep_dims = false;
  static const std::pair<int64_t, std::string> keep_dims_attr_info{1, "keep_dims"};
  if (!(ops::GetAttrValue(op, keep_dims_attr_info, keep_dims))) {
    std::string err_msg = GetInputInvalidErrMsg("keep_dims");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (reduce_ops::CommonReduceInferWithAttrAxes(op, input_x_idx, output_y_idx, attr_axis, keep_dims)) {
    return GRAPH_SUCCESS;
  }
  std::string err_msg = OtherErrMsg("infershape failed");
  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
  return GRAPH_FAILED;
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
    std::string err_msg = UpdateParamErrMsg("y1 or y2");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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

IMPLEMT_COMMON_INFERFUNC(FusedMulAddNInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter FusedMulAddNInferShape");
  if (OneInOneOutDynamicInfer(op, "x1", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(FusedMulAddN, FusedMulAddNInferShape);
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
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("num_axes", num_axes)) {
    std::string err_msg = GetInputInvalidErrMsg("num_axes");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("bias_from_blob", bias_from_blob)) {
    std::string err_msg = GetInputInvalidErrMsg("bias_from_blob");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("num_axes", num_axes)) {
    std::string err_msg = GetInputInvalidErrMsg("num_axes");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("bias_from_blob", bias_from_blob)) {
    std::string err_msg = GetInputInvalidErrMsg("bias_from_blob");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t length_x = dims_x.size();
  int64_t length_bias = dims_bias.size();

  if ((axis >= length_x) || (axis < (-length_x))) {
    string minvalue = std::to_string(-length_x);
    string maxvalue = std::to_string(length_x - 1);
    string excepted_value = ConcatString("in the range of [", minvalue,",", maxvalue,"]");
    std::string err_msg = GetAttrValueErrMsg("axis", std::to_string(axis), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (num_axes < -1) {
    std::string err_msg = GetAttrValueErrMsg("num_axes", std::to_string(num_axes), ConcatString("num_axes >= -1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
        string err_msg1 = ConcatString("length_bias and bias_num must be equal, length_bias:",length_bias, ", bias_num:",bias_num); 
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    } else if (num_axes == 0) {
      if (bias_dim_num != 0) {
        string err_msg1 = ConcatString("bias must be a scalar, bias_dim_num:",bias_dim_num); 
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    } else if (num_axes > 0) {
      int64_t num_axis = axis_ + num_axes;
      if (num_axis > length_x) {
        string err_msg1 = ConcatString("bias shape extends x shape when applied, num_axis:",num_axis, ", length_x:",length_x); 
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
      if (length_bias != num_axes) {
        string err_msg1 = ConcatString("length_bias and bias_num must be equal, length_bias:",length_bias, ", num_axes:",num_axes); 
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    }
  } else {
    if (bias_dim_num != 0) {
      int64_t bias_num = axis_ + length_bias;
      if (bias_num > length_x) {
        string err_msg1 = ConcatString("bias shape extends x shape when applied, bias_num:",bias_num, ", length_x:",length_x);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputInvalidErrMsg("axes");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  bool keep_dims;
  if (ge::GRAPH_SUCCESS != op.GetAttr("keep_dims", keep_dims)) {
    std::string err_msg = GetInputInvalidErrMsg("keep_dims");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  int topk = op.get_attr_topk();
  if (topk < 1) {
    OP_LOGE(op.GetName().c_str(), "topk must be greater than 0, current topk is %d", topk);
    return GRAPH_FAILED;
  }
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
    string err_msg1 = ConcatString("the dtype of input_type_x1 and input_type_x2 must be same! input_type_x1:",input_type_x1, ", input_type_x2:",input_type_x2); 
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
	}
	return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(MulNoNan, MulNoNanVerify);

IMPLEMT_COMMON_INFERFUNC(MulNoNanInferShape) {
  bool is_dynamic_output = true;
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y",
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
    string err_msg1 = ConcatString("input x1 or x2 dims must bigger than 1, shape_x1.GetDimNum():",shape_x1.GetDimNum(), ", shape_x2.GetDimNum():",shape_x2.GetDimNum());
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::string reduction;
  op.GetAttr("reduction", reduction);
  if ((reduction != "mean") && (reduction != "sum") && (reduction != "none")) {
    string expected_reduction_list = ConcatString("mean, sum, reduction");
    std::string err_msg = GetInputFormatNotSupportErrMsg("reduction", expected_reduction_list, reduction);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = OtherErrMsg("input x1 and x2 shape can't broadcast");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // reduce aixs = 1
  x_dims_broadcast.erase(x_dims_broadcast.begin() + 1);

  Shape shape_x_broadcast(x_dims_broadcast);
  if (!BroadCastTwoShape(op, shape_x_broadcast, shape_tgt, tgt_dims_broadcast)) {
    std::string err_msg = OtherErrMsg("input target shape can't broadcast to x shape");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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

  auto y_desc = op_info->MutableOutputDesc("y");

  std::string reduction;
  op.GetAttr("reduction", reduction);
  if (reduction == "none") {
    y_desc->SetShape(x_desc->GetShape());
  } else {
    std::vector<int64_t> x_dims;
    y_desc->SetShape(GeShape(x_dims));
  }

  y_desc->SetDataType(x_dtype);
  
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(KLDiv, KLDivInferShape);
VERIFY_FUNC_REG(KLDiv, KLDivVerify);
// ---------------------KLDiv End------------------------

// ----------------TensorMove Begin-------------------
IMPLEMT_COMMON_INFERFUNC(TensorMoveInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(TensorMove, TensorMoveInferShape);
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

  Format input_format = op.GetInputDesc(input_name1).GetFormat();

  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();

  if (shape_x.GetShapeSize() != shape_y.GetShapeSize()) {
    OP_LOGE(op.GetName().c_str(), "The ShapeSize of input_x does not match input_y.");
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
    OP_LOGE(op.GetName().c_str(), "input_x input_y tensor dtype does not match.");
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
      OP_LOGE("ChangeShape", "[ERROR] dims_fst and dims_sec can not be broadcast");
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

bool InferShapeAndTypeAddcdivAndAddcmul(Operator& op,
                                        const string& input_name1,
                                        const string& input_name2,
                                        const string& input_name3,
                                        const string& output_name) {
  AscendString op_name_str;
  if (GRAPH_SUCCESS !=op.GetName(op_name_str)) {
    OP_LOGE(op_name_str.GetString(), "get op name faild!");
    return false;
  }
  const char *op_name = op_name_str.GetString();
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensordesc_input1 = op_desc->MutableInputDesc(input_name1);
  GeTensorDescPtr tensordesc_input2 = op_desc->MutableInputDesc(input_name2);
  GeTensorDescPtr tensordesc_input3 = op_desc->MutableInputDesc(input_name3);
  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc(output_name);
  CHECK(op_desc == nullptr ||
        tensordesc_output == nullptr ||
        tensordesc_input1 == nullptr ||
        tensordesc_input2 == nullptr ||
        tensordesc_input3 == nullptr,
        OP_LOGE(op_name, "invalid OpDesc."), return GRAPH_FAILED);
  DataType input_dtype = tensordesc_input1->GetDataType();
  
  // output Desc
  tensordesc_output->SetDataType(input_dtype);
  
  // shape
  ge::GeShape shape_x = tensordesc_input1->GetShape();
  ge::GeShape shape_y = tensordesc_input2->GetShape();
  ge::GeShape shape_z = tensordesc_input3->GetShape();
  OP_LOGI(op_name, "shape %s: %s, shape %s: %s, shape %s: %s.",
          input_name1.c_str(), to_string(shape_x).c_str(),
          input_name2.c_str(), to_string(shape_y).c_str(),
          input_name3.c_str(), to_string(shape_z).c_str());
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();
  std::vector<int64_t> dims_z = shape_z.GetDims();
  
  // unknown rank
  if (IsUnknownRankShape(dims_x) || IsUnknownRankShape(dims_y) || IsUnknownRankShape(dims_z)) {
    tensordesc_output->SetShape(ge::GeShape(UNKNOWN_RANK));
    OP_LOGI(op_name, "output shape is UNKOWN RANK, output dtype is:%d.",
            input_dtype);
    return true;
  }
  
  // range
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  tensordesc_input1->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_y;
  tensordesc_input2->GetShapeRange(shape_range_y);
  std::vector<std::pair<int64_t, int64_t>> shape_range_z;
  tensordesc_input3->GetShapeRange(shape_range_z);
  
  std::vector<int64_t> dim_vec;
  std::vector<std::pair<int64_t, int64_t>> vec_range;
  dim_vec = dims_x;
  vec_range = shape_range_x;
  MakeUpShapeRange(dims_x, shape_range_x);
  if (!TwoShapeAndRangeBroadcastIntegration(op, dim_vec, vec_range, dims_y, shape_range_y, "x1", "x2")) {
    return false;
  }
  if (!TwoShapeAndRangeBroadcastIntegration(op, dim_vec, vec_range, dims_z, shape_range_z,
                                            "x1_broadcast", "input_data")) {
    return false;
  }
  ge::GeShape output_shape = ge::GeShape(dim_vec);
  tensordesc_output->SetShape(output_shape);
  tensordesc_output->SetShapeRange(vec_range);
  return true;
}

// ----------------Addcdiv begin-------------------
IMPLEMT_VERIFIER(Addcdiv, AddcdivVerify) {
  AscendString op_name_str;
  op.GetName(op_name_str);
  const char *op_name = op_name_str.GetString();
  // the data type of input_data, x1 and x2 should be same.
  if (op.GetInputDescByName("x1").GetDataType() !=
          op.GetInputDescByName("input_data").GetDataType() ||
      op.GetInputDescByName("x2").GetDataType() !=
          op.GetInputDescByName("input_data").GetDataType()) {
    OP_LOGE(op_name,
            "input_data data type and x1, x2 match failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(AddcdivInferShape) {
  if (InferShapeAndTypeAddcdivAndAddcmul(op, "x1", "x2", "input_data", "y")) {
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
  AscendString op_name_str;
  op.GetName(op_name_str);
  const char *op_name = op_name_str.GetString();
  // the data type of input_data,x1 and x2 should be same.
  if (op.GetInputDescByName("x1").GetDataType() !=
          op.GetInputDescByName("input_data").GetDataType() ||
      op.GetInputDescByName("x2").GetDataType() !=
          op.GetInputDescByName("input_data").GetDataType()) {
    OP_LOGE(op_name,
            "input_data data type and x1, x2 match failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(AddcmulInferShape) {
  if (InferShapeAndTypeAddcdivAndAddcmul(op, "x1", "x2", "input_data", "y")) {
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

    if (input_shape.GetShapeSize() != mask_shape.GetShapeSize()) {
        OP_LOGE(op.GetName().c_str(), "shapesize of x not match mask");
        return GRAPH_FAILED;
    }

    bool is_dynamic_output = true;
    if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x", "mask", "y", is_dynamic_output)) {
        return GRAPH_SUCCESS;
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

// ----------------Dawsn--------------------
COMMON_INFER_FUNC_REG(Dawsn, OneInOneOutCommonInferShape);
// --------------Dawsn END-----------------

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

// ----------------FresnelCos--------------------
COMMON_INFER_FUNC_REG(FresnelCos, OneInOneOutCommonInferShape);
// --------------FresnelCos END-----------------

// ----------------FresnelSin--------------------
COMMON_INFER_FUNC_REG(FresnelSin, OneInOneOutCommonInferShape);
// --------------FresnelSin END-----------------

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

// ----------------Expint---------------------
COMMON_INFER_FUNC_REG(Expint, OneInOneOutCommonInferShape);
// ----------------Expint END-----------------

// -------------------DataCompare----------------------
IMPLEMT_VERIFIER(DataCompare, DataCompareVerify) {
  float atol_data;
  if (ge::GRAPH_SUCCESS != op.GetAttr("atol", atol_data)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of DataCompare!");
    return GRAPH_FAILED;
  }
  if (atol_data < 0) {
    OP_LOGE(op.GetName().c_str(), "atol should >= 0!");
    return GRAPH_FAILED;
  }

  float rtol_data;
  if (ge::GRAPH_SUCCESS != op.GetAttr("rtol", rtol_data)) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr failed of DataCompare!");
    return GRAPH_FAILED;
  }
  if (rtol_data < 0) {
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
      OP_LOGE(op.GetName().c_str(), "The dim of input_x and input_y not match.");
      return false;
  }

  if(dims_input1.size() != 1) {
      OP_LOGE(op.GetName().c_str(), "The dim of input must be 1");
      return false;
  }

  if(dims_input1[0] != dims_input2[0]) {
      OP_LOGE(op.GetName().c_str(), "The 0-dim of input_x and input_y not match.");
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
      OP_LOGE(op.GetName().c_str(), "The dataType of input_x and input_y not match.");
      return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Dot, DotInferShape);
VERIFY_FUNC_REG(Dot, DotVerify);
// ---------------Dot END-------------------

// ---------------IsClose Begin-----------------
static bool InferShapeAndTypeIsClose(Operator &op,
                                     const string &input_name1,
                                     const string &input_name2,
                                     const string &output_name){
  TensorDesc v_output_desc = op.GetOutputDesc(output_name);

  Format input_format = op.GetInputDesc(input_name1).GetFormat();
  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
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
    if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1)) {
      return false;
    }

    int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
    dim_vec.push_back(dims);
  }
  ge::Shape output_shape = ge::Shape(dim_vec);

  v_output_desc.SetShape(output_shape);
  v_output_desc.SetDataType(DT_BOOL);
  v_output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, v_output_desc);

  return true;
}
IMPLEMT_VERIFIER(IsClose, IsCloseVerify){
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(IsCloseInferShape){
  if (InferShapeAndTypeIsClose(op, "x1", "x2", "y")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
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
        OP_LOGE(op.GetName().c_str(), "The dim size of var should biger than updates(indices) 1.");
        return false;
    }
  
    if ((1 == shape_var_list.size()) && (1 != shape_updates_list.size())) {
        OP_LOGE(op.GetName().c_str(), "The dim size of var should equal updates(indices) when size=1.");
        return false;
    }
  
    if (shape_indices_list.size() != shape_updates_list.size()) {
        OP_LOGE(op.GetName().c_str(), "The dim size of indices and updates not match.");
        return false;
    }

    for (size_t i = 0; i < shape_indices.GetDimNum(); i++) {
        if (shape_indices.GetDim(i) != shape_updates.GetDim(i)) {
            OP_LOGE(op.GetName().c_str(), "The dim value of indices and updates not match.");
            return false;
        }
    }

    if (shape_var_list.size() > 1) {
        for (size_t i = 0; i < shape_indices.GetDimNum(); i++) {
            if (((static_cast<int32_t>(i) < dims) && (shape_indices.GetDim(i) != shape_var.GetDim(i))) ||
                ((static_cast<int32_t>(i) >= dims) && (shape_indices.GetDim(i) != shape_var.GetDim(i + 1)))) {
                OP_LOGE(op.GetName().c_str(), "The dim value of var and updates not match.");
                return false;
            }
        }
    }

    DataType var_dtype = input_var_desc.GetDataType();
    DataType updates_dtype = input_updates_desc.GetDataType();
    if (var_dtype != updates_dtype) {
        OP_LOGE(op.GetName().c_str(), "The dtype of var and updates not match.");
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

// ----------------Spence--------------------
COMMON_INFER_FUNC_REG(Spence, OneInOneOutCommonInferShape);
// --------------Spence END-----------------

// ----------------AddMatMatElements-------------------
IMPLEMT_VERIFIER(AddMatMatElements, AddMatMatElementsVerify) {
  if (!CheckTwoInputDtypeSame(op, "c", "a") || !CheckTwoInputDtypeSame(op, "c", "b")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AddMatMatElementsInferShape) {
  if (OneInOneOutDynamicInfer(op, "c", {"c"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(AddMatMatElements, AddMatMatElementsInferShape);
VERIFY_FUNC_REG(AddMatMatElements, AddMatMatElementsVerify);

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
        OP_LOGE(op.GetName().c_str(), "The dim size of var should biger than updates(indices) 1.");
        return false;
    }

    if ((1 == shape_var_list.size()) && (1 != shape_updates_list.size())) {
        OP_LOGE(op.GetName().c_str(), "The dim size of var should equal updates(indices) when size=1.");
        return false;
    }
  
    if (shape_indices_list.size() != shape_updates_list.size()) {
        OP_LOGE(op.GetName().c_str(), "The dim size of indices and updates not match.");
        return false;
    }

    for (size_t i = 0; i < shape_indices.GetDimNum(); i++) {
        if (shape_indices.GetDim(i) != shape_updates.GetDim(i)) {
            OP_LOGE(op.GetName().c_str(), "The dim value of indices and updates not match.");
            return false;
        }
    }

    for (size_t i = 0; i < shape_var.GetDimNum(); i++) {
        if (shape_var.GetDim(i) != shape_assist.GetDim(i)) {
            OP_LOGE(op.GetName().c_str(), "The dim value of var and assist not match.");
            return false;
        }
    }

    if (shape_var_list.size() > 1) {
        for (size_t i = 0; i < shape_indices.GetDimNum(); i++) {
            if (((static_cast<int32_t>(i) < dims) && (shape_indices.GetDim(i) != shape_var.GetDim(i))) ||
                ((static_cast<int32_t>(i) >= dims) && (shape_indices.GetDim(i) != shape_var.GetDim(i + 1)))) {
                OP_LOGE(op.GetName().c_str(), "The dim value of var and updates not match.");
                return false;
            }
        }
    }

    DataType var_dtype = input_var_desc.GetDataType();
    DataType updates_dtype = input_updates_desc.GetDataType();
    if (var_dtype != updates_dtype) {
        OP_LOGE(op.GetName().c_str(), "The dtype of var and updates not match.");
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
    // Valid dim value [-shape_x, shape_x - 1] in Python,
    // which needs to be converted to [0, shape_x - 1] here in C++.
    if (attr_dim < 0) {
      attr_dim += dims_x.size();
    }
    // Shape of output Tensor is the same as input except for
    // the axis that dim points to.
    // Example.
    // Shape of input [2,3,4,5], dim = 0, then shape of output is [3,4,5]
    for (size_t i = 0UL; i < dims_x.size(); i++) {
      if (static_cast<int64_t>(i) != attr_dim) {
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
// ----------------------ApplyAdamV2----------------------
IMPLEMT_COMMON_INFERFUNC(ApplyAdamV2InferShape) {
  TensorDesc tensordesc_output_0 = op.GetOutputDesc("var");
  TensorDesc tensordesc_output_1 = op.GetOutputDesc("m");
  TensorDesc tensordesc_output_2 = op.GetOutputDesc("v");
  TensorDesc tensordesc_input = op.GetInputDesc("var");
  auto result_shape = tensordesc_input.GetShape();
  auto result_type = tensordesc_input.GetDataType();
  tensordesc_output_0.SetShape(result_shape);
  tensordesc_output_0.SetDataType(result_type);
  tensordesc_output_1.SetShape(result_shape);
  tensordesc_output_1.SetDataType(result_type);
  tensordesc_output_2.SetShape(result_shape);
  tensordesc_output_2.SetDataType(result_type);
  (void)op.UpdateOutputDesc("var", tensordesc_output_0);
  (void)op.UpdateOutputDesc("m", tensordesc_output_1);
  (void)op.UpdateOutputDesc("v", tensordesc_output_2);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdamV2, ApplyAdamV2InferShape);
//-----------------------ApplyAdamV2 END---------------------
}  // namespace ge
