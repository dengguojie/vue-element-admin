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
 * \file matrix_calculation_ops.cpp
 * \brief
 */
#include "inc/matrix_calculation_ops.h"

#include <string>
#include <vector>

#include "util/util.h"
#include "util/common_shape_fns.h"
#include "util/array_ops_shape_fns.h"
#include "util/error_util.h"
#include "graph/utils/node_utils.h"

namespace ge {
// ----------------FullyConnection-------------------
IMPLEMT_VERIFIER(FullyConnection, FullyConnectionVerify) {
  auto xShape = op.get_input_desc_x().GetShape().GetDims();
  auto wShape = op.get_input_desc_w().GetShape().GetDims();
  bool transpose = op.get_attr_transpose();
  int xDim = xShape.size();
  int axis = op.get_attr_axis();
  int axis_new;

  if (axis < 0) {
    axis_new = axis + xDim;
  } else {
    axis_new = axis;
  }

  // check axis
  if (axis_new != 1 && axis_new != 2) {
    string realvalue = ConcatString(axis_new);
    OpsAttrValueErrReport(op.GetName().c_str(), "axis", "1 or 2", realvalue);
    return GRAPH_FAILED;
  }

  int kShape = 1;
  int reduceStart;
  if (axis_new == 2) {
    reduceStart = 2;
  } else {
    reduceStart = 1;
  }
  for (int i = reduceStart; i < xDim; i++) {
    kShape *= xShape[i];
  }

  // check wShape size
  if (wShape.size() != 2) {
    string realvalue = ConcatString(wShape.size());
    OpsAttrValueErrReport(op.GetName().c_str(), "wShape size", "2", realvalue);
    return GRAPH_FAILED;
  }

  // check km and kn shape
  if (wShape.size() == 2) {
    if (!transpose) {
      if (kShape != wShape[1]) {
        OP_LOGE(op.GetName().c_str(), "weight K must equal to input K!\n");
        OpsTwoInputShapeErrReport(op.GetName(), "X Shape", "W Shape", "weight K must equal to input K!");
        return GRAPH_FAILED;
      }
    } else {
      if (kShape != wShape[0]) {
        OpsTwoInputShapeErrReport(op.GetName(), "X Shape", "W Shape", "weight K must equal to input K!");
        OP_LOGE(op.GetName().c_str(), "weight K must equal to input K!\n");
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(FullyConnection, FullyConnectionInfer) {
  auto outDesc = op.GetOutputDesc("y");
  auto weightDesc = op.GetInputDesc("w");
  auto xDesc = op.GetInputDesc("x");

  auto xShape = op.GetInputDesc("x").GetShape().GetDims();
  auto wShape = op.GetInputDesc("w").GetShape().GetDims();
  auto xDtype = op.GetInputDesc("x").GetDataType();
  bool transpose = op.get_attr_transpose();
  int axis = op.get_attr_axis();
  int xDim = xShape.size();
  int axis_new;

  if (axis < 0) {
    axis_new = axis + xDim;
  } else {
    axis_new = axis;
  }
  if (axis_new != 1 && axis_new != 2) {
    OpsAttrValueErrReport(op.GetName(), "axis", "1 or 2", ConcatString(axis_new));
    OP_LOGE(op.GetName().c_str(), "Attr axis is wrong, this axis is not supported!\n");
    return GRAPH_FAILED;
  }
  op.SetAttr("axis", axis_new);

  vector<int64_t> changedWeightShape;
  vector<int64_t> yShape;
  vector<int64_t> changedXShape;

  if (axis_new == 2) {
    if (yShape.empty()) {
      yShape.push_back(xShape[0]);
      yShape.push_back(xShape[1]);
    }
    changedXShape.push_back(xShape[0]);
    changedXShape.push_back(xShape[1]);
    if (xShape.size() >= 3) {
      int km_shape = 1;
      for (int i = 2; i < xDim; i++) {
        km_shape *= xShape[i];
      }
      changedXShape.push_back(km_shape);
    } else {
      OP_LOGE(op.GetName().c_str(), "Not enough info about M and K!\n");
      return GRAPH_FAILED;
    }

    if (!transpose) {
      changedWeightShape.push_back(wShape[0]);
      changedWeightShape.push_back(changedXShape[2]);
      changedWeightShape.push_back(1);
      changedWeightShape.push_back(1);
      yShape.push_back(wShape[0]);
    } else {
      changedWeightShape.push_back(changedXShape[2]);
      changedWeightShape.push_back(1);
      changedWeightShape.push_back(1);
      changedWeightShape.push_back(wShape[1]);
      yShape.push_back(wShape[1]);
      weightDesc.SetFormat(ge::FORMAT_CHWN);
      weightDesc.SetOriginFormat(ge::FORMAT_CHWN);
    }

    xDesc.SetShape(ge::Shape(changedXShape));
    xDesc.SetOriginShape(ge::Shape(changedXShape));
    (void)op.UpdateInputDesc("x", xDesc);
  } else {
    if (yShape.empty()) {
      yShape.push_back(xShape[0]);
    }
    if (xShape.size() == 2) {
      xShape.push_back(1);
      xShape.push_back(1);
    } else if (xShape.size() == 3) {
      xShape.push_back(1);
    }

    if (!transpose) {
      changedWeightShape.push_back(wShape[0]);
      changedWeightShape.push_back(xShape[1]);
      changedWeightShape.push_back(xShape[2]);
      changedWeightShape.push_back(xShape[3]);
      yShape.push_back(wShape[0]);
    } else {
      changedWeightShape.push_back(xShape[1]);
      changedWeightShape.push_back(xShape[2]);
      changedWeightShape.push_back(xShape[3]);
      changedWeightShape.push_back(wShape[1]);
      yShape.push_back(wShape[1]);
      weightDesc.SetFormat(ge::FORMAT_CHWN);
      weightDesc.SetOriginFormat(ge::FORMAT_CHWN);
    }

    if (xShape[0] == 1) {
      yShape.push_back(1);
      yShape.push_back(1);
    }
  }

  weightDesc.SetShape(ge::Shape(changedWeightShape));
  weightDesc.SetOriginShape(ge::Shape(changedWeightShape));
  (void)op.UpdateInputDesc("w", weightDesc);

  outDesc.SetShape(ge::Shape(yShape));
  if (xDtype == ge::DT_INT8) {
    outDesc.SetDataType(ge::DT_INT32);
  } else {
    outDesc.SetDataType(ge::DataType(xDtype));
  }
  (void)op.UpdateOutputDesc("y", outDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FullyConnection, FullyConnectionInfer);

VERIFY_FUNC_REG(FullyConnection, FullyConnectionVerify);

// ----------------FullyConnectionCompress-------------------
IMPLEMT_VERIFIER(FullyConnectionCompress, FullyConnectionCompressVerify) {
  auto xShape = op.get_input_desc_x().GetShape().GetDims();
  auto wShape = op.get_input_desc_w().GetShape().GetDims();
  bool transpose = op.get_attr_transpose();
  int xDim = xShape.size();

  int kShape = 1;
  for (int i = 1; i < xDim; i++) {
    kShape *= xShape[i];
  }

  // check wShape size
  if (wShape.size() != 1 && wShape.size() != 2) {
    OpsOneOutputShapeErrReport(op.GetName(), "W shape", "wShape Compress size must equal to 1 or 2!");
    OP_LOGE(op.GetName().c_str(), "wShape Compress size must equal to 1 or 2!\n");
    return GRAPH_FAILED;
  }

  // check km and kn shape
  if (wShape.size() == 2) {
    if (!transpose) {
      if (kShape != wShape[1]) {
        OpsTwoInputShapeErrReport(op.GetName(), "X Shape", "W Shape", "weight Compress K must equal to input K!");
        OP_LOGE(op.GetName().c_str(), "weight Compress K must equal to input K!\n");
        return GRAPH_FAILED;
      }
    } else {
      if (kShape != wShape[0]) {
        OpsTwoInputShapeErrReport(op.GetName(), "X Shape", "W Shape", "weight Compress K must equal to input K!");
        OP_LOGE(op.GetName().c_str(), "weight Compress K must equal to input K!\n");
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(FullyConnectionCompress, FullyConnectionCompressInfer) {
  auto outDesc = op.GetOutputDesc("y");
  auto weightDesc = op.GetInputDesc("w");

  auto xShape = op.GetInputDesc("x").GetShape().GetDims();
  auto wShape = op.GetInputDesc("w").GetShape().GetDims();
  auto xDtype = op.GetInputDesc("x").GetDataType();
  bool transpose = op.get_attr_transpose();

  if (xShape.size() < 1 || wShape.size() < 1) {
    OpsTwoInputShapeErrReport(op.GetName(), "X Shape", "W Shape", "Input Shape or Weight Shape should >= 1!");
    OP_LOGE(op.GetName().c_str(), "Invalid Shape size, xShape size is %u, wShape size is %u.", xShape.size(),
            wShape.size());
    return GRAPH_FAILED;
  }

  vector<int64_t> changedWeightShape;
  vector<int64_t> yShape;
  if (yShape.empty()) {
    yShape.push_back(xShape[0]);
  }

  if (xShape.size() == 2) {
    xShape.push_back(1);
    xShape.push_back(1);
  } else if (xShape.size() == 3) {
    xShape.push_back(1);
  }
  if (!transpose) {
    changedWeightShape.push_back(wShape[0]);
    changedWeightShape.push_back(xShape[1]);
    changedWeightShape.push_back(xShape[2]);
    changedWeightShape.push_back(xShape[3]);
    yShape.push_back(wShape[0]);
  } else {
    changedWeightShape.push_back(xShape[1]);
    changedWeightShape.push_back(xShape[2]);
    changedWeightShape.push_back(xShape[3]);
    changedWeightShape.push_back(wShape[1]);
    yShape.push_back(wShape[1]);
    weightDesc.SetFormat(ge::FORMAT_CHWN);
    weightDesc.SetOriginFormat(ge::FORMAT_CHWN);
  }

  weightDesc.SetShape(ge::Shape(changedWeightShape));
  weightDesc.SetOriginShape(ge::Shape(changedWeightShape));
  (void)op.UpdateInputDesc("w", weightDesc);

  outDesc.SetShape(ge::Shape(yShape));
  if (xDtype == ge::DT_INT8) {
    outDesc.SetDataType(ge::DT_INT32);
  } else {
    outDesc.SetDataType(ge::DataType(xDtype));
  }
  (void)op.UpdateOutputDesc("y", outDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FullyConnectionCompress, FullyConnectionCompressInfer);

VERIFY_FUNC_REG(FullyConnectionCompress, FullyConnectionCompressVerify);

// ----------------Matmul-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(MatMul, MatMulVerify) {
  std::vector<DataType> support_list;
  support_list.push_back(DT_FLOAT16);
  support_list.push_back(DT_FLOAT);
  support_list.push_back(DT_INT32);
  if (CheckInputDataType(op, "x1", support_list) == false) {
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "x2", support_list) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MatMulInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  ge::TensorDesc inputTensorDescX = op.GetInputDesc("x1");
  ge::TensorDesc inputTensorDescY = op.GetInputDesc("x2");
  std::vector<std::pair<int64_t, int64_t>> shape_x_range;
  std::vector<std::pair<int64_t, int64_t>> shape_y_range;

  ge::Shape shapeX = inputTensorDescX.GetShape();
  ge::Shape shapeY = inputTensorDescY.GetShape();
  auto shape_x_vec = inputTensorDescX.GetShape().GetDims();
  auto shape_y_vec = inputTensorDescY.GetShape().GetDims();

  // whether have unkown dims
  bool is_unkown_rank_x = shape_x_vec == UNKNOWN_RANK ? true : false;
  bool is_unkown_rank_y = shape_y_vec == UNKNOWN_RANK ? true : false;

  if (is_unkown_rank_x) {
    shape_x_vec = {-1, -1};
    shapeX = ge::Shape(shape_x_vec);
  } else {
    inputTensorDescX.GetShapeRange(shape_x_range);
  }

  if (is_unkown_rank_y) {
    shape_y_vec = {-1, -1};
    shapeY = ge::Shape(shape_y_vec);
  } else {
    inputTensorDescY.GetShapeRange(shape_y_range);
  }

  DataType dtype = inputTensorDescX.GetDataType();
  if (dtype == DT_FLOAT) {
    OP_LOGW(op.GetName().c_str(), "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  }
  OP_LOGW(op.GetName().c_str(), "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  bool transposeA = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr("transpose_x1", transposeA)) {
    OpsGetAttrErrReport(op.GetName(), "transpose_x1");
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]Matmul GetOpAttr transpose_x1 failed!");
    return GRAPH_FAILED;
  }

  bool transposeB = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr("transpose_x2", transposeB)) {
    OpsGetAttrErrReport(op.GetName(), "transpose_x2");
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]Matmul GetOpAttr transpose_x2 failed!");
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> shape_out_range;
  MakeUpShapeRange(shape_x_vec, shape_x_range);
  MakeUpShapeRange(shape_y_vec, shape_y_range);
  std::vector<int64_t> dimVector;
  int64_t tensor_ak = 0;
  int64_t tensor_bk = 0;
  if (transposeA) {
    if (transposeB) {
      dimVector.push_back(shapeX.GetDim(1));
      shape_out_range.push_back(shape_x_range[1]);
      dimVector.push_back(shapeY.GetDim(0));
      shape_out_range.push_back(shape_y_range[0]);
      tensor_ak = shapeX.GetDim(0);
      tensor_bk = shapeY.GetDim(1);
    } else {
      dimVector.push_back(shapeX.GetDim(1));
      shape_out_range.push_back(shape_x_range[1]);
      dimVector.push_back(shapeY.GetDim(1));
      shape_out_range.push_back(shape_y_range[1]);
      tensor_ak = shapeX.GetDim(0);
      tensor_bk = shapeY.GetDim(0);
    }
  } else {
    if (transposeB) {
      dimVector.push_back(shapeX.GetDim(0));
      shape_out_range.push_back(shape_x_range[0]);
      dimVector.push_back(shapeY.GetDim(0));
      shape_out_range.push_back(shape_y_range[0]);
      tensor_ak = shapeX.GetDim(1);
      tensor_bk = shapeY.GetDim(1);
    } else {
      dimVector.push_back(shapeX.GetDim(0));
      shape_out_range.push_back(shape_x_range[0]);
      dimVector.push_back(shapeY.GetDim(1));
      shape_out_range.push_back(shape_y_range[1]);
      tensor_ak = shapeX.GetDim(1);
      tensor_bk = shapeY.GetDim(0);
    }
  }
  if (tensor_ak != tensor_bk && tensor_ak != -1 && tensor_bk != -1) {
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]The input shape is not right!");
    return GRAPH_FAILED;
  }
  ge::Shape outputShape(dimVector);

  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetOriginShape(outputShape);
  tensordesc_output.SetDataType(op.GetInputDesc("x1").GetDataType());
  tensordesc_output.SetShapeRange(shape_out_range);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(MatMul, MatMulInferShape);

// Registered verify function
VERIFY_FUNC_REG(MatMul, MatMulVerify);
// ----------------Matmul-------------------
// ----------------Matmul-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(MatMulV2, MatMulV2Verify) {
  std::vector<DataType> support_list;
  support_list.push_back(DT_FLOAT16);
  support_list.push_back(DT_FLOAT);
  support_list.push_back(DT_INT32);
  support_list.push_back(DT_INT8);
  if (CheckInputDataType(op, "x1", support_list) == false) {
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "x2", support_list) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MatMulV2InferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  ge::TensorDesc inputTensorDescX = op.GetInputDesc("x1");
  ge::TensorDesc inputTensorDescY = op.GetInputDesc("x2");

  ge::Shape shapeX = inputTensorDescX.GetShape();
  ge::Shape shapeY = inputTensorDescY.GetShape();

  DataType dtype = inputTensorDescX.GetDataType();
  if (dtype == DT_FLOAT) {
    OP_LOGW(op.GetName().c_str(), "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  }
  OP_LOGW(op.GetName().c_str(), "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  bool transposeA = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr("transpose_x1", transposeA)) {
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]Matmul GetOpAttr transpose_x1 failed!");
    return GRAPH_FAILED;
  }

  bool transposeB = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr("transpose_x2", transposeB)) {
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]Matmul GetOpAttr transpose_x2 failed!");
    return GRAPH_FAILED;
  }
  if (shapeX.GetDims().size() != 2 && shapeX.GetDims().size() != 4){
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]Matmul the first input dims is not 2!");
    return GRAPH_FAILED;
  }
  if (shapeY.GetDims().size() != 2 && shapeY.GetDims().size() != 4){
    OP_LOGE(op.GetName().c_str(), "[Plugin][ERROR]Matmul the second input dims is not 2!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dimVector;
  if (transposeA) {
    if (transposeB) {
      dimVector.push_back(shapeX.GetDim(1));
      dimVector.push_back(shapeY.GetDim(0));
    } else {
      dimVector.push_back(shapeX.GetDim(1));
      dimVector.push_back(shapeY.GetDim(1));
    }
  } else {
    if (transposeB) {
      dimVector.push_back(shapeX.GetDim(0));
      dimVector.push_back(shapeY.GetDim(0));
    } else {
      dimVector.push_back(shapeX.GetDim(0));
      dimVector.push_back(shapeY.GetDim(1));
    }
  }

  ge::Shape outputShape(dimVector);
  auto input_format = FORMAT_ND;
  auto input_format_1 = FORMAT_ND;
  inputTensorDescX.SetFormat(input_format_1);
  inputTensorDescX.SetOriginFormat(input_format_1);
  inputTensorDescY.SetOriginFormat(input_format);
  inputTensorDescY.SetFormat(input_format);
  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetFormat(input_format_1);
  tensordesc_output.SetOriginFormat(input_format_1);
  if (op.GetInputDesc("x1").GetDataType() == ge::DT_INT8) {
    tensordesc_output.SetDataType(ge::DT_INT32);
  } else {
    tensordesc_output.SetDataType(op.GetInputDesc("x1").GetDataType());
  }
  (void)op.UpdateInputDesc("x1", inputTensorDescX);
  (void)op.UpdateInputDesc("x2", inputTensorDescY);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFORMAT_FUNC(MatMulV2, MatMulV2InferFormat) {
  OP_LOGD(op.GetName().c_str(), "[MatMulV2 Inferformat] Finaly input format is %d", FORMAT_ND);

  ge::TensorDesc tensordesc_input = op.GetInputDesc("x1");
  tensordesc_input.SetOriginFormat(FORMAT_ND);
  tensordesc_input.SetFormat(FORMAT_ND);
  (void)op.UpdateInputDesc("x1", tensordesc_input);

  ge::TensorDesc tensordesc_input_2 = op.GetInputDesc("x2");
  tensordesc_input_2.SetOriginFormat(FORMAT_ND);
  tensordesc_input_2.SetFormat(FORMAT_ND);
  (void)op.UpdateInputDesc("x2", tensordesc_input_2);

  return GRAPH_SUCCESS;
}

INFER_FORMAT_FUNC_REG(MatMulV2, MatMulV2InferFormat);
// Registered inferfunction
COMMON_INFER_FUNC_REG(MatMulV2, MatMulV2InferShape);

// Registered verify function
VERIFY_FUNC_REG(MatMulV2, MatMulV2Verify);
// ----------------Matmul-------------------
// ----------------GEMM-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(GEMM, GemmVerify) {
  std::vector<DataType> support_list;
  std::vector<DataType> support_list_ab;

  support_list.push_back(DT_FLOAT16);
  support_list.push_back(DT_INT8);

  support_list_ab.push_back(DT_FLOAT);
  support_list_ab.push_back(DT_INT32);
  support_list_ab.push_back(DT_FLOAT16);

  if (CheckInputDataType(op, "a", support_list) == false) {
    TbeInputDataTypeErrReport(op.GetName().c_str(), "a", "float16,int8",
                              DataTypeToStringDesc(op.GetInputDesc("a").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "b", support_list) == false) {
    TbeInputDataTypeErrReport(op.GetName().c_str(), "b", "float16,int8",
                              DataTypeToStringDesc(op.GetInputDesc("b").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "c", support_list_ab) == false) {
    TbeInputDataTypeErrReport(op.GetName().c_str(), "c", "float32,int32,float16",
                              DataTypeToStringDesc(op.GetInputDesc("c").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "alpha", support_list_ab) == false) {
    TbeInputDataTypeErrReport(op.GetName().c_str(), "alpha", "float32,int32,float16",
                              DataTypeToStringDesc(op.GetInputDesc("alpha").GetDataType()));
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "beta", support_list_ab) == false) {
    TbeInputDataTypeErrReport(op.GetName().c_str(), "beta", "float32,int32,float16",
                              DataTypeToStringDesc(op.GetInputDesc("beta").GetDataType()));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(GemmInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  ge::TensorDesc inputTensorDescC = op.GetInputDesc("c");
  DataType dtype = inputTensorDescC.GetDataType();
  ge::Shape shapeC = inputTensorDescC.GetShape();
  ge::Shape outputShape(shapeC);
  tensordesc_output.SetDataType(dtype);

  tensordesc_output.SetShape(outputShape);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(GEMM, GemmInferShape);

// Registered verify function
VERIFY_FUNC_REG(GEMM, GemmVerify);
// ----------------GEMM-------------------

// ----------------BatchMatMul-------------------

// Check the dtype and attr of the input tensor description.
// ----------------BatchMatMul-------------------

// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(BatchMatMul, BatchMatMulVerify) {
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(BatchMatMulInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  ge::TensorDesc inputTensorDescX = op.GetInputDesc("x1");
  ge::TensorDesc inputTensorDescY = op.GetInputDesc("x2");

  ge::Shape shapeX = inputTensorDescX.GetShape();
  ge::Shape shapeY = inputTensorDescY.GetShape();

  size_t dimNumX = shapeX.GetDimNum();

  ge::TensorDesc outputTensorDesc = inputTensorDescX;

  bool transposeA = false;
  bool transposeB = false;

  if (ge::GRAPH_SUCCESS != op.GetAttr("adj_x1", transposeA)) {
    OpsGetAttrErrReport(op.GetName(), "transposeA");
    printf("GetOpAttr transpose_a or transpose_a failed!");
    return GRAPH_FAILED;
  }
  if (ge::GRAPH_SUCCESS != op.GetAttr("adj_x2", transposeB)) {
    OpsGetAttrErrReport(op.GetName(), "transposeB");
    printf("GetOpAttr transpose_a or transpose_b failed!");
    return GRAPH_FAILED;
  }
  if (dimNumX < 3 || dimNumX > 8) {
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dimVector;
  int dimM = 0;
  int dimN = 0;
  int dimNum = dimNumX;
  if (transposeA == true) {
    if (transposeB == true) {
      dimM = dimNum - 1;
      dimN = dimNum - 2;
    } else {
      dimM = dimNum - 1;
      dimN = dimNum - 1;
    }
  } else if (transposeB == true) {
    dimM = dimNum - 2;
    dimN = dimNum - 2;
  } else {
    dimM = dimNum - 2;
    dimN = dimNum - 1;
  }
  for (int i = 0; i < dimNum - 2; i++) {
    dimVector.push_back(shapeX.GetDim(i));
  }
  dimVector.push_back(shapeX.GetDim(dimM));
  dimVector.push_back(shapeY.GetDim(dimN));

  ge::Shape outputShape(dimVector);

  tensordesc_output.SetShape(outputShape);
  tensordesc_output.SetDataType(op.GetInputDesc("x1").GetDataType());

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(BatchMatMul, BatchMatMulInferShape);

// Registered verify function
VERIFY_FUNC_REG(BatchMatMul, BatchMatMulVerify);


// ----------------BatchMatMulV2-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(BatchMatMulV2, BatchMatMulV2Verify) {
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(BatchMatMulV2InferShape) {
    TensorDesc g_tensordesc_output = op.GetOutputDesc("y");
    ge::TensorDesc g_inputTensorDescX = op.GetInputDesc("x1");
    ge::TensorDesc g_inputTensorDescY = op.GetInputDesc("x2");

    ge::Shape g_shapeX = g_inputTensorDescX.GetShape();
    ge::Shape g_shapeY = g_inputTensorDescY.GetShape();

    size_t g_dimNumX = g_shapeX.GetDimNum();
    size_t g_dimNumY = g_shapeY.GetDimNum();

    ge::TensorDesc g_outputTensorDesc = g_inputTensorDescX;

    bool g_transposeA = false;
    bool g_transposeB = false;

    if (ge::GRAPH_SUCCESS != op.GetAttr("adj_x1", g_transposeA)) {
        OpsGetAttrErrReport(op.GetName(), "g_transposeA");
        printf("GetOpAttr transpose_a or transpose_a failed!");
        return GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != op.GetAttr("adj_x2", g_transposeB)) {
        OpsGetAttrErrReport(op.GetName(), "g_transposeB");
        printf("GetOpAttr transpose_a or transpose_b failed!");
        return GRAPH_FAILED;
    }
    if (g_dimNumX < 3 || g_dimNumX > 8) {
        return GRAPH_FAILED;
    }
    std::vector<int64_t> g_dimVector;
    int g_dimM = 0;
    int g_dimN = 0;
    int g_dimNum = g_dimNumX;
    if (g_dimNumX < g_dimNumY) {
        g_dimNum = g_dimNumY;
    }
    if (g_transposeA == true) {
        if (g_transposeB == true) {
            g_dimM = g_dimNumX - 1;
            g_dimN = g_dimNumY - 2;
        } else {
            g_dimM = g_dimNumX - 1;
            g_dimN = g_dimNumY - 1;
        }
    } else if (g_transposeB == true) {
        g_dimM = g_dimNumX - 2;
        g_dimN = g_dimNumY - 2;
    } else {
        g_dimM = g_dimNumX - 2;
        g_dimN = g_dimNumY - 1;
    }

    if (g_dimNumX < g_dimNumY) {
        for (int i = 0; i < g_dimNum - 2; i++) {
            g_dimVector.push_back(g_shapeY.GetDim(i));
        }
    } else {
        for (int i = 0; i < g_dimNum - 2; i++) {
            g_dimVector.push_back(g_shapeX.GetDim(i));
        }
    }
    for (int i = 0; i < g_dimNum - 2; i++) {
        g_dimVector.push_back(g_shapeX.GetDim(i));
    }
    g_dimVector.push_back(g_shapeX.GetDim(g_dimM));
    g_dimVector.push_back(g_shapeY.GetDim(g_dimN));

    ge::Shape g_outputShape(g_dimVector);

    g_tensordesc_output.SetShape(g_outputShape);
    g_tensordesc_output.SetDataType(op.GetInputDesc("x1").GetDataType());

    (void)op.UpdateOutputDesc("y", g_tensordesc_output);
    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(BatchMatMulV2, BatchMatMulV2InferShape);

// Registered verify function
VERIFY_FUNC_REG(BatchMatMulV2, BatchMatMulV2Verify);

// - ---------------L2Loss-------------------
IMPLEMT_COMMON_INFERFUNC(L2LossInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  DataType input_dtype = input_desc->GetDataType();
  auto output_desc = op_info->MutableOutputDesc("y");

  std::vector<std::pair<int64_t, int64_t>> input_range;
  std::vector<int64_t> shape_vector;
  output_desc->SetShape(GeShape(shape_vector));
  output_desc->SetOriginShape(GeShape(shape_vector));
  output_desc->SetShapeRange(input_range);
  output_desc->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(L2Loss, L2LossInferShape);
// --------------L2Loss END-----------------

// ----------------DiagPart-------------------
IMPLEMT_COMMON_INFERFUNC(DiagPartInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < shape.GetDimNum() / 2; i++) {
    dim_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DiagPart, DiagPartInferShape);
// ----------------DiagPart END-------------------

// ----------------DiagPartD-------------------
IMPLEMT_COMMON_INFERFUNC(DiagPartDInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < shape.GetDimNum() / 2; i++) {
    dim_vector.push_back(shape.GetDim(i));
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DiagPartD, DiagPartDInferShape);
// ----------------DiagPartD END-------------------

// ---------------MatrixDiag-------------------
IMPLEMT_COMMON_INFERFUNC(MatrixDiagInferShape) {
  Shape input_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  int64_t dimsInput = input_shape.GetDimNum() - 1;
  int64_t dimNums1 = input_shape.GetDim(dimsInput);

  vector<int64_t> dimInfo = input_shape.GetDims();
  std::vector<int64_t> dim_vec;
  for (size_t j = 0; j < input_shape.GetDimNum(); ++j) {
    dim_vec.push_back(dimInfo[j]);
  }
  dim_vec.push_back(dimNums1);
  td.SetShape(ge::Shape(dim_vec));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixDiag, MatrixDiagVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiag, MatrixDiagInferShape);
VERIFY_FUNC_REG(MatrixDiag, MatrixDiagVerify);
// ----------------MatrixDiag END----------------

// ---------------MatrixDiagD-------------------
IMPLEMT_COMMON_INFERFUNC(MatrixDiagDInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  Shape assist_shape = op.GetInputDesc("assist").GetShape();
  TensorDesc td = op.GetOutputDesc("y");
  std::vector<int64_t> dims_x = x_shape.GetDims();
  std::vector<int64_t> dims_assist = assist_shape.GetDims();
  if (dims_x.size() < dims_assist.size()) {
    std::vector<int64_t> dims_tmp = dims_x;
    dims_x = dims_assist;
    dims_assist = dims_tmp;
  }

  if (dims_x.size() != dims_assist.size()) {
    int dec = dims_x.size() - dims_assist.size();
    for (int i = 0; i < dec; i++) {
      dims_assist.insert(dims_assist.begin(), (int64_t)1);
    }
  }

  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_assist[i]) && (dims_x[i] != 1) && (dims_assist[i] != 1)) {
      OpsInputShapeBroadcastErrReport(op.GetName(), "x", "assist", ConcatString(dims_x[i]),
                                      ConcatString(dims_assist[i]));
      OP_LOGE(op.GetName().c_str(),
              "The %s op dimensions does not "
              "match the broadcast rule(%lu %lu).",
              op.GetName().c_str(), dims_x[i], dims_assist[i]);
    }

    int64_t dims = dims_x[i] > dims_assist[i] ? dims_x[i] : dims_assist[i];
    dim_vec.push_back(dims);
  }
  td.SetShape(ge::Shape(dim_vec));
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixDiagD, MatrixDiagDVerify) {
  DataType input_diagonal_dtype = op.GetInputDesc(0).GetDataType();
  DataType input_help_dtype = op.GetInputDesc(1).GetDataType();
  if (input_diagonal_dtype != input_help_dtype) {
    OpsTwoInputDtypeErrReport(op.GetName(), "input_diagonal", "input_help", ConcatString(input_diagonal_dtype),
                              ConcatString(input_help_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the matrix_diag op inputs "
            "should have the same dtype!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagD, MatrixDiagDInferShape);
VERIFY_FUNC_REG(MatrixDiagD, MatrixDiagDVerify);
// ----------------MatrixDiagD END----------------

// ----------------MatrixDiagPart--------------------
IMPLEMT_COMMON_INFERFUNC(MatrixDiagPartInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Format input_format = op.GetInputDesc("x").GetFormat();
  std::vector<int64_t> dim_vector;
  int64_t dimsInput_1 = shape.GetDimNum() - 1;
  int64_t dimsInput_2 = shape.GetDimNum() - 2;
  int64_t dimNums_1 = shape.GetDim(dimsInput_1);
  int64_t dimNums_2 = shape.GetDim(dimsInput_2);
  if (dimNums_1 > dimNums_2) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
  } else {
    for (size_t i = 0; i < shape.GetDimNum() - 2; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
    dim_vector.push_back(dimNums_1);
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  td.SetFormat(input_format);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixDiagPart, MatrixDiagPartVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagPart, MatrixDiagPartInferShape);
VERIFY_FUNC_REG(MatrixDiagPart, MatrixDiagPartVerify);
// ------------------MatrixDiagPart END---------------------

// ----------------MatrixDiagPartD--------------------
IMPLEMT_COMMON_INFERFUNC(MatrixDiagPartDInferShape) {
  Shape shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Format input_format = op.GetInputDesc("x").GetFormat();
  std::vector<int64_t> dim_vector;
  int64_t dimsInput_1 = shape.GetDimNum() - 1;
  int64_t dimsInput_2 = shape.GetDimNum() - 2;
  int64_t dimNums_1 = shape.GetDim(dimsInput_1);
  int64_t dimNums_2 = shape.GetDim(dimsInput_2);
  if (dimNums_1 > dimNums_2) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
  } else {
    for (size_t i = 0; i < shape.GetDimNum() - 2; i++) {
      dim_vector.push_back(shape.GetDim(i));
    }
    dim_vector.push_back(dimNums_1);
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  td.SetFormat(input_format);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixDiagPartD, MatrixDiagPartDVerify) {
  DataType input_diagonal_dtype = op.GetInputDesc(0).GetDataType();
  DataType input_help_dtype = op.GetInputDesc(1).GetDataType();
  if (input_diagonal_dtype != input_help_dtype) {
    OpsTwoInputDtypeErrReport(op.GetName(), "input_diagonal", "input_help", ConcatString(input_diagonal_dtype),
                              ConcatString(input_help_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the matrix_diag_part op inputs "
            "should have the same dtype!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagPartD, MatrixDiagPartDInferShape);
VERIFY_FUNC_REG(MatrixDiagPartD, MatrixDiagPartDVerify);
// ------------------MatrixDiagPart ENDD---------------------

// ---------------MatrixSetDiag--------------
IMPLEMT_COMMON_INFERFUNC(MatrixSetDiagInferShape) {
  Shape input_shape = op.GetInputDesc(0).GetShape();
  DataType input_dtype = op.GetInputDesc(0).GetDataType();
  TensorDesc td = op.GetOutputDesc(0);
  td.SetShape(ge::Shape(input_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixSetDiag, MatrixSetDiagVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSetDiag, MatrixSetDiagInferShape);
VERIFY_FUNC_REG(MatrixSetDiag, MatrixSetDiagVerify);
// ----------------MatrixSetDiag END----------------

// ---------------MatrixSetDiagD--------------
IMPLEMT_COMMON_INFERFUNC(MatrixSetDiagDInferShape) {
  Shape input_shape = op.GetInputDesc(0).GetShape();
  DataType input_dtype = op.GetInputDesc(0).GetDataType();
  TensorDesc td = op.GetOutputDesc(0);
  td.SetShape(ge::Shape(input_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MatrixSetDiagD, MatrixSetDiagDVerify) {
  DataType input_matrix_dtype = op.GetInputDesc(0).GetDataType();
  DataType input_diagonal_dtype = op.GetInputDesc(1).GetDataType();
  DataType input_help_dtype = op.GetInputDesc(2).GetDataType();
  if ((input_matrix_dtype != input_diagonal_dtype) || (input_matrix_dtype != input_help_dtype)) {
    OpsTwoInputDtypeErrReport(op.GetName(), "input_matrix", "input_help", ConcatString(input_matrix_dtype),
                              ConcatString(input_help_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the matrix_set_part op inputs "
            "should have the same dtype!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSetDiagD, MatrixSetDiagDInferShape);
VERIFY_FUNC_REG(MatrixSetDiagD, MatrixSetDiagDVerify);
// ----------------MatrixSetDiag ENDD----------------

// -----------------ScatterNdUpdate-----------------
IMPLEMT_INFERFUNC(ScatterNdUpdate, ScatterNdUpdateInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterNdUpdate, ScatterNdUpdateInferShape);
// -------------------ScatterNdUpdate END----------------

// -----------------TensorScatterUpdate-----------------
IMPLEMT_VERIFIER(TensorScatterUpdate, TensorScatterUpdateVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(TensorScatterUpdate, TensorScatterUpdateInferShape) {
  Shape var_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorScatterUpdate, TensorScatterUpdateInferShape);
VERIFY_FUNC_REG(TensorScatterUpdate, TensorScatterUpdateVerify);
// -------------------TensorScatterUpdate END----------------

// ------------------ScatterAdd---------------------
IMPLEMT_VERIFIER(ScatterAdd, ScatterAddVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ScatterAddInferShape)
  // main part of shape infer
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ScatterAdd, ScatterAddInferShape);
VERIFY_FUNC_REG(ScatterAdd, ScatterAddVerify);
// --------------ScatterAdd END------------------

IMPLEMT_COMMON_INFERFUNC(ScatterDivInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ScatterDiv, ScatterDivVerify) {
  DataType var_dtype = op.GetInputDesc(0).GetDataType();
  DataType updates_dtype = op.GetInputDesc(2).GetDataType();
  if (var_dtype != updates_dtype) {
    OpsTwoInputDtypeErrReport(op.GetName(), "var", "updates", ConcatString(var_dtype), ConcatString(updates_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the scatter_div op inputs "
            "should have the same dtype!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterDiv, ScatterDivInferShape);
VERIFY_FUNC_REG(ScatterDiv, ScatterDivVerify);

// ----------------ScatterNdAdd------------
IMPLEMT_VERIFIER(ScatterNdAdd, ScatterNdAddVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterNdAddInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterNdAdd, ScatterNdAddInferShape);
VERIFY_FUNC_REG(ScatterNdAdd, ScatterNdAddVerify);
// ------------------ScatterNdAdd END------------------

// ----------------TensorScatterAdd------------
IMPLEMT_VERIFIER(TensorScatterAdd, TensorScatterAddVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TensorScatterAddInferShape) {
  Shape var_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TensorScatterAdd, TensorScatterAddInferShape);
VERIFY_FUNC_REG(TensorScatterAdd, TensorScatterAddVerify);
// ------------------TensorScatterAdd END------------------

// -------------------ScatterNdSub-------------------
IMPLEMT_VERIFIER(ScatterNdSub, ScatterNdSubVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterNdSubInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterNdSub, ScatterNdSubInferShape);
VERIFY_FUNC_REG(ScatterNdSub, ScatterNdSubVerify);
// ---------------ScatterNdSub END-----------------

// -------------------TensorScatterSub-------------------
IMPLEMT_VERIFIER(TensorScatterSub, TensorScatterSubVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TensorScatterSubInferShape) {
  Shape var_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TensorScatterSub, TensorScatterSubInferShape);
VERIFY_FUNC_REG(TensorScatterSub, TensorScatterSubVerify);
// ---------------TensorScatterSub END-----------------

// ----------------ScatterSub---------------------
IMPLEMT_VERIFIER(ScatterSub, ScatterSubVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ScatterSubInferShape)
  // main part of shape infer
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ScatterSub, ScatterSubInferShape);
VERIFY_FUNC_REG(ScatterSub, ScatterSubVerify);
// --------------------ScatterSub END-----------------

// ---------------ConfusionMatrix-----------------
IMPLEMT_COMMON_INFERFUNC(ConfusionMatrixInferShape) {
  int64_t num_classes;
  auto output_dtype = DT_FLOAT;
  std::string output_dtype_str = "float32";
  if (op.GetAttr("num_classes", num_classes) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "num_classes");
    OP_LOGE(op.GetName().c_str(), "Op get attr num_classes failed");
    return GRAPH_FAILED;
  }
  if (op.GetAttr("dtype", output_dtype_str) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "dtype");
    OP_LOGE(op.GetName().c_str(), "Op get attr dtype failed");
    return GRAPH_FAILED;
  }
  if (output_dtype_str == "float32") {
    output_dtype = DT_FLOAT;
  } else if (output_dtype_str == "int32") {
    output_dtype = DT_INT32;
  } else if (output_dtype_str == "int8") {
    output_dtype = DT_INT8;
  } else if (output_dtype_str == "float16") {
    output_dtype = DT_FLOAT16;
  } else if (output_dtype_str == "uint8") {
    output_dtype = DT_UINT8;
  } else {
    string expected_data_type_list = ConcatString("float32, int32, int8, float16, uint8");
    OpsInputDtypeErrReport(op.GetName(), "dtype", expected_data_type_list, output_dtype_str);
    OP_LOGE(" don't supports this dtype.");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> out_shape;
  out_shape.push_back(num_classes);
  out_shape.push_back(num_classes);
  vector<int64_t> y_shape(out_shape);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(y_shape));
  td.SetDataType((DataType)output_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ConfusionMatrix, ConfusionMatrixInferShape);
// ------------------ConfusionMatrix END------------------

IMPLEMT_COMMON_INFERFUNC(ScatterMulInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ScatterMul, ScatterMulVerify) {
  DataType var_dtype = op.GetInputDesc(0).GetDataType();
  DataType updates_dtype = op.GetInputDesc(2).GetDataType();
  if (var_dtype != updates_dtype) {
    OpsTwoInputDtypeErrReport(op.GetName(), "var", "updates", ConcatString(var_dtype), ConcatString(updates_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the scatter_mul op inputs "
            "should have the same dtype!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterMul, ScatterMulInferShape);
VERIFY_FUNC_REG(ScatterMul, ScatterMulVerify);

// ------------------ScatterUpdate---------------------
IMPLEMT_VERIFIER(ScatterUpdate, ScatterUpdateVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(ScatterUpdateInferShape)
  // main part of shape infer
  ge::GeShape var_shape = op_desc->MutableInputDesc("var")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op_desc->MutableInputDesc("var")->GetShapeRange(var_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("var")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("var");
  td->SetShape(var_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(var_shape_range);
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(ScatterUpdate, ScatterUpdateInferShape);
VERIFY_FUNC_REG(ScatterUpdate, ScatterUpdateVerify);
// --------------ScatterUpdate END------------------

IMPLEMT_COMMON_INFERFUNC(ScatterMinInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ScatterMin, ScatterMinVerify) {
  DataType var_dtype = op.GetInputDesc(0).GetDataType();
  DataType updates_dtype = op.GetInputDesc(2).GetDataType();
  if (var_dtype != updates_dtype) {
    OpsTwoInputDtypeErrReport(op.GetName(), "var", "updates", ConcatString(var_dtype), ConcatString(updates_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the scatter_min op inputs "
            "should have the same dtype!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterMin, ScatterMinInferShape);
VERIFY_FUNC_REG(ScatterMin, ScatterMinVerify);

IMPLEMT_COMMON_INFERFUNC(ScatterMaxInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ScatterMax, ScatterMaxVerify) {
  DataType var_dtype = op.GetInputDesc(0).GetDataType();
  DataType updates_dtype = op.GetInputDesc(2).GetDataType();
  if (var_dtype != updates_dtype) {
    OpsTwoInputDtypeErrReport(op.GetName(), "var", "updates", ConcatString(var_dtype), ConcatString(updates_dtype));
    OP_LOGE(op.GetName().c_str(),
            "the scatter_max op inputs "
            "should have the same dtype!\n");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ScatterMax, ScatterMaxInferShape);
VERIFY_FUNC_REG(ScatterMax, ScatterMaxVerify);

bool FullyDefined(Shape s) {
  auto dims = s.GetDims();
  if (dims == ge::UNKNOWN_SHAPE) {
    return false;
  }
  for (auto& dim : dims) {
    if (dim == ge::UNKNOWN_DIM) {
      return false;
    }
  }
  return true;
}

IMPLEMT_COMMON_INFERFUNC(MatrixDiagPartV2InferShape) {
  Shape input_shape;
  auto input_tensor_desc = op.GetInputDesc(0);
  if (WithRankAtLeast(input_tensor_desc, 2, input_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input input must be at least 2-D, real rank is %lld.",
            input_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(1);
  if (WithRankAtMost(k_tensor_desc, 1, k_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input k must be at most 1-D, real rank is %lld.",
            k_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape padding_value_shape;
  auto padding_value_tensor_desc = op.GetInputDesc(2);
  if (WithRank(padding_value_tensor_desc, 0, padding_value_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input padding_value rank must be 0, real rank is %lld.",
            padding_value_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("diagonal");

  Tensor k_tensor;
  if (op.GetInputConstData("k", k_tensor) != GRAPH_SUCCESS || !RankKnown(input_shape) || !FullyDefined(k_shape)) {
    output_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
    output_desc.SetDataType(input_tensor_desc.GetDataType());
    (void)op.UpdateOutputDesc("diagonal", output_desc);
    return GRAPH_SUCCESS;
  }

  int32_t lower_diag_index = 0;
  int32_t upper_diag_index = 0;
  if (k_shape.GetDimNum() == 0) {
    lower_diag_index = *(reinterpret_cast<int32_t*>(k_tensor.GetData()));
    upper_diag_index = lower_diag_index;
  } else {
    auto k_dims = k_shape.GetDims();
    int64_t num_elements = k_dims[0];
    if (num_elements == 1) {
      int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
      lower_diag_index = *data;
      upper_diag_index = lower_diag_index;
    } else if (num_elements == 2) {
      int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
      lower_diag_index = *(data);
      upper_diag_index = *(data + 1);
    } else {
      OP_LOGE(op.GetName().c_str(), "diag_index must be a vector with one or two elements. It has %lld elements.",
              num_elements);
      return GRAPH_PARAM_INVALID;
    }
  }

  if (lower_diag_index > upper_diag_index) {
    OP_LOGE(op.GetName().c_str(), "lower_diag_index is greater than upper_diag_index");
    return GRAPH_PARAM_INVALID;
  }

  auto input_dims = input_shape.GetDims();
  const int32_t input_rank = input_shape.GetDimNum();
  const int32_t num_rows = input_dims[input_rank - 2];
  const int32_t num_cols = input_dims[input_rank - 1];
  int64_t max_diag_len = ge::UNKNOWN_DIM;
  if (num_rows != ge::UNKNOWN_DIM && num_cols != ge::UNKNOWN_DIM) {
    if (lower_diag_index != 0 && (-num_rows >= lower_diag_index || lower_diag_index >= num_cols)) {
      OP_LOGE(op.GetName().c_str(), "lower_diag_index %lld is out of bound, num_rows: %lld, num_cols: %lld.",
              lower_diag_index, num_rows, num_cols);
      return GRAPH_PARAM_INVALID;
    }
    if (upper_diag_index != 0 && (-num_rows >= upper_diag_index || upper_diag_index >= num_cols)) {
      OP_LOGE(op.GetName().c_str(), "upper_diag_index %lld is out of bound, num_rows: %lld, num_cols: %lld.",
              upper_diag_index, num_rows, num_cols);
      return GRAPH_PARAM_INVALID;
    }
    max_diag_len = std::min(num_rows + std::min(upper_diag_index, 0), num_cols - std::max(lower_diag_index, 0));
  }

  std::vector<int64_t> output_dims;
  for (int32_t i = 0; i < input_rank - 2; ++i) {
    output_dims.push_back(input_dims[i]);
  }
  if (lower_diag_index < upper_diag_index) {
    output_dims.push_back(upper_diag_index - lower_diag_index + 1);
  }
  output_dims.push_back(max_diag_len);
  output_desc.SetShape(Shape(output_dims));
  output_desc.SetDataType(input_tensor_desc.GetDataType());
  (void)op.UpdateOutputDesc("diagonal", output_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagPartV2, MatrixDiagPartV2InferShape);

IMPLEMT_COMMON_INFERFUNC(MatrixSetDiagV2InferShape) {
  Shape input_shape;
  auto input_tensor_desc = op.GetInputDesc(0);
  if (WithRankAtLeast(input_tensor_desc, 2, input_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input input must be at least 2-D, real rank is %lld.",
            input_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape diagonal_shape;
  auto diagonal_tensor_desc = op.GetInputDesc(1);
  if (WithRankAtLeast(diagonal_tensor_desc, 1, diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input diagonal must be at least 1-D, real rank is %lld.",
            diagonal_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(2);
  if (WithRankAtMost(k_tensor_desc, 1, k_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input k rank must be at most 1-D, real rank is %lld.",
            k_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  int32_t lower_diag_index = 0;
  int32_t upper_diag_index = 0;
  bool k_index_known = false;
  Tensor k_tensor;
  if (op.GetInputConstData("k", k_tensor) == GRAPH_SUCCESS && FullyDefined(k_shape)) {
    k_index_known = true;
    if (k_shape.GetDimNum() == 0) {
      lower_diag_index = *(reinterpret_cast<int32_t*>(k_tensor.GetData()));
      upper_diag_index = lower_diag_index;
    } else {
      auto k_dims = k_shape.GetDims();
      int64_t num_elements = k_dims[0];
      if (num_elements == 1) {
        int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
        lower_diag_index = *data;
        upper_diag_index = lower_diag_index;
      } else if (num_elements == 2) {
        int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
        lower_diag_index = *(data);
        upper_diag_index = *(data + 1);
      } else {
        OP_LOGE(op.GetName().c_str(), "diag_index must be a vector with one or two elements. It has %lld elements",
                num_elements);
        return GRAPH_PARAM_INVALID;
      }
    }

    if (lower_diag_index > upper_diag_index) {
      OP_LOGE(op.GetName().c_str(), "lower_diag_index is greater than upper_diag_index");
      return GRAPH_PARAM_INVALID;
    }
  }

  if (RankKnown(input_shape)) {
    auto input_rank = input_shape.GetDimNum();
    if (k_index_known) {
      if (WithRank(diagonal_tensor_desc, (lower_diag_index == upper_diag_index ? input_rank - 1 : input_rank),
                   diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input diagonal must be fit with input matrix rank %lld, real rank is %lld.",
                input_rank, diagonal_shape.GetDimNum());
        return GRAPH_FAILED;
      } else {
        if (WithRankAtLeast(diagonal_tensor_desc, input_rank - 1, diagonal_shape, op.GetName().c_str()) !=
            GRAPH_SUCCESS) {
          OP_LOGE(op.GetName().c_str(), "input diagonal must be at least %lld-D, real rank is %lld.", input_rank - 1,
                  diagonal_shape.GetDimNum());
          return GRAPH_FAILED;
        }

        if (WithRankAtMost(diagonal_tensor_desc, input_rank, diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
          OP_LOGE(op.GetName().c_str(), "input diagonal must be at most %lld-D, real rank is %lld.", input_rank,
                  diagonal_shape.GetDimNum());
          return GRAPH_FAILED;
        }
      }

      auto input_dims = input_shape.GetDims();
      const int32_t num_rows = input_dims[input_rank - 2];
      const int32_t num_cols = input_dims[input_rank - 1];
      if (num_rows != ge::UNKNOWN_DIM && num_cols != ge::UNKNOWN_DIM) {
        if (lower_diag_index != 0 && (-num_rows >= lower_diag_index || lower_diag_index >= num_cols)) {
          OP_LOGE(op.GetName().c_str(), "lower_diag_index is out of bound.");
          return GRAPH_PARAM_INVALID;
        }
        if (upper_diag_index != 0 && (-num_rows >= upper_diag_index || upper_diag_index >= num_cols)) {
          OP_LOGE(op.GetName().c_str(), "upper_diag_index is out of bound.");
          return GRAPH_PARAM_INVALID;
        }
      }
    }
  }

  auto output_desc = op.GetOutputDesc("output");
  Shape output_shape = input_shape;
  if (RankKnown(diagonal_shape) && !FullyDefined(input_shape)) {
    Shape diagonal_prefix_shape;
    if (SubShape(diagonal_shape, 0, (lower_diag_index == upper_diag_index ? -1 : -2), 1, diagonal_prefix_shape,
                 op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to get subshape from diagonal_shape");
      return GRAPH_FAILED;
    }

    if (Concatenate(diagonal_prefix_shape, UnknownShapeOfRank(2), diagonal_shape) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to concatenate diag_prefix_shape and 2-D unknown_shape");
      return GRAPH_FAILED;
    }

    if (Merge(input_shape, diagonal_shape, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to merge input_shape and diagonal_shape.");
      return GRAPH_FAILED;
    }
  }
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(input_tensor_desc.GetDataType());
  (void)op.UpdateOutputDesc("output", output_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSetDiagV2, MatrixSetDiagV2InferShape);

IMPLEMT_COMMON_INFERFUNC(MatrixDiagV2InferShape) {
  Shape diagonal_shape;
  auto diagonal_tensor_desc = op.GetInputDesc(0);
  if (WithRankAtLeast(diagonal_tensor_desc, 1, diagonal_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input diagonal must be at least 1-D, real rank is %lld.",
            diagonal_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape k_shape;
  auto k_tensor_desc = op.GetInputDesc(1);
  if (WithRankAtMost(k_tensor_desc, 1, k_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input k rank must be at most 1-D, real rank is %lld.",
            k_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape num_rows_shape;
  auto num_rows_tensor_desc = op.GetInputDesc(2);
  if (WithRank(num_rows_tensor_desc, 0, num_rows_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input num_rows rank must be 0, real rank is %lld.",
            num_rows_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape num_cols_shape;
  auto num_cols_tensor_desc = op.GetInputDesc(3);
  if (WithRank(num_cols_tensor_desc, 0, num_cols_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input num_cols rank must be 0, real rank is %lld.",
            num_cols_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  Shape padding_value_shape;
  auto padding_value_tensor_desc = op.GetInputDesc(4);
  if (WithRank(padding_value_tensor_desc, 0, padding_value_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input padding_value rank must be 0, real rank is %lld.",
            padding_value_tensor_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto output_desc = op.GetOutputDesc("output");
  Tensor k_tensor;
  if (op.GetInputConstData("k", k_tensor) != GRAPH_SUCCESS || !RankKnown(diagonal_shape) || !FullyDefined(k_shape)) {
    output_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
    output_desc.SetDataType(diagonal_tensor_desc.GetDataType());
    (void)op.UpdateOutputDesc("output", output_desc);
    return GRAPH_SUCCESS;
  }

  int32_t lower_diag_index = 0;
  int32_t upper_diag_index = 0;
  if (k_shape.GetDimNum() == 0) {
    lower_diag_index = *(reinterpret_cast<int32_t*>(k_tensor.GetData()));
    upper_diag_index = lower_diag_index;
  } else {
    auto k_dims = k_shape.GetDims();
    int32_t num_elements = k_dims[0];
    if (num_elements == 1) {
      int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
      lower_diag_index = *data;
      upper_diag_index = lower_diag_index;
    } else if (num_elements == 2) {
      int32_t* data = reinterpret_cast<int32_t*>(k_tensor.GetData());
      lower_diag_index = *(data);
      upper_diag_index = *(data + 1);
    } else {
      OP_LOGE(op.GetName().c_str(), "diag_index must be a vector with one or two elements. It has %lld elements.",
              num_elements);
      return GRAPH_PARAM_INVALID;
    }
  }

  if (lower_diag_index > upper_diag_index) {
    OP_LOGE(op.GetName().c_str(), "lower_diag_index is greater than upper_diag_index.");
    return GRAPH_PARAM_INVALID;
  }

  auto diagonal_dims = diagonal_shape.GetDims();
  const int32_t diagonal_rank = diagonal_shape.GetDimNum();
  if (lower_diag_index < upper_diag_index) {
    const int64_t num_diags = diagonal_dims[diagonal_rank - 2];
    const int64_t other_dim = diagonal_dims[diagonal_rank - 1];
    if (num_diags != (upper_diag_index - lower_diag_index + 1)) {
      OP_LOGE(op.GetName().c_str(),
              "The number of rows of `diagonal` doesn't match the number of \
               diagonals implied from `d_lower` and `d_upper` \
               num_diags = %lld, d_lower = %lld, d_upper = %lld, other_dim = %lld",
              num_diags, lower_diag_index, upper_diag_index, other_dim);
      return GRAPH_PARAM_INVALID;
    }
  }

  int32_t num_rows = ge::UNKNOWN_DIM;
  Tensor num_rows_tensor;
  if (op.GetInputConstData("num_rows", num_rows_tensor) == GRAPH_SUCCESS) {
    num_rows = *(reinterpret_cast<int32_t*>(num_rows_tensor.GetData()));
  }

  int32_t num_cols = ge::UNKNOWN_DIM;
  Tensor num_cols_tensor;
  if (op.GetInputConstData("num_cols", num_cols_tensor) == GRAPH_SUCCESS) {
    num_cols = *(reinterpret_cast<int32_t*>(num_cols_tensor.GetData()));
  }

  const int32_t max_diag_len = diagonal_dims[diagonal_rank - 1];
  const int32_t min_num_rows = max_diag_len - std::min(upper_diag_index, 0);
  const int32_t min_num_cols = max_diag_len + std::max(lower_diag_index, 0);

  if (num_rows == ge::UNKNOWN_DIM && num_cols == ge::UNKNOWN_DIM) {
    num_rows = std::max(min_num_rows, min_num_cols);
    num_cols = num_rows;
  }

  if (num_rows == ge::UNKNOWN_DIM) {
    num_rows = min_num_rows;
  } else if (num_rows < min_num_rows) {
    OP_LOGE(op.GetName().c_str(), "num_rows %d is too small.", num_rows);
    return GRAPH_PARAM_INVALID;
  }

  if (num_cols == ge::UNKNOWN_DIM) {
    num_cols = min_num_cols;
  } else if (num_cols < min_num_cols) {
    OP_LOGE(op.GetName().c_str(), "num_cols %d is too small.", num_cols);
    return GRAPH_PARAM_INVALID;
  }

  if (num_rows != min_num_rows && num_cols != min_num_cols) {
    OP_LOGE(op.GetName().c_str(),
            "num_rows and num_cols are not consistent with lower_diag_index, \
             upper_diag_index, and the length of the given diagonals. \
             num_rows = %lld != min_num_rows = %lld, num_cols = %lld != min_num_cols = %lld",
            num_rows, min_num_rows, num_cols, min_num_cols);
    return GRAPH_PARAM_INVALID;
  }

  Shape output_shape;
  OP_LOGE(op.GetName().c_str(), "num_rows: ", num_rows, " num_cols: ", num_cols);
  if (lower_diag_index == upper_diag_index) {
    if (ReplaceDim(diagonal_shape, diagonal_rank - 1, num_rows, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to replacedim from diagonal_shape.");
      return GRAPH_FAILED;
    }
    if (Concatenate(output_shape, Shape({num_cols}), output_shape) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to concatenate betweent outputshape and shape({num_cols}).");
      return GRAPH_FAILED;
    }
  } else {
    if (ReplaceDim(diagonal_shape, diagonal_rank - 2, num_rows, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to replacedim from diagonal_shape.");
      return GRAPH_FAILED;
    }
    if (ReplaceDim(output_shape, diagonal_rank - 1, num_cols, output_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "failed to replacedim from output_shape.");
      return GRAPH_FAILED;
    }
  }
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(diagonal_tensor_desc.GetDataType());
  (void)op.UpdateOutputDesc("output", output_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDiagV2, MatrixDiagV2InferShape);

}  // namespace ge
