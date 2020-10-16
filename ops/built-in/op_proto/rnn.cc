/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file rnn.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include "inc/rnn.h"
#include <cmath>
#include <vector>
#include <string>
#include <string.h>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"
namespace ge {
IMPLEMT_VERIFIER(DynamicLSTM, DynamicLSTMVerify) {

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicLSTM, DynamicLSTMInferShape) {
  ge::TensorDesc inputXTensorDesc = op.GetInputDesc("x");
  ge::TensorDesc inputWTensorDesc = op.GetInputDesc("w");
  ge::Shape shapeX = inputXTensorDesc.GetShape();
  ge::Shape shapeW = inputWTensorDesc.GetShape();
  DataType inputXDtype = inputXTensorDesc.GetDataType();

  TensorDesc outputHTensorDesc = op.GetOutputDesc("output_h");

  inputWTensorDesc.SetFormat(ge::FORMAT_HWCN);
  inputWTensorDesc.SetOriginFormat(ge::FORMAT_HWCN);

  int64_t dim_num = shapeX.GetDimNum();
  int64_t batchSize = 0;
  int64_t hiddenSize = 0;
  int64_t num_step = 0;
  if (dim_num == 3) {
    num_step = shapeX.GetDims().at(0);
    batchSize = shapeX.GetDims().at(1);
    hiddenSize = shapeW.GetDims().at(1) / 4;
  } else {
    OP_LOGE(op.GetName().c_str(),
      "The input shape of X is not right, please check!");
    return GRAPH_FAILED;
  }

  vector<int64_t> OutputHDims = {num_step, batchSize, hiddenSize};
  outputHTensorDesc.SetShape(ge::Shape(OutputHDims));
  outputHTensorDesc.SetDataType(inputXDtype);

  (void) op.UpdateOutputDesc("output_h", outputHTensorDesc);
  (void) op.UpdateInputDesc("w", inputWTensorDesc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicLSTM, DynamicLSTMInferShape);
VERIFY_FUNC_REG(DynamicLSTM, DynamicLSTMVerify);

IMPLEMT_VERIFIER(BasicLSTMCell, BasicLSTMCellVerify) {

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BasicLSTMCell, BasicLSTMCellInferShape) {
  ge::TensorDesc inputHTensorDesc = op.GetInputDesc("h");
  ge::TensorDesc inputCTensorDesc = op.GetInputDesc("c");
   ge::TensorDesc inputWTensorDesc = op.GetInputDesc("w");
  ge::Shape shape = inputCTensorDesc.GetShape();
  ge::Shape shapeH = inputHTensorDesc.GetShape();
  DataType inputHDtype = inputHTensorDesc.GetDataType();
  DataType inputCDtype = inputCTensorDesc.GetDataType();

  inputWTensorDesc.SetFormat(ge::FORMAT_HWCN);
  inputWTensorDesc.SetOriginFormat(ge::FORMAT_HWCN);


  TensorDesc outputCtTensorDesc = op.GetOutputDesc("ct");
  TensorDesc outputHtTensorDesc = op.GetOutputDesc("ht");
  TensorDesc outputItTensorDesc = op.GetOutputDesc("it");
  TensorDesc outputJtTensorDesc = op.GetOutputDesc("jt");
  TensorDesc outputFtTensorDesc = op.GetOutputDesc("ft");
  TensorDesc outputOtTensorDesc = op.GetOutputDesc("ot");
  TensorDesc outputTanhctTensorDesc = op.GetOutputDesc("tanhct");

  outputCtTensorDesc.SetShape(shape);
  outputCtTensorDesc.SetDataType(inputCDtype);
  outputHtTensorDesc.SetShape(shapeH);
  outputHtTensorDesc.SetDataType(inputHDtype);
  outputItTensorDesc.SetShape(shape);
  outputItTensorDesc.SetDataType(inputCDtype);
  outputJtTensorDesc.SetShape(shape);
  outputJtTensorDesc.SetDataType(inputCDtype);
  outputFtTensorDesc.SetShape(shape);
  outputFtTensorDesc.SetDataType(inputCDtype);
  outputOtTensorDesc.SetShape(shape);
  outputOtTensorDesc.SetDataType(inputCDtype);
  outputTanhctTensorDesc.SetShape(shape);
  outputTanhctTensorDesc.SetDataType(inputCDtype);

  (void) op.UpdateOutputDesc("ct", outputCtTensorDesc);
  (void) op.UpdateOutputDesc("ht", outputHtTensorDesc);
  (void) op.UpdateOutputDesc("it", outputItTensorDesc);
  (void) op.UpdateOutputDesc("jt", outputJtTensorDesc);
  (void) op.UpdateOutputDesc("ft", outputFtTensorDesc);
  (void) op.UpdateOutputDesc("ot", outputOtTensorDesc);
  (void) op.UpdateOutputDesc("tanhct", outputTanhctTensorDesc);
  (void) op.UpdateInputDesc("w", inputWTensorDesc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BasicLSTMCell, BasicLSTMCellInferShape);
VERIFY_FUNC_REG(BasicLSTMCell, BasicLSTMCellVerify);

IMPLEMT_VERIFIER(BasicLSTMCellCStateGrad, BasicLSTMCellCStateGradVerify) {

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BasicLSTMCellCStateGrad, BasicLSTMCellCStateGradInferShape) {
  ge::TensorDesc inputCTensorDesc = op.GetInputDesc("c");

  ge::Shape inputCShape = inputCTensorDesc.GetShape();
  DataType inputCDtype = inputCTensorDesc.GetDataType();
  int64_t dim_num = inputCShape.GetDimNum();
  int64_t batch_size = 0;
  int64_t hidden_size = 0;
  if (dim_num == 2) {
    batch_size = inputCShape.GetDims().at(0);
    hidden_size = inputCShape.GetDims().at(1);
  } else {
    OpsAttrValueErrReport(op.GetName(), "c's dim", "2", Strcat(dim_num));
    OP_LOGE(op.GetName().c_str(),
      "The input shape of C is not right, please check!");
    return GRAPH_FAILED;
  }

  vector<int64_t> newDims;
  newDims = {batch_size, hidden_size*4};

  vector<int64_t> oldDims;
  oldDims = {batch_size, hidden_size};
  TensorDesc outputDgateTensorDesc = op.GetOutputDesc("dgate");
  TensorDesc outputDct1TensorDesc = op.GetOutputDesc("dct_1");

  outputDgateTensorDesc.SetShape(ge::Shape(newDims));
  outputDgateTensorDesc.SetDataType(inputCDtype);
  outputDct1TensorDesc.SetShape(ge::Shape(oldDims));
  outputDct1TensorDesc.SetDataType(inputCDtype);

  (void) op.UpdateOutputDesc("dgate", outputDgateTensorDesc);
  (void) op.UpdateOutputDesc("dct_1", outputDct1TensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BasicLSTMCellCStateGrad, BasicLSTMCellCStateGradInferShape);
VERIFY_FUNC_REG(BasicLSTMCellCStateGrad, BasicLSTMCellCStateGradVerify);

IMPLEMT_VERIFIER(BasicLSTMCellWeightGrad, BasicLSTMCellWeightGradVerify) {

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BasicLSTMCellWeightGrad, BasicLSTMCellWeightGradInferShape) {
  ge::TensorDesc inputxTensorDesc = op.GetInputDesc("x");
  ge::Shape inputXShape = inputxTensorDesc.GetShape();
  ge::TensorDesc inputHTensorDesc = op.GetInputDesc("h");
  ge::Shape inputHShape = inputHTensorDesc.GetShape();
  ge::TensorDesc inputDgateTensorDesc = op.GetInputDesc("dgate");
  DataType inputDgateDtype = inputDgateTensorDesc.GetDataType();
  TensorDesc outputDwTensorDesc = op.GetOutputDesc("dw");
  TensorDesc outputDbTensorDesc = op.GetOutputDesc("db");

  int64_t dim_num_x = inputXShape.GetDimNum();
  int64_t dim_num_h = inputHShape.GetDimNum();
  int64_t inputSize = 0;
  int64_t hiddenSize = 0;
  if ((dim_num_x == 2) && (dim_num_h == 2)) {
    inputSize = inputXShape.GetDims().at(1);
    hiddenSize = inputHShape.GetDims().at(1);
  } else {
    OP_LOGE(op.GetName().c_str(),
      "The input shape of X or H is not right, please check!");
    return GRAPH_FAILED;
  }

  Format outputdw_format = outputDwTensorDesc.GetFormat();
  vector<int64_t> dwDims = {};
  vector<int64_t> dbDims = {};
  if (outputdw_format == FORMAT_HWCN) {
    dwDims = {inputSize + hiddenSize, hiddenSize*4};
    dbDims = {hiddenSize*4};
  } else if (outputdw_format == FORMAT_ND) {
    outputDwTensorDesc.SetFormat(ge::FORMAT_HWCN);
    outputDwTensorDesc.SetOriginFormat(ge::FORMAT_HWCN);
    dwDims = {inputSize + hiddenSize, hiddenSize*4};
    dbDims = {hiddenSize*4};
  } else if (outputdw_format == FORMAT_NCHW) {
    dwDims = {hiddenSize*4, inputSize + hiddenSize, 1, 1};
    dbDims = {hiddenSize*4, 1, 1, 1};
  } else if (outputdw_format == FORMAT_NHWC) {
    dwDims = {hiddenSize*4, 1, 1, inputSize + hiddenSize};
    dbDims = {hiddenSize*4, 1, 1, 1};
  } else {
    string expected_format_list = Strcat("FORMAT_HWCN, FORMAT_NCHW, FORMAT_NHWC, FORMAT_ND");
    OpsInputFormatErrReport(op.GetName(), "dw", expected_format_list, Strcat(outputdw_format));
    OP_LOGE(op.GetName().c_str(), "Not supported the format of dw");
  }

  outputDwTensorDesc.SetShape(ge::Shape(dwDims));
  outputDwTensorDesc.SetDataType(inputDgateDtype);
  outputDbTensorDesc.SetShape(ge::Shape(dbDims));
  outputDbTensorDesc.SetDataType(inputDgateDtype);

  (void) op.UpdateOutputDesc("dw", outputDwTensorDesc);
  (void) op.UpdateOutputDesc("db", outputDbTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BasicLSTMCellWeightGrad, BasicLSTMCellWeightGradInferShape);
VERIFY_FUNC_REG(BasicLSTMCellWeightGrad, BasicLSTMCellWeightGradVerify);

IMPLEMT_VERIFIER(BasicLSTMCellInputGrad, BasicLSTMCellInputGradVerify) {

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BasicLSTMCellInputGrad, BasicLSTMCellInputGradInferShape) {
  ge::TensorDesc inputDgateTensorDesc = op.GetInputDesc("dgate");
  ge::Shape inputDgateShape = inputDgateTensorDesc.GetShape();
  DataType inputDgateDtype = inputDgateTensorDesc.GetDataType();
  ge::TensorDesc inputWTensorDesc = op.GetInputDesc("w");
  ge::Shape inputWShape = inputWTensorDesc.GetShape();

  int64_t dim_num = inputDgateShape.GetDimNum();
  int64_t batchSize = 0;
  int64_t hiddenSize = 0;
  int64_t inputSize = 0;
  if (dim_num == 2) {
    batchSize = inputDgateShape.GetDims().at(0);
    hiddenSize = inputDgateShape.GetDims().at(1) / 4;
  } else {
    OpsAttrValueErrReport(op.GetName(), "Dgate's dim", "2", Strcat(dim_num));
    OP_LOGE(op.GetName().c_str(),
      "The input shape of Dgate is not right, please check!");
    return GRAPH_FAILED;
  }

  int64_t dim_num_w = inputWShape.GetDimNum();
  if (dim_num_w == 2) {
    inputSize = inputWShape.GetDims().at(0) - hiddenSize;
  } else {
    OP_LOGE(op.GetName().c_str(),
      "The input shape or dataformat of W is not right, please check!");
    return GRAPH_FAILED;
  }

  inputWTensorDesc.SetFormat(ge::FORMAT_HWCN);
  inputWTensorDesc.SetOriginFormat(ge::FORMAT_HWCN);
  (void) op.UpdateInputDesc("w", inputWTensorDesc);

  vector<int64_t> dxtDims = {batchSize, inputSize};
  vector<int64_t> dhtDims = {batchSize, hiddenSize};

  TensorDesc outputDxtTensorDesc = op.GetOutputDesc("dxt");
  TensorDesc outputDhtTensorDesc = op.GetOutputDesc("dht");

  outputDxtTensorDesc.SetShape(ge::Shape(dxtDims));
  outputDxtTensorDesc.SetDataType(inputDgateDtype);
  outputDhtTensorDesc.SetShape(ge::Shape(dhtDims));
  outputDhtTensorDesc.SetDataType(inputDgateDtype);

  (void) op.UpdateOutputDesc("dxt", outputDxtTensorDesc);
  (void) op.UpdateOutputDesc("dht", outputDhtTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BasicLSTMCellInputGrad, BasicLSTMCellInputGradInferShape);
VERIFY_FUNC_REG(BasicLSTMCellInputGrad, BasicLSTMCellInputGradVerify);

//----------------BasicLSTMCell Op-------------------

IMPLEMT_VERIFIER(RNN, RNNVerify) {

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(RNN, RNNInferShape) {
  ge::TensorDesc inputXTensorDesc = op.GetInputDesc("x");
  ge::Shape shapeX = inputXTensorDesc.GetShape();
  DataType dtypeDst = op.GetInputDesc("bias_h").GetDataType();

  int64_t dimNumsX = shapeX.GetDimNum();

  int64_t batchSize = 0;
  if (dimNumsX == 3) {
    batchSize = shapeX.GetDims().at(1);
  } else {
    OpsAttrValueErrReport(op.GetName(), "x's dim'", "3", Strcat(dimNumsX));
    OP_LOGE(op.GetName().c_str(),
      "The input shape of x is not right, please check!");
    return GRAPH_FAILED;
  }

  int64_t time_size = shapeX.GetDims().at(0);

  int64_t hiddenSize = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("num_output", hiddenSize)) {
    OpsGetAttrErrReport(op.GetName(), "num_output");
    return GRAPH_FAILED;
  }
  vector<int64_t> dimsO = {time_size, batchSize, hiddenSize};
  vector<int64_t> dimsHt = {1, batchSize, hiddenSize};

  TensorDesc outputOTensorDesc = op.GetOutputDesc("o");
  TensorDesc outputHtTensorDesc = op.GetOutputDesc("h_t");

  outputOTensorDesc.SetShape(ge::Shape(dimsO));
  outputOTensorDesc.SetDataType(dtypeDst);
  outputHtTensorDesc.SetShape(ge::Shape(dimsHt));
  outputHtTensorDesc.SetDataType(dtypeDst);

  (void) op.UpdateOutputDesc("o", outputOTensorDesc);
  (void) op.UpdateOutputDesc("h_t", outputHtTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RNN, RNNInferShape);
VERIFY_FUNC_REG(RNN, RNNVerify);

IMPLEMT_VERIFIER(BasicRNNCell, BasicRNNCellVerify) {

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BasicRNNCell, BasicRNNCellInferShape) {
  ge::TensorDesc inputXTensorDesc = op.GetInputDesc("x");
  ge::Shape shapeX = inputXTensorDesc.GetShape();

  DataType dtypeDst = op.GetInputDesc("bias_h").GetDataType();

  int64_t dimNumsX = shapeX.GetDimNum();

  int64_t hiddenSize = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("num_output", hiddenSize)) {
    return GRAPH_FAILED;
  }

  int64_t batchSize = 0;
  if (dimNumsX == 2) {
    batchSize = shapeX.GetDims().at(0);
  } else {
    OP_LOGE(op.GetName().c_str(),
      "The input shape of x is not right, please check!");
    return GRAPH_FAILED;
  }
  vector<int64_t> dimsOt = {batchSize, hiddenSize};
  vector<int64_t> dimsHt = {batchSize, hiddenSize};

  TensorDesc outputOtTensorDesc = op.GetOutputDesc("o_t");
  TensorDesc outputHtTensorDesc = op.GetOutputDesc("h_t");

  outputOtTensorDesc.SetShape(ge::Shape(dimsOt));
  outputOtTensorDesc.SetDataType(dtypeDst);
  outputHtTensorDesc.SetShape(ge::Shape(dimsHt));
  outputHtTensorDesc.SetDataType(dtypeDst);

  (void) op.UpdateOutputDesc("o_t", outputOtTensorDesc);
  (void) op.UpdateOutputDesc("h_t", outputHtTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BasicRNNCell, BasicRNNCellInferShape);
VERIFY_FUNC_REG(BasicRNNCell, BasicRNNCellVerify);

}
