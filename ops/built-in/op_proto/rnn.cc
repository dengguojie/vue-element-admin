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
 * \file rnn.cpp
 * \brief
 */
#include "inc/rnn.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"

namespace ge {
IMPLEMT_VERIFIER(DynamicRNN, DynamicRNNVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicRNN, DynamicRNNInferShape) {
  ge::TensorDesc inputXTensorDesc = op.GetInputDesc("x");
  ge::TensorDesc inputWTensorDesc = op.GetInputDesc("w");
  ge::TensorDesc inputBTensorDesc = op.GetInputDesc("b");
  ge::Shape shapeX = inputXTensorDesc.GetShape();
  ge::Shape shapeW = inputWTensorDesc.GetShape();
  DataType inputXDtype = inputXTensorDesc.GetDataType();
  DataType inputBDtype = inputBTensorDesc.GetDataType();
  TensorDesc outputYTensorDesc = op.GetOutputDesc("y");
  TensorDesc outputHTensorDesc = op.GetOutputDesc("output_h");
  TensorDesc outputCTensorDesc = op.GetOutputDesc("output_c");
  TensorDesc outputITensorDesc = op.GetOutputDesc("i");
  TensorDesc outputJTensorDesc = op.GetOutputDesc("j");
  TensorDesc outputFTensorDesc = op.GetOutputDesc("f");
  TensorDesc outputOTensorDesc = op.GetOutputDesc("o");
  TensorDesc outputTanhcTensorDesc = op.GetOutputDesc("tanhc");

  int64_t dim_num = shapeX.GetDimNum();
  int64_t batchSize = 0;
  int64_t hiddenSize = 0;
  int64_t num_step = 0;
  if (dim_num == 3) {
    num_step = shapeX.GetDims().at(0);
    batchSize = shapeX.GetDims().at(1);
    hiddenSize = shapeW.GetDims().at(1) / 4;
  } else {
    OpsOneInputShapeErrReport(op.GetName(), "X Shape Dim", "The input shape of X not equal 3!");
    OP_LOGE(op.GetName().c_str(), "The input shape of X not equal 3, please check!");
    return GRAPH_FAILED;
  }

  vector<int64_t> outputHDims = {num_step, batchSize, hiddenSize};

  outputYTensorDesc.SetShape(ge::Shape(outputHDims));
  outputHTensorDesc.SetShape(ge::Shape(outputHDims));
  outputCTensorDesc.SetShape(ge::Shape(outputHDims));
  outputITensorDesc.SetShape(ge::Shape(outputHDims));
  outputJTensorDesc.SetShape(ge::Shape(outputHDims));
  outputFTensorDesc.SetShape(ge::Shape(outputHDims));
  outputOTensorDesc.SetShape(ge::Shape(outputHDims));
  outputTanhcTensorDesc.SetShape(ge::Shape(outputHDims));

  outputYTensorDesc.SetDataType(inputBDtype);
  outputHTensorDesc.SetDataType(inputXDtype);
  outputCTensorDesc.SetDataType(inputBDtype);
  outputITensorDesc.SetDataType(inputBDtype);
  outputJTensorDesc.SetDataType(inputBDtype);
  outputFTensorDesc.SetDataType(inputBDtype);
  outputOTensorDesc.SetDataType(inputBDtype);
  outputTanhcTensorDesc.SetDataType(inputBDtype);

  (void)op.UpdateOutputDesc("y", outputYTensorDesc);
  (void)op.UpdateOutputDesc("output_h", outputHTensorDesc);
  (void)op.UpdateOutputDesc("output_c", outputCTensorDesc);
  (void)op.UpdateOutputDesc("i", outputITensorDesc);
  (void)op.UpdateOutputDesc("j", outputJTensorDesc);
  (void)op.UpdateOutputDesc("f", outputFTensorDesc);
  (void)op.UpdateOutputDesc("o", outputOTensorDesc);
  (void)op.UpdateOutputDesc("tanhc", outputTanhcTensorDesc);

  inputWTensorDesc.SetFormat(ge::FORMAT_HWCN);
  (void)op.UpdateInputDesc("w", inputWTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicRNN, DynamicRNNInferShape);
VERIFY_FUNC_REG(DynamicRNN, DynamicRNNVerify);

IMPLEMT_VERIFIER(DynamicRNNGrad, DynamicRNNGradVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicRNNGrad, DynamicRNNGradInferShape) {
  ge::TensorDesc inputXTensorDesc = op.GetInputDesc("x");
  ge::TensorDesc inputHTensorDesc = op.GetInputDesc("h");
  ge::Shape shapeX = inputXTensorDesc.GetShape();
  ge::Shape shapeH = inputHTensorDesc.GetShape();
  DataType dtype = inputXTensorDesc.GetDataType();

  TensorDesc outputDxtTensorDesc = op.GetOutputDesc("dx");
  TensorDesc outputDht_1TensorDesc = op.GetOutputDesc("dh_prev");
  TensorDesc outputDct_1TensorDesc = op.GetOutputDesc("dc_prev");
  TensorDesc outputDwTensorDesc = op.GetOutputDesc("dw");
  TensorDesc outputDbTensorDesc = op.GetOutputDesc("db");

  int64_t dim_num = shapeX.GetDimNum();
  int64_t batch_size = 0;
  int64_t input_size = 0;
  int64_t hidden_size = 0;
  if (dim_num == 3) {
    batch_size = shapeX.GetDims().at(1);
    input_size = shapeX.GetDims().at(2);
    hidden_size = shapeH.GetDims().at(2);
  } else {
    OpsOneInputShapeErrReport(op.GetName(), "The input shape of X", "not right");
    OP_LOGE(op.GetName().c_str(), "The input shape of X is not right, please check!");
    return GRAPH_FAILED;
  }

  outputDxtTensorDesc.SetShape(shapeX);
  outputDxtTensorDesc.SetDataType(dtype);

  vector<int64_t> outputDht_1Dims = {batch_size, hidden_size};
  outputDht_1TensorDesc.SetShape(ge::Shape(outputDht_1Dims));
  outputDht_1TensorDesc.SetDataType(dtype);

  vector<int64_t> outputDct_1Dims = {batch_size, hidden_size};
  outputDct_1TensorDesc.SetShape(ge::Shape(outputDct_1Dims));
  outputDct_1TensorDesc.SetDataType(dtype);

  vector<int64_t> outputDwDims = {input_size + hidden_size, 4 * hidden_size};
  outputDwTensorDesc.SetShape(ge::Shape(outputDwDims));
  outputDwTensorDesc.SetDataType(dtype);

  vector<int64_t> outputDbDims = {4 * hidden_size};
  outputDbTensorDesc.SetShape(ge::Shape(outputDbDims));
  outputDbTensorDesc.SetDataType(dtype);

  (void)op.UpdateOutputDesc("dx", outputDxtTensorDesc);
  (void)op.UpdateOutputDesc("dh_prev", outputDht_1TensorDesc);
  (void)op.UpdateOutputDesc("dc_prev", outputDct_1TensorDesc);
  (void)op.UpdateOutputDesc("dw", outputDwTensorDesc);
  (void)op.UpdateOutputDesc("db", outputDbTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicRNNGrad, DynamicRNNGradInferShape);
VERIFY_FUNC_REG(DynamicRNNGrad, DynamicRNNGradVerify);

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
    OpsOneInputShapeErrReport(op.GetName(), "X Shape Dim", "The input shape of X not equal 3!");
    OP_LOGE(op.GetName().c_str(), "The input shape of X not equal 3, please check!");
    return GRAPH_FAILED;
  }

  vector<int64_t> OutputHDims = {num_step, batchSize, hiddenSize};
  outputHTensorDesc.SetShape(ge::Shape(OutputHDims));
  outputHTensorDesc.SetDataType(inputXDtype);

  (void)op.UpdateOutputDesc("output_h", outputHTensorDesc);
  (void)op.UpdateInputDesc("w", inputWTensorDesc);
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

  (void)op.UpdateOutputDesc("ct", outputCtTensorDesc);
  (void)op.UpdateOutputDesc("ht", outputHtTensorDesc);
  (void)op.UpdateOutputDesc("it", outputItTensorDesc);
  (void)op.UpdateOutputDesc("jt", outputJtTensorDesc);
  (void)op.UpdateOutputDesc("ft", outputFtTensorDesc);
  (void)op.UpdateOutputDesc("ot", outputOtTensorDesc);
  (void)op.UpdateOutputDesc("tanhct", outputTanhctTensorDesc);
  (void)op.UpdateInputDesc("w", inputWTensorDesc);
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
    OpsAttrValueErrReport(op.GetName(), "c's dim", "2", ConcatString(dim_num));
    OP_LOGE(op.GetName().c_str(), "The input shape of C is not right, please check!");
    return GRAPH_FAILED;
  }

  vector<int64_t> newDims;
  newDims = {batch_size, hidden_size * 4};

  vector<int64_t> oldDims;
  oldDims = {batch_size, hidden_size};
  TensorDesc outputDgateTensorDesc = op.GetOutputDesc("dgate");
  TensorDesc outputDct1TensorDesc = op.GetOutputDesc("dct_1");

  outputDgateTensorDesc.SetShape(ge::Shape(newDims));
  outputDgateTensorDesc.SetDataType(inputCDtype);
  outputDct1TensorDesc.SetShape(ge::Shape(oldDims));
  outputDct1TensorDesc.SetDataType(inputCDtype);

  (void)op.UpdateOutputDesc("dgate", outputDgateTensorDesc);
  (void)op.UpdateOutputDesc("dct_1", outputDct1TensorDesc);

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
    OpsTwoInputShapeErrReport(op.GetName(), "X Shape Dim", "H Shape Dim", "The input shape of X and H should be 2!");
    OP_LOGE(op.GetName().c_str(), "The input shape of X and H should be 2, please check!");
    return GRAPH_FAILED;
  }

  Format outputdw_format = outputDwTensorDesc.GetFormat();
  vector<int64_t> dwDims = {};
  vector<int64_t> dbDims = {};
  if (outputdw_format == FORMAT_HWCN) {
    dwDims = {inputSize + hiddenSize, hiddenSize * 4};
    dbDims = {hiddenSize * 4};
  } else if (outputdw_format == FORMAT_ND) {
    outputDwTensorDesc.SetFormat(ge::FORMAT_HWCN);
    outputDwTensorDesc.SetOriginFormat(ge::FORMAT_HWCN);
    dwDims = {inputSize + hiddenSize, hiddenSize * 4};
    dbDims = {hiddenSize * 4};
  } else if (outputdw_format == FORMAT_NCHW) {
    dwDims = {hiddenSize * 4, inputSize + hiddenSize, 1, 1};
    dbDims = {hiddenSize * 4, 1, 1, 1};
  } else if (outputdw_format == FORMAT_NHWC) {
    dwDims = {hiddenSize * 4, 1, 1, inputSize + hiddenSize};
    dbDims = {hiddenSize * 4, 1, 1, 1};
  } else {
    string expected_format_list = ConcatString("FORMAT_HWCN, FORMAT_NCHW, FORMAT_NHWC, FORMAT_ND");
    OpsInputFormatErrReport(op.GetName(), "dw", expected_format_list, ConcatString(outputdw_format));
    OP_LOGE(op.GetName().c_str(), "Not supported the format of dw");
  }

  outputDwTensorDesc.SetShape(ge::Shape(dwDims));
  outputDwTensorDesc.SetDataType(inputDgateDtype);
  outputDbTensorDesc.SetShape(ge::Shape(dbDims));
  outputDbTensorDesc.SetDataType(inputDgateDtype);

  (void)op.UpdateOutputDesc("dw", outputDwTensorDesc);
  (void)op.UpdateOutputDesc("db", outputDbTensorDesc);

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
    OpsAttrValueErrReport(op.GetName(), "Dgate's dim", "2", ConcatString(dim_num));
    OP_LOGE(op.GetName().c_str(), "The input shape of Dgate is not right, please check!");
    return GRAPH_FAILED;
  }

  int64_t dim_num_w = inputWShape.GetDimNum();
  if (dim_num_w == 2) {
    inputSize = inputWShape.GetDims().at(0) - hiddenSize;
  } else {
    OpsOneInputShapeErrReport(op.GetName(), "W Shape Dim", "The input shape of W should be 2, please check!");
    OP_LOGE(op.GetName().c_str(), "The input shape of W should be 2, please check!");
    return GRAPH_FAILED;
  }

  inputWTensorDesc.SetFormat(ge::FORMAT_HWCN);
  inputWTensorDesc.SetOriginFormat(ge::FORMAT_HWCN);
  (void)op.UpdateInputDesc("w", inputWTensorDesc);

  vector<int64_t> dxtDims = {batchSize, inputSize};
  vector<int64_t> dhtDims = {batchSize, hiddenSize};

  TensorDesc outputDxtTensorDesc = op.GetOutputDesc("dxt");
  TensorDesc outputDhtTensorDesc = op.GetOutputDesc("dht");

  outputDxtTensorDesc.SetShape(ge::Shape(dxtDims));
  outputDxtTensorDesc.SetDataType(inputDgateDtype);
  outputDhtTensorDesc.SetShape(ge::Shape(dhtDims));
  outputDhtTensorDesc.SetDataType(inputDgateDtype);

  (void)op.UpdateOutputDesc("dxt", outputDxtTensorDesc);
  (void)op.UpdateOutputDesc("dht", outputDhtTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BasicLSTMCellInputGrad, BasicLSTMCellInputGradInferShape);
VERIFY_FUNC_REG(BasicLSTMCellInputGrad, BasicLSTMCellInputGradVerify);

// ----------------BasicLSTMCell Op-------------------

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
    OpsAttrValueErrReport(op.GetName(), "x's dim'", "3", ConcatString(dimNumsX));
    OP_LOGE(op.GetName().c_str(), "The input shape of x is not right, please check!");
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

  (void)op.UpdateOutputDesc("o", outputOTensorDesc);
  (void)op.UpdateOutputDesc("h_t", outputHtTensorDesc);

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
    OpsOneInputShapeErrReport(op.GetName(), "X Shape Dim", "The input shape of X should be 2!");
    OP_LOGE(op.GetName().c_str(), "The input shape of X should be 2, please check!");
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

  (void)op.UpdateOutputDesc("o_t", outputOtTensorDesc);
  (void)op.UpdateOutputDesc("h_t", outputHtTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BasicRNNCell, BasicRNNCellInferShape);
VERIFY_FUNC_REG(BasicRNNCell, BasicRNNCellVerify);

IMPLEMT_VERIFIER(DynamicGRU, DynamicGRUVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicGRU, DynamicGRUInferShape) {
  ge::TensorDesc x_tensor_desc = op.GetInputDesc("x");
  ge::TensorDesc w_tensor_desc = op.GetInputDesc("w");
  ge::TensorDesc cw_tensor_desc = op.GetInputDesc("cw");
  ge::TensorDesc b_tensor_desc = op.GetInputDesc("b");
  ge::Shape shape_x = x_tensor_desc.GetShape();
  ge::Shape shape_w = w_tensor_desc.GetShape();
  DataType bias_dtype = b_tensor_desc.GetDataType();

  TensorDesc y_tensor_desc = op.GetOutputDesc("y");
  TensorDesc h_tensor_desc = op.GetOutputDesc("output_h");
  TensorDesc r_tensor_desc = op.GetOutputDesc("r");
  TensorDesc i_tensor_desc = op.GetOutputDesc("i");
  TensorDesc n_tensor_desc = op.GetOutputDesc("n");

  int64_t dim_num = shape_x.GetDimNum();
  CHECK(dim_num != 3, OP_LOGE(op.GetName().c_str(), "The dimension count of x should be 3, please check!");
        OpsOneInputShapeErrReport(op.GetName(), "x", "The dimension count of x should be 3"),
        return GRAPH_FAILED);

  int64_t num_step = shape_x.GetDims().at(0);
  int64_t batch_size = shape_x.GetDims().at(1);
  int64_t hidden_size = shape_w.GetDims().at(1) / 2;

  vector<int64_t> output_dims = {num_step, batch_size, hidden_size};

  y_tensor_desc.SetShape(ge::Shape(output_dims));
  h_tensor_desc.SetShape(ge::Shape(output_dims));
  r_tensor_desc.SetShape(ge::Shape(output_dims));
  i_tensor_desc.SetShape(ge::Shape(output_dims));
  n_tensor_desc.SetShape(ge::Shape(output_dims));

  y_tensor_desc.SetDataType(bias_dtype);
  h_tensor_desc.SetDataType(bias_dtype);
  r_tensor_desc.SetDataType(bias_dtype);
  i_tensor_desc.SetDataType(bias_dtype);
  n_tensor_desc.SetDataType(bias_dtype);

  CHECK(op.UpdateOutputDesc("y", y_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);

  CHECK(op.UpdateOutputDesc("output_h", h_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("r", r_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("i", i_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("n", n_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);

  w_tensor_desc.SetFormat(ge::FORMAT_HWCN);
  cw_tensor_desc.SetFormat(ge::FORMAT_HWCN);
  CHECK(op.UpdateInputDesc("w", w_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateInputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateInputDesc("cw", cw_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicGRU, DynamicGRUInferShape);
VERIFY_FUNC_REG(DynamicGRU, DynamicGRUVerify);

IMPLEMT_VERIFIER(DynamicGRUV2, DynamicGRUV2Verify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicGRUV2, DynamicGRUV2InferShape) {
  TensorDesc x_tensor_desc = op.GetInputDesc("x");
  TensorDesc w_input_tensor_desc = op.GetInputDesc("weight_input");
  TensorDesc w_hidden_tensor_desc = op.GetInputDesc("weight_hidden");
  Shape shape_x = x_tensor_desc.GetShape();
  Shape shape_w_hidden = w_hidden_tensor_desc.GetShape();

  DataType bias_dtype = x_tensor_desc.GetDataType();
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc->MutableInputDesc("bias_input") != nullptr) {
    bias_dtype = op.GetInputDesc("bias_input").GetDataType();
  } else if (op_desc->MutableInputDesc("bias_hidden") != nullptr) {
    bias_dtype = op.GetInputDesc("bias_hidden").GetDataType();
  } else if (op_desc->MutableInputDesc("init_h") != nullptr) {
    bias_dtype = op.GetInputDesc("init_h").GetDataType();
  }

  TensorDesc y_tensor_desc = op.GetOutputDesc("y");
  TensorDesc h_tensor_desc = op.GetOutputDesc("output_h");
  TensorDesc update_tensor_desc = op.GetOutputDesc("update");
  TensorDesc reset_tensor_desc = op.GetOutputDesc("reset");
  TensorDesc new_tensor_desc = op.GetOutputDesc("new");
  TensorDesc hn_tensor_desc = op.GetOutputDesc("hidden_new");

  int64_t dim_num = shape_x.GetDimNum();
  CHECK(dim_num != 3, OP_LOGE(op.GetName().c_str(), "The dimension count of x should be 3, please check!");
        OpsOneInputShapeErrReport(op.GetName(), "x", "The dimension count of x should be 3"), return GRAPH_FAILED);

  int64_t num_step = shape_x.GetDims().at(0);
  int64_t batch_size = shape_x.GetDims().at(1);
  int64_t hidden_size = shape_w_hidden.GetDims().at(0);

  vector<int64_t> output_dims = {num_step, batch_size, hidden_size};

  y_tensor_desc.SetShape(Shape(output_dims));
  h_tensor_desc.SetShape(Shape(output_dims));
  update_tensor_desc.SetShape(Shape(output_dims));
  reset_tensor_desc.SetShape(Shape(output_dims));
  new_tensor_desc.SetShape(Shape(output_dims));
  hn_tensor_desc.SetShape(Shape(output_dims));

  y_tensor_desc.SetDataType(bias_dtype);
  h_tensor_desc.SetDataType(bias_dtype);
  update_tensor_desc.SetDataType(bias_dtype);
  reset_tensor_desc.SetDataType(bias_dtype);
  new_tensor_desc.SetDataType(bias_dtype);
  hn_tensor_desc.SetDataType(bias_dtype);

  CHECK(op.UpdateOutputDesc("y", y_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("output_h", h_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("update", update_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("reset", reset_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("new", new_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("hidden_new", new_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);

  w_input_tensor_desc.SetFormat(FORMAT_HWCN);
  w_hidden_tensor_desc.SetFormat(FORMAT_HWCN);
  CHECK(op.UpdateInputDesc("weight_input", w_input_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateIntputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateInputDesc("weight_hidden", w_hidden_tensor_desc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicGRUV2, DynamicGRUV2InferShape);
VERIFY_FUNC_REG(DynamicGRUV2, DynamicGRUV2Verify);

IMPLEMT_VERIFIER(DynamicGRUV2Grad, DynamicGRUV2GradVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicGRUV2Grad, DynamicGRUV2GradInferShape) {
  ge::TensorDesc inputXTensorDesc = op.GetInputDesc("x");
  ge::TensorDesc inputHTensorDesc = op.GetInputDesc("h");
  ge::Shape shapeX = inputXTensorDesc.GetShape();
  ge::Shape shapeH = inputHTensorDesc.GetShape();
  DataType dtype = inputXTensorDesc.GetDataType();

  int64_t dim_num = shapeX.GetDimNum();
  int64_t batch_size = 0;
  int64_t input_size = 0;
  int64_t hidden_size = 0;
  if (dim_num == 3) {
    batch_size = shapeX.GetDims().at(1);
    input_size = shapeX.GetDims().at(2);
    hidden_size = shapeH.GetDims().at(2);
  } else {
    OpsOneInputShapeErrReport(op.GetName(),  "The input shape of X", "not right");
    OP_LOGE(op.GetName().c_str(),
      "The input shape of X is not right, please check!");
    return GRAPH_FAILED;
  }

  TensorDesc outputDxtTensorDesc = op.GetOutputDesc("dx");
  TensorDesc outputDht_1TensorDesc = op.GetOutputDesc("dh_prev");
  TensorDesc outputDwxTensorDesc = op.GetOutputDesc("dw_input");
  TensorDesc outputDwhTensorDesc = op.GetOutputDesc("dw_hidden");
  TensorDesc outputDbxTensorDesc = op.GetOutputDesc("db_input");
  TensorDesc outputDbhTensorDesc = op.GetOutputDesc("db_hidden");

  outputDxtTensorDesc.SetShape(shapeX);
  outputDxtTensorDesc.SetDataType(dtype);

  vector<int64_t> outputDht_1Dims = {batch_size, hidden_size};
  outputDht_1TensorDesc.SetShape(ge::Shape(outputDht_1Dims));
  outputDht_1TensorDesc.SetDataType(dtype);

  vector<int64_t> outputDwxDims = {input_size, 3 * hidden_size};
  outputDwxTensorDesc.SetShape(ge::Shape(outputDwxDims));
  outputDwxTensorDesc.SetDataType(dtype);

  vector<int64_t> outputDwhDims = {hidden_size, 3 * hidden_size};
  outputDwhTensorDesc.SetShape(ge::Shape(outputDwhDims));
  outputDwhTensorDesc.SetDataType(dtype);

  vector<int64_t> outputDbDims = {3 * hidden_size};
  outputDbxTensorDesc.SetShape(ge::Shape(outputDbDims));
  outputDbxTensorDesc.SetDataType(dtype);

  outputDbhTensorDesc.SetShape(ge::Shape(outputDbDims));
  outputDbhTensorDesc.SetDataType(dtype);

  CHECK(op.UpdateOutputDesc("dx", outputDxtTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("dh_prev", outputDht_1TensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("dw_input", outputDwxTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("dw_hidden", outputDwhTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("db_input", outputDbxTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("db_hidden", outputDbhTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicGRUV2Grad, DynamicGRUV2GradInferShape);
VERIFY_FUNC_REG(DynamicGRUV2Grad, DynamicGRUV2GradVerify);

// ----------------EmbeddingDenseGrad Begin-------------------
bool InferShapeAndTypeEmbeddingDenseGrad(Operator &op,
                                         const string &input_name1,
                                         const string &input_name2,
                                         const string &output_name)
{
    TensorDesc v_output_desc = op.GetOutputDesc(output_name);
    DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
    Format input_format = op.GetInputDesc(input_name1).GetFormat();
    ge::Shape shapeX = op.GetInputDesc(input_name1).GetShape();
    std::vector<int64_t> dims_x = shapeX.GetDims();

    int64_t num_weights_value = 0;
    std::vector<int64_t> dim_vec;
    op.GetAttr("num_weights", num_weights_value);
    OP_LOGI(op.GetName().c_str(), "the attr num_weights is %d",
            num_weights_value);
    if (num_weights_value == 0)
    {
        OP_LOGE(op.GetName().c_str(), "num_weights not be zero");
        return false;
    }
    dim_vec.push_back(num_weights_value);
    if (IsUnknownRankShape(dims_x))
    {
        dim_vec.push_back(-1);
    }
    else
    {
        dim_vec.push_back(dims_x.back());
    }

    ge::Shape output_shape = ge::Shape(dim_vec);
    v_output_desc.SetShape(output_shape);
    v_output_desc.SetDataType(input_dtype);
    v_output_desc.SetFormat(input_format);
    if (IsUnknown(dim_vec))
    {
        std::vector<std::pair<int64_t, int64_t>> input_range;
        std::vector<std::pair<int64_t, int64_t>> output_shape_range;
        op.GetInputDesc(input_name1).GetShapeRange(input_range);
        if (input_range.empty())
        {
            output_shape_range = {{num_weights_value, num_weights_value}, {1, -1}};
        }
        else
        {
            output_shape_range = {{num_weights_value, num_weights_value}, input_range.back()};
        }
        v_output_desc.SetShapeRange(output_shape_range);
    }
    CHECK(op.UpdateOutputDesc(output_name, v_output_desc) != GRAPH_SUCCESS,
          OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
    return true;
}

IMPLEMT_VERIFIER(EmbeddingDenseGrad, EmbeddingDenseGradVerify)
{
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(g_embeddingDenseGradInferShape)
{
    if (InferShapeAndTypeEmbeddingDenseGrad(op, "grad", "indices", "y"))
    {
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(EmbeddingDenseGrad, g_embeddingDenseGradInferShape);
// Registered verify function
VERIFY_FUNC_REG(EmbeddingDenseGrad, EmbeddingDenseGradVerify);
// ----------------EmbeddingDenseGrad END---------------------

IMPLEMT_VERIFIER(CommonLSTM, CommonLSTMVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(CommonLSTM, CommonLSTMInferShape) {
  ge::TensorDesc inputXTensorDesc = op.GetInputDesc("x");
  ge::TensorDesc inputWTensorDesc = op.GetInputDesc("w");
  ge::Shape shapeX = inputXTensorDesc.GetShape();
  ge::Shape shapeW = inputWTensorDesc.GetShape();
  DataType inputXDtype = inputXTensorDesc.GetDataType();
  TensorDesc outputYTensorDesc = op.GetOutputDesc("y");
  TensorDesc outputHTensorDesc = op.GetOutputDesc("y_h");
  TensorDesc outputCTensorDesc = op.GetOutputDesc("y_c");

  int64_t dim_num = shapeX.GetDimNum();
  int64_t batchSize = 0;
  int64_t hiddenSize = 0;
  int64_t numStep = 0;
  int64_t numDirections = 0;
  if (dim_num == 3) {
    numStep = shapeX.GetDims().at(0);
    batchSize = shapeX.GetDims().at(1);
  } else {
    OpsOneInputShapeErrReport(op.GetName(), "X Shape Dim", "The input shape of X not equal 3!");
    OP_LOGE(op.GetName().c_str(), "The input shape of X not equal 3, please check!");
    return GRAPH_FAILED;
  }

  // shapeW [num_directions, 4*hidden_size, input_size]
  numDirections = shapeW.GetDims().at(0);
  hiddenSize = shapeW.GetDims().at(1) / 4;

  vector<int64_t> outputYDims = {numStep, numDirections, batchSize, hiddenSize};

  vector<int64_t> outputHCDims = {numDirections, batchSize, hiddenSize};

  outputYTensorDesc.SetShape(ge::Shape(outputYDims));
  outputHTensorDesc.SetShape(ge::Shape(outputHCDims));
  outputCTensorDesc.SetShape(ge::Shape(outputHCDims));

  outputYTensorDesc.SetDataType(inputXDtype);
  outputHTensorDesc.SetDataType(inputXDtype);
  outputCTensorDesc.SetDataType(inputXDtype);

  (void)op.UpdateOutputDesc("y", outputYTensorDesc);
  (void)op.UpdateOutputDesc("y_h", outputHTensorDesc);
  (void)op.UpdateOutputDesc("y_c", outputCTensorDesc);

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(CommonLSTM, CommonLSTMInferShape);
VERIFY_FUNC_REG(CommonLSTM, CommonLSTMVerify);

IMPLEMT_INFERFUNC(CommonGRU, CommonGRUInferShape) {
  TensorDesc x_tensor_desc = op.GetInputDesc("x");
  TensorDesc w_tensor_desc = op.GetInputDesc("w");
  TensorDesc r_tensor_desc = op.GetInputDesc("r");
  Shape shape_x = x_tensor_desc.GetShape();
  Shape shape_w = w_tensor_desc.GetShape();

  DataType bias_dtype = DT_FLOAT;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc->MutableInputDesc("b") != nullptr) {
    bias_dtype = op.GetInputDesc("b").GetDataType();
  } else if (op_desc->MutableInputDesc("init_h") != nullptr) {
    bias_dtype = op.GetInputDesc("init_h").GetDataType();
  }

  TensorDesc y_tensor_desc = op.GetOutputDesc("y");
  TensorDesc y_h_tensor_desc = op.GetOutputDesc("y_h");

  int64_t dim_num = shape_x.GetDimNum();
  if (dim_num != 3) {
    OP_LOGE(op.GetName().c_str(), "The dimension count of x should be 3, please check!");
    OpsOneInputShapeErrReport(op.GetName(), "x", "The dimension count of x should be 3");
    return GRAPH_FAILED;
  }

  int64_t seq_length = shape_x.GetDims().at(0);
  int64_t batch_size = shape_x.GetDims().at(1);
  int64_t num_directions = shape_w.GetDims().at(0);
  int64_t hidden_size = shape_w.GetDims().at(1) / 3;

  vector<int64_t> y_dims = {seq_length, num_directions, batch_size, hidden_size};
  vector<int64_t> y_h_dims = {num_directions, batch_size, hidden_size};

  y_tensor_desc.SetShape(Shape(y_dims));
  y_h_tensor_desc.SetShape(Shape(y_h_dims));

  y_tensor_desc.SetDataType(bias_dtype);
  y_h_tensor_desc.SetDataType(bias_dtype);

  (void) op.UpdateOutputDesc("y", y_tensor_desc);
  (void) op.UpdateOutputDesc("y_h", y_h_tensor_desc);

  w_tensor_desc.SetFormat(FORMAT_HWCN);
  r_tensor_desc.SetFormat(FORMAT_HWCN);
  (void) op.UpdateInputDesc("w", w_tensor_desc);
  (void) op.UpdateInputDesc("r", r_tensor_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CommonGRU, CommonGRUInferShape);
// ----------------EmbeddingBag-------------------

int64_t get_batch_dim(bool include_last_offset, int64_t offset_lens) {
    int64_t output_dim = 0;
    if (include_last_offset) {
        output_dim = offset_lens -1;
    } else {
        output_dim = offset_lens;
    }
    return output_dim;
}

IMPLEMT_COMMON_INFERFUNC(EmbeddingBagInferShape) {
    OP_LOGI(op.GetName().c_str(), " EmbeddingBag inferShape begin!");
    // get weight info
    TensorDesc weight_desc = op.GetInputDesc("weight");
    auto weight_shape = weight_desc.GetShape().GetDims();
    DataType weight_dtype = weight_desc.GetDataType();

    int64_t batch_dim = 0;
    int64_t embedding_dim = weight_shape[1];
    // get indices info
    TensorDesc indices_desc = op.GetInputDesc("indices");
    auto indices_shape = indices_desc.GetShape().GetDims();

    // get offsets info
    TensorDesc offsets_desc = op.GetInputDesc("offsets");
    auto offsets_shape = offsets_desc.GetShape().GetDims();

    // get offset_lens
    int64_t offset_lens = static_cast<int64_t>(offsets_shape[0]);

    bool include_last_offset;
    if (op.GetAttr("include_last_offset", include_last_offset) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), " get attr include_last_offset failed");
        return GRAPH_FAILED;
    }
    // get batch_dim
    int64_t indices_size = static_cast<int64_t>(indices_shape.size());
    if (indices_size == 2) {
        batch_dim = indices_shape[0];
    } else {
        batch_dim = get_batch_dim(include_last_offset, offset_lens);
    }
    
    vector<int64_t> embedding_output_shape;
   // push embedding bag dim to output shape
    embedding_output_shape.push_back(batch_dim);
    embedding_output_shape.push_back(embedding_dim);

    // update output info
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(ge::Shape(embedding_output_shape));
    output_desc.SetDataType(weight_dtype);
    output_desc.SetOriginFormat(ge::FORMAT_ND);
    output_desc.SetFormat(ge::FORMAT_ND);
    (void)op.UpdateOutputDesc("y", output_desc);

    OP_LOGI(op.GetName().c_str(), " EmbeddingBag inferShape end!");
    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(EmbeddingBag, EmbeddingBagVerify) {
    return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(EmbeddingBag, EmbeddingBagInferShape);
VERIFY_FUNC_REG(EmbeddingBag, EmbeddingBagVerify);
// ----------------EmbeddingBag-------------------
}  // namespace ge
