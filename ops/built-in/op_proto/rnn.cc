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

  std::string direction = "UNIDIRECTIONAL";
  vector<int64_t> outputHDims;
  if (GRAPH_SUCCESS != op.GetAttr("direction", direction)) {
    OP_LOGD(op.GetName().c_str(), "Direction is UNIDIRECTIONAL.");
  }
  if (direction == "BIDIRECTIONAL") {
    OP_LOGD(op.GetName().c_str(), "Direction is BIDIRECTIONAL.");

    outputHDims = {num_step, batchSize, 2 * hiddenSize};
    Shape outputShape(outputHDims);
    outputYTensorDesc.SetShape(outputShape);
    outputHTensorDesc.SetShape(outputShape);
    outputCTensorDesc.SetShape(outputShape);
    outputHDims = {2 * num_step, batchSize, hiddenSize};
    Shape outputShape_(outputHDims);
    outputITensorDesc.SetShape(outputShape_);
    outputJTensorDesc.SetShape(outputShape_);
    outputFTensorDesc.SetShape(outputShape_);
    outputOTensorDesc.SetShape(outputShape_);
    outputTanhcTensorDesc.SetShape(outputShape_);
  } else {
    outputHDims = {num_step, batchSize, hiddenSize};
    Shape outputShape(outputHDims);
    outputYTensorDesc.SetShape(outputShape);
    outputHTensorDesc.SetShape(outputShape);
    outputCTensorDesc.SetShape(outputShape);
    outputITensorDesc.SetShape(outputShape);
    outputJTensorDesc.SetShape(outputShape);
    outputFTensorDesc.SetShape(outputShape);
    outputOTensorDesc.SetShape(outputShape);
    outputTanhcTensorDesc.SetShape(outputShape);
  }

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

  (void)op.UpdateInputDesc("w", inputWTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicRNN, DynamicRNNInferShape);
VERIFY_FUNC_REG(DynamicRNN, DynamicRNNVerify);

IMPLEMT_VERIFIER(DynamicRNNV2, DynamicRNNV2Verify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicRNNV2, DynamicRNNV2InferShape) {
  ge::TensorDesc inputXTensorDesc = op.GetInputDescByName("x");
  ge::TensorDesc inputWiTensorDesc = op.GetInputDescByName("weight_input");
  ge::TensorDesc inputWhTensorDesc = op.GetInputDescByName("weight_hidden");
  ge::Shape shapeX = inputXTensorDesc.GetShape();
  ge::Shape shapeWi = inputWiTensorDesc.GetShape();
  ge::Shape shapeWh = inputWhTensorDesc.GetShape();
  DataType inputXDtype = inputXTensorDesc.GetDataType();
  // bias is optional, default dtype is x's dtype
  DataType inputBDtype = inputXDtype;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc->MutableInputDesc("b") != nullptr) {
    inputBDtype = op.GetInputDescByName("b").GetDataType();
  } else if (op_desc->MutableInputDesc("init_c") != nullptr) {
    inputBDtype = op.GetInputDescByName("init_c").GetDataType();
  }
  TensorDesc outputYTensorDesc = op.GetOutputDescByName("y");
  TensorDesc outputHTensorDesc = op.GetOutputDescByName("output_h");
  TensorDesc outputCTensorDesc = op.GetOutputDescByName("output_c");
  TensorDesc outputITensorDesc = op.GetOutputDescByName("i");
  TensorDesc outputJTensorDesc = op.GetOutputDescByName("j");
  TensorDesc outputFTensorDesc = op.GetOutputDescByName("f");
  TensorDesc outputOTensorDesc = op.GetOutputDescByName("o");
  TensorDesc outputTanhcTensorDesc = op.GetOutputDescByName("tanhc");

  int64_t dim_num = shapeX.GetDimNum();
  int64_t batchSize = 0;
  int64_t hiddenSize = 0;
  int64_t num_step = 0;
  int64_t inputSize = 0;
  if (dim_num == 3) {
    num_step = shapeX.GetDims().at(0);
    batchSize = shapeX.GetDims().at(1);
    inputSize = shapeX.GetDims().at(2);
    hiddenSize = shapeWi.GetDims().at(1) / 4;
  } else {
    AscendString OpName;
    op.GetName(OpName);
    OpsOneInputShapeErrReport(OpName.GetString(), "X Shape Dim", "The input shape of X not equal 3!");
    OP_LOGE(OpName.GetString(), "The input shape of X not equal 3, please check!");
    return GRAPH_FAILED;
  }

  vector<int64_t> outputYDims = {num_step, batchSize, hiddenSize};
  vector<int64_t> outputHCDims = {1, batchSize, hiddenSize};

  outputYTensorDesc.SetShape(ge::Shape(outputYDims));
  outputHTensorDesc.SetShape(ge::Shape(outputHCDims));
  outputCTensorDesc.SetShape(ge::Shape(outputHCDims));
  outputITensorDesc.SetShape(ge::Shape(outputYDims));
  outputJTensorDesc.SetShape(ge::Shape(outputYDims));
  outputFTensorDesc.SetShape(ge::Shape(outputYDims));
  outputOTensorDesc.SetShape(ge::Shape(outputYDims));
  outputTanhcTensorDesc.SetShape(ge::Shape(outputYDims));

  outputYTensorDesc.SetDataType(inputBDtype);
  outputHTensorDesc.SetDataType(inputXDtype);
  outputCTensorDesc.SetDataType(inputBDtype);
  outputITensorDesc.SetDataType(inputBDtype);
  outputJTensorDesc.SetDataType(inputBDtype);
  outputFTensorDesc.SetDataType(inputBDtype);
  outputOTensorDesc.SetDataType(inputBDtype);
  outputTanhcTensorDesc.SetDataType(inputBDtype);

  CHECK(op.UpdateOutputDesc("y", outputYTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("output_h", outputHTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("output_c", outputCTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("i", outputITensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("j", outputJTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("f", outputFTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("o", outputOTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("tanhc", outputTanhcTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);

  ge::AttrUtils::SetInt(op_desc, "input_size", inputSize);
  ge::AttrUtils::SetInt(op_desc, "hidden_size", hiddenSize);
  CHECK(op.UpdateInputDesc("weight_input", inputWiTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  CHECK(op.UpdateInputDesc("weight_hidden", inputWhTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed."), return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicRNNV2, DynamicRNNV2InferShape);
VERIFY_FUNC_REG(DynamicRNNV2, DynamicRNNV2Verify);

IMPLEMT_VERIFIER(DynamicLSTMGradCell, DynamicLSTMGradCellVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicLSTMGradCell, DynamicLSTMGradCellInferShape) {
  ge::TensorDesc inputDYTensorDesc = op.GetInputDesc("dy");
  ge::Shape shapeDY = inputDYTensorDesc.GetShape();
  DataType dtype = inputDYTensorDesc.GetDataType();

  int64_t dim_num = shapeDY.GetDims().size();
  int64_t batch_size = 0;
  int64_t output_dim_size = 0;
  if (dim_num == 3) {
    batch_size = shapeDY.GetDims().at(1);
    output_dim_size = shapeDY.GetDims().at(2);
  } else {
    OpsOneInputShapeErrReport(op.GetName(), "The input shape of dy", "not right");
    OP_LOGE(op.GetName().c_str(), "The input shape of dy is not right, please check!");
    return GRAPH_FAILED;
  }

  int64_t output_nz_dim_size = (output_dim_size + 15) / 16;
  TensorDesc outputDgateTensorDesc = op.GetOutputDesc("dgate");
  TensorDesc outputDct1TensorDesc = op.GetOutputDesc("dct_1");

  vector<int64_t> outputDgateDims = {batch_size, 4 * output_nz_dim_size * 16};
  outputDgateTensorDesc.SetShape(ge::Shape(outputDgateDims));
  outputDgateTensorDesc.SetDataType(dtype);

  vector<int64_t> outputDct1HDims = {batch_size, output_dim_size};
  outputDct1TensorDesc.SetShape(ge::Shape(outputDct1HDims));
  outputDct1TensorDesc.SetDataType(dtype);

  CHECK(op.UpdateOutputDesc("dgate", outputDgateTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc dgate failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("dct_1", outputDct1TensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc dct_1 failed."), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DynamicLSTMGradCell, DynamicLSTMGradCellInferShape);
VERIFY_FUNC_REG(DynamicLSTMGradCell, DynamicLSTMGradCellVerify);

IMPLEMT_VERIFIER(DynamicGRUCellGrad, DynamicGRUCellGradVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicGRUCellGrad, DynamicGRUCellGradInferShape) {
  ge::TensorDesc inputDYTensorDesc = op.GetInputDesc("dy");
  ge::Shape shapeDY = inputDYTensorDesc.GetShape();
  DataType dtype = inputDYTensorDesc.GetDataType();

  int64_t dim_num = shapeDY.GetDims().size();
  int64_t batch_size = 0;
  int64_t output_dim_size = 0;
  if(dim_num != 3) {
    OpsOneInputShapeErrReport(op.GetName(), "The input shape of dy", "not right");
    OP_LOGE(op.GetName().c_str(), "The input shape of dy is not right, please check!");
    return GRAPH_FAILED;
  }
  batch_size = shapeDY.GetDims().at(1);
  output_dim_size = shapeDY.GetDims().at(2);
  
  TensorDesc outputDhPrevTensorDesc = op.GetOutputDesc("dh_prev");
  TensorDesc outputDgateTensorDesc = op.GetOutputDesc("dgate_h");
  TensorDesc outputDnxXTensorDesc = op.GetOutputDesc("dnt_x");

  vector<int64_t> outputDhPrevHDims = {1, batch_size, output_dim_size};
  outputDhPrevTensorDesc.SetShape(ge::Shape(outputDhPrevHDims));
  outputDhPrevTensorDesc.SetDataType(dtype);

  vector<int64_t> outputDgateDims = {1, batch_size, 3 * output_dim_size};
  outputDgateTensorDesc.SetShape(ge::Shape(outputDgateDims));
  outputDgateTensorDesc.SetDataType(dtype);

  outputDnxXTensorDesc.SetShape(ge::Shape(outputDhPrevHDims));
  outputDnxXTensorDesc.SetDataType(dtype);

  CHECK(op.UpdateOutputDesc("dh_prev", outputDhPrevTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc dh_prev failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("dgate_h", outputDgateTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc dgate_h failed."), return GRAPH_FAILED);
  CHECK(op.UpdateOutputDesc("dnt_x", outputDnxXTensorDesc) != GRAPH_SUCCESS,
        OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc dnt_x failed."), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DynamicGRUCellGrad, DynamicGRUCellGradInferShape);
VERIFY_FUNC_REG(DynamicGRUCellGrad, DynamicGRUCellGradVerify);

IMPLEMT_VERIFIER(DynamicRNNV3, DynamicRNNV3Verify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicRNNV3, DynamicRNNV3InferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "GetOpDescFromOperator return nullptr!");
    return GRAPH_FAILED;
  }
  auto inputProjectTensorDesc = op_desc->MutableInputDesc("project");
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
  int64_t inputSize = 0;
  int64_t num_step = 0;
  int64_t stateSize = hiddenSize;
  if (inputProjectTensorDesc != nullptr) {
    stateSize = inputProjectTensorDesc->GetShape().GetDims().at(1);
  }
  if (dim_num == 3) {
    num_step = shapeX.GetDims().at(0);
    batchSize = shapeX.GetDims().at(1);
    hiddenSize = shapeW.GetDims().at(1) / 4;
    inputSize = shapeW.GetDims().at(0) - hiddenSize;
  } else {
    OpsOneInputShapeErrReport(op.GetName(), "X Shape Dim", "The input shape of X not equal 3!");
    OP_LOGE(op.GetName().c_str(), "The input shape of X not equal 3, please check!");
    return GRAPH_FAILED;
  }

  vector<int64_t> outputHDims = {num_step, batchSize, hiddenSize};
  vector<int64_t> outputHStateDims = {num_step, batchSize, stateSize};
  outputYTensorDesc.SetShape(ge::Shape(outputHStateDims));
  outputHTensorDesc.SetShape(ge::Shape(outputHStateDims));
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

  inputWTensorDesc.SetFormat(ge::FORMAT_ND);
  inputBTensorDesc.SetFormat(ge::FORMAT_ND);
  ge::AttrUtils::SetInt(op_desc, "input_size", inputSize);
  ge::AttrUtils::SetInt(op_desc, "hidden_size", hiddenSize);
  (void)op.UpdateInputDesc("w", inputWTensorDesc);
  (void)op.UpdateInputDesc("b", inputBTensorDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicRNNV3, DynamicRNNV3InferShape);
VERIFY_FUNC_REG(DynamicRNNV3, DynamicRNNV3Verify);


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
    std::string err_msg = GetAttrValueErrMsg("dim_num", std::to_string(dim_num), ConcatString("c's dim is 2"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = OtherErrMsg(ConcatString("dim_num_x:",dim_num_x, "&&", "dim_num_h:",dim_num_h, "The input shape of X and H should be 2"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetInputFormatNotSupportErrMsg("outputdw_format", expected_format_list, ConcatString(outputdw_format));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetAttrValueErrMsg("dim_num", std::to_string(dim_num), ConcatString("Dgate's dim is 2"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t dim_num_w = inputWShape.GetDimNum();
  if (dim_num_w == 2) {
    inputSize = inputWShape.GetDims().at(0) - hiddenSize;
  } else {
    std::string err_msg = OtherErrMsg(ConcatString("dim_num_w:",dim_num_w,"The input shape of W should be 2, please check!"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);   
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
    std::string err_msg = GetAttrValueErrMsg("dimNumsX", std::to_string(dimNumsX), ConcatString("x's shape is 3"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    std::string err_msg = GetAttrValueErrMsg("dimNumsX", std::to_string(dimNumsX), ConcatString("x's shape is 3"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  int64_t input_size = shape_x.GetDims().at(2);
  int64_t hidden_size = shape_w_hidden.GetDims().at(0);
  bool flag_transdatarnn = false;
  if ((input_size % 16 != 0) || (hidden_size % 16 != 0)) {
    flag_transdatarnn = true;
  }

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

  if (flag_transdatarnn) {
    w_input_tensor_desc.SetFormat(FORMAT_ND);
    w_hidden_tensor_desc.SetFormat(FORMAT_ND);
    ge::AttrUtils::SetInt(op_desc, "input_size", input_size);
    ge::AttrUtils::SetInt(op_desc, "hidden_size", hidden_size);
  } else {
    w_input_tensor_desc.SetFormat(FORMAT_HWCN);
    w_hidden_tensor_desc.SetFormat(FORMAT_HWCN);
  }
  
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
                                         const int64_t& input_idx_1, 
                                         const int64_t& input_idx_2,
                                         const int64_t& output_idx)
{
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    CHECK(op_desc == nullptr,
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), OtherErrMsg("invalid OpDesc.")),
      return false);
    auto tensordesc_input_1 = op_desc->MutableInputDesc(input_idx_1);
    auto tensordesc_output = op_desc->MutableOutputDesc(output_idx);
    CHECK(tensordesc_output == nullptr || tensordesc_input_1 == nullptr,
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), OtherErrMsg("invalid tensordesc.")),
      return false);

    // set output
    DataType input_dtype = tensordesc_input_1->GetDataType();
    Format input_format = tensordesc_input_1->GetFormat();
    tensordesc_output->SetDataType(input_dtype);
    tensordesc_output->SetFormat(input_format);

    const GeShape &shape_x = tensordesc_input_1->MutableShape();
    GeShape &shape_y = tensordesc_output->MutableShape();
    OP_LOGI(op.GetName().c_str(), "shape_x: %s, shape_y: %s.", to_string(shape_x).c_str(), to_string(shape_y).c_str());

    int64_t num_weights_value = 0;
    op.GetAttr("num_weights", num_weights_value);
    OP_LOGI("EmbeddingDenseGrad", "the attr num_weights is %d", num_weights_value);
    int64_t output_shape_len = 2;
    int64_t input_shape_len = shape_x.GetDimNum();
    shape_y.SetDimNum(output_shape_len);
    shape_y.SetDim(0, num_weights_value);
    if (shape_x.IsUnknownDimNum())
    {
        shape_y.SetDim(1, -1);
    }
    else
    {
        shape_y.SetDim(1, shape_x.GetDim(input_shape_len - 1));
    }

    tensordesc_output->SetShape(shape_y);
    if (shape_x.IsUnknownShape())
    {
        std::vector<std::pair<int64_t, int64_t>> input_range;
        std::vector<std::pair<int64_t, int64_t>> output_shape_range;
        tensordesc_input_1->GetShapeRange(input_range);
        if (input_range.empty())
        {
            output_shape_range = {{num_weights_value, num_weights_value}, {1, -1}};
        }
        else
        {
            output_shape_range = {{num_weights_value, num_weights_value}, input_range.back()};
        }
        tensordesc_output->SetShapeRange(output_shape_range);
    }
    return true;
}

IMPLEMT_VERIFIER(EmbeddingDenseGrad, EmbeddingDenseGradVerify)
{
    return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(g_embeddingDenseGradInferShape)
{   

    if (InferShapeAndTypeEmbeddingDenseGrad(op, 0, 1, 0))
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

// ----------------RnnGenMaskV2 Begin-------------------
bool InferShapeAndTypeRnnGenMaskV2(Operator &op,
                                         const char* seq_length,
                                         const char* x,
                                         const char* seq_mask)
{
  OP_LOGI(op.GetName().c_str(), " RnnGenMaskV2 inferShape begin!");
  TensorDesc tensordesc_input = op.GetInputDescByName(seq_length);
  TensorDesc tensordesc_input_x = op.GetInputDescByName(x);
  TensorDesc tensordesc_output = op.GetOutputDescByName(seq_mask);

  int64_t hidden_size;
  if (op.GetAttr("hidden_size", hidden_size) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "hidden_size");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr hidden_size failed.");
    return GRAPH_FAILED;
  }
  OP_LOGI("RnnGenMaskV2", "the attr hidden_size is %d", hidden_size);
  ge::Shape length_shape = tensordesc_input.GetShape();
  std::vector<int64_t> dim_length = length_shape.GetDims();

  if(dim_length.size() != 1){
    OP_LOGE("RnnGenMaskV2", "Unexcepeted Input Shape.");
    return false;
  }

  ge::Shape length_input_x = tensordesc_input_x.GetShape();
  std::vector<int64_t> dim_input_x = length_input_x.GetDims();

  int64_t num_step = dim_input_x[0];
  int64_t batch_size = dim_input_x[1];

  std::vector<int64_t> dim_mask = {num_step, batch_size, hidden_size};

  tensordesc_output.SetShape(Shape(dim_mask));
  tensordesc_output.SetDataType(DT_FLOAT16);
  tensordesc_output.SetFormat(tensordesc_input.GetFormat());

  if (IsUnknown(dim_length) || IsUnknown(dim_input_x))
  {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    op.GetInputDescByName(seq_length).GetShapeRange(input_range);

    std::vector<std::pair<int64_t, int64_t>> input_x_range;
    op.GetInputDescByName(x).GetShapeRange(input_x_range);

    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    if (input_range.empty()){
      output_shape_range = {{1, -1},{1, -1},{1, -1}};
    }
    else{
      output_shape_range = {input_x_range[0], input_x_range[1],
                            {hidden_size, hidden_size}};
    }
    tensordesc_output.SetShapeRange(output_shape_range);
  }

  (void)op.UpdateOutputDesc("seq_mask", tensordesc_output);
  OP_LOGI(op.GetName().c_str(), " RnnGenMaskV2 inferShape end!");
  return true;
}

IMPLEMT_VERIFIER(RnnGenMaskV2, RnnGenMaskV2Verify)
{
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(g_rnnGenMaskV2InferShape)
{
  if (InferShapeAndTypeRnnGenMaskV2(op, "seq_length", "x","seq_mask")){
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(RnnGenMaskV2, g_rnnGenMaskV2InferShape);
// Registered verify function
VERIFY_FUNC_REG(RnnGenMaskV2, RnnGenMaskV2Verify);
// ----------------RnnGenMaskV2 END---------------------

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

// ----------------LSTMP-------------------
IMPLEMT_VERIFIER(LSTMP, LSTMPVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(LSTMP, LSTMPInferShape) {
  auto inputx = op.GetInputDesc("x");
  auto inputwx = op.GetInputDesc("wx");
  auto inputwr = op.GetInputDesc("wr");
  auto shapex = inputx.GetShape();
  auto shapewx = inputwx.GetShape();
  auto shapewr = inputwr.GetShape();
  if (shapewx.GetDimNum() != 2 && shapewr.GetDimNum() != 2) {
    OpsOneInputShapeErrReport(op.GetName(), "wx/wr Shape Dim", "The input shape  not equal 2!");
    OP_LOGE(op.GetName().c_str(), "The input shape of wx/wr not equal 2, please check!");
    return GRAPH_FAILED;
  }

  int64_t dim_num = shapex.GetDimNum();
  int64_t batch_size = 0;
  int64_t hidden_size = 0;
  int64_t num_step = 0;
  int64_t state = 0;
  bool time_major = false;
  op.GetAttr("time_major", time_major); 
  if (dim_num == 3) {
    batch_size = shapex.GetDims().at(0);
    num_step = shapex.GetDims().at(1);
    if (time_major == true) {
      num_step = shapex.GetDims().at(0);
      batch_size = shapex.GetDims().at(1);
    }
    hidden_size = shapewx.GetDims().at(0) / 4;
    state = shapewr.GetDims().at(1);
  } else {
    OpsOneInputShapeErrReport(op.GetName(), "X Shape Dim", "The input shape of X not equal 3!");
    OP_LOGE(op.GetName().c_str(), "The input shape of X not equal 3, please check!");
    return GRAPH_FAILED;
  }
  
  auto x_dtype = inputx.GetDataType();
  auto output_y = op.GetOutputDesc("y");
  std::vector<int64_t> dims_y = {batch_size, num_step, state};
  if (time_major == true) {
    dims_y = {num_step, batch_size, state};
  }
  output_y.SetShape(ge::Shape(dims_y));
  output_y.SetDataType(x_dtype);

  auto output_rt = op.GetOutputDesc("output_h");
  std::vector<int64_t> dims_rt = {1, batch_size, state};
  output_rt.SetShape(ge::Shape(dims_rt));
  output_rt.SetDataType(x_dtype);

  auto output_ct = op.GetOutputDesc("output_c");
  std::vector<int64_t> dims_ct = {1, batch_size, hidden_size};
  output_ct.SetShape(ge::Shape(dims_ct));
  output_ct.SetDataType(x_dtype);

  op.UpdateOutputDesc("y", output_y);
  op.UpdateOutputDesc("output_h", output_rt);
  op.UpdateOutputDesc("output_c", output_ct);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(LSTMP, LSTMPInferShape);
VERIFY_FUNC_REG(LSTMP, LSTMPVerify);
// ----------------LSTMP-------------------

}  // namespace ge

