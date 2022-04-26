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
 * \file stateless_random_ops.cpp
 * \brief
 */
#include "inc/stateless_random_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(StatelessMultinomial, StatelessMultinomialInfer) {
  auto logitsTensor = op.get_input_desc_logits();
  auto num_samplesTensor = op.get_input_desc_num_samples();
  auto seedTensor = op.get_input_desc_seed();

  Shape seed;
  std::string error_msg;
  if (WithRank(seedTensor, 1, seed, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[seed] must be 1, but get ",
        seedTensor.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (seed.GetDim(0) != 2) {
    error_msg = ConcatString("the value of dim[0] for input[seed] ",
        "must be 2, but the real value is", seed.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape logits_shape;
  Shape unused;
  if (WithRank(logitsTensor, 2, logits_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[logits] must be 2, but get ",
        logitsTensor.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(num_samplesTensor, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[num_samples] must be 0, but get ",
        num_samplesTensor.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  DataType dtype = op.get_input_desc_num_samples().GetDataType();
  if (dtype != DT_INT32) {
    error_msg = ConcatString("the dtype of input[num_samples] must be DT_INT32,",
        " but get ", DTypeStr(dtype), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Tensor numSamplesTensor;
  if (op.GetInputConstData("num_samples", numSamplesTensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get input[num_samples] const data fialed."));
    return GRAPH_FAILED;
  }
  int64_t numSamples;
  if (MakeDimForScalarInput(numSamplesTensor, numSamples, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeDimForScalarInput function ",
        "make dim for input[num_samples] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  (void)Matrix(logits_shape.GetDim(0), numSamples, shape);

  TensorDesc outputDesc = op.GetOutputDesc("y");
  DataType output_type;
  if (op.GetAttr("output_dtype", output_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get attr[output_dtype] fialed."));
  }

  outputDesc.SetDataType(output_type);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatelessMultinomial, StatelessMultinomialInfer);

IMPLEMT_INFERFUNC(StatelessRandomUniformInt, StatelessRandomUniformIntInfer) {
  Shape unused;
  std::string error_msg;
  TensorDesc minval_desc = op.GetInputDesc(2);
  if (WithRank(minval_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[minval] must be 0, but get ",
        minval_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  TensorDesc maxval_desc = op.GetInputDesc(3);
  if (WithRank(maxval_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[maxval] must be 0, but get ",
        maxval_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape seed;
  TensorDesc seed_desc = op.GetInputDesc(1);
  if (WithRank(seed_desc, 1, seed, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[seed] must be 1, but get ",
        seed_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (seed.GetDim(0) != 2) {
    error_msg = ConcatString("the value of dim[0] for input[seed] must be 2, ",
        "but the real value is ", seed.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("failed to get input[shape] const data."));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeShapeFromShapeTensor function,",
        " make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDesc("y");
  DataType output_type = op.GetInputDesc("minval").GetDataType();
  outputDesc.SetDataType(output_type);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatelessRandomUniformInt, StatelessRandomUniformIntInfer);

}  // namespace ge
