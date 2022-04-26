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
 * \file stateful_random_ops.cpp
 * \brief
 */
#include "inc/stateful_random_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(NonDeterministicInts, NonDeterministicIntsInfer) {
  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get shape_tensor error.");
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get shape error.");
    return GRAPH_FAILED;
  }

  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr dtype error.");
    return GRAPH_FAILED;
  }
  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetDataType(dtype);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(NonDeterministicInts, NonDeterministicIntsInfer);

IMPLEMT_INFERFUNC(RngSkip, RngSkipInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(1), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        1, DebugString(op.GetInputDesc(1).GetShape().GetDims()),
        "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape unused_shape1;
  if (WithRank(op.GetInputDesc(2), 0, unused_shape1, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        2, DebugString(op.GetInputDesc(2).GetShape().GetDims()),
        "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RngSkip, RngSkipInfer);

IMPLEMT_INFERFUNC(StatefulRandomBinomial, StatefulRandomBinomialInfer) {
  Shape unused;
  if (WithRankAtMost(op.GetInputDesc(3), 1, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
      "failed to call WithRankAtMost function, ",
      "input[counts] rank must be at most 1D, but got rank[",
      op.GetInputDesc(3).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(4), 1, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
      "failed to call WithRankAtMost function, ",
      "input[probs] rank must be at most 1D, but got rank[",
      op.GetInputDesc(4).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
      std::string("get const data[shape] failed"));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
     AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call MakeShapeFromShapeTensor function failed to make "
        "shape by input[shape] data"));
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDesc("y");
  DataType output_type;
  if (op.GetAttr("dtype", output_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
      std::string("get attr[dtype] failed"));
  }
  outputDesc.SetDataType(output_type);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulRandomBinomial, StatefulRandomBinomialInfer);

IMPLEMT_INFERFUNC(StatefulStandardNormalV2, StatefulStandardNormalV2Infer) {
  Shape unused;
  std::string error_msg;
  TensorDesc algorithm_desc = op.GetInputDesc(1);
  if (WithRank(algorithm_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString(
        "failed to call WithRank function, the rank of input[algorithm] must be 0,",
        " but get ", algorithm_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("failed to get input[shape] constant data."));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeShapeFromShapeTensor function, ",
        "make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetDataType(DT_FLOAT);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulStandardNormalV2, StatefulStandardNormalV2Infer);

IMPLEMT_INFERFUNC(StatefulTruncatedNormal, StatefulTruncatedNormalInfer) {
  Shape unused;
  std::string error_msg;
  TensorDesc algorithm_desc = op.GetInputDesc(1);
  if (WithRank(algorithm_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString(
        "failed to call WithRank function, the rank of input[algorithm] must be 0,",
        " but get ", algorithm_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("failed to get input[shape] constant data."));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeShapeFromShapeTensor function, ",
        "make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetDataType(DT_FLOAT);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulTruncatedNormal, StatefulTruncatedNormalInfer);

IMPLEMT_INFERFUNC(StatefulUniform, StatefulUniformInfer) {
  Shape unused;
  std::string error_msg;
  TensorDesc algorithm_desc = op.GetInputDesc(1);
  if (WithRank(algorithm_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString(
        "failed to call WithRank function, the rank of input[algorithm] must be 0",
        " but get ", algorithm_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("failed to get input[shape] constant data."));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeShapeFromShapeTensor function, ",
        "make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetDataType(DT_FLOAT);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulUniform, StatefulUniformInfer);

IMPLEMT_INFERFUNC(StatefulUniformFullInt, StatefulUniformFullIntInfer) {
  Shape unused;
  std::string error_msg;
  TensorDesc algorithm_desc = op.GetInputDesc(1);
  if (WithRank(algorithm_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString(
        "failed to call WithRank function, the rank of input[algorithm] must be 0",
        " but get ", algorithm_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("failed to get input[shape] constant data."));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeShapeFromShapeTensor function, ",
        "make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetDataType(DT_UINT64);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulUniformFullInt, StatefulUniformFullIntInfer);

IMPLEMT_INFERFUNC(StatefulUniformInt, StatefulUniformIntInfer) {
  Shape unused;
  std::string error_msg;
  TensorDesc algorithm_desc = op.GetInputDesc(1);
  if (WithRank(algorithm_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString(
        "failed to call WithRank function, the rank of input[algorithm] must be 0",
        " but get ", algorithm_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  TensorDesc minval_desc = op.GetInputDesc(3);
  if (WithRank(minval_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString(
        "failed to call WithRank function, the rank of input[minval] must be 0",
        " but get ", minval_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  TensorDesc maxval_desc = op.GetInputDesc(4);
  if (WithRank(maxval_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString(
        "failed to call WithRank function, the rank of input[maxval] must be 0",
        " but get ", maxval_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("failed to get input[shape] constant data."));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeShapeFromShapeTensor function, ",
        "make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetDataType(DT_INT64);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatefulUniformInt, StatefulUniformIntInfer);

IMPLEMT_INFERFUNC(RngReadAndSkipV2, RngReadAndSkipV2Infer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto value_in_desc = op_desc->MutableInputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> range;
  if (value_in_desc->GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto value_out_desc = op_desc->MutableOutputDesc(0);
  value_out_desc->SetShape(value_in_desc->GetShape());
  value_out_desc->SetShapeRange(range);
  value_out_desc->SetDataType(DT_INT64);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(RngReadAndSkipV2, RngReadAndSkipV2Verify) {
  const auto op_name = TbeGetName(op);
  Shape alg_shape;
  if (WithRank(op.GetInputDesc(1), 0, alg_shape, op_name.c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op_name, err_msg);
    return GRAPH_FAILED;
  }

  Shape delta_shape;
  if (WithRank(op.GetInputDesc(2), 0, delta_shape, op_name.c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op_name, err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(RngReadAndSkipV2, RngReadAndSkipV2Infer);
VERIFY_FUNC_REG(RngReadAndSkipV2, RngReadAndSkipV2Verify);


}  // namespace ge
