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
 * \file ctc_ops.cpp
 * \brief
 */
#include "inc/ctc_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(CTCLoss, CTCLossInfer) {
  Shape inputs;
  Shape labels_indices;
  Shape labels_values;
  Shape sequence_length;
  if (WithRank(op.GetInputDesc(0), 3, inputs, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "3D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 2, labels_indices, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "2D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 1, labels_values, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), "1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 1, sequence_length, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3, DebugString(op.GetInputDesc(3).GetShape().GetDims()), "1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t dim1 = labels_indices.GetDim(0);
  int64_t dim2 = labels_values.GetDim(0);
  int64_t unused = 0;
  if (Merge(dim1, dim2, unused) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Merge function, 0th dim[",
                                       dim1, "] of input[labels_indices] not equal 0th dim[",
                                       dim2, "] of input[labels_values]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  int64_t dim3 = inputs.GetDim(1);
  int64_t dim4 = sequence_length.GetDim(0);
  int64_t batch_size = 0;
  if (Merge(dim3, dim4, batch_size) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Merge function, 1th dim[",
                                       dim3, "] of input[inputs] not equal 0th dim[",
                                       dim4, "] of input[sequence_length]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  inputs.SetDim(1, batch_size);

  DataType type = op.GetInputDesc("inputs").GetDataType();
  TensorDesc loss_desc = op.GetOutputDesc("loss");
  loss_desc.SetShape(Shape({batch_size}));
  loss_desc.SetDataType(type);
  if (op.UpdateOutputDesc("loss", loss_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[loss] desc failed."));
    return GRAPH_FAILED;
  }
  TensorDesc gradient_desc = op.GetOutputDesc("gradient");
  gradient_desc.SetShape(Shape(inputs));
  gradient_desc.SetDataType(type);
  if (op.UpdateOutputDesc("gradient", gradient_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[gradient] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CTCLoss, CTCLossInfer);

IMPLEMT_INFERFUNC(CTCGreedyDecoder, CTCGreedyDecoderInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  auto inputs_desc = op_desc->MutableInputDesc(0);
  GeShape inputs_shape;
  if (WithRank(inputs_desc, 3, inputs_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "3D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto sequence_length_desc = op_desc->MutableInputDesc(1);
  GeShape sequence_length_shape;
  if (WithRank(sequence_length_desc, 1, sequence_length_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t batch_size = UNKNOWN_DIM;
  if (Merge(inputs_shape.GetDim(1), sequence_length_shape.GetDim(0), batch_size) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Merge function, 1th dim[",
                                       inputs_shape.GetDim(1), "] of input[inputs] not equal 0th dim[",
                                       sequence_length_shape.GetDim(0), "] of input[sequence_length]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t total_decoded_outputs = UNKNOWN_DIM;

  auto decoded_indices_desc = op_desc->MutableOutputDesc("decoded_indices");
  (void)FillOpDesc(decoded_indices_desc, GeShape({total_decoded_outputs, 2}), DT_INT64);

  auto decoded_values_desc = op_desc->MutableOutputDesc("decoded_values");
  (void)FillOpDesc(decoded_values_desc, GeShape({total_decoded_outputs}), DT_INT64);

  auto decoded_shape_desc = op_desc->MutableOutputDesc("decoded_shape");
  (void)FillOpDesc(decoded_shape_desc, GeShape({2}), DT_INT64);

  auto log_probability_desc = op_desc->MutableOutputDesc("log_probability");
  (void)FillOpDesc(log_probability_desc, GeShape({batch_size, 1}), inputs_desc->GetDataType());

  return GRAPH_SUCCESS;
}


INFER_FUNC_REG(CTCGreedyDecoder, CTCGreedyDecoderInfer);

IMPLEMT_INFERFUNC(CTCBeamSearchDecoder, CTCBeamSearchDecoderInfer) {
  Shape inputs_shape;
  auto inputs_desc = op.GetInputDesc(0);
  if (WithRank(inputs_desc, 3, inputs_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(inputs_desc.GetShape().GetDims()), "3D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape sequence_length_shape;
  auto sequence_length_desc = op.GetInputDesc(1);
  if (WithRank(sequence_length_desc, 1, sequence_length_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(sequence_length_desc.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t batch_size;
  if (Merge(inputs_shape.GetDim(1), sequence_length_shape.GetDim(0), batch_size) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Merge function, 1th dim[", inputs_shape.GetDim(1),
                                       "] of input[inputs] not equal 0th dim[", sequence_length_shape.GetDim(0),
                                       "] of input[sequence_length]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int32_t top_paths;
  if (op.GetAttr("top_paths", top_paths) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("get attr[top_paths] failed"));
    return GRAPH_FAILED;
  }

  for (int i = 0; i < top_paths; ++i) {
    auto temp_desc = op.GetDynamicOutputDesc("decoded_indices", i);
    temp_desc.SetShape(Shape({UNKNOWN_DIM, 2}));
    temp_desc.SetDataType(DT_INT64);
    if (op.UpdateDynamicOutputDesc("decoded_indices", i, temp_desc) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("update description for output decoded_indices[", i,"] failed");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  for (int i = 0; i < top_paths; ++i) {
    auto temp_desc = op.GetDynamicOutputDesc("decoded_values", i);
    temp_desc.SetShape(Shape({UNKNOWN_DIM}));
    temp_desc.SetDataType(DT_INT64);
    if (op.UpdateDynamicOutputDesc("decoded_values", i, temp_desc) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("update description for dynimic output decoded_values[", i,"] failed");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  for (int i = 0; i < top_paths; ++i) {
    auto temp_desc = op.GetDynamicOutputDesc("decoded_shape", i);
    temp_desc.SetShape(Shape({2}));
    temp_desc.SetDataType(DT_INT64);
    if (op.UpdateDynamicOutputDesc("decoded_shape", i, temp_desc) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("update description for dynimic output decoded_shape[", i,"] failed");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  auto log_probability_desc = op.GetOutputDesc("log_probability");
  log_probability_desc.SetShape(Shape({batch_size, top_paths}));
  log_probability_desc.SetDataType(inputs_desc.GetDataType());
  if (op.UpdateOutputDesc("log_probability", log_probability_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), string("update description for output[log_probability] failed"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CTCBeamSearchDecoder, CTCBeamSearchDecoderInfer);

IMPLEMT_INFERFUNC(CTCLossV2, CTCLossV2Infer) {
  const char *op_name = "CTCLossV2";
  OP_LOGD(op_name, "CTCLossV2Infer begin.");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto tensordesc_log_probs = op_desc->GetInputDescPtr(0);
  auto dtype = tensordesc_log_probs->GetDataType();
  auto log_probs_shape = tensordesc_log_probs->GetShape();
  int64_t T = log_probs_shape.GetDim(0);
  int64_t N = log_probs_shape.GetDim(1);
  int64_t C = log_probs_shape.GetDim(2);
  int64_t S = 1;

  auto tensordesc_targets = op_desc->GetInputDescPtr(1);
  auto targets_shape = tensordesc_targets->GetShape();
  int64_t targets_dims_len = targets_shape.GetDimNum();
  auto tensordesc_target_lengths = op_desc->GetInputDescPtr(3);
  auto target_lengths_shape = tensordesc_target_lengths->GetShape();

  if (targets_dims_len == 2) {
    S = targets_shape.GetDim(1);
  } else {
    int64_t blank_new;
    op.GetAttr("blank", blank_new);
    S = blank_new / C;
    int64_t blank = blank_new % C;
    op.SetAttr("blank", blank);
  }
  auto tensordesc_neg_log_likelihood = op_desc->MutableOutputDesc(0);
  
  tensordesc_neg_log_likelihood->SetShape(target_lengths_shape);
  tensordesc_neg_log_likelihood->SetDataType(dtype);

  auto tensordesc_log_alpha = op_desc->MutableOutputDesc(1);
  tensordesc_log_alpha->SetDataType(dtype);
  GeShape &output_shape = tensordesc_log_alpha->MutableShape();
  int64_t log_alpha_dim_len = 3;
  std::vector<int64_t> log_alpha_dims = {N, T, 2 * S + 1};
  output_shape.SetDimNum(log_alpha_dim_len);
  for (int64_t i = 0; i < log_alpha_dim_len; i++) {
    output_shape.SetDim(i, log_alpha_dims[i]);
  }

  OP_LOGD(op_name, "CTCLossV2Infer end.");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CTCLossV2, CTCLossV2Infer);

IMPLEMT_INFERFUNC(CTCLossV2Grad, CTCLossV2GradInfer) {
  const char *op_name = "CTCLossV2Grad";
  OP_LOGD(op_name, "CTCLossV2GradInfer begin.");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto tensordesc_log_probs = op_desc->GetInputDescPtr(1);
  auto dtype = tensordesc_log_probs->GetDataType();
  auto log_probs_shape = tensordesc_log_probs->GetShape();

  auto tensordesc_grad = op_desc->MutableOutputDesc(0);
 
  tensordesc_grad->SetShape(log_probs_shape);
  tensordesc_grad->SetDataType(dtype);
  OP_LOGD(op_name, "CTCLossV2GradInfer begin.");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CTCLossV2Grad, CTCLossV2GradInfer);
}  // namespace ge
