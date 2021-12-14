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
 * \file nn_batch_norm_ops.cpp
 * \brief
 */
#include "inc/nn_batch_norm_ops.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"
#include "graph/utils/node_utils.h"

namespace ge {
// -----------------------------BatchNorm------------------------------
IMPLEMT_VERIFIER(BatchNorm, BatchNormVerify) {
  if (!CheckTwoInputDtypeSame(op, "scale", "offset")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BatchNorm, BatchNormInferShape) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NHWC" && data_format != "NCHW") {
      string expected_format_list = ConcatString("NHWC, NCHW");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (!OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "scale", {"batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"})) {
    return GRAPH_FAILED;
  }
  std::vector<int64_t> oShapeVector;
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto output_desc = op_info->MutableOutputDesc("reserve_space_3");
  if (output_desc != nullptr) {
    output_desc->SetShape(GeShape(oShapeVector));
    output_desc->SetDataType(DT_FLOAT);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BatchNorm, BatchNormInferShape);
VERIFY_FUNC_REG(BatchNorm, BatchNormVerify);
// -----------------------------BatchNorm END----------------------------

// -----------------------------BatchNorm3D------------------------------
IMPLEMT_VERIFIER(BatchNorm3D, BatchNorm3DVerify) {
  if (!CheckTwoInputDtypeSame(op, "scale", "offset")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BatchNorm3D, BatchNorm3DInferShape) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NDHWC" && data_format != "NCDHW") {
      string expected_format_list = ConcatString("NDHWC, NCDHW");
      OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
      OP_LOGE(op.GetName().c_str(),
              "data_format only "
              "support 'NDHWC' and 'NCDHW'.");
      return GRAPH_FAILED;
    }
  }
  if (!OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "scale", {"batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BatchNorm3D, BatchNorm3DInferShape);
VERIFY_FUNC_REG(BatchNorm3D, BatchNorm3DVerify);
// -----------------------------BatchNorm3D END----------------------------

// ----------------SyncBatchNormGatherStatsWithCounts Begin-------------------
IMPLEMT_INFERFUNC(SyncBatchNormGatherStatsWithCounts,
                  SyncBatchNormGatherStatsWithCountsInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter SyncBatchNormGatherStatsWithCounts proto inferfunction!");
  TensorDesc tensordesc_input = op.GetInputDesc("running_var");
  auto input_shape = tensordesc_input.GetShape().GetDims();
  DataType input_dtype = tensordesc_input.GetDataType();

  TensorDesc tensordesc_output_1 = op.GetOutputDesc("invert_std");
  tensordesc_output_1.SetDataType(input_dtype);

  TensorDesc tensordesc_output_2 = op.GetOutputDesc("running_var_update");
  tensordesc_output_2.SetDataType(input_dtype);

  AscendString op_name;
  if (GRAPH_SUCCESS != op.GetName(op_name)) {
    OP_LOGE("SyncBatchNormGatherStatsWithCounts", "op_name get failed.");
    return GRAPH_FAILED;
  }
  const char* op_name_c = op_name.GetString();
  OP_LOGI(op.GetName().c_str(), "SyncBatchNormGatherStatsWithCounts op_name get successed.");

  tensordesc_output_1.SetShape(ge::Shape(input_shape));
  tensordesc_output_2.SetShape(ge::Shape(input_shape));
  (void)op.UpdateOutputDesc("invert_std", tensordesc_output_1);
  (void)op.UpdateOutputDesc("running_var_update", tensordesc_output_2);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SyncBatchNormGatherStatsWithCounts,
               SyncBatchNormGatherStatsWithCountsInferShape);
// ----------------SyncBatchNormGatherStatsWithCounts END---------------------

// ----------------SyncBNTrainingUpdate Begin-------------------
IMPLEMT_INFERFUNC(SyncBNTrainingUpdate,
                  SyncBNTrainingUpdateInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter SyncBNTrainingUpdate proto inferfunction!");
  TensorDesc tensordesc_input = op.GetInputDesc("running_mean");
  auto input_shape = tensordesc_input.GetShape().GetDims();
  DataType input_dtype = tensordesc_input.GetDataType();

  TensorDesc tensordesc_output = op.GetOutputDesc("running_mean_update");
  tensordesc_output.SetDataType(input_dtype);

  AscendString op_name;
  if (GRAPH_SUCCESS != op.GetName(op_name)) {
    OP_LOGE("SyncBNTrainingUpdate", "op_name get failed.");
    return GRAPH_FAILED;
  }
  const char* op_name_c = op_name.GetString();
  OP_LOGI(op.GetName().c_str(), "SyncBNTrainingUpdate op_name get successed.");

  tensordesc_output.SetShape(ge::Shape(input_shape));
  (void)op.UpdateOutputDesc("running_mean_update", tensordesc_output);

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(SyncBNTrainingUpdate, SyncBNTrainingUpdateInferShape);
// ----------------SyncBNTrainingUpdate End-------------------

// ----------------SyncBatchNormBackwardReduce Op-------------------
IMPLEMT_VERIFIER(SyncBatchNormBackwardReduce, SyncBatchNormBackwardReduceVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(SyncBatchNormBackwardReduce, SyncBatchNormBackwardReduceInferShape) {
  auto x_shape = op.GetInputDesc("sum_dy").GetShape().GetDims();
  DataType x_dtype = op.GetInputDesc("sum_dy").GetDataType();

  TensorDesc sum_dy_xmu_desc = op.GetOutputDesc("sum_dy_xmu");
  sum_dy_xmu_desc.SetShape(ge::Shape(x_shape));
  sum_dy_xmu_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("sum_dy_xmu", sum_dy_xmu_desc);
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(ge::Shape(x_shape));
  y_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", y_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SyncBatchNormBackwardReduce, SyncBatchNormBackwardReduceInferShape);
VERIFY_FUNC_REG(SyncBatchNormBackwardReduce, SyncBatchNormBackwardReduceVerify);
// ----------------SyncBatchNormBackwardReduce END-------------------

// ----------------SyncBatchNormBackwardElemt Op-------------------
IMPLEMT_VERIFIER(SyncBatchNormBackwardElemt, SyncBatchNormBackwardElemtVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(SyncBatchNormBackwardElemt, SyncBatchNormBackwardElemtInferShape) {
  auto x_shape = op.GetInputDesc("grad_output").GetShape().GetDims();
  DataType x_dtype = op.GetInputDesc("grad_output").GetDataType();
  TensorDesc grad_input_desc = op.GetOutputDesc("grad_input");
  grad_input_desc.SetShape(ge::Shape(x_shape));
  grad_input_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("grad_input", grad_input_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SyncBatchNormBackwardElemt, SyncBatchNormBackwardElemtInferShape);
VERIFY_FUNC_REG(SyncBatchNormBackwardElemt, SyncBatchNormBackwardElemtVerify);
// ----------------SyncBatchNormBackwardElemt END-------------------

// -----------------------------BatchNormExt2------------------------------
IMPLEMT_VERIFIER(BatchNormExt2, BatchNormExt2Verify) {
  if (!CheckTwoInputDtypeSame(op, "input_scale", "input_offset")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BatchNormExt2, BatchNormExt2InferShape) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NHWC" && data_format != "NCHW") {
      string expected_format_list = ConcatString("NHWC, NCHW");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  auto x_shape = op.GetInputDesc("input_x").GetShape().GetDims();
  DataType x_dtype = op.GetInputDesc("input_x").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("output_y");
  y_desc.SetShape(ge::Shape(x_shape));
  y_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("output_y", y_desc);

  auto scale_shape = op.GetInputDesc("input_scale").GetShape().GetDims();
  DataType scale_dtype = op.GetInputDesc("input_scale").GetDataType();

  TensorDesc batch_mean_desc = op.GetOutputDesc("output_mean");
  batch_mean_desc.SetShape(ge::Shape(scale_shape));
  batch_mean_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("output_mean", batch_mean_desc);

  TensorDesc batch_variance_desc = op.GetOutputDesc("output_variance");
  batch_variance_desc.SetShape(ge::Shape(scale_shape));
  batch_variance_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("output_variance", batch_variance_desc);

  TensorDesc reserve_space_1_desc = op.GetOutputDesc("output_reserve_space_1");
  reserve_space_1_desc.SetShape(ge::Shape(scale_shape));
  reserve_space_1_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("output_reserve_space_1", reserve_space_1_desc);

  TensorDesc reserve_space_2_desc = op.GetOutputDesc("output_reserve_space_2");
  reserve_space_2_desc.SetShape(ge::Shape(scale_shape));
  reserve_space_2_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("output_reserve_space_2", reserve_space_2_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BatchNormExt2, BatchNormExt2InferShape);
VERIFY_FUNC_REG(BatchNormExt2, BatchNormExt2Verify);
// -----------------------------BatchNormExt2 END----------------------------

// ---------------------------BatchNormGrad------------------------------
IMPLEMT_VERIFIER(BatchNormGrad, BatchNormGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y_backprop", "x")) {
    return GRAPH_FAILED;
  }
  if ((!CheckTwoInputDtypeSame(op, "scale", "reserve_space_1")) ||
      (!CheckTwoInputDtypeSame(op, "scale", "reserve_space_2"))) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BatchNormGrad, BatchNormGradInferShape) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NHWC" && data_format != "NCHW") {
      string expected_format_list = ConcatString("NHWC, NCHW");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  if (!TwoInOneOutDynamicInferNoBroadcast(op, "x", "y_backprop", {"x_backprop"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "scale", {"scale_backprop", "offset_backprop"})) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> oShapeVector;
  // update reserve_space_4
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto output_desc = op_info->MutableOutputDesc("reserve_space_4");
  output_desc->SetShape(GeShape(oShapeVector));
  output_desc->SetDataType(DT_FLOAT);

  // update reserve_space_5
  output_desc = op_info->MutableOutputDesc("reserve_space_5");
  output_desc->SetShape(GeShape(oShapeVector));
  output_desc->SetDataType(DT_FLOAT);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BatchNormGrad, BatchNormGradInferShape);
VERIFY_FUNC_REG(BatchNormGrad, BatchNormGradVerify);
// ---------------------------BatchNormGrad END-----------------------------

// ---------------------------BatchNorm3DGrad------------------------------
IMPLEMT_VERIFIER(BatchNorm3DGrad, BatchNorm3DGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y_backprop", "x")) {
    return GRAPH_FAILED;
  }
  if ((!CheckTwoInputDtypeSame(op, "scale", "reserve_space_1")) ||
      (!CheckTwoInputDtypeSame(op, "scale", "reserve_space_2"))) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BatchNorm3DGrad, BatchNorm3DGradInferShape) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NDHWC" && data_format != "NCDHW") {
      string expected_format_list = ConcatString("NDHWC, NCDHW");
      OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
      OP_LOGE(op.GetName().c_str(),
              "data_format only "
              "support 'NDHWC' and 'NCDHW'.");
      return GRAPH_FAILED;
    }
  }

  if (!TwoInOneOutDynamicInferNoBroadcast(op, "x", "y_backprop", {"x_backprop"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "scale", {"scale_backprop", "offset_backprop"})) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> oShapeVector;
  // update reserve_space_4
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto output_desc = op_info->MutableOutputDesc("reserve_space_4");
  output_desc->SetShape(GeShape(oShapeVector));
  output_desc->SetDataType(DT_FLOAT);

  // update reserve_space_5
  output_desc = op_info->MutableOutputDesc("reserve_space_5");
  output_desc->SetShape(GeShape(oShapeVector));
  output_desc->SetDataType(DT_FLOAT);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BatchNorm3DGrad, BatchNorm3DGradInferShape);
VERIFY_FUNC_REG(BatchNorm3DGrad, BatchNorm3DGradVerify);
// ---------------------------BatchNorm3DGrad END-----------------------------

// ---------------------------BatchNormGradExt2------------------------------
IMPLEMT_VERIFIER(BatchNormGradExt2, BatchNormGradExt2Verify) {
  if (!CheckTwoInputDtypeSame(op, "y_backprop", "x")) {
    return GRAPH_FAILED;
  }
  if ((!CheckTwoInputDtypeSame(op, "scale", "reserve_space_1")) ||
      (!CheckTwoInputDtypeSame(op, "scale", "reserve_space_2"))) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BatchNormGradExt2, BatchNormGradExt2InferShape) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NHWC" && data_format != "NCHW") {
      string expected_format_list = ConcatString("NHWC, NCHW");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  auto x_shape = op.GetInputDesc("x").GetShape().GetDims();
  DataType x_dtype = op.GetInputDesc("x").GetDataType();

  TensorDesc x_backprop_desc = op.GetOutputDesc("x_backprop");
  x_backprop_desc.SetShape(ge::Shape(x_shape));
  x_backprop_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("x_backprop", x_backprop_desc);

  auto scale_shape = op.GetInputDesc("scale").GetShape().GetDims();
  DataType scale_dtype = op.GetInputDesc("scale").GetDataType();

  TensorDesc scale_backprop_desc = op.GetOutputDesc("scale_backprop");
  scale_backprop_desc.SetShape(ge::Shape(scale_shape));
  scale_backprop_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("scale_backprop", scale_backprop_desc);

  TensorDesc offset_backprop_desc = op.GetOutputDesc("offset_backprop");
  offset_backprop_desc.SetShape(ge::Shape(scale_shape));
  offset_backprop_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("offset_backprop", offset_backprop_desc);

  std::vector<int64_t> oShapeVector;
  Shape oShape(oShapeVector);

  TensorDesc reserve_space_3_desc = op.GetOutputDesc("reserve_space_3");
  reserve_space_3_desc.SetShape(ge::Shape(oShape));
  reserve_space_3_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("reserve_space_3", reserve_space_3_desc);

  TensorDesc reserve_space_4_desc = op.GetOutputDesc("reserve_space_4");
  reserve_space_4_desc.SetShape(ge::Shape(oShape));
  reserve_space_4_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("reserve_space_4", reserve_space_4_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BatchNormGradExt2, BatchNormGradExt2InferShape);
VERIFY_FUNC_REG(BatchNormGradExt2, BatchNormGradExt2Verify);
// ---------------------------BatchNormGradExt2 END-----------------------------

// -------------------------L2Normalize-----------------------------
IMPLEMT_INFERFUNC(L2Normalize, L2NormalizeInferShape) {
  auto outShape = op.GetInputDesc("x").GetShape();
  auto outDtype = op.GetInputDesc("x").GetDataType();

  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(outShape);
  td.SetDataType(outDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(L2Normalize, L2NormalizeInferShape);
// -------------------------L2Normalize END-----------------------------

// -------------------------L2NormalizeGrad-----------------------------
IMPLEMT_INFERFUNC(L2NormalizeGrad, L2NormalizeGradInferShape) {
  auto outShape = op.GetInputDesc("x").GetShape();
  auto outDtype = op.GetInputDesc("x").GetDataType();

  TensorDesc td = op.GetOutputDesc("dx");
  td.SetShape(outShape);
  td.SetDataType(outDtype);
  (void)op.UpdateOutputDesc("dx", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(L2NormalizeGrad, L2NormalizeGradInferShape);
// -------------------------L2NormalizeGrad END-----------------------------
// namespace domi

IMPLEMT_VERIFIER(BNInference, BNInferenceVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BNInference, BNInferenceInferShape) {
  auto x_shape = op.GetInputDesc("x").GetShape().GetDims();
  DataType x_dtype = op.GetInputDesc("x").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(ge::Shape(x_shape));
  y_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", y_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BNInference, BNInferenceInferShape);
VERIFY_FUNC_REG(BNInference, BNInferenceVerify);

// ----------------------BNInferenceD
IMPLEMT_VERIFIER(BNInferenceD, BNInferenceDVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BNInferenceD, BNInferenceDInferShape) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NHWC" && data_format != "NCHW") {
      string expected_format_list = ConcatString("NHWC, NCHW");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (!OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BNInferenceD, BNInferenceDInferShape);
VERIFY_FUNC_REG(BNInferenceD, BNInferenceDVerify);
// ----------------------BNInferenceD END

}  // namespace ge
