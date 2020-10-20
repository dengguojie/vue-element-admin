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
      OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
      OP_LOGE(op.GetName().c_str(),
              "data_format only "
              "support 'NHWC' and 'NCHW'.");
      return GRAPH_FAILED;
    }
  }
  auto x_shape = op.GetInputDesc("x").GetShape().GetDims();
  DataType x_dtype = op.GetInputDesc("x").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(ge::Shape(x_shape));
  y_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", y_desc);

  auto scale_shape = op.GetInputDesc("scale").GetShape().GetDims();
  DataType scale_dtype = op.GetInputDesc("scale").GetDataType();

  TensorDesc batch_mean_desc = op.GetOutputDesc("batch_mean");
  batch_mean_desc.SetShape(ge::Shape(scale_shape));
  batch_mean_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("batch_mean", batch_mean_desc);

  TensorDesc batch_variance_desc = op.GetOutputDesc("batch_variance");
  batch_variance_desc.SetShape(ge::Shape(scale_shape));
  batch_variance_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("batch_variance", batch_variance_desc);

  TensorDesc reserve_space_1_desc = op.GetOutputDesc("reserve_space_1");
  reserve_space_1_desc.SetShape(ge::Shape(scale_shape));
  reserve_space_1_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("reserve_space_1", reserve_space_1_desc);

  TensorDesc reserve_space_2_desc = op.GetOutputDesc("reserve_space_2");
  reserve_space_2_desc.SetShape(ge::Shape(scale_shape));
  reserve_space_2_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("reserve_space_2", reserve_space_2_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BatchNorm, BatchNormInferShape);
VERIFY_FUNC_REG(BatchNorm, BatchNormVerify);
// -----------------------------BatchNorm END----------------------------

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
      OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
      OP_LOGE(op.GetName().c_str(),
              "data_format only "
              "support 'NHWC' and 'NCHW'.");
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
      OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
      OP_LOGE(op.GetName().c_str(),
              "data_format only "
              "support 'NHWC' and 'NCHW'.");
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

  TensorDesc reserve_space_4_desc = op.GetOutputDesc("reserve_space_4");
  reserve_space_4_desc.SetShape(ge::Shape(oShape));
  reserve_space_4_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("reserve_space_4", reserve_space_4_desc);

  TensorDesc reserve_space_5_desc = op.GetOutputDesc("reserve_space_5");
  reserve_space_5_desc.SetShape(ge::Shape(oShape));
  reserve_space_5_desc.SetDataType(scale_dtype);
  (void)op.UpdateOutputDesc("reserve_space_5", reserve_space_5_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BatchNormGrad, BatchNormGradInferShape);
VERIFY_FUNC_REG(BatchNormGrad, BatchNormGradVerify);
// ---------------------------BatchNormGrad END-----------------------------

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
      OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
      OP_LOGE(op.GetName().c_str(),
              "data_format only "
              "support 'NHWC' and 'NCHW'.");
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

IMPLEMT_VERIFIER(BNInferenceD, BNInferenceDVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BNInferenceD, BNInferenceDInferShape) {
  auto x_shape = op.GetInputDesc("x").GetShape().GetDims();
  DataType x_dtype = op.GetInputDesc("x").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(ge::Shape(x_shape));
  y_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BNInferenceD, BNInferenceDInferShape);
VERIFY_FUNC_REG(BNInferenceD, BNInferenceDVerify);

}  // namespace ge
