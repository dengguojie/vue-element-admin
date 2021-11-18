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
 * \file nonlinear_fuc_ops.cpp
 * \brief
 */
#include "inc/nonlinear_fuc_ops.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"

namespace ge {

IMPLEMT_COMMON_INFERFUNC(OneInOneOutCommonInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
// ----------------------GeluGrad----------------------
IMPLEMT_VERIFIER(GeluGrad, GeluGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "dy") || !CheckTwoInputDtypeSame(op, "x", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(GeluGradInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"z"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(GeluGrad, GeluGradInferShape);
VERIFY_FUNC_REG(GeluGrad, GeluGradVerify);
// ----------------------GeluGrad END----------------------
//-----------------------ELUGRADV2-------------------------
IMPLEMT_COMMON_INFERFUNC(EluGradV2InferShape){
    TensorDesc output_desc = op.GetOutputDesc("y");
    DataType input_dtype = op.GetInputDesc("activations").GetDataType();
    Format input_format = op.GetInputDesc("activations").GetFormat();
    ge::Shape input_shape = op.GetInputDesc("activations").GetShape();
    output_desc.SetDataType(input_dtype);
    output_desc.SetFormat(input_format);
    output_desc.SetShape(input_shape);
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(EluGradV2, EluGradV2Verify){
    // Check Whether the two input tensor data types are consistent
    DataType input_type_grads = op.GetInputDesc("grads").GetDataType();
    DataType input_type_activations = op.GetInputDesc("activations").GetDataType();
    if(input_type_activations != input_type_grads) {
        OP_LOGE(op.GetName().c_str(), "Input dtypes are not the same.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(EluGradV2, EluGradV2InferShape);
VERIFY_FUNC_REG(EluGradV2, EluGradV2Verify);
//-----------------------ELUGRADV2 END---------------------
// ----------------------Gelu----------------------
IMPLEMT_COMMON_INFERFUNC(GeluInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Gelu, GeluInferShape);
// ----------------------Gelu END----------------------

// ----------------HardSwish-------------------
COMMON_INFER_FUNC_REG(HardSwish, OneInOneOutCommonInferShape);
// --------------HardSwish END-----------------

// ----------------Swish-------------------
COMMON_INFER_FUNC_REG(Swish, OneInOneOutCommonInferShape);
// --------------Swish END-----------------

// ----------------------SwishGrad----------------------
IMPLEMT_COMMON_INFERFUNC(SwishGradInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"grad_x"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(SwishGrad, SwishGradInferShape);
// ----------------------SwishGrad END----------------------

// ----------------HardSwishGrad------------------
IMPLEMT_COMMON_INFERFUNC(HardSwishGradInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(HardSwishGrad, HardSwishGradInferShape);
// -------------HardSwishGrad END-------------------

// ----------------------FastGeluGrad----------------------
IMPLEMT_VERIFIER(FastGeluGrad, FastGeluGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FastGeluGradInferShape) {
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "dy", "x", {"z"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FastGeluGrad, FastGeluGradInferShape);
VERIFY_FUNC_REG(FastGeluGrad, FastGeluGradVerify);
// ----------------------FastGeluGrad END----------------------

// ----------------------FastGelu----------------------
IMPLEMT_COMMON_INFERFUNC(FastGeluInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(FastGelu, FastGeluInferShape);
// ----------------------FastGelu END------------------

// ----------------TanhGrad Op Begin----------------
IMPLEMT_COMMON_INFERFUNC(TanhGradInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("z");
  tensordesc_output.SetShape(op.GetInputDesc("y").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("y").GetDataType());
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op.GetInputDesc("y").GetShapeRange(shape_range_x);
  tensordesc_output.SetShapeRange(shape_range_x);
  (void)op.UpdateOutputDesc("z", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TanhGrad, TanhGradInferShape);
// ----------------TanhGrad Op End-------------------

// ----------------PRelu-------------------
IMPLEMT_COMMON_INFERFUNC(PReluInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(PRelu, PReluInferShape);
// ----------------PRelu End---------------

// ----------------PReluGrad---------------
IMPLEMT_VERIFIER(PReluGrad, PReluGradVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(PReluGrad, PReluGradInferShape) {
  auto outShape = op.GetInputDesc("grads").GetShape();
  auto outDtype = op.GetInputDesc("grads").GetDataType();
  TensorDesc td = op.GetOutputDesc("dx");
  td.SetShape(outShape);
  td.SetDataType(outDtype);
  auto outShapeOne = op.GetInputDesc("weights").GetShape();
  auto outDtypeOne = op.GetInputDesc("weights").GetDataType();
  TensorDesc tdOne = op.GetOutputDesc("da");
  tdOne.SetShape(outShapeOne);
  tdOne.SetDataType(outDtypeOne);

  (void)op.UpdateOutputDesc("dx", td);
  (void)op.UpdateOutputDesc("da", tdOne);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PReluGrad, PReluGradInferShape);
VERIFY_FUNC_REG(PReluGrad, PReluGradVerify);
// ----------------PReluGrad End---------------
// ----------------Tanh Op Begin-----------------
COMMON_INFER_FUNC_REG(Tanh, OneInOneOutCommonInferShape);
// ----------------Tanh Op End-------------------

// ----------------Relu-------------------
COMMON_INFER_FUNC_REG(Relu, OneInOneOutCommonInferShape);
// --------------Relu END-----------------

// ----------------ReluV2-------------------
IMPLEMT_COMMON_INFERFUNC(ReluV2InferShape) {
  OP_LOGI(op.GetName().c_str(), "enter relu_v2 op_proto inferfunction!!!");
  if (!OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_FAILED;
  }

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto xDesc = op_info->MutableInputDesc("x");
  auto outDesc_mask = op_info->MutableOutputDesc("mask");

  vector<int64_t> input_shape = xDesc->MutableShape().GetDims();
  Shape origin_shape = op.GetInputDesc("x").GetOriginShape();
  auto origin_format = op.GetInputDesc("x").GetOriginFormat();

  std::vector<int64_t> dims_mask;
  std::vector<int64_t> dims = origin_shape.GetDims();
  if (dims.size() != 4) {
    OP_LOGW("The origin shape dim is must be 4, which is needed by DreluFusionPass");
    return GRAPH_FAILED;
  }
  if (origin_format == FORMAT_NHWC) {
    OP_LOGI(op.GetName().c_str(), "The format is NHWC");
    for (unsigned int i = 0; i < dims.size() - 1; i++) {
      if (1 == i) {
        if (xDesc->GetDataType() == DT_UINT8 || xDesc->GetDataType() == DT_INT8) {
          dims_mask.push_back((origin_shape.GetDim(3) + 31) / 32);
        } else {
          dims_mask.push_back((origin_shape.GetDim(3) + 15) / 16);
        }
      }
      dims_mask.push_back(origin_shape.GetDim(i));
    }
  } else if (origin_format == FORMAT_NCHW) {
    OP_LOGI(op.GetName().c_str(), "The format is NCHW");
    for (unsigned int i = 0; i < dims.size(); i++) {
      if (1 == i) {
        if (xDesc->GetDataType() == DT_UINT8 || xDesc->GetDataType() == DT_INT8) {
          dims_mask.push_back((origin_shape.GetDim(1) + 31) / 32);
        } else {
          dims_mask.push_back((origin_shape.GetDim(1) + 15) / 16);
        }
      } else {
        dims_mask.push_back(origin_shape.GetDim(i));
      }
    }
  } else {
    string expected_format_list = ConcatString("FORMAT_NHWC, FORMAT_NCHW");
    OpsInputFormatErrReport(op.GetName(), "x", expected_format_list, ConcatString(origin_format));
    OP_LOGI("The format only support NHWC and NCHW.");
    return GRAPH_FAILED;
  }
  if (xDesc->GetDataType() == DT_UINT8 || xDesc->GetDataType() == DT_INT8) {
    dims_mask.push_back(4);
  } else {
    dims_mask.push_back(2);
  }

  outDesc_mask->SetShape(GeShape(dims_mask));
  outDesc_mask->SetDataType(DataType(DT_UINT8));
  if (IsUnknown(input_shape)) {
    std::vector<std::pair<int64_t, int64_t>> mask_range;
    MakeUpShapeRange(dims_mask, mask_range);
    outDesc_mask->SetShapeRange(mask_range);
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReluV2, ReluV2InferShape);
// ----------------ReluV2 END-------------------

// ----------------BNLL-------------------
COMMON_INFER_FUNC_REG(BNLL, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------BNLL END-----------------

// ----------------Elu-------------------
IMPLEMT_COMMON_INFERFUNC(EluInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter EluInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Elu, EluInferShape);
// --------------Elu END-----------------

// ----------------Celu-------------------
IMPLEMT_COMMON_INFERFUNC(CeluInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter CeluInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Celu, CeluInferShape);
// --------------Celu END-----------------

// ----------------EluGrad-------------------
IMPLEMT_VERIFIER(EluGrad, EluGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "grads", "activations")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(EluGrad, EluGradVerify);

IMPLEMT_COMMON_INFERFUNC(EluGradInferShape) {
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "grads", "activations", {"y"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(EluGrad, EluGradInferShape);
// --------------EluGrad END-----------------

// ----------------Relu6Grad------------------
IMPLEMT_VERIFIER(Relu6Grad, Relu6GradVerify) {
  if (!CheckTwoInputDtypeSame(op, "features", "gradients")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(Relu6GradInferShape) {
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "gradients", "features", {"backprops"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Relu6Grad, Relu6GradInferShape);
VERIFY_FUNC_REG(Relu6Grad, Relu6GradVerify);
// -------------Relu6Grad END-------------------

// ----------------Relu6-------------------
IMPLEMT_COMMON_INFERFUNC(Relu6InferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Relu6, Relu6InferShape);
// ----------------Relu6 END-------------------

// ----------------Begin Relu6D-------------------
IMPLEMT_COMMON_INFERFUNC(Relu6DInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

IMPLEMT_VERIFIER(Relu6D, Relu6DVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter Relu6D verifyFunction!");

  // check input const attr for scale
  std::vector<float> const_attr;
  if (!GetConstAttr(op, {"scale"}, const_attr)) {
    std::string err_msg = GetInputInvalidErrMsg("scale");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(Relu6D, Relu6DInferShape);

// Registered verify function
VERIFY_FUNC_REG(Relu6D, Relu6DVerify);
// ----------------Relu6D END-------------------

// ----------------SigmoidGrad----------------
IMPLEMT_VERIFIER(SigmoidGrad, SigmoidGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SigmoidGradInferShape) {
  Shape shape_x = op.GetInputDesc("y").GetShape();
  DataType input_dtype = op.GetInputDesc("y").GetDataType();
  std::vector<std::pair<int64_t, int64_t>> shape_range_y;
  op.GetInputDesc("y").GetShapeRange(shape_range_y);
  TensorDesc tensordesc_output = op.GetOutputDesc("z");
  tensordesc_output.SetShape(shape_x);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetShapeRange(shape_range_y);
  if (op.UpdateOutputDesc("z", tensordesc_output) != GRAPH_SUCCESS) {
    std::string err_msg = UpdateParamErrMsg("z");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SigmoidGrad, SigmoidGradInferShape);
VERIFY_FUNC_REG(SigmoidGrad, SigmoidGradVerify);
// ----------------SigmoidGrad-------------------

// ----------------Softplus-------------------
IMPLEMT_COMMON_INFERFUNC(SoftplusInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Softplus, SoftplusInferShape);
// --------------Softplus END-----------------

// ----------------SoftplusGrad-------------------
IMPLEMT_VERIFIER(SoftplusGrad, SoftplusGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "gradients", "features")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SoftplusGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "gradients", "features", "backprops", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(SoftplusGrad, SoftplusGradInferShape);
VERIFY_FUNC_REG(SoftplusGrad, SoftplusGradVerify);
// ----------------SoftplusGrad END-------------------

// ----------------SoftSign ----------------
IMPLEMT_COMMON_INFERFUNC(SoftsignInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Softsign, SoftsignInferShape);
// ----------------SoftSign END-------------------

// ----------------Selu-------------------
IMPLEMT_COMMON_INFERFUNC(SeluInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Selu, SeluInferShape);
// ----------------Selu END-------------------

// ----------------ReluGrad-------------------
IMPLEMT_VERIFIER(ReluGrad, ReluGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "gradients", "features")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ReluGradInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "gradients", "features", "backprops")) {
    return GRAPH_FAILED;
  }
  if (!InferShapeRangeTwoInOneOutBroadcase(op, "gradients", "features", "backprops")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReluGrad, ReluGradInferShape);
VERIFY_FUNC_REG(ReluGrad, ReluGradVerify);
// ----------------ReluGrad END-----------------

// ----------------ReluGradV2-------------------
IMPLEMT_COMMON_INFERFUNC(ReluGradV2InferShape) {
  if (OneInOneOutDynamicInfer(op, "gradients", {"backprops"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(ReluGradV2, ReluGradV2InferShape);

// ----------------ReluGradV2 END-----------------

// ----------------LeakyReluGrad-------------------
IMPLEMT_VERIFIER(LeakyReluGrad, LeakyReluGradVerify) {
  OP_LOGI(op.GetName().c_str(), "enter LeakyReluGrad verify");
  if (!CheckTwoInputDtypeSame(op, "gradients", "features")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(LeakyReluGradInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("backprops");
  tensordesc_output.SetShape(op.GetInputDesc("gradients").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("gradients").GetDataType());
  std::vector<std::pair<int64_t, int64_t>> shape_range_grad;
  op.GetInputDesc("gradients").GetShapeRange(shape_range_grad);
  tensordesc_output.SetShapeRange(shape_range_grad);
  (void)op.UpdateOutputDesc("backprops", tensordesc_output);
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(LeakyReluGrad, LeakyReluGradVerify);
COMMON_INFER_FUNC_REG(LeakyReluGrad, LeakyReluGradInferShape);
// ----------------LeakyReluGrad END----------------

// ----------------ThresholdGradV2D-------------------
bool InferShapeAndTypeThresholdGradV2D(Operator& op, const string& input_name1, const string& input_name2,
                                       const string& output_name) {
  TensorDesc vOutputDesc = op.GetOutputDesc(output_name);

  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();
  ge::Shape shapeX = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shapeY = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dimsX = shapeX.GetDims();
  std::vector<int64_t> dimsY = shapeY.GetDims();
  if (dimsX.size() < dimsY.size()) {
    std::vector<int64_t> dimsTmp = dimsX;
    dimsX = dimsY;
    dimsY = dimsTmp;
  }

  if (dimsX.size() != dimsY.size()) {
    int dec = dimsX.size() - dimsY.size();
    for (int i = 0; i < dec; i++) {
      dimsY.insert(dimsY.begin(), (int64_t)1);
    }
  }

  std::vector<int64_t> dimVec;
  for (size_t i = 0; i < dimsX.size(); i++) {
    if ((dimsX[i] != dimsY[i]) && (dimsX[i] != 1) && (dimsY[i] != 1)) {
      return false;
    }

    int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
    dimVec.push_back(dims);
  }
  ge::Shape outputShape = ge::Shape(dimVec);

  vOutputDesc.SetShape(outputShape);
  vOutputDesc.SetDataType(input_dtype);
  vOutputDesc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, vOutputDesc);

  return true;
}

IMPLEMT_VERIFIER(ThresholdGradV2D, ThresholdGradV2DVerify) {
  if (op.GetInputDesc("gradients").GetDataType() != op.GetInputDesc("features").GetDataType()) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ThresholdGradV2DInferShape) {
  bool is_dynamic_output = true;
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "gradients", "features", "backprops", is_dynamic_output)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ThresholdGradV2D, ThresholdGradV2DInferShape);

// Registered verify function
VERIFY_FUNC_REG(ThresholdGradV2D, ThresholdGradV2DVerify);
// ----------------ThresholdGradV2D-------------------
// ------------ThresholdV2D Op Start----------------
IMPLEMT_VERIFIER(ThresholdV2D, ThresholdV2DVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(ThresholdV2DInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");

  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());
  tensordesc_output.SetFormat(op.GetInputDesc("x").GetFormat());
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ThresholdV2D, ThresholdV2DInferShape);
VERIFY_FUNC_REG(ThresholdV2D, ThresholdV2DVerify);

// ------------ThresholdV2D Op End----------------

// ------------Mish Op Start----------------
IMPLEMT_COMMON_INFERFUNC(MishInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Mish, MishInferShape);
// ------------Mish Op End----------------

// ------------Mish Grad Op Start----------------
IMPLEMT_COMMON_INFERFUNC(MishGradInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("x_grad");
  TensorDesc tensordesc_input = op.GetInputDesc("x");
  tensordesc_output.SetShape(tensordesc_input.GetShape());
  tensordesc_output.SetDataType(tensordesc_input.GetDataType());
  (void)op.UpdateOutputDesc("x_grad", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MishGrad, MishGradInferShape);
// ------------Mish Grad Op End----------------

// ----------------HardtanhGrad Begin-------------------
IMPLEMT_INFERFUNC(HardtanhGrad, HardtanhGradInferShape) {
  DataType result_type = op.GetInputDescByName("result").GetDataType();
  DataType grad_type = op.GetInputDescByName("grad").GetDataType();

  AscendString op_name;
  if (GRAPH_SUCCESS != op.GetName(op_name)) {
    OP_LOGE("HardtanhGrad", "op_name get failed.");
    return GRAPH_FAILED;
  }
  const char* op_name_c = op_name.GetString();

  if (result_type != grad_type) {
    OP_LOGE(op_name_c, "result'dtype is not same as grad'dtype.");
    return GRAPH_FAILED;
  }

  if (OneInOneOutDynamicInfer(op, "result", {"y"})) {
    return GRAPH_SUCCESS;
  }

  OP_LOGE(op_name_c, "shape of y is not same as shape of result.");
  return GRAPH_FAILED;
}
INFER_FUNC_REG(HardtanhGrad, HardtanhGradInferShape);
// ----------------HardtanhGrad END---------------------

// ----------------SoftplusV2 Begin-------------------
IMPLEMT_INFERFUNC(SoftplusV2, SoftplusV2InferShape) {
  TensorDesc tensordesc_input = op.GetInputDesc("x");
  Shape input_shape = tensordesc_input.GetShape();
  Format input_format = tensordesc_input.GetFormat();
  DataType input_dtype = tensordesc_input.GetDataType();

  TensorDesc tensordesc_output = op.GetOutputDesc("y");

  tensordesc_output.SetShape(input_shape);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetFormat(input_format);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
INFER_FUNC_REG(SoftplusV2, SoftplusV2InferShape);
// ----------------SoftplusV2 END---------------------

// ----------------SoftplusV2Grad Begin-------------------
IMPLEMT_INFERFUNC(SoftplusV2Grad, SoftplusV2GradInferShape) {
  TensorDesc tensordesc_input1 = op.GetInputDesc("input_gradients");
  Shape input_shape1 = tensordesc_input1.GetShape();
  Format input_format1 = tensordesc_input1.GetFormat();
  DataType input_dtype1 = tensordesc_input1.GetDataType();
  std::vector<int64_t> dims_input1 = input_shape1.GetDims();
  TensorDesc tensordesc_input2 = op.GetInputDesc("input_features");
  Shape input_shape2 = tensordesc_input2.GetShape();
  std::vector<int64_t> dims_input2 = input_shape2.GetDims();

  if (dims_input1.size() != dims_input2.size()) {
    OP_LOGE(op.GetName().c_str(), "Input shapes are not the same.");
    return GRAPH_FAILED;
  }

  TensorDesc tensordesc_output = op.GetOutputDesc("output_backprops");
  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_input1.size(); i++) {
    if ((dims_input1[i] != dims_input2[i]) && (dims_input1[i] != 1) &&
        (dims_input2[i] != 1)) {
      OP_LOGE(op.GetName().c_str(), "Input shapes are not compatible.");
      return GRAPH_FAILED;
    }

    int64_t dims =
        dims_input1[i] > dims_input2[i] ? dims_input1[i] : dims_input2[i];
    dim_vec.push_back(dims);
  }
  ge::Shape output_shape = ge::Shape(dim_vec);
  tensordesc_output.SetShape(output_shape);
  tensordesc_output.SetDataType(input_dtype1);
  tensordesc_output.SetFormat(input_format1);
  (void)op.UpdateOutputDesc("output_backprops", tensordesc_output);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SoftplusV2Grad, SoftplusV2GradVerify) {
  // check input tensors' dtype which needed to be same
  if (op.GetInputDesc("input_gradients").GetDataType() !=
      op.GetInputDesc("input_features").GetDataType()) {
    OP_LOGE(op.GetName().c_str(), "Input dtypes are not the same.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
INFER_FUNC_REG(SoftplusV2Grad, SoftplusV2GradInferShape);
// Registered verify function
VERIFY_FUNC_REG(SoftplusV2Grad, SoftplusV2GradVerify);
// ----------------SoftplusV2Grad END---------------------

// ----------------ThresholdedRelu Begin-------------------
IMPLEMT_COMMON_INFERFUNC(ThresholdedReluInferShape) {
  ge::TensorDesc input_desc = op.GetInputDesc(0);
  ge::TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(input_desc.GetShape());
  output_desc.SetFormat(input_desc.GetFormat());
  output_desc.SetDataType(input_desc.GetDataType());
  op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ThresholdedRelu, ThresholdedReluVerify) {
  float alpha = 1.0;
  auto ret = op.GetAttr("alpha", alpha);
  if (ret != ge::GRAPH_SUCCESS) {
    OP_LOGE("ThresholdedReluVerify", "OP GetAttr alpha fail.");
    return GRAPH_FAILED;
  }
  ge::TensorDesc input_desc = op.GetInputDesc(0);
  ge::DataType data_type = input_desc.GetDataType();
  if (data_type != DT_FLOAT16 && data_type != DT_FLOAT) {
    OP_LOGE("ThresholdedReluVerify", "Input DataType is not fp16 or fp32");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(ThresholdedRelu, ThresholdedReluInferShape);
VERIFY_FUNC_REG(ThresholdedRelu, ThresholdedReluVerify);
// ----------------ThresholdedRelu END---------------------

// ----------------HardShrink Begin-------------------
IMPLEMT_COMMON_INFERFUNC(HardShrinkInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("output_y");
  DataType predict_dtype = op.GetInputDesc("input_x").GetDataType();
  Format predict_format = op.GetInputDesc("input_x").GetFormat();
  ge::Shape output_shape = op.GetInputDesc("input_x").GetShape();
  output_desc.SetDataType(predict_dtype);
  output_desc.SetFormat(predict_format);
  output_desc.SetShape(output_shape);
  (void)op.UpdateOutputDesc("output_y", output_desc);
      return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HardShrink, HardShrinkVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(HardShrink, HardShrinkInferShape);
VERIFY_FUNC_REG(HardShrink, HardShrinkVerify);
// ----------------HardShrink END---------------------

// ----------------HardShrinkGrad Begin-------------------
IMPLEMT_COMMON_INFERFUNC(HardShrinkGradInferShape)
{
    TensorDesc output_desc = op.GetOutputDesc("backprops");
    DataType dtype_x = op.GetInputDesc("features").GetDataType();
    Format format_x = op.GetInputDesc("features").GetFormat();
    ge::Shape shape_x = op.GetInputDesc("features").GetShape();
    ge::Shape shape_grad = op.GetInputDesc("gradients").GetShape();
    std::vector<int64_t> dims_x = shape_x.GetDims();
    std::vector<int64_t> dims_grad = shape_grad.GetDims();
    if (dims_x.size() < dims_grad.size()) {
        std::vector<int64_t> dims_tmp = dims_x;
        dims_x = dims_grad;
        dims_grad = dims_tmp;
    }
    if (dims_x.size() != dims_grad.size()) {
        int dec = dims_x.size() - dims_grad.size();
        for (int i = 0; i < dec; i++) {
            dims_grad.insert(dims_grad.begin(), (int64_t)1);
        }
    }
    std::vector<int64_t> dim_vec;
    for (size_t i = 0; i < dims_x.size(); i++) {
        if ((dims_x[i] != dims_grad[i]) && (dims_x[i] != 1) && (dims_grad[i] != 1)) {
            OP_LOGE("HardShrinkGrad", "Input's dim must be same and dim not equal 1");
            return GRAPH_FAILED;
        }
        int64_t dims = dims_x[i] > dims_grad[i] ? dims_x[i] : dims_grad[i];
        dim_vec.push_back(dims);
    }
    ge::Shape output_shape = ge::Shape(dim_vec);
    output_desc.SetDataType(dtype_x);
    output_desc.SetFormat(format_x);
    output_desc.SetShape(output_shape);
    (void)op.UpdateOutputDesc("backprops", output_desc);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HardShrinkGrad, HardShrinkGradVerify)
{
    if (op.GetInputDesc("features").GetDataType() != op.GetInputDesc("gradients").GetDataType()) {
        OP_LOGE(op.GetName().c_str(), "Input two tensor's dtype must be same.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(HardShrinkGrad, HardShrinkGradInferShape);
VERIFY_FUNC_REG(HardShrinkGrad, HardShrinkGradVerify);
// ----------------HardShrinkGrad END---------------------

// ----------------HardSigmoid Begin-------------------
IMPLEMT_INFERFUNC(HardSigmoid,HardSigmoidInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("output_y");
  tensordesc_output.SetShape(op.GetInputDesc("input_x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("input_x").GetDataType());
  tensordesc_output.SetFormat(op.GetInputDesc("input_x").GetFormat());

  (void)op.UpdateOutputDesc("output_y", tensordesc_output);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HardSigmoid, HardSigmoidInferShape);
// ----------------HardSigmoid END---------------------

// ----------------SoftShrink Begin-------------------
IMPLEMT_COMMON_INFERFUNC(SoftShrinkInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("output_y");
  DataType predict_dtype = op.GetInputDesc("input_x").GetDataType();
  Format predict_format = op.GetInputDesc("input_x").GetFormat();
  ge::Shape output_shape = op.GetInputDesc("input_x").GetShape();
  output_desc.SetDataType(predict_dtype);
  output_desc.SetFormat(predict_format);
  output_desc.SetShape(output_shape);
  (void)op.UpdateOutputDesc("output_y", output_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SoftShrink, SoftShrinkVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SoftShrink, SoftShrinkInferShape);
VERIFY_FUNC_REG(SoftShrink, SoftShrinkVerify);
// ----------------SoftShrink END---------------------

// ----------------SoftShrinkGrad Begin-------------------
IMPLEMT_COMMON_INFERFUNC(SoftShrinkGradInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("output_y");
  DataType dtype_x = op.GetInputDesc("input_x").GetDataType();
  Format format_x = op.GetInputDesc("input_x").GetFormat();
  ge::Shape shape_x = op.GetInputDesc("input_x").GetShape();
  ge::Shape shape_grad = op.GetInputDesc("input_grad").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_grad = shape_grad.GetDims();
  if (dims_x.size() < dims_grad.size()) {
    std::vector<int64_t> dims_tmp = dims_x;
    dims_x = dims_grad;
    dims_grad = dims_tmp;
  }
  if (dims_x.size() != dims_grad.size()) {
    int dec = dims_x.size() - dims_grad.size();
    for (int i = 0; i < dec; i++) {
      dims_grad.insert(dims_grad.begin(), (int64_t)1);
    }
  }
  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if ((dims_x[i] != dims_grad[i]) && (dims_x[i] != 1) && (dims_grad[i] != 1)) {
      OP_LOGE(op.GetName().c_str(), "Input shapes are not compatible.");
      return GRAPH_FAILED;
    }
    int64_t dims = dims_x[i] > dims_grad[i] ? dims_x[i] : dims_grad[i];
    dim_vec.push_back(dims);
  }
  ge::Shape output_shape = ge::Shape(dim_vec);
  output_desc.SetDataType(dtype_x);
  output_desc.SetFormat(format_x);
  output_desc.SetShape(output_shape);
  (void)op.UpdateOutputDesc("output_y", output_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SoftShrinkGrad, SoftShrinkGradVerify) {
  if (op.GetInputDesc("input_x").GetDataType() != op.GetInputDesc("input_grad").GetDataType()) {
    OP_LOGE(op.GetName().c_str(), "Input dtypes are not the same.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SoftShrinkGrad, SoftShrinkGradInferShape);
VERIFY_FUNC_REG(SoftShrinkGrad, SoftShrinkGradVerify);
// ----------------SoftShrinkGrad END-----------

// ----------------Sigmoid----------------------
IMPLEMT_COMMON_INFERFUNC(SigmoidInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter SigmoidInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(Sigmoid, SigmoidInferShape);
// ----------------Sigmoid Op End---------------

// ----------------LeakyRelu--------------------
IMPLEMT_COMMON_INFERFUNC(LeakyReluInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter LeakyReluInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(LeakyRelu, LeakyReluInferShape);
// ----------------LeakyRelu END-----------------

// ----------------LogSigmoidGrad begin--------------------
IMPLEMT_VERIFIER(LogSigmoidGrad, LogSigmoidGradVerify) {
  if (op.GetInputDesc("grads").GetDataType() != op.GetInputDesc("features").GetDataType()) {
    OP_LOGI(op.GetName().c_str(), "Input and output's dtype mast be same.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(LogSigmoidGradInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("backprops");
  std::vector<int64_t> dims_grads = op.GetInputDesc("grads").GetShape().GetDims();
  std::vector<int64_t> dims_features = op.GetInputDesc("features").GetShape().GetDims();

  if (dims_grads.size() < dims_features.size()) {
    std::vector<int64_t> dims_tmp = dims_grads;
    dims_grads = dims_features;
    dims_features = dims_tmp;
  }
  if (dims_grads.size() != dims_features.size()) {
    int dec = dims_grads.size() - dims_features.size();
    for (int i = 0; i < dec; i++) {
      dims_features.insert(dims_features.begin(), (int64_t)1);
    }
  }
  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < dims_grads.size(); i++) {
    if ((dims_grads[i] != dims_features[i]) && (dims_grads[i] != 1) && (dims_features[i] != 1)) {
      OP_LOGI(op.GetName().c_str(), "Input and output's dimvalue mast be different,mast not be 1.");
      return GRAPH_FAILED;
    }
    int64_t dims = dims_grads[i] > dims_features[i] ? dims_grads[i] : dims_features[i];
    dim_vec.push_back(dims);
  }
  output_desc.SetShape(ge::Shape(dim_vec));
  DataType input_dtype = op.GetInputDesc("grads").GetDataType();
  output_desc.SetDataType(input_dtype);
  Format input_format = op.GetInputDesc("grads").GetFormat();
  output_desc.SetFormat(input_format);
  (void)op.UpdateOutputDesc("backprops", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LogSigmoidGrad, LogSigmoidGradInferShape);
VERIFY_FUNC_REG(LogSigmoidGrad, LogSigmoidGradVerify);
// ----------------LogSigmoidGrad end----------------------

// ----------------LogSigmoid--------------------
IMPLEMT_COMMON_INFERFUNC(LogSigmoidInferShape) {
  Shape input_shape = op.GetInputDesc("x").GetShape();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(input_shape);
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  output_desc.SetDataType(input_dtype);
  Format input_format = op.GetInputDesc("x").GetFormat();
  output_desc.SetFormat(input_format);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LogSigmoid, LogSigmoidInferShape);

// ----------------HardSigmoidLossGrad--------------------
IMPLEMT_COMMON_INFERFUNC(HardSigmoidGradInferShape) {
    TensorDesc output_desc = op.GetOutputDesc("y");
    ge::Shape shape_x = op.GetInputDesc("input_x").GetShape();
    DataType dtype_grad = op.GetInputDesc("grads").GetDataType();
    Format formatGrad = op.GetInputDesc("grads").GetFormat();
    ge::Shape shape_grad = op.GetInputDesc("grads").GetShape();
    std::vector<int64_t> dims_x = shape_x.GetDims();
    std::vector<int64_t> dims_grad = shape_grad.GetDims();
    if (dims_x.size() != dims_grad.size()) {
        OP_LOGE(op.GetName().c_str(), "the two inputs size not equal!\n");
        return GRAPH_FAILED;
    }
    output_desc.SetDataType(dtype_grad);
    output_desc.SetFormat(formatGrad);
    output_desc.SetShape(shape_grad);
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
}
    
IMPLEMT_VERIFIER (HardSigmoidGrad, HardSigmoidGradVerify){
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(HardSigmoidGrad, HardSigmoidGradInferShape);
VERIFY_FUNC_REG(HardSigmoidGrad, HardSigmoidGradVerify);

// ----------------HardSigmoidLossGrad END--------------------

// ----------------Shrink Begin-------------------
IMPLEMT_COMMON_INFERFUNC(ShrinkInferShape) {
  const char* input_x = "input_x";
  const char* output_y = "output_y";
  TensorDesc output_desc = op.GetOutputDescByName(output_y);
  DataType predict_dtype = op.GetInputDescByName(input_x).GetDataType();
  Format predict_format = op.GetInputDescByName(input_x).GetFormat();
  ge::Shape output_shape = op.GetInputDescByName(input_x).GetShape();
  output_desc.SetDataType(predict_dtype);
  output_desc.SetFormat(predict_format);
  output_desc.SetShape(output_shape);
  (void)op.UpdateOutputDesc(output_y, output_desc);
      return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Shrink, ShrinkVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Shrink, ShrinkInferShape);
VERIFY_FUNC_REG(Shrink, ShrinkVerify);
// ----------------Shrink END---------------------

// ---------------------ThresholdV2------------------------
IMPLEMT_INFERFUNC(ThresholdV2, ThresholdV2InferShape) {
  TensorDesc tensordesc_input = op.GetInputDescByName("x");
  Shape input_shape = tensordesc_input.GetShape();
  DataType input_dtype = tensordesc_input.GetDataType();
  Format input_format = tensordesc_input.GetFormat();
  std::vector<std::pair<int64_t, int64_t>> input_range;
  tensordesc_input.GetShapeRange(input_range);

  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  tensordesc_output.SetShape(input_shape);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetFormat(input_format);
  tensordesc_output.SetShapeRange(input_range);

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ThresholdV2, ThresholdV2Verify) {
  DataType input_type_x = op.GetInputDescByName("x").GetDataType();
  DataType input_type_threshold = op.GetInputDescByName("threshold").GetDataType();
  if (input_type_x != input_type_threshold) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ThresholdV2, ThresholdV2InferShape);
VERIFY_FUNC_REG(ThresholdV2, ThresholdV2Verify);
// ---------------------ThresholdV2------------------------
}  // namespace ge
