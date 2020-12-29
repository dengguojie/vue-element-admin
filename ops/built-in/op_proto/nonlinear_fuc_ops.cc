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
  TensorDesc tensordesc_output = op.GetOutputDesc("z");
  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("dy").GetDataType());
  (void)op.UpdateOutputDesc("z", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GeluGrad, GeluGradInferShape);
VERIFY_FUNC_REG(GeluGrad, GeluGradVerify);
// ----------------------GeluGrad END----------------------

// ----------------------Gelu----------------------
IMPLEMT_COMMON_INFERFUNC(GeluInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");

  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Gelu, GeluInferShape);
// ----------------------Gelu END----------------------

// ----------------------FastGeluGrad----------------------
IMPLEMT_VERIFIER(FastGeluGrad, FastGeluGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FastGeluGradInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("z");
  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("dy").GetDataType());
  (void)op.UpdateOutputDesc("z", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FastGeluGrad, FastGeluGradInferShape);
VERIFY_FUNC_REG(FastGeluGrad, FastGeluGradVerify);
// ----------------------FastGeluGrad END----------------------

// ----------------------FastGelu----------------------
IMPLEMT_COMMON_INFERFUNC(FastGeluInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");

  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FastGelu, FastGeluInferShape);
// ----------------------Gelu END----------------------

// ----------------TanhGrad Op Begin----------------
IMPLEMT_COMMON_INFERFUNC(TanhGradInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("z");
  tensordesc_output.SetShape(op.GetInputDesc("y").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("y").GetDataType());

  (void)op.UpdateOutputDesc("z", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TanhGrad, TanhGradInferShape);
// ----------------TanhGrad Op End-------------------

// ----------------PRelu-------------------
IMPLEMT_INFERFUNC(PRelu, PReluInferShape) {
  auto outShape = op.GetInputDesc("x").GetShape();
  auto outDtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(outShape);
  td.SetDataType(outDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PRelu, PReluInferShape);
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
  OP_LOGI(op.GetName().c_str(), "enter op_proto inferfunction!!!");
  TensorDesc xDesc = op.GetInputDesc("x");
  TensorDesc outDesc = op.GetOutputDesc("y");
  TensorDesc outDesc_mask = op.GetOutputDesc("mask");

  Shape shape = xDesc.GetShape();
  Shape origin_shape = xDesc.GetOriginShape();

  outDesc.SetShape(shape);
  outDesc.SetDataType(xDesc.GetDataType());
  (void)op.UpdateOutputDesc("y", outDesc);

  std::vector<int64_t> dims_mask;
  std::vector<int64_t> dims = origin_shape.GetDims();
  if (dims.size() != 4) {
    OpsAttrValueErrReport(op.GetName(), "x's origin shape dim", "4", ConcatString(dims.size()));
    OP_LOGI("The origin shape dim is must be 4");
    return GRAPH_FAILED;
  }
  if (xDesc.GetOriginFormat() == FORMAT_NHWC) {
    OP_LOGI(op.GetName().c_str(), "The format is NHWC");
    for (unsigned int i = 0; i < dims.size() - 1; i++) {
      if (1 == i) {
        if (xDesc.GetDataType() == DT_UINT8 || xDesc.GetDataType() == DT_INT8) {
          dims_mask.push_back((origin_shape.GetDim(3) + 31) / 32);
        } else {
          dims_mask.push_back((origin_shape.GetDim(3) + 15) / 16);
        }
      }
      dims_mask.push_back(origin_shape.GetDim(i));
    }
  } else if (xDesc.GetOriginFormat() == FORMAT_NCHW) {
    OP_LOGI(op.GetName().c_str(), "The format is NCHW");
    for (unsigned int i = 0; i < dims.size(); i++) {
      if (1 == i) {
        if (xDesc.GetDataType() == DT_UINT8 || xDesc.GetDataType() == DT_INT8) {
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
    OpsInputFormatErrReport(op.GetName(), "x", expected_format_list, ConcatString(xDesc.GetOriginFormat()));
    OP_LOGI("The format only support NHWC and NCHW.");
    return GRAPH_FAILED;
  }
  if (xDesc.GetDataType() == DT_UINT8 || xDesc.GetDataType() == DT_INT8) {
    dims_mask.push_back(4);
  } else {
    dims_mask.push_back(2);
  }

  outDesc_mask.SetShape(Shape(dims_mask));
  outDesc_mask.SetDataType(DataType(DT_UINT8));
  (void)op.UpdateOutputDesc("mask", outDesc_mask);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ReluV2, ReluV2InferShape);
// ----------------ReluV2 END-------------------

// ----------------BNLL-------------------
COMMON_INFER_FUNC_REG(BNLL, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------BNLL END-----------------

// ----------------Elu-------------------
COMMON_INFER_FUNC_REG(Elu, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------Elu END-----------------

// ----------------EluGrad-------------------
IMPLEMT_VERIFIER(EluGrad, EluGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "grads", "activations")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(EluGrad, ELMTWISE_INFER_SHAPEANDTYPE("grads", "y"));
VERIFY_FUNC_REG(EluGrad, EluGradVerify);
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
  Shape input_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(input_shape);
  tensordesc_output.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", tensordesc_output) != GRAPH_SUCCESS) {
    OpsOPUpdateErrReport(op.GetName(), "y");
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Relu6D, Relu6DVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter Relu6D verifyFunction!");

  // check input const attr for scale
  std::vector<float> const_attr;
  if (!GetConstAttr(op, {"scale"}, const_attr)) {
    OP_LOGE(op.GetName().c_str(), "The GetOpAttr ConstValue failed!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(Relu6D, Relu6DInferShape);

// Registered verify function
VERIFY_FUNC_REG(Relu6D, Relu6DVerify);
// ----------------Relu6D END-------------------

// ----------------Softplus-------------------
IMPLEMT_COMMON_INFERFUNC(SoftplusInferShape) {
  Shape features_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(features_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
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
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "gradients", "features", "backprops")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(SoftplusGrad, SoftplusGradInferShape);
VERIFY_FUNC_REG(SoftplusGrad, SoftplusGradVerify);
// ----------------SoftplusGrad END-------------------

// ----------------SoftSign ----------------
IMPLEMT_COMMON_INFERFUNC(SoftsignInferShape) {
  Shape features_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(features_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Softsign, SoftsignInferShape);
// ----------------SoftSign END-------------------

// ----------------Selu-------------------
IMPLEMT_COMMON_INFERFUNC(SeluInferShape) {
  Shape features_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(features_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
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

IMPLEMT_VERIFIER(LeakyRelu, LeakyReluVerify) {
  OP_LOGI(op.GetName().c_str(), "enter LeakyRelu verify");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LeakyRelu, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
VERIFY_FUNC_REG(LeakyRelu, LeakyReluVerify);

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
  // vOutputDesc.push_back(op.GetInputDesc(0));
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
  if (InferShapeAndTypeThresholdGradV2D(op, "gradients", "features", "backprops")) {
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

// ----------------HardtanhGrad Begin-------------------
IMPLEMT_VERIFIER(HardtanhGrad, HardtanhGradVerify) {
  DataType input_type_x = op.GetInputDesc("result").GetDataType();
  DataType input_type_y = op.GetInputDesc("grad").GetDataType();
  if (input_type_x != input_type_y) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(HardtanhGradInferShape) {
  TensorDesc output_desc = op.GetOutputDesc("y");

  auto tensor_desc = op.GetInputDesc("result");
  auto tensor_shape = tensor_desc.GetShape();
  output_desc.SetShape(tensor_shape);

  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(HardtanhGrad, HardtanhGradInferShape);
VERIFY_FUNC_REG(HardtanhGrad, HardtanhGradVerify);
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

}  // namespace ge
