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
 * \file nn_training_ops.cpp
 * \brief
 */
#include "inc/nn_training_ops.h"
#include <string>
#include <vector>
#include <map>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"

namespace ge {

// Obtains the output tensor description for Apply_op
void ApplyInferShapeAndDtype(Operator& op, const string& input_name, const string& output_name) {
  TensorDesc out_desc = op.GetOutputDesc(output_name);
  TensorDesc in_desc = op.GetInputDesc(input_name);

  out_desc.SetShape(in_desc.GetShape());
  out_desc.SetDataType(in_desc.GetDataType());
  if (op.UpdateOutputDesc(output_name, out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed, maybe output name error!");
  }
}

void DynamicApplyInferShapeRange(Operator& op, const string& input_name, const string& output_name) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr input_tensor_desc = op_desc->MutableInputDesc(input_name);
  GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc(output_name);
  if (output_tensor_desc == nullptr){
    return;
  }
  std::vector<std::pair<int64_t, int64_t>> input_shape_range;
  if (input_tensor_desc->GetShapeRange(input_shape_range) == GRAPH_SUCCESS) {
    output_tensor_desc->SetShapeRange(input_shape_range);
  }
}

// Set ref port for ref input without ref output
void SetRefInput(Operator& op, const string& input_name) {
  const OpDescPtr opDesc = OpDescUtils::GetOpDescFromOperator(op);
  int inputIndex = opDesc->GetInputIndexByName(input_name);
  if (inputIndex < 0) {
    OP_LOGE(op.GetName().c_str(), "SetRefInput failed, can not find input %s!", input_name.c_str());
  }
  GeTensorDesc inputTensorDesc = opDesc->GetInputDesc(inputIndex);
  vector<uint32_t> index = {static_cast<uint32_t>(inputIndex)};
  inputTensorDesc.SetRefPortByIndex(index);
  if (opDesc->UpdateInputDesc(static_cast<uint32_t>(inputIndex), inputTensorDesc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc failed, maybe input name error!");
  }
}

// Check input and attr of the input tensor description.
bool ApplyVerifyFunc(const ge::Operator& op, const std::vector<std::string>& inputTensorList,
                     const std::vector<std::string>& inputScalarList) {
  // check shape of Tensor
  auto var_dims = op.GetInputDesc(inputTensorList[0]).GetShape().GetDims();
  if (var_dims.size() > 8 || var_dims.size() < 0) {
    OP_LOGE(op.GetName().c_str(), "var only support 0 ~ 8 dims!");
    return GRAPH_FAILED;
  }
  if (IsUnknown(var_dims)) {
    OP_LOGW(op.GetName().c_str(), "this is dynamic shape, will exit ApplyVerifyFunc");
    return true;
  }
  for (std::size_t i = 1; i < inputTensorList.size(); i++) {
    auto tmp_dims = op.GetInputDesc(inputTensorList[i]).GetShape().GetDims();
    if (IsUnknown(tmp_dims)) {
      OP_LOGW(op.GetName().c_str(), "this is dynamic shape, will continue ApplyVerifyFunc");
      continue;
    }
    if (tmp_dims != var_dims) {
      OP_LOGE(op.GetName().c_str(), "the shape of %s must equal with %s", inputTensorList[i].c_str(),
              inputTensorList[0].c_str());
      return false;
    }
  }

  // check shape of Scalar
  for (std::size_t j = 0; j < inputScalarList.size(); j++) {
    auto scalar_dims = op.GetInputDesc(inputScalarList[j]).GetShape().GetDims();
    if (scalar_dims.size() > 1) {
      OP_LOGE(op.GetName().c_str(), "The input %s must be scalar!", inputScalarList[j].c_str());
      return false;
    }
  }
  return true;
}

// ----------------ApplyAdaMax Op-------------------
IMPLEMT_VERIFIER(ApplyAdaMax, ApplyAdaMaxVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdaMax proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("m");
  inputTensorList.push_back("v");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("beta1_power");
  inputScalarList.push_back("beta1");
  inputScalarList.push_back("beta2");
  inputScalarList.push_back("epsilon");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ApplyAdaMaxInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdaMax op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdaMax, ApplyAdaMaxInferShape);
VERIFY_FUNC_REG(ApplyAdaMax, ApplyAdaMaxVerify);
// ----------------ApplyAdaMax END-------------------

// ----------------ApplyAdaMaxD Op-------------------
IMPLEMT_VERIFIER(ApplyAdaMaxD, ApplyAdaMaxDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdaMaxD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("m");
  inputTensorList.push_back("v");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("beta1_power");
  inputScalarList.push_back("beta1");
  inputScalarList.push_back("beta2");
  inputScalarList.push_back("epsilon");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ApplyAdaMaxDInferShape) {
  if ((OneInOneOutDynamicInfer(op, "var", {"var"})) && (OneInOneOutDynamicInfer(op, 
      "m", {"m"})) && (OneInOneOutDynamicInfer(op, "v", {"v"}))) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(ApplyAdaMaxD, ApplyAdaMaxDInferShape);
VERIFY_FUNC_REG(ApplyAdaMaxD, ApplyAdaMaxDVerify);
// ----------------ApplyAdaMax END-------------------

// ----------------SparseApplyAdagradD Op----------------
IMPLEMT_COMMON_INFERFUNC(SparseApplyAdagradDInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  Shape accum_shape = op.GetInputDesc("accum").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc out_var_desc = op.GetOutputDesc("var");
  TensorDesc out_accum_desc = op.GetOutputDesc("accum");
  out_var_desc.SetShape(ge::Shape(var_shape));
  out_accum_desc.SetShape(ge::Shape(accum_shape));
  out_var_desc.SetDataType(input_dtype);
  out_accum_desc.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("var", out_var_desc) != GRAPH_SUCCESS ||
      op.UpdateOutputDesc("accum", out_accum_desc) != GRAPH_SUCCESS) {
    std::string err_msg = UpdateParamErrMsg("accum");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseApplyAdagradD, SparseApplyAdagradDVerify) {
  DataType var_dtype = op.GetInputDesc("var").GetDataType();
  DataType accum_dtype = op.GetInputDesc("accum").GetDataType();
  if (var_dtype != accum_dtype) {
    std::string err_msg = OtherErrMsg("The sparse_apply_adagrad op inputs should have the same dtype!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseApplyAdagradD, SparseApplyAdagradDInferShape);
VERIFY_FUNC_REG(SparseApplyAdagradD, SparseApplyAdagradDVerify);
// ----------------SparseApplyAdagradD END------------

// ----------------SparseApplyAdagrad Op----------------
IMPLEMT_COMMON_INFERFUNC(SparseApplyAdagradInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc out_var_desc = op.GetOutputDesc("var");
  out_var_desc.SetShape(Shape(var_shape));
  out_var_desc.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("var", out_var_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseApplyAdagrad, SparseApplyAdagradVerify) {
  DataType var_dtype = op.GetInputDesc("var").GetDataType();
  DataType accum_dtype = op.GetInputDesc("accum").GetDataType();
  if (var_dtype != accum_dtype) {
    OP_LOGE(op.GetName().c_str(), "The sparse_apply_adagrad op inputs should have the same dtype!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseApplyAdagrad, SparseApplyAdagradInferShape);
VERIFY_FUNC_REG(SparseApplyAdagrad, SparseApplyAdagradVerify);
// ----------------SparseApplyAdagrad END-----------------

// ----------------SparseApplyAdagradV2D Op----------------
IMPLEMT_COMMON_INFERFUNC(SparseApplyAdagradV2DInferShape) {
  ApplyInferShapeAndDtype(op, "var", "var");
  ApplyInferShapeAndDtype(op, "accum", "accum");
  DynamicApplyInferShapeRange(op, "var", "var");
  DynamicApplyInferShapeRange(op, "accum", "accum");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseApplyAdagradV2D, SparseApplyAdagradV2DVerify) {
  const std::map<std::string, std::vector<DataType>> kInputTensorMap{{"var", {DT_FLOAT}}, {"accum", {DT_FLOAT}}};
  // input tensor params, must have same shape and dtype
  if (!CheckInputDtypeAndShape(op, kInputTensorMap)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseApplyAdagradV2D, SparseApplyAdagradV2DInferShape);
VERIFY_FUNC_REG(SparseApplyAdagradV2D, SparseApplyAdagradV2DVerify);
// ----------------SparseApplyAdagradV2D END------------

// ----------------SparseApplyAdagradV2 Op----------------
IMPLEMT_COMMON_INFERFUNC(SparseApplyAdagradV2InferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc out_var_desc = op.GetOutputDesc("var");
  out_var_desc.SetShape(Shape(var_shape));
  out_var_desc.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("var", out_var_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseApplyAdagradV2, SparseApplyAdagradV2Verify) {
  const std::map<std::string, std::vector<DataType>> kInputTensorMap{{"var", {DT_FLOAT}}, {"accum", {DT_FLOAT}}};
  // input tensor params, must have same shape and dtype
  if (!CheckInputDtypeAndShape(op, kInputTensorMap)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseApplyAdagradV2, SparseApplyAdagradV2InferShape);
VERIFY_FUNC_REG(SparseApplyAdagradV2, SparseApplyAdagradV2Verify);
// ----------------SparseApplyAdagradV2 END-----------------

// ----------------ApplyAdagrad Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyAdagrad, ApplyAdagradVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdagrad proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdagradInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdagrad op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdagrad, ApplyAdagradInferShape);
VERIFY_FUNC_REG(ApplyAdagrad, ApplyAdagradVerify);
// ----------------ApplyAdagrad END-------------------

// ----------------ApplyAdagradD Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyAdagradD, ApplyAdagradDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdagradD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdagradDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyCenteredRMSPropD op_proto inferfunction!");
  if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "accum", {"accum"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdagradD, ApplyAdagradDInferShape);
VERIFY_FUNC_REG(ApplyAdagradD, ApplyAdagradDVerify);
// ----------------ApplyAdagradD END-------------------

// ----------------ApplyAdagradV2 Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyAdagradV2, ApplyAdagradV2Verify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdagradV2 proto verifyFunction!");
  const std::vector<std::string> kInputTensorList{"var", "accum", "grad"};
  const std::vector<std::string> kInputScalarList{"lr", "epsilon"};
  if (!ApplyVerifyFunc(op, kInputTensorList, kInputScalarList)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdagradV2InferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdagradV2 op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdagradV2, ApplyAdagradV2InferShape);
VERIFY_FUNC_REG(ApplyAdagradV2, ApplyAdagradV2Verify);
// ----------------ApplyAdagradV2 END-------------------

// ----------------ApplyAdagradV2D Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyAdagradV2D, ApplyAdagradV2DVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdagradV2D proto verifyFunction!");
  const std::vector<std::string> kInputTensorList{"var", "accum", "grad"};
  const std::vector<std::string> kInputScalarList{"lr"};
  std::vector<float> const_attr;
  if (!GetConstAttr(op, {"epsilon"}, const_attr)) {
    OP_LOGE(op.GetName().c_str(), "The GetOpAttr ConstValue failed!");
  }
  if (!ApplyVerifyFunc(op, kInputTensorList, kInputScalarList)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(ApplyAdagradV2DInferShape) {
  if (OneInOneOutDynamicInfer(op, "var", {"var"})) {
  if (OneInOneOutDynamicInfer(op, "var", {"accum"})){
    return GRAPH_SUCCESS;
  }
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(ApplyAdagradV2D, ApplyAdagradV2DInferShape);
VERIFY_FUNC_REG(ApplyAdagradV2D, ApplyAdagradV2DVerify);
// ----------------ApplyAdagradV2D END-------------------

// ----------------ApplyAdagradDA Op-------------------
IMPLEMT_VERIFIER(ApplyAdagradDA, ApplyAdagradDAVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdagradDA proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("gradient_accumulator");
  inputTensorList.push_back("gradient_squared_accumulator");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("l1");
  inputScalarList.push_back("l2");
  inputScalarList.push_back("global_step");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdagradDAInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdagradDA op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdagradDA, ApplyAdagradDAInferShape);
VERIFY_FUNC_REG(ApplyAdagradDA, ApplyAdagradDAVerify);
// ----------------ApplyAdagradDA END-------------------

// ----------------ApplyAdagradDAD Op-------------------
IMPLEMT_VERIFIER(ApplyAdagradDAD, ApplyAdagradDADVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdagradDAD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("gradient_accumulator");
  inputTensorList.push_back("gradient_squared_accumulator");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("l1");
  inputScalarList.push_back("l2");
  inputScalarList.push_back("global_step");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdagradDADInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyCenteredRMSPropD op_proto inferfunction!");
  if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "gradient_accumulator", {"gradient_accumulator"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "gradient_squared_accumulator", {"gradient_squared_accumulator"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdagradDAD, ApplyAdagradDADInferShape);
VERIFY_FUNC_REG(ApplyAdagradDAD, ApplyAdagradDADVerify);
// ----------------ApplyAdagradDAD END-------------------

// ----------------ApplyAddSign Op-------------------
IMPLEMT_VERIFIER(ApplyAddSign, ApplyAddSignVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAddSign proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("m");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("alpha");
  inputScalarList.push_back("sign_decay");
  inputScalarList.push_back("beta");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAddSignInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAddSign op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAddSign, ApplyAddSignInferShape);
VERIFY_FUNC_REG(ApplyAddSign, ApplyAddSignVerify);
// ----------------ApplyAddSign END-------------------

// ----------------ApplyAddSignD Op-------------------
IMPLEMT_VERIFIER(ApplyAddSignD, ApplyAddSignDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAddSignD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("m");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("alpha");
  inputScalarList.push_back("sign_decay");
  inputScalarList.push_back("beta");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAddSignDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAddSignD op_proto inferfunction!");
  if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "m", {"m"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAddSignD, ApplyAddSignDInferShape);
VERIFY_FUNC_REG(ApplyAddSignD, ApplyAddSignDVerify);
// ----------------ApplyAddSignD END-------------------

// ----------------ApplyCenteredRMSProp Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyCenteredRMSProp, ApplyCenteredRMSPropVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyCenteredRMSProp proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("mg");
  inputTensorList.push_back("ms");
  inputTensorList.push_back("mom");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("rho");
  inputScalarList.push_back("momentum");
  inputScalarList.push_back("epsilon");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyCenteredRMSPropInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyCenteredRMSProp op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyCenteredRMSProp, ApplyCenteredRMSPropInferShape);
VERIFY_FUNC_REG(ApplyCenteredRMSProp, ApplyCenteredRMSPropVerify);
// ----------------ApplyCenteredRMSProp END-------------------

// ----------------ApplyCenteredRMSPropD Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyCenteredRMSPropD, ApplyCenteredRMSPropDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyCenteredRMSPropD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("mg");
  inputTensorList.push_back("ms");
  inputTensorList.push_back("mom");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("rho");
  inputScalarList.push_back("momentum");
  inputScalarList.push_back("epsilon");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyCenteredRMSPropDInferShape) {
    OP_LOGI(op.GetName().c_str(), "Enter ApplyCenteredRMSPropD op_proto inferfunction!");
    if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
        return GRAPH_FAILED;
    }
    if (!OneInOneOutDynamicInfer(op, "mg", {"mg"})) {
        return GRAPH_FAILED;
    }
    if (!OneInOneOutDynamicInfer(op, "ms", {"ms"})) {
        return GRAPH_FAILED;
    }
    if (!OneInOneOutDynamicInfer(op, "mom", {"mom"})) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyCenteredRMSPropD, ApplyCenteredRMSPropDInferShape);
VERIFY_FUNC_REG(ApplyCenteredRMSPropD, ApplyCenteredRMSPropDVerify);
// ----------------ApplyCenteredRMSPropD END-------------------

// ----------------ApplyGradientDescent Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyGradientDescent, ApplyGradientDescentVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyGradientDescent proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("delta");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("alpha");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyGradientDescentInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyGradientDescent op_proto inferfunction!");
  if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyGradientDescent, ApplyGradientDescentInferShape);
VERIFY_FUNC_REG(ApplyGradientDescent, ApplyGradientDescentVerify);
// ----------------ApplyGradientDescent END-------------------

// ----------------ApplyMomentum Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyMomentum, ApplyMomentumVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyMomentum proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("momentum");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyMomentumInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyMomentum op_proto inferfunction!");
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "var", "accum", {"var"})) {
    return GRAPH_FAILED;
  }
  SetRefInput(op, "accum");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyMomentum, ApplyMomentumInferShape);
VERIFY_FUNC_REG(ApplyMomentum, ApplyMomentumVerify);
// ----------------ApplyMomentum END-------------------

// ----------------ApplyMomentumD Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyMomentumD, ApplyMomentumDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyMomentumD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("momentum");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyMomentumDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyMomentumD op_proto inferfunction!");
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "var", "accum", {"var", "accum"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyMomentumD, ApplyMomentumDInferShape);
VERIFY_FUNC_REG(ApplyMomentumD, ApplyMomentumDVerify);
// ----------------ApplyMomentumD END-------------------

// ----------------ApplyKerasMomentum Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyKerasMomentum, ApplyKerasMomentumVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyKerasMomentum proto verifyFunction!");
  const std::vector<std::string> kInputTensorList{"var", "accum", "grad"};
  const std::vector<std::string> kInputScalarList{"lr", "momentum"};

  if (!ApplyVerifyFunc(op, kInputTensorList, kInputScalarList)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyKerasMomentumInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyKerasMomentum op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyKerasMomentum, ApplyKerasMomentumInferShape);
VERIFY_FUNC_REG(ApplyKerasMomentum, ApplyKerasMomentumVerify);
// ----------------ApplyKerasMomentum END-------------------

// ----------------ApplyKerasMomentumD Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyKerasMomentumD, ApplyKerasMomentumDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyKerasMomentum proto verifyFunction!");
  const std::vector<std::string> kInputTensorList{"var", "accum", "grad"};
  const std::vector<std::string> kInputScalarList{"lr", "momentum"};
  if (!ApplyVerifyFunc(op, kInputTensorList, kInputScalarList)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyKerasMomentumDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyKerasMomentum op_proto inferfunction!");
  if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "accum", {"accum"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyKerasMomentumD, ApplyKerasMomentumDInferShape);
VERIFY_FUNC_REG(ApplyKerasMomentumD, ApplyKerasMomentumDVerify);
// ----------------ApplyKerasMomentumD END-------------------

// ----------------ApplyPowerSign Op-------------------
IMPLEMT_VERIFIER(ApplyPowerSign, ApplyPowerSignVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyPowerSign proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("m");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("logbase");
  inputScalarList.push_back("sign_decay");
  inputScalarList.push_back("beta");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyPowerSignInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyPowerSign op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyPowerSign, ApplyPowerSignInferShape);
VERIFY_FUNC_REG(ApplyPowerSign, ApplyPowerSignVerify);
// ----------------ApplyPowerSign END-------------------

// ----------------ApplyPowerSignD Op-------------------
IMPLEMT_VERIFIER(ApplyPowerSignD, ApplyPowerSignDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyPowerSignD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("m");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("logbase");
  inputScalarList.push_back("sign_decay");
  inputScalarList.push_back("beta");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyPowerSignDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyPowerSignD op_proto inferfunction!");
  if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "m", {"m"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyPowerSignD, ApplyPowerSignDInferShape);
VERIFY_FUNC_REG(ApplyPowerSignD, ApplyPowerSignDVerify);
// ----------------ApplyPowerSignD END-------------------

// ----------------ApplyProximalGradientDescent Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyProximalGradientDescent, ApplyProximalGradientDescentVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyProximalGradientDescent proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("delta");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("alpha");
  inputScalarList.push_back("l1");
  inputScalarList.push_back("l2");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyProximalGradientDescentInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyProximalGradientDescent op_proto inferfunction!");
  if (OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyProximalGradientDescent, ApplyProximalGradientDescentInferShape);
VERIFY_FUNC_REG(ApplyProximalGradientDescent, ApplyProximalGradientDescentVerify);
// ----------------ApplyProximalGradientDescent END-------------------

// ----------------DataFormatDimMap Op-------------------
IMPLEMT_VERIFIER(DataFormatDimMap, DataFormatDimMapVerify) {
  OP_LOGI(op.GetName().c_str(), "the op begin verify");
  std::string src_format;
  if (op.GetAttr("src_format", src_format) == GRAPH_SUCCESS) {
    if (src_format.size() != 4) {
      std::string err_msg = GetAttrValueErrMsg("src_format", std::to_string(src_format.size()), ConcatString("4"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  } else {
    return GRAPH_FAILED;
  }
  std::string dst_format;
  if (op.GetAttr("dst_format", dst_format) == GRAPH_SUCCESS) {
    if (dst_format.size() != 4) {
      std::string err_msg = GetAttrValueErrMsg("dst_format", std::to_string(dst_format.size()), ConcatString("4"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  } else {
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "the op verify success");
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(DataFormatDimMap, DataFormatDimMapVerify);

IMPLEMT_COMMON_INFERFUNC(DataFormatDimMapInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter DataFormatDimMapInferShape");
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(DataFormatDimMap, DataFormatDimMapInferShape);
// ----------------DataFormatDimMap End-------------------

// ----------------ApplyProximalAdagrad Op-------------------
IMPLEMT_VERIFIER(ApplyProximalAdagrad, ApplyProximalAdagradVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyProximalAdagrad proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("l1");
  inputScalarList.push_back("l2");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyProximalAdagradInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyProximalAdagrad op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyProximalAdagrad, ApplyProximalAdagradInferShape);
VERIFY_FUNC_REG(ApplyProximalAdagrad, ApplyProximalAdagradVerify);
// ----------------ApplyProximalAdagrad END-------------------

// ----------------ApplyProximalAdagradD Op-------------------
IMPLEMT_VERIFIER(ApplyProximalAdagradD, ApplyProximalAdagradDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyProximalAdagradD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("l1");
  inputScalarList.push_back("l2");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyProximalAdagradDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyProximalAdagradD op_proto inferfunction!");
  if (OneInOneOutDynamicInfer(op, "var", {"var"})) {
    if (OneInOneOutDynamicInfer(op, "accum", {"accum"})) {
    return GRAPH_SUCCESS;
    }
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ApplyProximalAdagradD, ApplyProximalAdagradDInferShape);
VERIFY_FUNC_REG(ApplyProximalAdagradD, ApplyProximalAdagradDVerify);
// ----------------ApplyProximalAdagradD END-------------------

// ----------------SparseApplyProximalAdagrad Op-------------------
// Registered inferfunction
COMMON_INFER_FUNC_REG(SparseApplyProximalAdagrad, ELMTWISE_INFER_SHAPEANDTYPE("var", "var"));
// ----------------SparseApplyProximalAdagrad END-------------------

// ----------------SparseApplyProximalAdagradD Op-------------------
// Registered inferfunction
IMPLEMT_COMMON_INFERFUNC(SparseApplyProximalAdagradDShape) {
  OP_LOGI(op.GetName().c_str(), "Enter SparseApplyProximalAdagradD op_proto inferfunction!");
  // var_out
  TensorDesc var_tensordesc_output = op.GetOutputDesc("var");
  var_tensordesc_output.SetShape(op.GetInputDesc("var").GetShape());
  var_tensordesc_output.SetDataType(op.GetInputDesc("var").GetDataType());
  std::vector<std::pair<int64_t, int64_t>> shape_range_var;
  op.GetInputDesc("var").GetShapeRange(shape_range_var);
  var_tensordesc_output.SetShapeRange(shape_range_var);
  (void)op.UpdateOutputDesc("var", var_tensordesc_output);
  // accum_out
  TensorDesc accum_tensordesc_output = op.GetOutputDesc("accum");
  accum_tensordesc_output.SetShape(op.GetInputDesc("accum").GetShape());
  accum_tensordesc_output.SetDataType(op.GetInputDesc("accum").GetDataType());
  std::vector<std::pair<int64_t, int64_t>> shape_range_accum;
  op.GetInputDesc("accum").GetShapeRange(shape_range_accum);
  accum_tensordesc_output.SetShapeRange(shape_range_accum);
  (void)op.UpdateOutputDesc("accum", accum_tensordesc_output);
  OP_LOGI(op.GetName().c_str(), "Leave SparseApplyProximalAdagradD op_proto inferfunction!");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseApplyProximalAdagradD, SparseApplyProximalAdagradDShape);
// ----------------SparseApplyProximalAdagradD END-------------------

// ----------------ApplyFtrl Op-------------------
IMPLEMT_VERIFIER(ApplyFtrl, ApplyFtrlVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyFtrl proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("linear");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("l1");
  inputScalarList.push_back("l2");
  inputScalarList.push_back("lr_power");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyFtrlInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyFtrl op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyFtrl, ApplyFtrlInferShape);
VERIFY_FUNC_REG(ApplyFtrl, ApplyFtrlVerify);
// ----------------ApplyFtrl END-------------------

// ----------------ApplyFtrlD Op-------------------
IMPLEMT_VERIFIER(ApplyFtrlD, ApplyFtrlDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyFtrlD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("linear");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("l1");
  inputScalarList.push_back("l2");
  inputScalarList.push_back("lr_power");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyFtrlDInferShape) {
    OP_LOGI(op.GetName().c_str(), "Enter ApplyFtrlD op_proto inferfunction!");
    if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
        return GRAPH_FAILED;
    }
    if (!OneInOneOutDynamicInfer(op, "accum", {"accum"})) {
        return GRAPH_FAILED;
    }
    if (!OneInOneOutDynamicInfer(op, "linear", {"linear"})) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyFtrlD, ApplyFtrlDInferShape);
VERIFY_FUNC_REG(ApplyFtrlD, ApplyFtrlDVerify);
// ----------------ApplyFtrlD END-------------------

// ----------------ApplyFtrlV2 Op-------------------
IMPLEMT_VERIFIER(ApplyFtrlV2, ApplyFtrlV2Verify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyFtrlV2 proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("linear");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("l1");
  inputScalarList.push_back("l2");
  inputScalarList.push_back("l2_shrinkage");
  inputScalarList.push_back("lr_power");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyFtrlV2InferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyFtrlV2 op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyFtrlV2, ApplyFtrlV2InferShape);
VERIFY_FUNC_REG(ApplyFtrlV2, ApplyFtrlV2Verify);
// ----------------ApplyFtrlV2 END-------------------

// ----------------ApplyFtrlV2D Op-------------------
IMPLEMT_VERIFIER(ApplyFtrlV2D, ApplyFtrlV2DVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyFtrlV2D proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("linear");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("l1");
  inputScalarList.push_back("l2");
  inputScalarList.push_back("l2_shrinkage");
  inputScalarList.push_back("lr_power");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyFtrlV2DInferShape) {
    OP_LOGI(op.GetName().c_str(), "Enter ApplyFtrlV2D op_proto inferfunction!");
    if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
        return GRAPH_FAILED;
    }
    if (!OneInOneOutDynamicInfer(op, "accum", {"accum"})) {
        return GRAPH_FAILED;
    }
    if (!OneInOneOutDynamicInfer(op, "linear", {"linear"})) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyFtrlV2D, ApplyFtrlV2DInferShape);
VERIFY_FUNC_REG(ApplyFtrlV2D, ApplyFtrlV2DVerify);
// ----------------ApplyFtrlV2D END-------------------

// ----------------ApplyAdadelta Op-------------------
IMPLEMT_VERIFIER(ApplyAdadelta, ApplyAdadeltaVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdadelta proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("accum_update");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("rho");
  inputScalarList.push_back("epsilon");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdadeltaInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdadelta op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdadelta, ApplyAdadeltaInferShape);
VERIFY_FUNC_REG(ApplyAdadelta, ApplyAdadeltaVerify);
// ----------------ApplyAdadelta END-------------------

// ----------------ApplyAdadeltaD Op-------------------
IMPLEMT_VERIFIER(ApplyAdadeltaD, ApplyAdadeltaDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdadeltaD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("accum");
  inputTensorList.push_back("accum_update");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("lr");
  inputScalarList.push_back("rho");
  inputScalarList.push_back("epsilon");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdadeltaDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdadeltaD op_proto inferfunction!");
  if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "accum", {"accum"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "accum_update", {"accum_update"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdadeltaD, ApplyAdadeltaDInferShape);
VERIFY_FUNC_REG(ApplyAdadeltaD, ApplyAdadeltaDVerify);
// ----------------ApplyAdadeltaD END-------------------

// ----------------ApplyAdam Op-------------------
bool ApplyAdamSetNd(Operator& op, std::string& input) {
  auto tensor_desc = op.GetInputDesc(input);
  tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc.SetFormat(ge::FORMAT_ND);

  (void)op.UpdateInputDesc(input, tensor_desc);
  return true;
}

IMPLEMT_INFERFORMAT_FUNC(ApplyAdam, ApplyAdamInferFormat) {
  OP_LOGD(op.GetName().c_str(), "Enter ApplyAdam op_proto infer format function!");
  // scalar edge do set ND (the third to the eighth)
  std::vector<std::string> inputs = {"beta1_power", "beta2_power", "lr", "beta1", "beta2", "epsilon"};
  for (size_t i = 0; i < inputs.size(); i++) {
    if (ApplyAdamSetNd(op, inputs[i]) != true) {
      OP_LOGE(op.GetName().c_str(), "ApplyAdam's scalar input failed to set ND format.");
      return GRAPH_FAILED;
    }
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->DefaultInferFormat();

  for (size_t i = 0; i < inputs.size(); i++) {
    if (ApplyAdamSetNd(op, inputs[i]) != true) {
      OP_LOGE(op.GetName().c_str(), "ApplyAdam's scalar input failed to set ND format.");
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ApplyAdam, ApplyAdamVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdam proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("m");
  inputTensorList.push_back("v");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("beta1_power");
  inputScalarList.push_back("beta2_power");
  inputScalarList.push_back("lr");
  inputScalarList.push_back("beta1");
  inputScalarList.push_back("beta2");
  inputScalarList.push_back("epsilon");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdamInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdam op_proto inferfunction!");
  if (OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ApplyAdam, ApplyAdamInferShape);
VERIFY_FUNC_REG(ApplyAdam, ApplyAdamVerify);
INFER_FORMAT_FUNC_REG(ApplyAdam, ApplyAdamInferFormat);
// ----------------ApplyAdam END-------------------

// ----------------ApplyAdamD Op-------------------
IMPLEMT_VERIFIER(ApplyAdamD, ApplyAdamDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdamD proto verifyFunction!");
  std::vector<std::string> inputTensorList;
  inputTensorList.push_back("var");
  inputTensorList.push_back("m");
  inputTensorList.push_back("v");
  inputTensorList.push_back("grad");
  std::vector<std::string> inputScalarList;
  inputScalarList.push_back("beta1_power");
  inputScalarList.push_back("beta2_power");
  inputScalarList.push_back("lr");
  inputScalarList.push_back("beta1");
  inputScalarList.push_back("beta2");
  inputScalarList.push_back("epsilon");

  if (ApplyVerifyFunc(op, inputTensorList, inputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdamDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdamD op_proto inferfunction!");
  if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "m", {"m"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "v", {"v"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdamD, ApplyAdamDInferShape);
VERIFY_FUNC_REG(ApplyAdamD, ApplyAdamDVerify);
// ----------------ApplyAdamD END-------------------

// ----------------ApplyRMSProp Op-------------------
// Check the dtype, input and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyRMSProp, ApplyRMSPropVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyRMSProp proto verifyFunction!");
  const std::vector<std::string> kInputTensorList{"var", "ms", "mom", "grad"};
  const std::vector<std::string> kInputScalarList{"lr", "rho", "momentum", "epsilon"};

  if (ApplyVerifyFunc(op, kInputTensorList, kInputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyRMSPropInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyRMSProp op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ApplyRMSProp, ApplyRMSPropInferShape);

// Registered verify function
VERIFY_FUNC_REG(ApplyRMSProp, ApplyRMSPropVerify);
// ----------------ApplyRMSProp END-------------------

// ----------------ApplyRMSPropD Op-------------------
// Check the dtype, input and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyRMSPropD, ApplyRMSPropDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyRMSPropD proto verifyFunction!");
  // check input const attr for rho, momentum, epsilon
  std::vector<float> const_attr;
  if (!GetConstAttr(op, {"rho", "momentum", "epsilon"}, const_attr)) {
    OP_LOGE(op.GetName().c_str(), "The GetOpAttr ConstValue failed!");
  }

  const std::vector<std::string> kInputTensorList{"var", "ms", "mom", "grad"};
  const std::vector<std::string> kInputScalarList{"lr"};
  if (ApplyVerifyFunc(op, kInputTensorList, kInputScalarList) == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyRMSPropDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyRMSPropD proto inferfunction!");
  if (OneInOneOutDynamicInfer(op, "var", {"var"})) {
    if (OneInOneOutDynamicInfer(op, "ms", {"ms"})) {
      if (OneInOneOutDynamicInfer(op, "mom", {"mom"})) {
        return GRAPH_SUCCESS;
      }
    }
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ApplyRMSPropD, ApplyRMSPropDInferShape);

// Registered verify function
VERIFY_FUNC_REG(ApplyRMSPropD, ApplyRMSPropDVerify);
// ----------------ApplyRMSPropD END-------------------

// ----------------SparseApplyRMSProp Op-------------------
// Check the dtype, input and attr of the input tensor description.
IMPLEMT_VERIFIER(SparseApplyRMSProp, SparseApplyRMSPropVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter SparseApplyRMSProp verifyFunction!");
  const std::vector<std::string> kInputTensorList{"var", "ms", "mom"};
  const std::vector<std::string> kInputScalarList{"lr", "rho", "momentum", "epsilon"};
  if (!ApplyVerifyFunc(op, kInputTensorList, kInputScalarList)) {
    return GRAPH_FAILED;
  }

  auto vector_dims = op.GetInputDesc("indices").GetShape().GetDims();
  if (vector_dims.size() != 1) {
    OP_LOGE(op.GetName().c_str(), "Input indices must be one-dimensional");
    return GRAPH_FAILED;
  }

  vector<int64_t> var_dims = op.GetInputDesc("var").GetShape().GetDims();
  vector<int64_t> grad_dims = op.GetInputDesc("grad").GetShape().GetDims();

  for (unsigned int dim_index = 1; dim_index < var_dims.size(); dim_index++) {
    if (var_dims[dim_index] != grad_dims[dim_index]) {
      OP_LOGE(op.GetName().c_str(), "Input var and grad must match in dimension (%u)", dim_index);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(SparseApplyRMSPropInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter SparseApplyRMSProp inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(SparseApplyRMSProp, SparseApplyRMSPropInferShape);

// Registered verify function
VERIFY_FUNC_REG(SparseApplyRMSProp, SparseApplyRMSPropVerify);
// ----------------SparseApplyRMSProp END-------------------

// ----------------SparseApplyRMSPropD Op-------------------
// Check the dtype, input and attr of the input tensor description.
IMPLEMT_VERIFIER(SparseApplyRMSPropD, SparseApplyRMSPropDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter SparseApplyRMSPropD verifyFunction!");

  // check input const attr for rho, momentum, epsilon
  std::vector<float> constAttr;
  if (!GetConstAttr(op, {"rho", "momentum", "epsilon"}, constAttr)) {
    std::string err_msg = GetInputInvalidErrMsg("rho, momentum or epsilon");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  const std::vector<std::string> kInputTensorList{"var", "ms", "mom"};
  const std::vector<std::string> kInputScalarList{"lr"};
  if (!ApplyVerifyFunc(op, kInputTensorList, kInputScalarList)) {
    return GRAPH_FAILED;
  }

  auto vector_dims = op.GetInputDesc("indices").GetShape().GetDims();
  if (vector_dims.size() != 1) {
    std::string err_msg = GetShapeSizeErrMsg(5, ConcatString(vector_dims.size()), ConcatString("one-dimensional"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  vector<int64_t> var_dims = op.GetInputDesc("var").GetShape().GetDims();
  vector<int64_t> grad_dims = op.GetInputDesc("grad").GetShape().GetDims();

  for (unsigned int dim_index = 1; dim_index < var_dims.size(); dim_index++) {
    if (var_dims[dim_index] != grad_dims[dim_index]) {
      std::string err_msg = OtherErrMsg(ConcatString("Input var and grad must match in dimension ", dim_index));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(SparseApplyRMSPropDInferShape) {
  ApplyInferShapeAndDtype(op, "var", "var");
  ApplyInferShapeAndDtype(op, "accum", "accum");
  DynamicApplyInferShapeRange(op, "var", "var");
  DynamicApplyInferShapeRange(op, "accum", "accum");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseApplyRMSPropD, SparseApplyRMSPropDInferShape);
VERIFY_FUNC_REG(SparseApplyRMSPropD, SparseApplyRMSPropDVerify);

// ----------------SparseApplyRMSPropD Op End-------------------

// ----------------SparseApplyAdadelta Op Begin-------------------
// Check the dtype, input and attr of the input tensor description.
IMPLEMT_VERIFIER(SparseApplyAdadelta, SparseApplyAdadeltaVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter SparseApplyAdadelta verifyFunction!");
  const std::vector<std::string> kInputTensorList = {"var", "accum", "accum_update"};
  const std::vector<std::string> kInputScalarList = {"lr", "rho", "epsilon"};
  if (!ApplyVerifyFunc(op, kInputTensorList, kInputScalarList)) {
    return GRAPH_FAILED;
  }

  auto vector_dims = op.GetInputDesc("indices").GetShape().GetDims();
  if (vector_dims.size() != 1) {
    OP_LOGE(op.GetName().c_str(), "Input indices must be one-dimensional");
    return GRAPH_FAILED;
  }

  vector<int64_t> var_dims = op.GetInputDesc("var").GetShape().GetDims();
  vector<int64_t> grad_dims = op.GetInputDesc("grad").GetShape().GetDims();

  for (unsigned int dim_index = 1; dim_index < var_dims.size(); dim_index++) {
    if (var_dims[dim_index] != grad_dims[dim_index]) {
      OP_LOGE(op.GetName().c_str(), "Input var and grad must match in dimension (%u)", dim_index);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(SparseApplyAdadeltaInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter SparseApplyAdadelta inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(SparseApplyAdadelta, SparseApplyAdadeltaInferShape);

// Registered verify function
VERIFY_FUNC_REG(SparseApplyAdadelta, SparseApplyAdadeltaVerify);
// ----------------SparseApplyAdadelta op END-------------------

// ----------------SparseApplyAdadeltaD Op Begin-------------------
// Check the dtype, input and attr of the input tensor description.
IMPLEMT_VERIFIER(SparseApplyAdadeltaD, SparseApplyAdadeltaDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter SparseApplyAdadeltaD verifyFunction!");

  // check input const attr for rho, epsilon
  std::vector<float> const_attr;
  if (!GetConstAttr(op, {"epsilon"}, const_attr)) {
    std::string err_msg = GetInputInvalidErrMsg("epsilon");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  const std::vector<std::string> kInputTensorList = {"var", "accum", "accum_update"};
  const std::vector<std::string> kInputScalarList = {"lr", "rho"};
  if (!ApplyVerifyFunc(op, kInputTensorList, kInputScalarList)) {
    return GRAPH_FAILED;
  }

  auto vector_dims = op.GetInputDesc("indices").GetShape().GetDims();
  if (vector_dims.size() != 1) {
    std::string err_msg = GetShapeSizeErrMsg(6, ConcatString(vector_dims.size()), ConcatString(1));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  vector<int64_t> var_dims = op.GetInputDesc("var").GetShape().GetDims();
  vector<int64_t> grad_dims = op.GetInputDesc("grad").GetShape().GetDims();

  for (unsigned int dim_index = 1; dim_index < var_dims.size(); dim_index++) {
    if (var_dims[dim_index] != grad_dims[dim_index]) {
      std::string err_msg = OtherErrMsg(ConcatString("Input var and grad must match in dimension ", dim_index));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(SparseApplyAdadeltaDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter SparseApplyAdadeltaD inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  ApplyInferShapeAndDtype(op, "accum", "accum");
  ApplyInferShapeAndDtype(op, "accum", "accum_update");
  DynamicApplyInferShapeRange(op, "var", "var");
  DynamicApplyInferShapeRange(op, "accum", "accum");
  DynamicApplyInferShapeRange(op, "accum", "accum_update");
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(SparseApplyAdadeltaD, SparseApplyAdadeltaDInferShape);

// Registered verify function
VERIFY_FUNC_REG(SparseApplyAdadeltaD, SparseApplyAdadeltaDVerify);
// ----------------SparseApplyAdadeltaD END-------------------

// ----------------SGD------------------
bool CheckSgdDimension(const Operator& op) {
  OP_LOGI(op.GetName().c_str(), "Enter SGD op_proto inferfunction!");
  // input tensor dim check
  auto var_dims = op.GetInputDesc("parameters").GetShape().GetDims();
  if (var_dims.size() > 8 || var_dims.size() <= 0) {
    std::string err_msg = GetShapeSizeErrMsg(0, ConcatString(var_dims.size()), ConcatString("1 ~ 8 dims!"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return false;
  }
  return true;
}

// Check tensor description.
IMPLEMT_VERIFIER(SGD, SGDVerify) {
  if (!CheckSgdDimension(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(SGDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter SGD op_proto inferfunction!");
  float dampening, weight_decay;
  bool nesterov{false};
  if (op.GetAttr("dampening", dampening) == GRAPH_SUCCESS && op.GetAttr("nesterov", nesterov) == GRAPH_SUCCESS) {
    if (nesterov) {
      if (dampening != 0.0) {
        std::string err_msg = OtherErrMsg(ConcatString("Attr dampening(", dampening, ") must == 0 when nesterov == true"));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    }
  }

  if (op.GetAttr("weight_decay", weight_decay) == GRAPH_SUCCESS) {
    if (weight_decay < 0) {
      std::string err_msg = GetAttrValueErrMsg("weight_decay", ConcatString(weight_decay), ConcatString("more than or equal to 0"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }
  TensorDesc variable_desc = op.GetInputDesc("parameters");
  auto variabl_shape = variable_desc.GetShape().GetDims();
  DataType variabl_dtype = variable_desc.GetDataType();

  TensorDesc variable_update_desc = op.GetOutputDesc("parameters");
  variable_update_desc.SetShape(Shape(variabl_shape));
  variable_update_desc.SetDataType(variabl_dtype);
  if (op.UpdateOutputDesc("parameters", variable_update_desc) != GRAPH_SUCCESS) {
    std::string err_msg = UpdateParamErrMsg("parameters");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(SGD, SGDInferShape);
// Registered verify function
VERIFY_FUNC_REG(SGD, SGDVerify);
// ----------------SGD End------------------

// ----------------FusedMulApplyMomentum-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(FusedMulApplyMomentum, FusedMulApplyMomentumVerify) {
  const std::map<std::string, std::vector<DataType>> kInputTensorMap = {
      {"var", {DT_FLOAT16, DT_FLOAT}}, {"accum", {DT_FLOAT16, DT_FLOAT}}, {"x1", {DT_FLOAT16, DT_FLOAT}}};
  const std::map<std::string, std::vector<DataType>> kInputScalarMap = {{"lr", {DT_FLOAT16, DT_FLOAT}},
                                                                        {"momentum", {DT_FLOAT16, DT_FLOAT}}};

  if (!CheckInputDataType(op, "x2", {DT_FLOAT16, DT_FLOAT})) {
    return GRAPH_FAILED;
  }
  // input tensor params, must have same shape and dtype
  if (!CheckInputDtypeAndShape(op, kInputTensorMap) || !CheckInputDtypeAndShape(op, kInputScalarMap)) {
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "The op verify end");
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(FusedMulApplyMomentumInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter FusedMulApplyMomentum op_proto inferfunction!");
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "var", "accum", {"var", "accum"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(FusedMulApplyMomentum, FusedMulApplyMomentumInferShape);
VERIFY_FUNC_REG(FusedMulApplyMomentum, FusedMulApplyMomentumVerify);

// ----------------FusedMulApplyMomentumExtern-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(FusedMulApplyMomentumExtern, FusedMulApplyMomentumExternVerify) {
  if (!CheckInputDataType(op, "var", {DT_FLOAT})) {
    return GRAPH_FAILED;
  }
  if (!CheckInputDataType(op, "var_copy", {DT_FLOAT16})) {
    return GRAPH_FAILED;
  }
  const std::map<std::string, std::vector<DataType>> kInputTensorMap = {{"accum", {DT_FLOAT16, DT_FLOAT}},
                                                                        {"x1", {DT_FLOAT16, DT_FLOAT}}};
  const std::map<std::string, std::vector<DataType>> kInputScalarMap = {
      {"lr", {DT_FLOAT16, DT_FLOAT}}, {"momentum", {DT_FLOAT16, DT_FLOAT}}, {"x2", {DT_FLOAT16, DT_FLOAT}}};

  // input tensor params, must have same shape and dtype
  if (!CheckInputDtypeAndShape(op, kInputTensorMap) || !CheckInputDtypeAndShape(op, kInputScalarMap)) {
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(FusedMulApplyMomentumExternInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter FusedMulApplyMomentumExtern op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  ApplyInferShapeAndDtype(op, "var_copy", "var_copy");
  ApplyInferShapeAndDtype(op, "accum", "accum");
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(FusedMulApplyMomentumExtern, FusedMulApplyMomentumExternInferShape);

VERIFY_FUNC_REG(FusedMulApplyMomentumExtern, FusedMulApplyMomentumExternVerify);

// ----------------FusedMulApplyKerasMomentum-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(FusedMulApplyKerasMomentum, FusedMulApplyKerasMomentumVerify) {
  const std::map<std::string, std::vector<DataType>> kInputTensorMap = {
      {"var", {DT_FLOAT16, DT_FLOAT}}, {"accum", {DT_FLOAT16, DT_FLOAT}}, {"x1", {DT_FLOAT16, DT_FLOAT}}};
  const std::map<std::string, std::vector<DataType>> kInputScalarMap = {{"lr", {DT_FLOAT16, DT_FLOAT}},
                                                                        {"momentum", {DT_FLOAT16, DT_FLOAT}}};

  if (!CheckInputDataType(op, "x2", {DT_FLOAT16, DT_FLOAT})) {
    return GRAPH_FAILED;
  }
  // input tensor params, must have same shape and dtype
  if (!CheckInputDtypeAndShape(op, kInputTensorMap) || !CheckInputDtypeAndShape(op, kInputScalarMap)) {
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "The op verify end");
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(FusedMulApplyKerasMomentumInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter FusedMulApplyKerasMomentum op_proto inferfunction!");
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "var", "accum", {"var", "accum"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(FusedMulApplyKerasMomentum, FusedMulApplyKerasMomentumInferShape);
VERIFY_FUNC_REG(FusedMulApplyKerasMomentum, FusedMulApplyKerasMomentumVerify);

// ----------------LarsV2 Begin------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(LarsV2, LarsV2InferVerify) {
  std::map<std::string, std::vector<DataType>> inputTensorMap = {
      {"g", {DT_FLOAT}},
      {"w", {DT_FLOAT}},
  };

  // input tensor params, must have same shape and dtype
  if (!CheckInputDtypeAndShape(op, inputTensorMap)) {
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LarsV2, ELMTWISE_INFER_SHAPEANDTYPE("g", "g_new"));
VERIFY_FUNC_REG(LarsV2, LarsV2InferVerify);
// ----------------LarsV2 End-------------------

// ----------------LarsV2Update Begin------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(LarsV2Update, LarsV2UpdateInferVerify) {
  const std::map<std::string, std::vector<DataType>> kInputTensorMap = {{"g", {DT_FLOAT}}, {"w", {DT_FLOAT}}};

  // input tensor params, must have same shape and dtype
  if (!CheckInputDtypeAndShape(op, kInputTensorMap)) {
    return GRAPH_FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "The op verify end");
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(LarsV2UpdateInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "w", "g", "g_new", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LarsV2Update, LarsV2UpdateInferShape);
VERIFY_FUNC_REG(LarsV2Update, LarsV2UpdateInferVerify);
// ----------------LarsV2Update End-------------------

// ----------------SparseApplyFtrl-------------------
void DynamicApplyInferShapeAndDtype(Operator& op, const string& input_name, const string& output_name) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr Input_tensor_desc = op_desc->MutableInputDesc(input_name);
  GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc(output_name);

  auto input_shape = Input_tensor_desc->GetShape();
  DataType input_type = Input_tensor_desc->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> input_shape_range;
  Input_tensor_desc->GetShapeRange(input_shape_range);

  output_tensor_desc->SetShape(input_shape);
  output_tensor_desc->SetDataType(input_type);
  output_tensor_desc->SetShapeRange(input_shape_range);
}

IMPLEMT_COMMON_INFERFUNC(SparseApplyFtrlDInferShape) {
  DynamicApplyInferShapeAndDtype(op, "var", "var");
  DynamicApplyInferShapeAndDtype(op, "accum", "accum");
  DynamicApplyInferShapeAndDtype(op, "linear", "linear");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseApplyFtrlD, SparseApplyFtrlDVerify) {
  DataType var_dtype = op.GetInputDesc(0).GetDataType();
  DataType accum_dtype = op.GetInputDesc(1).GetDataType();
  if (var_dtype != accum_dtype) {
    std::string err_msg = OtherErrMsg("The sparse_apply_ftrl op inputs should have the same dtype!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  float lr, l1, l2, lr_power;
  if (op.GetAttr("lr", lr) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("lr");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (lr <= 0) {
    std::string err_msg = GetAttrValueErrMsg("lr", ConcatString(lr), ConcatString("more than 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.GetAttr("l1", l1) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("l1");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (l1 < 0) {
    std::string err_msg = GetAttrValueErrMsg("l1", ConcatString(l1), ConcatString("more than or equal to 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.GetAttr("l2", l2) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("l2");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (l2 < 0) {
    std::string err_msg = GetAttrValueErrMsg("l2", ConcatString(l2), ConcatString("more than or equal to 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.GetAttr("lr_power", lr_power) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("lr_power");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (lr_power > 0) {
    std::string err_msg = GetAttrValueErrMsg("lr_power", ConcatString(lr_power), ConcatString("less than or equal to 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseApplyFtrlD, SparseApplyFtrlDInferShape);
VERIFY_FUNC_REG(SparseApplyFtrlD, SparseApplyFtrlDVerify);

IMPLEMT_COMMON_INFERFUNC(SparseApplyFtrlInferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc out_tensor_desc = op.GetOutputDesc("var");
  out_tensor_desc.SetShape(Shape(var_shape));
  out_tensor_desc.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("var", out_tensor_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseApplyFtrl, SparseApplyFtrlVerify) {
  DataType var_dtype = op.GetInputDesc(0).GetDataType();
  DataType accum_dtype = op.GetInputDesc(1).GetDataType();
  if (var_dtype != accum_dtype) {
    OP_LOGE(op.GetName().c_str(), "The sparse_apply_ftrl op inputs should have the same dtype!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseApplyFtrl, SparseApplyFtrlInferShape);
VERIFY_FUNC_REG(SparseApplyFtrl, SparseApplyFtrlVerify);

// ----------------SparseApplyFtrl end-------------------

// ----------------SparseApplyFtrlV2-------------------

IMPLEMT_COMMON_INFERFUNC(SparseApplyFtrlV2DInferShape) {
  ApplyInferShapeAndDtype(op, "var", "var");
  ApplyInferShapeAndDtype(op, "accum", "accum");
  ApplyInferShapeAndDtype(op, "linear", "linear");
  DynamicApplyInferShapeRange(op, "var", "var");
  DynamicApplyInferShapeRange(op, "accum", "accum");
  DynamicApplyInferShapeRange(op, "linear", "linear");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseApplyFtrlV2D, SparseApplyFtrlV2DVerify) {
  DataType var_dtype = op.GetInputDesc("var").GetDataType();
  DataType accum_dtype = op.GetInputDesc("accum").GetDataType();
  if (var_dtype != accum_dtype) {
    std::string err_msg = OtherErrMsg("The sparse_apply_ftrl op inputs should have the same dtype!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  float lr, l1, l2, l2_shrinkage, lr_power;
  if (op.GetAttr("lr", lr) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("lr");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (lr <= 0) {
    std::string err_msg = GetAttrValueErrMsg("lr", ConcatString(lr), ConcatString("more than 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.GetAttr("l1", l1) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("l1");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (l1 < 0) {
    std::string err_msg = GetAttrValueErrMsg("l1", ConcatString(l1), ConcatString("more than or equal to 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.GetAttr("l2", l2) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("l2");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (l2 < 0) {
    std::string err_msg = GetAttrValueErrMsg("l2", ConcatString(l2), ConcatString("more than or equal to 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.GetAttr("l2_shrinkage", l2_shrinkage) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("l2_shrinkage");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (l2_shrinkage < 0) {
    std::string err_msg = GetAttrValueErrMsg("l2_shrinkage", ConcatString(l2_shrinkage), ConcatString("more than or equal to 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.GetAttr("lr_power", lr_power) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("lr_power");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (lr_power < 0) {
    std::string err_msg = GetAttrValueErrMsg("lr_power", ConcatString(lr_power), ConcatString("more than or equal to 0"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseApplyFtrlV2D, SparseApplyFtrlV2DInferShape);
VERIFY_FUNC_REG(SparseApplyFtrlV2D, SparseApplyFtrlV2DVerify);

IMPLEMT_COMMON_INFERFUNC(SparseApplyFtrlV2InferShape) {
  Shape var_shape = op.GetInputDesc("var").GetShape();
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc out_tensor_desc = op.GetOutputDesc("var");
  out_tensor_desc.SetShape(ge::Shape(var_shape));
  out_tensor_desc.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("var", out_tensor_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseApplyFtrlV2, SparseApplyFtrlV2Verify) {
  DataType var_dtype = op.GetInputDesc("var").GetDataType();
  DataType accum_dtype = op.GetInputDesc("accum").GetDataType();
  if (var_dtype != accum_dtype) {
    OP_LOGE(op.GetName().c_str(), "The sparse_apply_ftrl op inputs should have the same dtype!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseApplyFtrlV2, SparseApplyFtrlV2InferShape);
VERIFY_FUNC_REG(SparseApplyFtrlV2, SparseApplyFtrlV2Verify);

// ----------------SparseApplyFtrlV2 end-------------------

// ----------------ApplyAdamWithAmsgrad Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyAdamWithAmsgrad, ApplyAdamWithAmsgradVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdamWithAmsgrad proto verifyFunction!");
  const std::vector<std::string> kInputTensorList{"var", "m", "v", "vhat", "grad"};
  const std::vector<std::string> kInputScalarList{"lr", "beta1", "beta2", "beta1_power", "beta2_power", "epsilon"};

  if (!ApplyVerifyFunc(op, kInputTensorList, kInputScalarList)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdamWithAmsgradInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdamWithAmsgrad op_proto inferfunction!");
  ApplyInferShapeAndDtype(op, "var", "var");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdamWithAmsgrad, ApplyAdamWithAmsgradInferShape);
VERIFY_FUNC_REG(ApplyAdamWithAmsgrad, ApplyAdamWithAmsgradVerify);
// ----------------ApplyAdamWithAmsgrad END-------------------

// ----------------ApplyAdamWithAmsgradD Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(ApplyAdamWithAmsgradD, ApplyAdamWithAmsgradDVerify) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdamWithAmsgrad proto verifyFunction!");
  const std::vector<std::string> kInputTensorList{"var", "m", "v", "vhat", "grad"};
  const std::vector<std::string> kInputScalarList{"lr", "beta1_power", "beta2_power"};
  // check input const attr for beta1, beta2, epsilon
  std::vector<float> const_attr;
  if (!GetConstAttr(op, {"beta1", "beta2", "epsilon"}, const_attr)) {
    OP_LOGE(op.GetName().c_str(), "The GetOpAttr ConstValue failed!");
    return GRAPH_FAILED;
  }
  if (!ApplyVerifyFunc(op, kInputTensorList, kInputScalarList)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ApplyAdamWithAmsgradDInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter ApplyAdamWithAmsgrad op_proto inferfunction!");
  if (!OneInOneOutDynamicInfer(op, "var", {"var"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "m", {"m"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "v", {"v"})) {
    return GRAPH_FAILED;
  }
  if (!OneInOneOutDynamicInfer(op, "vhat", {"vhat"})) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ApplyAdamWithAmsgradD, ApplyAdamWithAmsgradDInferShape);
VERIFY_FUNC_REG(ApplyAdamWithAmsgradD, ApplyAdamWithAmsgradDVerify);
// ----------------ApplyAdamWithAmsgrad END-------------------

}  // namespace ge
