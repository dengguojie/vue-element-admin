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
 * \file nn_norm_ops.cpp
 * \brief
 */
#include "inc/nn_norm_ops.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"
#include "graph/utils/node_utils.h"
#include "common/inc/op_log.h"

namespace ge {
// --------------------------LogSoftmaxGrad-------------------------
IMPLEMT_COMMON_INFERFUNC(LogSoftmaxGradInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");

  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LogSoftmaxGrad, LogSoftmaxGradInferShape);
// --------------------------LogSoftmaxGrad END---------------------

//-------------SparseSoftmaxCrossEntropyWithLogits----------------
IMPLEMT_VERIFIER(SparseSoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogitsVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(SparseSoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogitsInferShape) {
  TensorDesc tensordesc_loss = op.GetOutputDesc(0);
  TensorDesc tensordesc_backprop = op.GetOutputDesc(1);
  ge::Shape shape_fetures = op.GetInputDesc(0).GetShape();
  ge::Shape shape_labels = op.GetInputDesc(1).GetShape();
  tensordesc_backprop.SetShape(shape_fetures);
  tensordesc_loss.SetShape(shape_labels);
  DataType input_dtype = op.GetInputDesc(0).GetDataType();
  tensordesc_loss.SetDataType(input_dtype);
  tensordesc_backprop.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("loss", tensordesc_loss);
  (void)op.UpdateOutputDesc("backprop", tensordesc_backprop);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseSoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogitsInferShape);
VERIFY_FUNC_REG(SparseSoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogitsVerify);
//-------------SparseSoftmaxCrossEntropyWithLogits----------------

// ---------------------------SoftmaxV2-----------------------------
IMPLEMT_COMMON_INFERFUNC(SoftmaxV2InferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(SoftmaxV2, SoftmaxV2InferShape);
// --------------------------SoftmaxV2 END------------------------------

// ---------------------------LogSoftmaxV2------------------------------
IMPLEMT_COMMON_INFERFUNC(LogSoftmaxV2InferShape) {
  if (OneInOneOutDynamicInfer(op, "logits", {"logsoftmax"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(LogSoftmaxV2, LogSoftmaxV2InferShape);
// -------------------------LogSoftmaxV2 END----------------------------

// ----------------SigmoidCrossEntropyWithLogitsGrad-------------------
IMPLEMT_COMMON_INFERFUNC(SigmoidCrossEntropyWithLogitsGradInferShape) {
  if (TwoInOneOutDynamicInferNoBroadcast(op, "predict", "target", {"gradient"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(SigmoidCrossEntropyWithLogitsGrad, SigmoidCrossEntropyWithLogitsGradInferShape);
// ---------------SigmoidCrossEntropyWithLogitsGrad END-----------------

// -------------------SigmoidCrossEntropyWithLogits---------------------
IMPLEMT_COMMON_INFERFUNC(SigmoidCrossEntropyWithLogitsInferShape) {
  if (TwoInOneOutDynamicInferNoBroadcast(op, "predict", "target", {"loss"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogitsInferShape);
// ------------------SigmoidCrossEntropyWithLogits END------------------

// -------------------SigmoidCrossEntropyWithLogitsV2---------------------
IMPLEMT_COMMON_INFERFUNC(SigmoidCrossEntropyWithLogitsV2InferShape) {

  std::string reduction = "mean";
  if (op.GetAttr("reduction", reduction) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("reduction");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (reduction == "none") {
        if (OneInOneOutDynamicInfer(op, "predict", {"loss"})) {
        return GRAPH_SUCCESS;
    }
        return GRAPH_FAILED;  
  } 
  else {
      // if reduction == "mean" or reduction == "sum" , output a scalar
      auto op_info = OpDescUtils::GetOpDescFromOperator(op);
      auto outputTensordesc = op_info->MutableOutputDesc("loss");
      auto predict_desc = op_info->MutableInputDesc("predict");
      DataType predict_dtype = predict_desc->GetDataType();
      std::vector<int64_t> o_shape;
      outputTensordesc->SetShape(GeShape(o_shape));
      outputTensordesc->SetDataType(predict_dtype);
      return GRAPH_SUCCESS;
    }
  }

COMMON_INFER_FUNC_REG(SigmoidCrossEntropyWithLogitsV2, SigmoidCrossEntropyWithLogitsV2InferShape);
// ------------------SigmoidCrossEntropyWithLogitsV2 END------------------

// ----------------SmoothL1LossGrad-------------------
IMPLEMT_COMMON_INFERFUNC(SmoothL1LossGradInferShape) {
  if (OneInOneOutDynamicInfer(op, "predict", {"gradient"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(SmoothL1LossGrad, SmoothL1LossGradInferShape);
// ----------------SmoothL1LossGrad END-------------------

// ----------------Roll Begin---------------------
IMPLEMT_COMMON_INFERFUNC(RollInferShape) {
  TensorDesc output_desc_y = op.GetOutputDesc("y");
  DataType predict_dtype = op.GetInputDesc("x").GetDataType();
  Format predict_format = op.GetInputDesc("x").GetFormat();
  ge::Shape output_shape = op.GetInputDesc("x").GetShape();
  output_desc_y.SetDataType(predict_dtype);
  output_desc_y.SetFormat(predict_format);
  output_desc_y.SetShape(output_shape);
  (void)op.UpdateOutputDesc("y", output_desc_y);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(Roll, RollInferShape);
// ----------------Roll END---------------------

// ----------------SmoothL1Loss-------------------
IMPLEMT_COMMON_INFERFUNC(SmoothL1LossInferShape) {
  if (OneInOneOutDynamicInfer(op, "predict", {"loss"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(SmoothL1Loss, SmoothL1LossInferShape);
// ----------------SmoothL1Loss END-------------------

// --------------------------BinaryCrossEntropy-------------------------
IMPLEMT_COMMON_INFERFUNC(BinaryCrossEntropyInferShape) {
  std::string reduceType = "mean";
  if (op.GetAttr("reduction", reduceType) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("reduction");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (reduceType == "none") {
    // if reduction == "none" , output shape == x.shape
    OP_LOGI(op.GetName().c_str(), "the attr reduction = none");
    if (OneInOneOutDynamicInfer(op, "x", {"output"})){
      return GRAPH_SUCCESS;
    }
    return GRAPH_SUCCESS;
  } else {
    // if reduction == "mean" or reduction == "sum" , output a scalar
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    if (op_info == nullptr) {
      std::string err_msg = GetAttrValueErrMsg("op_info", ConcatString(op_info), "not be nullptr");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    auto outputTensordesc = op_info->MutableOutputDesc("output");
    auto x_desc = op_info->MutableInputDesc("x");
    DataType x_dtype = x_desc->GetDataType();
    std::vector<int64_t> oShape;
    std::vector<std::pair<int64_t, int64_t>> oRange;
    outputTensordesc->SetShape(GeShape(oShape));
    outputTensordesc->SetShapeRange(oRange);
    outputTensordesc->SetDataType(x_dtype);
    return GRAPH_SUCCESS;
  }
}

COMMON_INFER_FUNC_REG(BinaryCrossEntropy, BinaryCrossEntropyInferShape);
// --------------------------BinaryCrossEntropy END---------------------

// --------------------------BinaryCrossEntropyGrad-------------------------
IMPLEMT_COMMON_INFERFUNC(BinaryCrossEntropyGradInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"output"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(BinaryCrossEntropyGrad, BinaryCrossEntropyGradInferShape);
// --------------------------BinaryCrossEntropyGrad END---------------------

//----------------SoftmaxCrossEntropyWithLogits-------------------
IMPLEMT_VERIFIER(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogitsVerify) {
  if (!CheckTwoInputDtypeSame(op, "features", "labels")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SoftmaxCrossEntropyWithLogitsInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "features", "labels", "backprop", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto backprop_output = op_desc->MutableOutputDesc("backprop");
  auto loss_output = op_desc->MutableOutputDesc("loss");

  auto backprop_shape = backprop_output->GetShape().GetDims();
  auto backprop_dtype = backprop_output->GetDataType();
  if (backprop_shape.size() > 1) {
    backprop_shape.pop_back();
  }
  loss_output->SetShape(GeShape(backprop_shape));
  loss_output->SetDataType(backprop_dtype);

  if (is_dynamic_output) {
    std::vector<std::pair<int64_t, int64_t>> backprop_range;
    backprop_output->GetShapeRange(backprop_range);
    if (backprop_range.size() > 1) {
      backprop_range.pop_back();
    }
    loss_output->SetShapeRange(backprop_range);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogitsInferShape);
VERIFY_FUNC_REG(SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogitsVerify);
// ----------------SoftmaxCrossEntropyWithLogits END---------------------

// ----------------Centralization-------------------
IMPLEMT_COMMON_INFERFUNC(CentralizationInferShape) {
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Centralization, CentralizationInferShape);
// ----------------Centralization END-------------------

// -----------------------------SoftmaxGrad------------------------------
IMPLEMT_COMMON_INFERFUNC(SoftmaxGradInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "softmax", "grad_softmax", "grad_x")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

IMPLEMT_VERIFIER(SoftmaxGrad, SoftmaxGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "softmax", "grad_softmax")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SoftmaxGrad, SoftmaxGradInferShape);
VERIFY_FUNC_REG(SoftmaxGrad, SoftmaxGradVerify);
// ------------------------------SoftmaxGrad END--------------------------

// --------------------------ConfusionSoftmaxGrad-------------------------
IMPLEMT_COMMON_INFERFUNC(ConfusionSoftmaxGradInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "grad", "x", "y") == false) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ConfusionSoftmaxGrad, ConfusionSoftmaxGradInferShape);
// ------------------------ConfusionSoftmaxGrad END-----------------------

// --------------------------SoftmaxGradExt-------------------------
IMPLEMT_COMMON_INFERFUNC(SoftmaxGradExtInferShape) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SoftmaxGradExt, SoftmaxGradExtInferShape);
// ------------------------SoftmaxGradExt END-----------------------

//------------------------MVN---------------------------
IMPLEMT_INFERFUNC(MVN, MVNInferShape) {
  auto outShape = op.GetInputDesc("x").GetShape();
  auto outDtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(outShape);
  td.SetDataType(outDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MVN, MVNVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MVN, MVNInferShape);
VERIFY_FUNC_REG(MVN, MVNVerify);
//-----------------------MVN---------------------------

//------------------------MVNV2---------------------------
IMPLEMT_INFERFUNC(MVNV2, MVNV2InferShape) {
  auto outShape = op.GetInputDesc("x").GetShape();
  auto outDtype = op.GetInputDesc("x").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(outShape);
  td.SetDataType(outDtype);
  CHECK(op.UpdateOutputDesc("y", td) != GRAPH_SUCCESS,
          GE_OP_LOGE(GRAPH_FAILED, "Update output desc of node[MVNV2] failed."), return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MVNV2, MVNV2Verify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MVNV2, MVNV2InferShape);
VERIFY_FUNC_REG(MVNV2, MVNV2Verify);
//-----------------------MVNV2---------------------------

//------------------------Normalize---------------------------
IMPLEMT_INFERFUNC(Normalize, NormalizeInfer) {
  auto input_dType = op.get_input_desc_x1().GetDataType();
  auto output_dType = input_dType;
  op.update_output_desc_y(
      TensorDesc(Shape(op.get_input_desc_x1().GetShape()), op.get_input_desc_x1().GetFormat(), output_dType));
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Normalize, NormalizeVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Normalize, NormalizeInfer);
VERIFY_FUNC_REG(Normalize, NormalizeVerify);
//------------------------Normalize---------------------------

//----------------------Renorm begin------------------------
IMPLEMT_COMMON_INFERFUNC(RenormInferShape) {
    TensorDesc output_desc = op.GetOutputDesc("y");
    DataType predict_dtype = op.GetInputDesc("x").GetDataType();
    Format predict_format = op.GetInputDesc("x").GetFormat();
    ge::Shape output_shape = op.GetInputDesc("x").GetShape();
    int64_t dim;
    op.GetAttr("dim", dim);
    for (int64_t i = 0; i < output_shape.GetDimNum(); i++) {
        if (i != dim) {
            output_shape.SetDim(i, 1);
        }
    }
    output_desc.SetDataType(predict_dtype);
    output_desc.SetFormat(predict_format);
    output_desc.SetShape(output_shape);
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Renorm, RenormVerify) {
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Renorm, RenormInferShape);
VERIFY_FUNC_REG(Renorm, RenormVerify);
//----------------------Renorm end------------------------

// ----------------------LayerNormGrad------------------------
IMPLEMT_COMMON_INFERFUNC(LayerNormGradInferShape) {
  TensorDesc td_output_pd_x = op.GetOutputDesc("pd_x");
  TensorDesc td_output2_pd_gamma = op.GetOutputDesc("pd_gamma");
  TensorDesc td_output3_pd_beta = op.GetOutputDesc("pd_beta");

  td_output_pd_x.SetShape(op.GetInputDesc("dy").GetShape());
  td_output_pd_x.SetDataType(op.GetInputDesc("dy").GetDataType());

  td_output2_pd_gamma.SetShape(op.GetInputDesc("gamma").GetShape());
  td_output2_pd_gamma.SetDataType(op.GetInputDesc("gamma").GetDataType());

  td_output3_pd_beta.SetShape(op.GetInputDesc("gamma").GetShape());
  td_output3_pd_beta.SetDataType(op.GetInputDesc("gamma").GetDataType());

  (void)op.UpdateOutputDesc("pd_x", td_output_pd_x);
  (void)op.UpdateOutputDesc("pd_gamma", td_output2_pd_gamma);
  (void)op.UpdateOutputDesc("pd_beta", td_output3_pd_beta);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LayerNormGrad, LayerNormGradInferShape);
// --------------------LayerNormGrad END----------------------

// ------------------------LayerNorm--------------------------
IMPLEMT_COMMON_INFERFUNC(LayerNormInferShape) {
  TensorDesc td_output_y = op.GetOutputDesc("y");
  TensorDesc td_output2_mean = op.GetOutputDesc("mean");
  TensorDesc td_output3_variance = op.GetOutputDesc("variance");

  ge::Shape output_shape1 = op.GetInputDesc("x").GetShape();
  ge::Shape output_shape2 = op.GetInputDesc("x").GetShape();

  int64_t begin_norm_axis = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("begin_norm_axis", begin_norm_axis)) {
    OpsGetAttrErrReport(op.GetName(), "begin_norm_axis");
    OP_LOGE("GetOpAttr beginNormAxis failed!");
    return GRAPH_FAILED;
  }

  size_t real_dim_num = output_shape1.GetDimNum();
  if (begin_norm_axis >= (int64_t)real_dim_num) {
    string excepted_value = ConcatString("less than x's dims [", (int64_t)real_dim_num, "]");
    OpsAttrValueErrReport(op.GetName(), "begin_norm_axis", excepted_value, ConcatString(begin_norm_axis));
    OP_LOGE(
        "[Plugin][ERROR]the op layernorm do not support beginNormAxis"
        "(%ld) large than shape dims(%lu)",
        begin_norm_axis, real_dim_num);
    return GRAPH_FAILED;
  }
  if (begin_norm_axis == -1) {
    begin_norm_axis = real_dim_num - 1;
  }
  for (size_t i = 0; i < real_dim_num; ++i) {
    if (i >= (size_t)begin_norm_axis) {
      output_shape2.SetDim(i, 1);
    } else {
      output_shape2.SetDim(i, output_shape1.GetDim(i));
    }
  }
  td_output_y.SetShape(output_shape1);
  td_output_y.SetDataType(op.GetInputDesc("x").GetDataType());

  td_output2_mean.SetShape(output_shape2);
  td_output2_mean.SetDataType(op.GetInputDesc("gamma").GetDataType());

  td_output3_variance.SetShape(output_shape2);
  td_output3_variance.SetDataType(op.GetInputDesc("gamma").GetDataType());

  (void)op.UpdateOutputDesc("y", td_output_y);
  (void)op.UpdateOutputDesc("mean", td_output2_mean);
  (void)op.UpdateOutputDesc("variance", td_output3_variance);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LayerNorm, LayerNormInferShape);
// ----------------------LayerNorm END--------------------------

// ----------------LayerNormBetaGammaBackprop--------------------
IMPLEMT_COMMON_INFERFUNC(LayerNormBetaGammaBackpropInferShape) {
  std::vector<int64_t> dims_tm;
  if (op.GetAttr("shape_gamma", dims_tm) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("shape_gamma");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }
  Shape valid_shape(dims_tm);

  TensorDesc tensordesc_output0_pd_gamma = op.GetOutputDesc("pd_gamma");
  TensorDesc tensordesc_output1_pd_beta = op.GetOutputDesc("pd_beta");

  tensordesc_output0_pd_gamma.SetShape(valid_shape);
  tensordesc_output0_pd_gamma.SetDataType(DT_FLOAT);

  tensordesc_output1_pd_beta.SetShape(valid_shape);
  tensordesc_output1_pd_beta.SetDataType(DT_FLOAT);

  (void)op.UpdateOutputDesc("pd_gamma", tensordesc_output0_pd_gamma);
  (void)op.UpdateOutputDesc("pd_beta", tensordesc_output1_pd_beta);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LayerNormBetaGammaBackprop, LayerNormBetaGammaBackpropInferShape);
// -------------------LayerNormBetaGammaBackprop END------------------

// --------------------------LayerNormXBackprop-----------------------
IMPLEMT_COMMON_INFERFUNC(LayerNormXBackpropInferShape) {
  TensorDesc tensordesc_output_pd_x = op.GetOutputDesc("pd_x");

  tensordesc_output_pd_x.SetShape(op.GetInputDesc("dy").GetShape());
  tensordesc_output_pd_x.SetDataType(op.GetInputDesc("dy").GetDataType());

  (void)op.UpdateOutputDesc("pd_x", tensordesc_output_pd_x);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LayerNormXBackprop, LayerNormXBackpropInferShape);
// ---------------------LayerNormXBackprop END--------------------------

// ----------------LayerNormBetaGammaBackpropV2--------------------
IMPLEMT_COMMON_INFERFUNC(LayerNormBetaGammaBackpropV2InferShape) {
  std::vector<int64_t> dims_tm;
  if (op.GetAttr("shape_gamma", dims_tm) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("shape_gamma");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }
  Shape valid_shape(dims_tm);

  TensorDesc tensordesc_output0_pd_gamma = op.GetOutputDesc("pd_gamma");
  TensorDesc tensordesc_output1_pd_beta = op.GetOutputDesc("pd_beta");

  tensordesc_output0_pd_gamma.SetShape(valid_shape);
  tensordesc_output0_pd_gamma.SetDataType(DT_FLOAT);

  tensordesc_output1_pd_beta.SetShape(valid_shape);
  tensordesc_output1_pd_beta.SetDataType(DT_FLOAT);

  (void)op.UpdateOutputDesc("pd_gamma", tensordesc_output0_pd_gamma);
  (void)op.UpdateOutputDesc("pd_beta", tensordesc_output1_pd_beta);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LayerNormBetaGammaBackpropV2, LayerNormBetaGammaBackpropV2InferShape);
// -------------------LayerNormBetaGammaBackpropV2 END------------------

// --------------------------LayerNormXBackpropV2-----------------------
IMPLEMT_COMMON_INFERFUNC(LayerNormXBackpropV2InferShape) {
  TensorDesc tensordesc_output_pd_x = op.GetOutputDesc("pd_x");
  TensorDesc tensordesc_output_res_for_gamma = op.GetOutputDesc("res_for_gamma");

  tensordesc_output_pd_x.SetShape(op.GetInputDesc("dy").GetShape());
  tensordesc_output_pd_x.SetDataType(op.GetInputDesc("dy").GetDataType());

  tensordesc_output_res_for_gamma.SetShape(op.GetInputDesc("dy").GetShape());
  tensordesc_output_res_for_gamma.SetDataType(DT_FLOAT);

  (void)op.UpdateOutputDesc("pd_x", tensordesc_output_pd_x);
  (void)op.UpdateOutputDesc("res_for_gamma", tensordesc_output_res_for_gamma);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LayerNormXBackpropV2, LayerNormXBackpropV2InferShape);
// ---------------------LayerNormXBackpropV2 END--------------------------

// ----------------DropOutDoMask Op Start-------------------
IMPLEMT_VERIFIER(DropOutDoMask, DropOutDoMaskVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "keep_prob")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DropOutDoMaskInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_info->MutableInputDesc("x");
  vector<int64_t> x_shape = x_desc->MutableShape().GetDims();
  DataType input_dtype = x_desc->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> input_range;
  x_desc->GetShapeRange(input_range);
  MakeUpShapeRange(x_shape, input_range);

  auto output_desc_y = op_info->MutableOutputDesc("y");
  output_desc_y->SetShape(GeShape(x_shape));
  output_desc_y->SetOriginShape(GeShape(x_shape));
  output_desc_y->SetShapeRange(input_range);
  output_desc_y->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DropOutDoMask, DropOutDoMaskInferShape);
VERIFY_FUNC_REG(DropOutDoMask, DropOutDoMaskVerify);
// ----------------DropOutDoMask Op End-------------------

// ----------------DropOutDoMaskV3Op Start-------------------
IMPLEMT_VERIFIER(DropOutDoMaskV3, DropOutDoMaskV3Verify) {
  if (!CheckTwoInputDtypeSame(op, "x", "keep_prob")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DropOutDoMaskV3InferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_info->MutableInputDesc("x");
  vector<int64_t> x_shape = x_desc->MutableShape().GetDims();
  DataType input_dtype = x_desc->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> input_range;
  x_desc->GetShapeRange(input_range);
  MakeUpShapeRange(x_shape, input_range);

  auto output_desc_y = op_info->MutableOutputDesc("y");
  output_desc_y->SetShape(GeShape(x_shape));
  output_desc_y->SetOriginShape(GeShape(x_shape));
  output_desc_y->SetShapeRange(input_range);
  output_desc_y->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DropOutDoMaskV3, DropOutDoMaskV3InferShape);
VERIFY_FUNC_REG(DropOutDoMaskV3, DropOutDoMaskV3Verify);
// ----------------DropOutDoMaskV3 Op End-------------------


// ----------------DropOutDoMaskV3D Op Start-------------------
IMPLEMT_VERIFIER(DropOutDoMaskV3D, DropOutDoMaskV3DVerify) {
  std::vector<float> constAttr;
  if(!GetConstAttr(op, {"keep_prob"}, constAttr)){
     OP_LOGE(op.GetName().c_str(), "The GetOpAttr ConstValue failed!");
     return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DropOutDoMaskV3DInferShape) {
  TensorDesc tensor_desc_output = op.GetOutputDesc("y");

  tensor_desc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensor_desc_output.SetDataType(op.GetInputDesc("x").GetDataType());
  (void)op.UpdateOutputDesc("y", tensor_desc_output);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DropOutDoMaskV3D, DropOutDoMaskV3DInferShape);
VERIFY_FUNC_REG(DropOutDoMaskV3D, DropOutDoMaskV3DVerify);
// ----------------DropOutDoMaskV3D Op End-------------------


//---------------------------------Scale------------------------------------
IMPLEMT_INFERFUNC(Scale, ScaleInferShape) {
  OP_LOGI("Scale", "infer shape begin---");
  DataType dtype_x = op.GetInputDesc("x").GetDataType();
  ge::Shape shape_x = op.GetInputDesc("x").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  // set output
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape_x);
  output_desc.SetDataType(dtype_x);
  (void)op.UpdateOutputDesc("y", output_desc);

  int64_t axis;
  int64_t num_axes;
  bool scale_from_blob;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("num_axes", num_axes)) {
    std::string err_msg = GetInputInvalidErrMsg("num_axes");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("scale_from_blob", scale_from_blob)) {
    std::string err_msg = GetInputInvalidErrMsg("scale_from_blob");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t length_x = dims_x.size();
  ge::Shape shape_scale = op.GetInputDesc("scale").GetShape();
  int64_t scale_dim_num = shape_scale.GetDimNum();
  std::vector<int64_t> dims_scale = shape_scale.GetDims();
  int64_t length_scale = dims_scale.size();

  int64_t axis_;
  if (axis < 0) {
    axis_ = length_x + axis;
  } else {
    axis_ = axis;
  }

  // add scale reshape
  std::vector<int64_t> scale_shape_new;
  if ((!scale_from_blob) && (scale_dim_num != 0)) {
    int64_t scale_check_num = axis_ + length_scale;
    if (scale_check_num > length_x) {
      string err_msg1 = ConcatString("scale shape extends x shape when check applied, scale_check_num:",scale_check_num, ", length_x:",length_x);
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    int64_t begin_idx = length_scale - 1;
    for (int64_t i = begin_idx; i >= 0; i--) {
      if (dims_x[axis_ + i] == dims_scale[i]) {
        for (int64_t j = 0; j <= i; j++) {
          scale_shape_new.push_back(dims_scale[j]);
        }
        break;
      } else if (dims_scale[i] > 1) {
        for (int64_t j = 0; j <= i; j++) {
          scale_shape_new.push_back(dims_scale[j]);
        }
        break;
      } else if (dims_scale[i] == 1) {
        continue;
      }
    }
  }

  // shape process
  if (dims_x.size() == 4 && scale_dim_num != 0) {
    std::vector<int64_t> dims_scale_tmp;

    if (scale_from_blob) {
      std::vector<int64_t> dims_scale_tmp1 = shape_scale.GetDims();
      int64_t tmp1_size = dims_scale_tmp1.size();
      for (int64_t i = 0; i < tmp1_size; i++) {
        dims_scale_tmp.push_back(dims_scale_tmp1[i]);
      }
      if (num_axes == -1) {
        for (int64_t i = 0; i < axis_; i++) {
          dims_scale_tmp.insert(dims_scale_tmp.begin(), (int64_t)1);
        }
      } else if (num_axes > 0) {
        int64_t left_length = length_x - num_axes - axis_;
        for (int64_t i = 0; i < axis_; i++) {
          dims_scale_tmp.insert(dims_scale_tmp.begin(), (int64_t)1);
        }
        for (int64_t i = 0; i < left_length; i++) {
          dims_scale_tmp.push_back((int64_t)1);
        }
      }
    } else {
      int64_t length_scale_new = scale_shape_new.size();
      for (int64_t i = 0; i < length_scale_new; i++) {
        dims_scale_tmp.push_back(scale_shape_new[i]);
      }
      int64_t left_length = length_x - length_scale_new - axis_;
      for (int64_t i = 0; i < axis_; i++) {
        dims_scale_tmp.insert(dims_scale_tmp.begin(), (int64_t)1);
      }
      for (int64_t i = 0; i < left_length; i++) {
        dims_scale_tmp.push_back((int64_t)1);
      }
    }

    // furthermore, bottom[1] may have the empty shape, regardless of the value of
    // axis, a scalar multiplier.
    if ((scale_dim_num == 1) && (dims_scale[0] == 1)) {
      dims_scale_tmp = {1};
    }

    // update scale shape
    ge::Shape output_scale_shape = ge::Shape(dims_scale_tmp);
    TensorDesc scale_desc = op.GetInputDesc("scale");
    scale_desc.SetShape(output_scale_shape);
    scale_desc.SetOriginShape(output_scale_shape);
    (void)op.UpdateInputDesc("scale", scale_desc);

    // update bias shape
    DataType dtype_bias = op.GetInputDesc("bias").GetDataType();
    Format format_bias = op.GetInputDesc("bias").GetFormat();
    if (!((dtype_bias == DT_UNDEFINED) && (format_bias == FORMAT_RESERVED))) {
      // bias input
      ge::Shape output_bias_shape = ge::Shape(dims_scale_tmp);
      TensorDesc bias_desc = op.GetInputDesc("bias");
      bias_desc.SetShape(output_bias_shape);
      bias_desc.SetOriginShape(output_bias_shape);
      (void)op.UpdateInputDesc("bias", bias_desc);
    }
  }

  OP_LOGI("Scale", "infer shape end---");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Scale, ScaleVerify) {
  ge::Shape shape_x = op.GetInputDesc("x").GetShape();
  ge::Shape shape_scale = op.GetInputDesc("scale").GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_scale = shape_scale.GetDims();
  int64_t scale_dim_num = shape_scale.GetDimNum();

  int64_t axis;
  int64_t num_axes;
  bool scale_from_blob;
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("num_axes", num_axes)) {
    std::string err_msg = GetInputInvalidErrMsg("num_axes");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("scale_from_blob", scale_from_blob)) {
    std::string err_msg = GetInputInvalidErrMsg("scale_from_blob");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t length_x = dims_x.size();
  int64_t length_scale = dims_scale.size();

  if ((axis >= length_x) || (axis < (-length_x))) {
    string minvalue = ConcatString(-length_x);
    string maxvalue = ConcatString(length_x - 1);
    string excepted_value = ConcatString("in the range of [", minvalue, ", ", maxvalue,"]");
    std::string err_msg = GetAttrValueErrMsg("axis", ConcatString(axis), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (num_axes < -1) {
    std::string err_msg = GetAttrValueErrMsg("num_axes", ConcatString(num_axes), ConcatString("non-negative or -1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t axis_;
  if (axis < 0) {
    axis_ = length_x + axis;
  } else {
    axis_ = axis;
  }

  // add scale reshape
  std::vector<int64_t> scale_shape_new;
  if ((!scale_from_blob) && (scale_dim_num != 0)) {
    int64_t scale_check_num = axis_ + length_scale;
    if (scale_check_num > length_x) {
      string err_msg1 = ConcatString("scale shape extends x shape when check applied, scale_check_num:",scale_check_num, ", length_x:",length_x);
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    int64_t begin_idx = length_scale - 1;
    for (int64_t i = begin_idx; i >= 0; i--) {
      if (dims_x[axis_ + i] == dims_scale[i]) {
        for (int64_t j = 0; j <= i; j++) {
          scale_shape_new.push_back(dims_scale[j]);
        }
        break;
      } else if (dims_scale[i] > 1) {
        for (int64_t j = 0; j <= i; j++) {
          scale_shape_new.push_back(dims_scale[j]);
        }
        break;
      } else if (dims_scale[i] == 1) {
        continue;
      }
    }
  }

  if (scale_from_blob) {
    if (num_axes == -1) {
      int64_t scale_num = length_x - axis_;
      if (length_scale != scale_num) {
        string err_msg1 = ConcatString("length_scale and scale_num must be equal, length_scale:",length_scale, ", scale_num:",scale_num);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
      for (int64_t i = 0; i < scale_num; i++) {
        if (dims_x[axis_ + i] != dims_scale[i]) {
        string err_msg1 = ConcatString("length_scale and scale_num must be equal, dims_x[axis_ + i]:",dims_x[axis_ + i], ", dims_scale[i]:",dims_scale[i]);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
        }
      }
    } else if (num_axes == 0) {
      if (scale_dim_num != 0) {
        std::string err_msg = GetAttrValueErrMsg("scale_dim_num", ConcatString(scale_dim_num), ConcatString(0));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
    } else if (num_axes > 0) {
      int64_t num_axis = axis_ + num_axes;
      if (num_axis > length_x) {
        string err_msg1 = ConcatString("scale shape extends x shape when applied, num_axis:",num_axis, ", length_x:",length_x);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
      if (length_scale != num_axes) {
        string err_msg1 = ConcatString("length_scale and num_axes must be equal, length_scale:",length_scale, ", num_axes:",num_axes);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
      for (int64_t i = 0; i < num_axes; i++) {
        if (dims_x[axis_ + i] != dims_scale[i]) {
          string err_msg1 = ConcatString("dimensions shape_x and shape_scale must be equal, dims_x[axis_ + i]:",dims_x[axis_ + i], ", dims_scale[i]:",dims_scale[i]);
          std::string err_msg = OtherErrMsg(err_msg1);
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_FAILED;
        }
      }
    }
  } else {
    int64_t scale_dim_num_new = scale_shape_new.size();
    int64_t length_scale_new = scale_dim_num_new;
    if (scale_dim_num_new != 0) {
      int64_t scale_num = axis_ + length_scale_new;
      if (scale_num > length_x) {
        string err_msg1 = ConcatString("scale shape extends x shape when applied, scale_num:",scale_num, ", length_x:",length_x);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
        return GRAPH_FAILED;
      }
      for (int64_t i = 0; i < length_scale_new; i++) {
        if (dims_x[axis_ + i] != scale_shape_new[i]) {
          string err_msg1 = ConcatString("dimensions shape_x and shape_scale must be equal, dims_x[axis_ + i]:",dims_x[axis_ + i], ", scale_shape_new[i]:",scale_shape_new[i]);
          std::string err_msg = OtherErrMsg(err_msg1);
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
          return GRAPH_FAILED;
        }
      }
    }
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Scale, ScaleInferShape);
VERIFY_FUNC_REG(Scale, ScaleVerify);
//-------------------------------------Scale--------------------------------------------

// ----------------LRNGrad   ------------------
IMPLEMT_VERIFIER(LRNGrad, LRNGradVerify) {
  if (!CheckInputsShapeDtypeSame(op, {"grads", "x", "y"})) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LRNGrad, ELMTWISE_INFER_SHAPEANDTYPE("grads", "z"));

VERIFY_FUNC_REG(LRNGrad, LRNGradVerify);

// ----------------LrnGrad END------------------

// ----------------LRN-------------------
COMMON_INFER_FUNC_REG(LRN, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));
// --------------LRN END-----------------

// ------------------------GroupNorm--------------------------
IMPLEMT_VERIFIER(GroupNorm, GroupNormVerify) {
  if (!CheckTwoInputDtypeSame(op, "scale", "offset")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(GroupNormInferShape) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NHWC" && data_format != "NCHW") {
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

COMMON_INFER_FUNC_REG(GroupNorm, GroupNormInferShape);
VERIFY_FUNC_REG(GroupNorm, GroupNormVerify);
// ----------------------GroupNorm END--------------------------

// ------------------------InstanceNormV2--------------------------
IMPLEMT_VERIFIER(InstanceNormV2, InstanceNormV2Verify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InstanceNormV2InferShape) {
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  DataType x_dtype = inputTensorDesc.GetDataType();

  std::vector<int64_t> dimVector;
  int64_t dimNum = shape.GetDimNum();

  std::vector<int64_t> dims_input;
  dims_input = shape.GetDims();

  for (int64_t item = 0; item < dimNum; ++item) {
    if (item == 2 || item == 3) {
      dimVector.push_back(1);
    } else {
      dimVector.push_back(dims_input[item]);
    }
  }

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(ge::Shape(dims_input));
  y_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", y_desc);

  TensorDesc batch_mean_desc = op.GetOutputDesc("batch_mean");
  batch_mean_desc.SetShape(ge::Shape(dimVector));
  batch_mean_desc.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("batch_mean", batch_mean_desc);

  TensorDesc batch_variance_desc = op.GetOutputDesc("batch_variance");
  batch_variance_desc.SetShape(ge::Shape(dimVector));
  batch_variance_desc.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("batch_variance", batch_variance_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InstanceNormV2, InstanceNormV2InferShape);
VERIFY_FUNC_REG(InstanceNormV2, InstanceNormV2Verify);
// ----------------------InstanceNormV2 END--------------------------

// ------------------------INInferV2D START--------------------------
IMPLEMT_VERIFIER(INInferV2D, INInferV2DVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(INInferV2DInferShape) {
  auto inputTensorDescX = op.GetInputDesc("x");
  auto shape = inputTensorDescX.GetShape();
  DataType x_dtype = inputTensorDescX.GetDataType();

  std::vector<int64_t> dimVector;
  int64_t dimNum = shape.GetDimNum();
  std::vector<int64_t> dims_input;
  dims_input = shape.GetDims();

  for (int64_t item = 0; item < dimNum; ++item) {
    if (item == 2 || item == 3) {
      dimVector.push_back(1);
    } else {
      dimVector.push_back(dims_input[item]);
    }
  }

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(ge::Shape(dims_input));
  y_desc.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", y_desc);

  TensorDesc batch_mean_desc = op.GetOutputDesc("batch_mean");
  batch_mean_desc.SetShape(ge::Shape(dimVector));
  batch_mean_desc.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("batch_mean", batch_mean_desc);

  TensorDesc batch_variance_desc = op.GetOutputDesc("batch_variance");
  batch_variance_desc.SetShape(ge::Shape(dimVector));
  batch_variance_desc.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("batch_variance", batch_variance_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(INInferV2D, INInferV2DInferShape);
VERIFY_FUNC_REG(INInferV2D, INInferV2DVerify);
// ------------------------INInferV2D END--------------------------

// ----------------InstanceNorm Begin-------------------
IMPLEMT_COMMON_INFERFUNC(InstanceNormInferShape) {
  // x desc
  TensorDesc tensor_desc_input = op.GetInputDesc("x");
  Shape input_shape = tensor_desc_input.GetShape();
  DataType input_dtype = tensor_desc_input.GetDataType();
  std::vector<int64_t> dims_input = input_shape.GetDims();
  int64_t dim_num = input_shape.GetDimNum();

  TensorDesc tensor_desc_output1 = op.GetOutputDesc("y");
  TensorDesc tensor_desc_output2 = op.GetOutputDesc("mean");
  TensorDesc tensor_desc_output3 = op.GetOutputDesc("variance");
  tensor_desc_output1.SetDataType(input_dtype);
  tensor_desc_output2.SetDataType(input_dtype);
  tensor_desc_output3.SetDataType(input_dtype);

  // construct the mean an variance output shape
  std::vector<int64_t> o_shape_vec;
  for (int i = 0; i < dim_num; i++) {
    if (i == 2 || i == 3) {
      o_shape_vec.push_back(1);
    } else {
      o_shape_vec.push_back(dims_input[i]);
    }
  }

  // update the tensor desc
  tensor_desc_output1.SetShape(input_shape);
  Shape oShape(o_shape_vec);
  tensor_desc_output2.SetShape(oShape);
  tensor_desc_output3.SetShape(oShape);
  (void)op.UpdateOutputDesc("y", tensor_desc_output1);
  (void)op.UpdateOutputDesc("mean", tensor_desc_output2);
  (void)op.UpdateOutputDesc("variance", tensor_desc_output3);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(InstanceNorm, InstanceNormVerify) {
  // get dtype
  std::string x_dtype, gamma_dtype, beta_dtype;
  x_dtype = op.GetInputDesc("x").GetDataType();
  gamma_dtype = op.GetInputDesc("gamma").GetDataType();
  beta_dtype = op.GetInputDesc("beta").GetDataType();

  // compare dtype
  if (x_dtype != gamma_dtype) {
    OP_LOGE(op.GetName().c_str(),
            "gamma's dtype: %s, is NOT same as x's dtype: %s!!",
            gamma_dtype.c_str(), x_dtype.c_str());
    return GRAPH_FAILED;
  }
  if (x_dtype != beta_dtype) {
    OP_LOGE(op.GetName().c_str(),
            "beta's dtype: %s,  is NOT same as x's dtype: %s!!",
            beta_dtype.c_str(), x_dtype.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InstanceNorm, InstanceNormInferShape);
VERIFY_FUNC_REG(InstanceNorm, InstanceNormVerify);
// ----------------InstanceNorm END---------------------

// ----------------KlDivLossGrad Begin-------------------
bool InferShapeAndTypeKlDivLossGrad(Operator& op, const string& input_name,
                                    const string& output_name) {
  TensorDesc output_desc = op.GetOutputDesc(output_name);
  DataType input_dtype = op.GetInputDesc(input_name).GetDataType();
  Format input_format = op.GetInputDesc(input_name).GetFormat();
  ge::Shape input_shape = op.GetInputDesc(input_name).GetShape();

  output_desc.SetShape(input_shape);
  output_desc.SetDataType(input_dtype);
  output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name, output_desc);
  return true;
}

IMPLEMT_COMMON_INFERFUNC(KlDivLossGradInferShape) {
  if (InferShapeAndTypeKlDivLossGrad(op, "input", "y")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(op.GetName().c_str(), "KL_DIV_LOSS_GRAD Infershape Failed");
  return GRAPH_FAILED;
}

IMPLEMT_VERIFIER(KlDivLossGrad, KlDivLossGradVerify) {
  if (op.GetInputDesc("grad").GetDataType() !=
          op.GetInputDesc("input").GetDataType() ||
      op.GetInputDesc("input").GetDataType() !=
          op.GetInputDesc("target").GetDataType()) {
    OP_LOGE(op.GetName().c_str(), "grad type is not same with input or target");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(KlDivLossGrad, KlDivLossGradInferShape);
VERIFY_FUNC_REG(KlDivLossGrad, KlDivLossGradVerify);
// ----------------KlDivLossGrad END---------------------

// ----------------L1LossGrad Begin-------------------
IMPLEMT_VERIFIER(L1LossGrad, L1LossGradVerify) {
  DataType grads_type = op.GetInputDesc("grads").GetDataType();
  DataType predict_type = op.GetInputDesc("predict").GetDataType();
  DataType label_type = op.GetInputDesc("label").GetDataType();

  if ((grads_type != DT_FLOAT16 && grads_type != DT_FLOAT) ||
      (label_type != DT_FLOAT16 && label_type != DT_FLOAT) ||
      (predict_type != DT_FLOAT16 && predict_type != DT_FLOAT)) {
    OP_LOGE(op.GetName().c_str(), "input dtype should be fp32 or fp 16");
    return GRAPH_FAILED;
  }
  if (grads_type != predict_type) {
    OP_LOGE(op.GetName().c_str(), "grads' dtype is NOT same as predict's dtype");
    return GRAPH_FAILED;
  }
  if (grads_type != label_type) {
    OP_LOGE(op.GetName().c_str(), "label's dtype is NOT same as other inputs'");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(L1LossGradInfer) {
  std::string reduction;
  if (op.GetAttr("reduction", reduction) == GRAPH_SUCCESS) {
    if (reduction != "none" && reduction != "mean" && reduction != "sum") {
      OP_LOGE(op.GetName().c_str(), "reduction is not in none, mean or sum");
      return GRAPH_FAILED;
    }
  }

  Shape grads_shape = op.GetInputDesc("grads").GetShape();
  Shape label_shape = op.GetInputDesc("label").GetShape();
  Shape predict_shape = op.GetInputDesc("predict").GetShape();
  if (predict_shape.GetDims().size() != grads_shape.GetDims().size() ||
      predict_shape.GetDims().size() != label_shape.GetDims().size()) {
    OP_LOGE(op.GetName().c_str(),
            "predict, grads and label are NOT in same size");
    return GRAPH_FAILED;
  }

  if (OneInOneOutDynamicInfer(op, "predict", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(L1LossGrad, L1LossGradInfer);
// Registered verify function
VERIFY_FUNC_REG(L1LossGrad, L1LossGradVerify);
// ----------------L1LossGrad END---------------------

// ----------------LpLoss Begin-------------------
IMPLEMT_VERIFIER(LpLoss, LpLossVerify) { return GRAPH_SUCCESS; }

IMPLEMT_COMMON_INFERFUNC(LpLossInferShape) {
  std::string reduction;
  if (op.GetAttr("reduction", reduction) == GRAPH_SUCCESS) {
    if (reduction != "none" && reduction != "mean" && reduction != "sum") {
      printf(op.GetName().c_str(),
             "Attr reduction only support 'none', 'mean', 'sum'");
      return GRAPH_FAILED;
    }
  }
  if (reduction == "none") {
    // if reduction == "none" , output shape == x.shape
    OP_LOGI(op.GetName().c_str(), "the attr reduction = none");
    if (OneInOneOutDynamicInfer(op, "predict", {"y"})){
      return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
  } else {
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    if (op_info == nullptr) {
      OP_LOGE(op.GetName().c_str(), "op_info should not be nullptr");
      return GRAPH_FAILED;
    }
    auto outputTensordesc = op_info->MutableOutputDesc("y");
    auto x_desc = op_info->MutableInputDesc("predict");
    DataType x_dtype = x_desc->GetDataType();
    std::vector<int64_t> o_shape;
    std::vector<std::pair<int64_t, int64_t>> o_range;
    outputTensordesc->SetShape(GeShape(o_shape));
    outputTensordesc->SetShapeRange(o_range);
    outputTensordesc->SetDataType(x_dtype);
    return GRAPH_SUCCESS;
  }
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(LpLoss, LpLossInferShape);
// Registered verify function
VERIFY_FUNC_REG(LpLoss, LpLossVerify);
// ----------------LpLoss END---------------------

// ----------------MseLossGrad Begin-------------------
IMPLEMT_COMMON_INFERFUNC(MseLossGradInferShape) {
  bool is_dynamic_output = true;

  auto tensor_predict = op.GetInputDesc("predict");
  auto predict_type = tensor_predict.GetDataType();
  auto tensor_label = op.GetInputDesc("label");
  auto label_type = tensor_label.GetDataType();

  if (predict_type != label_type) {
    OP_LOGE(op.GetName().c_str(), "predict dtype is not same as label's dtype.");
    return GRAPH_FAILED;
  }

  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "predict", "label", "y", is_dynamic_output)){
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(MseLossGrad, MseLossGradInferShape);
// ----------------MseLossGrad END---------------------

// ----------------MseLoss Begin-------------------
IMPLEMT_COMMON_INFERFUNC(MseLossInferShape) {
  std::string reduceType = "mean";
  if (op.GetAttr("reduction", reduceType) == GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "reduction");
    OP_LOGE(op.GetName().c_str(), "get attr reduction failed");
    return GRAPH_FAILED;
  }
  if (reduceType == "none") {
    // if reduction == "none" , output shape == x.shape
    OP_LOGI(op.GetName().c_str(), "the attr reduction = none");
    if (OneInOneOutDynamicInfer(op, "predict", {"y"})){
      return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
  } else {
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    if (op_info == nullptr) {
      OP_LOGE(op.GetName().c_str(), "op_info should not be nullptr");
      return GRAPH_FAILED;
    }
    auto outputTensordesc = op_info->MutableOutputDesc("y");
    auto x_desc = op_info->MutableInputDesc("predict");
    DataType x_dtype = x_desc->GetDataType();
    std::vector<int64_t> o_shape;
    std::vector<std::pair<int64_t, int64_t>> o_range;
    outputTensordesc->SetShape(GeShape(o_shape));
    outputTensordesc->SetShapeRange(o_range);
    outputTensordesc->SetDataType(x_dtype);
    return GRAPH_SUCCESS;
  }
}

COMMON_INFER_FUNC_REG(MseLoss, MseLossInferShape);
// ----------------MseLoss END---------------------

// ----------------SoftMarginLoss Begin-------------------
bool InferShapeAndTypeSoftMarginLoss(Operator& op, const string& input_name1, const string& input_name2,
                                     const string& output_name, const string& reduction) {
    TensorDesc v_output_desc = op.GetOutputDesc(output_name);
    DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
    Format input_format = op.GetInputDesc(input_name1).GetFormat();
    ge::Shape output_shape;
    std::string attr_value = "none";
    graphStatus attr = op.GetAttr("reduction", attr_value);

    if(attr_value == "none") {
        ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
        ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
        std::vector<int64_t> dims_x = shape_x.GetDims();
        std::vector<int64_t> dims_y = shape_y.GetDims();
        if (dims_x.size() < dims_y.size()) {
            std::vector<int64_t> dims_tmp = dims_x;
            dims_x = dims_y;
            dims_y = dims_tmp;
        }

        if (dims_x.size() != dims_y.size()) {
            int dec = dims_x.size() - dims_y.size();
            for (int i = 0; i < dec; i++) {
                dims_y.insert(dims_y.begin(), (int64_t)1);
            }
        }

        std::vector<int64_t> dim_vec;
        for (size_t i = 0; i < dims_x.size(); i++) {
            if ((dims_x[i] != dims_y[i]) && (dims_x[i] != 1) && (dims_y[i] != 1)) {
                OP_LOGE(op.GetName().c_str(), "The shape of input_x input_y must be broadcastable");
                return false;
            }
            int64_t dims = dims_x[i] > dims_y[i] ? dims_x[i] : dims_y[i];
            dim_vec.push_back(dims);
        }
        output_shape = ge::Shape(dim_vec);
    } else {
        std::vector<int64_t> dim_vec;
        dim_vec.push_back(1);
        output_shape = ge::Shape(dim_vec);
    }
    v_output_desc.SetShape(output_shape);
    v_output_desc.SetDataType(input_dtype);
    v_output_desc.SetFormat(input_format);
    op.UpdateOutputDesc(output_name, v_output_desc);
    return true;
}

IMPLEMT_VERIFIER(SoftMarginLoss, SoftMarginLossVerify) {
    if (op.GetInputDesc("input_x").GetDataType() != op.GetInputDesc("input_y").GetDataType()) {
        OP_LOGE(op.GetName().c_str(), "The dtype of input_x input_y should be same.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SoftMarginLossInferShape) {
    if(InferShapeAndTypeSoftMarginLoss(op, "input_x", "input_y", "output_z", "reduction")) {
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(SoftMarginLoss, SoftMarginLossInferShape);
VERIFY_FUNC_REG(SoftMarginLoss, SoftMarginLossVerify);
// ----------------SoftMarginLoss End-------------------

// ----------------SigmoidCrossEntropyWithLogitsGradV2 Begin-------------------
IMPLEMT_VERIFIER(SigmoidCrossEntropyWithLogitsGradV2,
                 SigmoidCrossEntropyWithLogitsGradV2Verity) {
  std::vector<int64_t> predict_shape_dim =
      op.GetInputDesc("predict").GetShape().GetDims();
  std::vector<int64_t> target_shape_dim =
      op.GetInputDesc("target").GetShape().GetDims();
  for (size_t i = 0; i < predict_shape_dim.size(); i++) {
    if ((predict_shape_dim[i] != target_shape_dim[i])) {
      printf(op.GetName().c_str(),
             "the input shape of predict and target should be same");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SigmoidCrossEntropyWithLogitsGradV2InferShape) {
  if (OneInOneOutDynamicInfer(op, "predict", {"gradient"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(SigmoidCrossEntropyWithLogitsGradV2,
                      SigmoidCrossEntropyWithLogitsGradV2InferShape);
VERIFY_FUNC_REG(SigmoidCrossEntropyWithLogitsGradV2,
                SigmoidCrossEntropyWithLogitsGradV2Verity);
// ----------------SigmoidCrossEntropyWithLogitsGradV2 END---------------------

// ----------------SmoothL1LossGradV2 Begin-------------------
IMPLEMT_INFERFUNC(SmoothL1LossGradV2, SmoothL1LossGradV2InferShape) {
  std::string reduction = "mean";
  if (GRAPH_SUCCESS == op.GetAttr("reduction", reduction)) {
    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
      OP_LOGE(op.GetName().c_str(), "The val of reduction is invalid.");
      return GRAPH_FAILED;
    }
  }

  if (OneInOneOutDynamicInfer(op, "predict", {"gradient"})) {
    return GRAPH_SUCCESS;
  }

  OP_LOGE(op.GetName().c_str(), "Infer Failed.");
  return GRAPH_FAILED;
}

INFER_FUNC_REG(SmoothL1LossGradV2, SmoothL1LossGradV2InferShape);
// ----------------SmoothL1LossGradV2 END---------------------

// ----------------SmoothL1LossV2 Begin-------------------
IMPLEMT_INFERFUNC(SmoothL1LossV2, SmoothL1LossV2InferShape) {
  TensorDesc tensordesc_input1 = op.GetInputDesc("predict");
  Shape input_shape1 = tensordesc_input1.GetShape();
  DataType input_dtype1 = tensordesc_input1.GetDataType();
  std::vector<int64_t> dims_input1 = input_shape1.GetDims();

  TensorDesc tensordesc_input2 = op.GetInputDesc("label");
  Shape input_shape2 = tensordesc_input2.GetShape();
  DataType input_dtype2 = tensordesc_input2.GetDataType();
  std::vector<int64_t> dims_input2 = input_shape2.GetDims();

  if (dims_input1.size() != dims_input2.size()) {
    OP_LOGE(op.GetName().c_str(), "Input Dims Unmatch.");
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("loss");

  std::string reduction_val = "mean";

  if (GRAPH_SUCCESS != op.GetAttr("reduction", reduction_val)) {
    OP_LOGE(op.GetName().c_str(), "Failed to get the val of reduction.");
    return GRAPH_FAILED;
  }

  if (reduction_val != "none" && reduction_val != "mean" &&
      reduction_val != "sum") {
    OP_LOGE(op.GetName().c_str(), "The val of reduction is invalid.");
    return GRAPH_FAILED;
  }

  if (reduction_val == "none") {
    tensordesc_output.SetShape(input_shape1);
  } else {
    std::vector<int64_t> dimVec = {1};
    Shape shape_one = Shape(dimVec);
    tensordesc_output.SetShape(shape_one);
  }

  tensordesc_output.SetDataType(input_dtype1);
  (void)op.UpdateOutputDesc("loss", tensordesc_output);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SmoothL1LossV2, SmoothL1LossV2Verify) {
  DataType input_type_x = op.GetInputDesc("predict").GetDataType();
  DataType input_type_y = op.GetInputDesc("label").GetDataType();
  if (input_type_x != input_type_y) {
    OP_LOGE(op.GetName().c_str(), "Input Type Unmatch.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SmoothL1LossV2, SmoothL1LossV2InferShape);
VERIFY_FUNC_REG(SmoothL1LossV2, SmoothL1LossV2Verify);
// ----------------SmoothL1LossV2 END---------------------
// ----------------PoissonNllLoss Begin----------------------------
bool InferShapeAndTypePoissonNllLoss(Operator& op, 
                                     const string& input_x, 
                                     const string& target, 
                                     const string& loss, 
                                     const string& reduction) {
    TensorDesc vOutputDesc = op.GetOutputDesc(loss);
    DataType inputDtype = op.GetInputDesc(input_x).GetDataType();
    Format inputFormat = op.GetInputDesc(input_x).GetFormat();
    ge::Shape shapeX = op.GetInputDesc(input_x).GetShape();
    ge::Shape shapeY = op.GetInputDesc(target).GetShape();
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
    if (reduction == "none") {
        for (size_t i = 0; i < dimsX.size(); i++) {
            if ((dimsX[i] != dimsY[i]) && (dimsX[i] != 1) && (dimsY[i] != 1)) {
                OP_LOGE(op.GetName().c_str(), "Input shapes are not compatible.");
                return false;
            }
            int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
            dimVec.push_back(dims);
        }
    } else if (reduction == "mean" || reduction == "sum") {
        int64_t one = 1;
        dimVec.push_back(one);
    } else {
        OP_LOGE(op.GetName().c_str(), "Parameter reduction expects 'none', 'mean' or 'sum'.");
        return false;
    }
    ge::Shape outputShape = ge::Shape(dimVec);
    vOutputDesc.SetShape(outputShape);
    vOutputDesc.SetDataType(inputDtype);
    vOutputDesc.SetFormat(inputFormat);
    op.UpdateOutputDesc(loss, vOutputDesc);
    return true;
}

//PoissonNllLoss
IMPLEMT_VERIFIER(PoissonNllLoss, PoissonNllLossVerify) {
    DataType input_type_input = op.GetInputDesc("input_x").GetDataType();
    DataType input_type_target = op.GetInputDesc("target").GetDataType();
    if (input_type_input != input_type_target) {
        OP_LOGE(op.GetName().c_str(), "Input dtypes are not the same.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

//Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(PoissonNllLossInferShape) {
    std::string reduction_str = "";
    op.GetAttr("reduction",reduction_str);
    if (InferShapeAndTypePoissonNllLoss(op, "input_x", "target", "loss", reduction_str)) {
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

//Registered inferfunction
COMMON_INFER_FUNC_REG(PoissonNllLoss, PoissonNllLossInferShape);

//Registered verify function
VERIFY_FUNC_REG(PoissonNllLoss, PoissonNllLossVerify);
//PoissonNllLoss
// ----------------PoissonNllLoss END------------------------------
// --------------------------RnnGenMask-------------------------
IMPLEMT_COMMON_INFERFUNC(RnnGenMaskInferShape) {
  TensorDesc tensordesc_input = op.GetInputDesc("seq_length");
  TensorDesc tensordesc_output = op.GetOutputDesc("seq_mask");

  Shape length_shape = tensordesc_input.GetShape();
  std::vector<int64_t> dim_length = length_shape.GetDims();

  if(dim_length.size() != 1){
    OP_LOGE(op.GetName().c_str(), "Unexcepeted Input Shape.");
    return GRAPH_FAILED;
  }
  int64_t batch_size = dim_length[0];

  int64_t num_step = 0;
  if(GRAPH_SUCCESS != op.GetAttr("num_step", num_step)){
    OP_LOGE(op.GetName().c_str(), "Failed to get the value of num_step.");
    return GRAPH_FAILED;
  }

  int64_t hidden_size = 0;
  if(GRAPH_SUCCESS != op.GetAttr("hidden_size", hidden_size)){
    OP_LOGE(op.GetName().c_str(), "Failed to get the value of hidden_size.");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dim_mask = {num_step, batch_size, hidden_size};

  tensordesc_output.SetShape(Shape(dim_mask));
  tensordesc_output.SetDataType(DT_FLOAT16);
  tensordesc_output.SetFormat(tensordesc_input.GetFormat());
  (void)op.UpdateOutputDesc("seq_mask", tensordesc_output);
  return GRAPH_SUCCESS;
}
 IMPLEMT_VERIFIER(RnnGenMask, RnnGenMaskVerify) {
    return GRAPH_SUCCESS;
}
//Registered inferfunction
COMMON_INFER_FUNC_REG(RnnGenMask, RnnGenMaskInferShape);

//Registered verify function
VERIFY_FUNC_REG(RnnGenMask, RnnGenMaskVerify);
// --------------------------RnnGenMask END---------------------
// --------------------------MultilabelMarginLoss-------------------------
IMPLEMT_VERIFIER(MultilabelMarginLoss, MultilabelMarginLossVerify) { return GRAPH_SUCCESS; }

IMPLEMT_INFERFUNC(MultilabelMarginLoss, MultilabelMarginLossInferShape) {
  string input_name1 = "x";
  string input_name2 = "target";
  string output_name1 = "y";
  string output_name2 = "is_target";

  DataType input_dtype = op.GetInputDesc(input_name1).GetDataType();
  Format input_format = op.GetInputDesc(input_name1).GetFormat();
  DataType target_dtype = op.GetInputDesc(input_name2).GetDataType();
  Format target_format = op.GetInputDesc(input_name2).GetFormat();

  ge::Shape shape_x = op.GetInputDesc(input_name1).GetShape();
  ge::Shape shape_y = op.GetInputDesc(input_name2).GetShape();
  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();

  if (dims_x.size() != dims_y.size()) {
    OP_LOGE(op.GetName().c_str(), "InputSize and OutputSize are not the same.");
    return GRAPH_FAILED;
  }

  if (dims_x.size() != 1 && dims_x.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "InputSize Must Equalt to 1 or 2");
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDesc(output_name1);
  TensorDesc isTargetDesc = op.GetOutputDesc(output_name2);

  std::string reduction = op.get_attr_reduction();
  std::vector<int64_t> outputDimVec;
  std::vector<int64_t> isTargetDimVec;
  for (size_t i = 0; i < dims_x.size(); i++) {
    if (dims_x[i] != dims_y[i]) {
      OP_LOGE(op.GetName().c_str(), "InputSize and OutputSize are not the same.");
      return GRAPH_FAILED;
    }
    isTargetDimVec.push_back(dims_x[i]);
  }

  if (reduction == "none" && dims_x.size() == 2) outputDimVec.push_back(dims_x[0]);

  ge::Shape outputShape = ge::Shape(outputDimVec);
  outputDesc.SetShape(outputShape);
  outputDesc.SetDataType(input_dtype);
  outputDesc.SetFormat(input_format);

  ge::Shape isTargetShape = ge::Shape(isTargetDimVec);
  isTargetDesc.SetShape(isTargetShape);
  isTargetDesc.SetDataType(target_dtype);
  isTargetDesc.SetFormat(target_format);

  op.UpdateOutputDesc(output_name1, outputDesc);
  op.UpdateOutputDesc(output_name2, isTargetDesc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MultilabelMarginLoss, MultilabelMarginLossInferShape);
VERIFY_FUNC_REG(MultilabelMarginLoss, MultilabelMarginLossVerify);
// ----------------MultiLabelMarginLoss end----------------------------------
// ----------------------NormalizeBatch----------------------
IMPLEMT_COMMON_INFERFUNC(NormalizeBatchInferShape) {
  TensorDesc tensordesc_output = op.GetOutputDesc("output_y");
  TensorDesc tensordesc_input = op.GetInputDesc("input_x");
  auto input_shape = tensordesc_input.GetShape();
  auto input_dtype = tensordesc_input.GetDataType();
  auto input_dims = input_shape.GetDims();
  if (input_dims.size() != 3) {
    OP_LOGE(op.GetName().c_str(), "input shape doesn't support");
    return GRAPH_FAILED;
  }
  tensordesc_output.SetShape(input_shape);
  tensordesc_output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("output_y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(NormalizeBatch, NormalizeBatchInferShape);
//-----------------------NormalizeBatch END---------------------
}  // namespace ge
