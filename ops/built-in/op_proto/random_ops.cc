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
 * \file random_ops.cpp
 * \brief
 */
#include "inc/random_ops.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/random_ops_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(Multinomial, MultinomialInfer) {
  Shape shape;
  Shape logits_shape;
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 2, logits_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  Tensor num_samples_tensor;
  if (op.GetInputConstData("num_samples", num_samples_tensor) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  int64_t num_samples;
  if (MakeDimForScalarInput(num_samples_tensor, num_samples, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  (void)Matrix(logits_shape.GetDim(0), num_samples, shape);

  Operator::OpType type;
  if (op.GetAttr("dtype", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
      std::string("get attr[dtype] failed"));
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(type);
  output_desc.SetShape(shape);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(Multinomial, MultinomialInfer);

IMPLEMT_INFERFUNC(MultinomialAliasDraw, MultinomialAliasDrawInfer) {
  Shape q_shape;
  Shape j_shape;
  if (WithRank(op.GetInputDesc(0), 1, q_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 1, j_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (q_shape.GetDim(0) != j_shape.GetDim(0)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),std::string("q_shape should be equal to j_shape"));
    return GRAPH_FAILED;
  }

  int64_t num;
  (void)op.GetAttr("num_samples", num);
  std::vector<int64_t> out_dims = {num};
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(DT_INT64);
  output_desc.SetShape(Shape(out_dims));
  op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MultinomialAliasDraw, MultinomialAliasDrawInfer);

IMPLEMT_INFERFUNC(MultinomialAliasSetup, MultinomialAliasSetupInfer) {
  TensorDesc x_input = op.GetInputDesc("probs");
  Shape x_shape;
  if (WithRankAtLeast(x_input, 1, x_shape, "MultinomialAliasSetup") != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  
  TensorDesc output_desc_q = op.GetOutputDesc("q");
  TensorDesc output_desc_j = op.GetOutputDesc("j");
  auto type = op.GetInputDesc("probs").GetDataType();
  output_desc_q.SetDataType(type);
  output_desc_q.SetShape(x_shape);
  output_desc_j.SetDataType(DT_INT64);
  output_desc_j.SetShape(x_shape);
  op.UpdateOutputDesc("q", output_desc_q);
  op.UpdateOutputDesc("j", output_desc_j);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MultinomialAliasSetup, MultinomialAliasSetupInfer);

IMPLEMT_INFERFUNC(ParameterizedTruncatedNormal, ParameterizedTruncatedNormalInfer) {
  Shape unused;
  std::string err_msg;
  if (WithRankAtMost(op.GetInputDesc(1), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "at most 1D");
    err_msg = std::string("failed to call WithRankAtMost, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(2), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), "at most 1D");
    err_msg = std::string("failed to call WithRankAtMost, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(3), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        3, DebugString(op.GetInputDesc(3).GetShape().GetDims()), "at most 1D");
    err_msg = std::string("failed to call WithRankAtMost, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(4), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        4, DebugString(op.GetInputDesc(4).GetShape().GetDims()), "at most 1D");
    err_msg = std::string("failed to call WithRankAtMost, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  return RandomShape(op, "shape", "y");
}

INFER_FUNC_REG(ParameterizedTruncatedNormal, ParameterizedTruncatedNormalInfer);

IMPLEMT_INFERFUNC(RandomGammaGrad, RandomGammaGradInfer) {
  auto type = op.GetInputDesc("alpha").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetDataType(type);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return BROADCAST_INFER("alpha", "sample", "y")(op);
}

INFER_FUNC_REG(RandomGammaGrad, RandomGammaGradInfer);

IMPLEMT_INFERFUNC(RandomGamma, RandomGammaInfer) {
  Shape shape;
  Shape alpha_shape;
  Tensor shape_tensor;
  auto type = op.GetInputDesc("alpha").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");

  std::vector<std::string> input_infer_depends = {"shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    output_desc.SetDataType(type);
    auto input_shape = op.GetInputDesc("shape").GetShape();
    if (input_shape.GetShapeSize() == ge::UNKNOWN_DIM) {
      output_desc.SetShape(ge::Shape(ge::UNKNOWN_RANK));
      output_desc.SetOriginShape(ge::Shape(ge::UNKNOWN_RANK));
    } else {
      int64_t rank = input_shape.GetShapeSize();
      std::vector<int64_t> out_shape;
      for (int64_t i = 0; i < rank; i++) {
        out_shape.push_back(ge::UNKNOWN_DIM);
      }
      output_desc.SetShape(ge::Shape(out_shape));
      output_desc.SetOriginShape(ge::Shape(out_shape));
    }
    return GRAPH_SUCCESS;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  alpha_shape = op.GetInputDesc(1).GetShape();
  (void)Concatenate(shape, alpha_shape, shape);

  output_desc.SetShape(shape);
  output_desc.SetDataType(type);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(RandomGamma, RandomGammaInfer);

IMPLEMT_INFERFUNC(Randperm, RandpermInferShape) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  
  TensorDesc td = op.GetOutputDescByName("out");  
  td.SetShape(scalar_shape);
  (void)op.UpdateOutputDesc("out", td);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Randperm, RandpermInferShape);

IMPLEMT_INFERFUNC(RandomPoisson, RandomPoissonInfer) {
  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  // concatenate
  TensorDesc rate_desc = op.GetInputDesc(1);
  Shape rate_shape = rate_desc.GetShape();
  (void)Concatenate(shape, rate_shape, shape);

  Operator::OpType type;
  if (op.GetAttr("dtype", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
      std::string("get attr[dtype] failed"));
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(type);
  output_desc.SetShape(shape);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(RandomPoisson, RandomPoissonInfer);

IMPLEMT_INFERFUNC(Exponential, ExponentialInfer) {

  TensorDesc tensordesc_output = op.GetOutputDesc("y");

  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Exponential, ExponentialInfer);

IMPLEMT_INFERFUNC(RandomShuffle, RandomShuffleInfer) {
  TensorDesc desc = op.GetInputDesc(0);
  Shape shape = desc.GetShape();

  auto type = op.GetInputDesc("x").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  std::vector<std::pair<int64_t,int64_t>> range;
  auto status = desc.GetShapeRange(range);
  if (status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  output_desc.SetShapeRange(range);
  output_desc.SetShape(shape);
  output_desc.SetDataType(type);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(RandomShuffle, RandomShuffleInfer);

IMPLEMT_INFERFUNC(RandomStandardNormal, RandomStandardNormalInfer) {
  return RandomShapeWithDataType(op, "shape", "dtype", "y");
}

INFER_FUNC_REG(RandomStandardNormal, RandomStandardNormalInfer);

// -------------------------Normal-----------------------------
IMPLEMT_INFERFUNC(Normal, NormalInferShape) {
  TensorDesc mean_desc = op.GetInputDescByName("mean");
  auto outShape = mean_desc.GetShape();
  auto outDtype = mean_desc.GetDataType();
  
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(outShape);
  td.SetDataType(outDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Normal, NormalInferShape);
// -------------------------Normal END-----------------------------

IMPLEMT_INFERFUNC(RandomUniformInt, RandomUniformIntInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  Tensor tensor;
  if (op.GetInputConstData("shape", tensor) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  Shape shape;
  if (MakeShapeFromShapeTensor(tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto type = op.GetInputDesc("min").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(type);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(RandomUniformInt, RandomUniformIntInfer);

IMPLEMT_INFERFUNC(RandomUniform, RandomUniformInfer) {
  return RandomShapeWithDataType(op, "shape", "dtype", "y");
}

INFER_FUNC_REG(RandomUniform, RandomUniformInfer);

IMPLEMT_INFERFUNC(TruncatedNormal, TruncatedNormalInfer) {
  return RandomShape(op, "shape", "y");
}

INFER_FUNC_REG(TruncatedNormal, TruncatedNormalInfer);

int64_t InferShapeGetShapeSizeBytes(int64_t shapeSize) {
  if (shapeSize < 0) {
    return UNKNOWN_DIM;
  }

  // 128 bits are generated at a time.
  int64_t n128s = shapeSize / 128;
  if (shapeSize % 128 != 0) {
    n128s++;
  }

  // 128 bits occupy 16 bytes.
  int64_t n8s = n128s * 16;

  return n8s;
}

graphStatus GetDropOutGenMaskShapeAndRange(Operator &op, ShapeAndRange &out) {
  std::vector<std::pair<int64_t, int64_t>> valueRange;
  int64_t down = 0;
  int64_t up = 0;

  out = {Shape(UNKNOWN_SHAPE), {}};
  ShapeAndRange shapeAndRange{Shape(UNKNOWN_SHAPE), {}};
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_shape_desc = opDesc->MutableInputDesc("shape");
  (void)input_shape_desc->GetValueRange(valueRange);
  OP_LOGI(op.GetName().c_str(), "DropOutGenMask::valueRange.size() = %d", valueRange.size());
  if (valueRange.size() == 0) {
    OP_LOGW(op.GetName().c_str(), "valueRange.size() == 0");
    return GRAPH_SUCCESS;
  }

  for (size_t i = 0; i < valueRange.size(); i++) {
    OP_LOGI(op.GetName().c_str(), "i = %d, first = %lld, second = %lld",
            i, valueRange[i].first, valueRange[i].second);
    if (valueRange[i].first < UNKNOWN_DIM || valueRange[i].second < UNKNOWN_DIM) {
      OP_LOGW(op.GetName().c_str(), "value < UNKNOWN_DIM");
      return GRAPH_SUCCESS;
    }

    if (i == 0) {
      down = valueRange[i].first;
      up = valueRange[i].second;
      continue;
    }

    if ((down == UNKNOWN_DIM) || (valueRange[i].first == UNKNOWN_DIM)) {
      down = UNKNOWN_DIM;
    } else {
      down = down * valueRange[i].first;
    }

    if ((up == UNKNOWN_DIM) || (valueRange[i].second == UNKNOWN_DIM)) {
      up = UNKNOWN_DIM;
    } else {
      up = up * valueRange[i].second; 
    }
    OP_LOGI(op.GetName().c_str(), "down = %lld, up = %lld", down, up);
  }

  Shape unknownShape(UNKNOWN_SHAPE);
  std::vector<std::pair<int64_t, int64_t>> shapeRange(1);
  shapeRange[0].first = InferShapeGetShapeSizeBytes(down);
  shapeRange[0].second = InferShapeGetShapeSizeBytes(up);
  shapeAndRange.shape_range_ = shapeRange;
  shapeAndRange.shape_ = unknownShape;
  out.shape_ = shapeAndRange.shape_;
  out.shape_range_ = shapeAndRange.shape_range_;

  OP_LOGI(op.GetName().c_str(), "down = %lld, up = %lld, first = %lld, second = %lld \n",
          down, up, shapeRange[0].first, shapeRange[0].second);

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DropOutGenMaskInfer) {
  Shape unused;

  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("call WithRank function failed, ",
        GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()),
            "scalar"));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<std::string> input_infer_depends = {"shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  Tensor shape_tensor;
  auto ret = op.GetInputConstData("shape", shape_tensor);
  OP_LOGI(op.GetName().c_str(), "GetInputConstData ret = %d \n", ret);
  if (ret != GRAPH_SUCCESS) {
    ShapeAndRange shapeAndRange;
    if (GetDropOutGenMaskShapeAndRange(op, shapeAndRange) != GRAPH_SUCCESS) {
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        ConcatString("call GetDropOutGenMaskShapeAndRange function failed "));
      return GRAPH_FAILED;
    }
    TensorDesc outputDesc = op.GetOutputDesc("y");
    outputDesc.SetShape(shapeAndRange.shape_);
    outputDesc.SetShapeRange(shapeAndRange.shape_range_);
    outputDesc.SetDataType(DT_UINT8);
    return op.UpdateOutputDesc("y", outputDesc);
  }

  Shape shape;
  if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        ConcatString("call MakeShapeFromShapeTensor function failed to make "
            "shape by input[shape] data"));
    return GRAPH_FAILED;
  }

  int64_t n8s = InferShapeGetShapeSizeBytes(shape.GetShapeSize());
  OP_LOGI(op.GetName().c_str(), "n8s = %lld, shape.GetShapeSize() = %lld", n8s, shape.GetShapeSize());
  std::vector<int64_t> out_dims = {n8s};
  Shape outShape(out_dims);

  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetShape(outShape);
  outputDesc.SetDataType(DT_UINT8);
  return op.UpdateOutputDesc("y", outputDesc);
}

COMMON_INFER_FUNC_REG(DropOutGenMask, DropOutGenMaskInfer);
COMMON_INFER_FUNC_REG(StatelessDropOutGenMask, DropOutGenMaskInfer);

IMPLEMT_VERIFIER(StatelessDropOutGenMask, StatelessDropOutGenMasktVerify) {
  Shape unused;
  AscendString name;
  (void)op.GetName(name);
  if (WithRank(op.GetInputDesc(1), 0, unused, name.GetString()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("call WithRank function failed, ",
        GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()),
            "scalar"));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(name.GetString(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 0, unused, name.GetString()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("call WithRank function failed, ",
        GetShapeErrMsg(1, DebugString(op.GetInputDesc(2).GetShape().GetDims()),
            "scalar"));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(name.GetString(), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(3), 0, unused, name.GetString()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("call WithRank function failed, ",
        GetShapeErrMsg(1, DebugString(op.GetInputDesc(3).GetShape().GetDims()),
            "scalar"));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(name.GetString(), err_msg);
    return GRAPH_FAILED;
  }
  
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(StatelessDropOutGenMask, StatelessDropOutGenMasktVerify);

IMPLEMT_INFERFUNC(DropOutGenMaskV3, DropOutGenMaskV3Infer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("call WithRank function failed, ",
        GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()),
            "scalar"));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<std::string> input_infer_depends = {"shape"};
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(ge::Shape(ge::UNKNOWN_RANK));
    return op.UpdateOutputDesc("y", output_desc);
  }

  Shape shape;
  if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        ConcatString("call MakeShapeFromShapeTensor function failed to make "
            "shape by input[shape] data"));
    return GRAPH_FAILED;
  }

  Shape out_shape;
  if (ShapeFullDefined(shape)) {
    for (const auto& dim : shape.GetDims()) {
      if (dim < 0) {
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
            ConcatString("invalid value[", dim, "] of 0th input",
            DebugString(shape.GetDims())));
        return GRAPH_FAILED;
      }
    }
    // aign to 16
    const int64_t align = 16;
    size_t rank = shape.GetDimNum();
    // [*batch, M, N] -> [*batch, N/16, M/16, 16, 16]
    if (rank >= 2 &&
        shape.GetDim(rank - 1) % align == 0 &&
        shape.GetDim(rank - 2) % align == 0) {
      int64_t last_dim = shape.GetDim(rank - 1);
      int64_t second_last_dim = shape.GetDim(rank - 2);
      last_dim /= align;
      second_last_dim /= align;
      out_shape = shape;
      out_shape.SetDim(rank - 1, second_last_dim);
      out_shape.SetDim(rank - 2, last_dim);
      Shape tmp_shape({align, align});
      (void)Concatenate(out_shape, tmp_shape, out_shape);
    } else {
      uint64_t random_count = static_cast<uint64_t>(shape.GetShapeSize());
      // due to align to 16
      if (random_count > (INT64_MAX - 15)) {
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
            ConcatString("required random count[", random_count,
            "] exceed INT64_MAX - 15"));
        return GRAPH_FAILED;
      }
      // align to 16
      random_count = (random_count + 15) & (~15);
      std::vector<int64_t> out_dims = {static_cast<int64_t>(random_count)};
      out_shape = ge::Shape(out_dims);
    }
  } else {
    out_shape = ge::Shape(ge::UNKNOWN_RANK);
  }

  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(out_shape);
  output_desc.SetDataType(DT_UINT8);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(DropOutGenMaskV3, DropOutGenMaskV3Infer);

IMPLEMT_VERIFIER(Dropout, DropoutVerify) {
  auto dropout_ratio = op.get_attr_dropout_ratio();
  bool flagDropoutRatio = (dropout_ratio < 1 && dropout_ratio > 0);
  if (!flagDropoutRatio) {
    OP_LOGE(op.GetName().c_str(), "dropout_ratio must be in (0,1)");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(Dropout, DropoutVerify);

IMPLEMT_INFERFUNC(Dropout, DropoutInfer) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");

  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Dropout, DropoutInfer);

// --------------------LinSpaceD-------------------------
IMPLEMT_COMMON_INFERFUNC(LinSpaceDInferShape) {
  Shape input_shape = op.GetInputDesc("assist").GetShape();
  DataType input_dtype = op.GetInputDesc("assist").GetDataType();
  TensorDesc td = op.GetOutputDesc("output");
  td.SetShape(ge::Shape(input_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("output", td);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LinSpaceD, LinSpaceDInferShape);
// ------------------LinSpaceD END-----------------------------

// ------------------LinSpace-------------------------------
template <typename T>
static void GetLinSpaceConstValue(const Operator& op, const Tensor& const_tensor, std::vector<int64_t>& const_data) {
  size_t size = 0;

  T* const_data_ptr = (T*)const_tensor.GetData();
  size = const_tensor.GetSize() / sizeof(T);
  for (size_t i = 0; i < size; ++i) {
    const_data.push_back((int64_t)((*(const_data_ptr + i))));
    AscendString op_name;
    (void)op.GetName(op_name);
    OP_LOGI(op_name.GetString(), "const data float fusion pass ====== %ld", (int64_t)(*(const_data_ptr + i)));
  }
}

IMPLEMT_COMMON_INFERFUNC(LinSpaceInferShape) {
  Tensor input_num_tensor;
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  if (op.GetInputConstData("num", input_num_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(op_name.GetString(), "LinSpaceInferShape: GetInputConstData fail!");
    std::cout<< "LinSpaceInferShape: GetInputConstData fail" <<std::endl;
    DataType input_dtype = op.GetInputDescByName("start").GetDataType();
    std::cout<< "LinSpaceInferShape: input_dtype = " << input_dtype <<std::endl;
    std::vector<std::string> input_infer_depends = {"num"};
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    op_desc->SetOpInferDepends(input_infer_depends);

    TensorDesc td = op.GetOutputDesc("output");
    td.SetShape(ge::Shape(ge::UNKNOWN_RANK));
    td.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("output", td);
    return GRAPH_SUCCESS;
  }
  std::vector<int64_t> num_shape_vec;
  if (num_shape_vec.empty()) {
    OP_LOGI(op_name.GetString(), "num_shape_vec is empty!");
  }

  DataType data_type = op.GetInputDescByName("num").GetDataType();
  switch (data_type) {
    case DT_INT32:
      GetLinSpaceConstValue<int32_t>(op, input_num_tensor, num_shape_vec);
      break;
    case DT_INT64:
      GetLinSpaceConstValue<int64_t>(op, input_num_tensor, num_shape_vec);
      break;
    default:
      OP_LOGE(op_name.GetString(), "num_data_type is not support fail!");
      return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "The num is %d \n", num_shape_vec[0]);

  DataType input_dtype = op.GetInputDesc("start").GetDataType();
  TensorDesc td = op.GetOutputDesc("output");
  td.SetShape(ge::Shape(num_shape_vec));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("output", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LinSpace, LinSpaceInferShape);
// -------------------LinSpace END-----------------------

IMPLEMT_INFERFUNC(RandomChoiceWithMask, RandomChoiceWithMaskInfer) {
  TensorDesc output_y_desc = op.GetOutputDesc("y");
  output_y_desc.SetDataType(DT_INT32);
  TensorDesc output_mask_desc = op.GetOutputDesc("mask");
  output_mask_desc.SetDataType(DT_BOOL);
  Shape input_shape = op.GetInputDesc("x").GetShape();
  int64_t rank = static_cast<int64_t>(input_shape.GetDimNum());
  std::vector<int64_t> input_dims = op.GetInputDesc("x").GetShape().GetDims();
  if (input_dims.empty()) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
      std::string("input x should not be empty."));
    return GRAPH_FAILED;
  } else if (input_dims == UNKNOWN_RANK) {
    output_y_desc.SetShape(Shape({UNKNOWN_DIM, UNKNOWN_DIM}));
    output_mask_desc.SetShape(Shape({UNKNOWN_DIM}));
  } else {
    int64_t count;
    if (ge::GRAPH_SUCCESS != op.GetAttr("count", count)) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("get attr[count] failed."));
    }
    if (count > 0) {
      output_y_desc.SetShape(Shape({count, rank}));
      output_mask_desc.SetShape(Shape({count}));
    } else if (count == 0) {
      output_y_desc.SetShape(Shape({UNKNOWN_DIM, rank}));
      output_mask_desc.SetShape(Shape({UNKNOWN_DIM}));
      int64_t min_dim = 1;
      int64_t max_dim = 1;
      bool unknowshape = false;
      for (const auto& dim : input_dims) {
        if (dim == UNKNOWN_DIM) {
          unknowshape = true;
          break;
        }
        max_dim *= dim;
      }
      if (!unknowshape) {
        std::vector<std::pair<int64_t, int64_t>> y_range;
        std::vector<std::pair<int64_t, int64_t>> mask_range;
        auto p1 = std::make_pair(min_dim, max_dim);
        auto p2 = std::make_pair(rank, rank);
        y_range.push_back(p1);
        y_range.push_back(p2);
        mask_range.push_back(p1);
        output_y_desc.SetShapeRange(y_range);
        output_mask_desc.SetShapeRange(mask_range);
      }
    } else {
      std::string err_msg = ConcatString(
        "check attr[count] value[", count, "] failed, ",
        "must greater or equal to 0");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  if (op.UpdateOutputDesc("y", output_y_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("mask", output_mask_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RandomChoiceWithMask, RandomChoiceWithMaskInfer);

IMPLEMT_VERIFIER(ShuffleChannel, ShuffleChannelVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ShuffleChannelInferShape) {
  auto output_desc = op.GetInputDesc("x");
  return op.UpdateOutputDesc("y", output_desc);
}

COMMON_INFER_FUNC_REG(ShuffleChannel, ShuffleChannelInferShape);
VERIFY_FUNC_REG(ShuffleChannel, ShuffleChannelVerify);

// ----------------MultinomialFuss Begin-------------------
static bool CheckDtype(const ge::Operator& op) {
  int DATA_TYPE_INT32_ONNX = 6;
  int DATA_TYPE_INT64_ONNX = 7;
  int data_type = DATA_TYPE_INT32_ONNX;
  op.GetAttr("dtype", data_type);
  if (data_type != DATA_TYPE_INT32_ONNX && data_type != DATA_TYPE_INT64_ONNX) {
    OP_LOGE("MultinomialFussVerify", "data_type not int32 or int64");
    return false;
  }
  return true;
}

IMPLEMT_COMMON_INFERFUNC(MultinomialFussInferShape) {
  int DATA_TYPE_INT32_ONNX = 6;
  ge::TensorDesc input_desc = op.GetInputDesc(0);
  ge::TensorDesc output_desc = op.GetOutputDesc(0);
  std::vector<int64_t> dims = input_desc.GetShape().GetDims();
  int sample_size = 1;
  op.GetAttr("sample_size", sample_size);
  int size = dims.size();
  dims[size - 1] = sample_size;
  int data_type = DATA_TYPE_INT32_ONNX;
  op.GetAttr("dtype", data_type);
  ge::DataType data_type_om = DT_INT64;
  if (data_type == DATA_TYPE_INT32_ONNX) {
    data_type_om = DT_INT32;
  }
  output_desc.SetDataType(data_type_om);
  output_desc.SetFormat(input_desc.GetFormat());
  output_desc.SetShape(Shape(dims));
  op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MultinomialFuss, MultinomialFussVerify) {
  if (!CheckDtype(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(MultinomialFuss, MultinomialFussInferShape);
VERIFY_FUNC_REG(MultinomialFuss, MultinomialFussVerify);
// ----------------MultinomialFuss END---------------------

// ------------------DropoutV2 Start-------------------------------
IMPLEMT_COMMON_INFERFUNC(DropoutV2InferShape)
{
    const int axis_n = 0;
    TensorDesc x_desc = op.GetInputDesc(axis_n);
    TensorDesc y_desc = op.GetOutputDesc(axis_n);
    y_desc.SetShape(x_desc.GetShape());
    y_desc.SetFormat(x_desc.GetFormat());
    y_desc.SetDataType(x_desc.GetDataType());
    op.UpdateOutputDesc("y", y_desc);

    const int axis_c = 1;
    TensorDesc seed_desc = op.GetInputDesc(axis_c);
    TensorDesc mask_desc = op.GetOutputDesc(axis_c);
    mask_desc.SetShape(x_desc.GetShape());
    mask_desc.SetFormat(x_desc.GetFormat());
    mask_desc.SetDataType(seed_desc.GetDataType());
    op.UpdateOutputDesc("mask", mask_desc);

    const int axis_h = 2;
    TensorDesc new_seed_desc = op.GetOutputDesc(axis_h);
    new_seed_desc.SetShape(seed_desc.GetShape());
    new_seed_desc.SetFormat(seed_desc.GetFormat());
    new_seed_desc.SetDataType(seed_desc.GetDataType());
    op.UpdateOutputDesc("seed", new_seed_desc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(DropoutV2, DropoutV2Verify)
{
    TensorDesc x_desc = op.GetInputDesc(0);
    DataType x_dtype = x_desc.GetDataType();
    if (x_dtype != DT_FLOAT16 && x_dtype != DT_FLOAT) {
        OP_LOGE(op.GetName().c_str(), "DropoutV2 x input should be float16 or float32");
        return GRAPH_FAILED;
    }

    TensorDesc seed_desc = op.GetInputDesc(1);
    DataType seed_dtype = seed_desc.GetDataType();
    if (seed_dtype != DT_FLOAT) {
        OP_LOGE(op.GetName().c_str(), "DropoutV2 seed input should be float32");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DropoutV2, DropoutV2InferShape);
VERIFY_FUNC_REG(DropoutV2, DropoutV2Verify);
// ------------------DropoutV2 End-------------------------------

// ------------------Bernoulli Start-------------------------------
IMPLEMT_COMMON_INFERFUNC(BernoulliInferShape)
{
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x", "p", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Bernoulli, BernoulliInferShape);
// ------------------Bernoulli End-------------------------------

// ----------------------Uniform----------------------
IMPLEMT_COMMON_INFERFUNC(UniformInferShape) {

    TensorDesc tensordesc_input = op.GetInputDesc("x");
    (void)op.UpdateOutputDesc("y", tensordesc_input);

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Uniform, UniformInferShape);
//-----------------------Uniform END---------------------

// ------------------ContinuationIndicator Start-------------------------------
IMPLEMT_COMMON_INFERFUNC(ContinuationIndicatorInferShape) {
  int64_t time_step;
  (void)op.GetAttr("time_step", time_step);
  int64_t batch_size;
  (void)op.GetAttr("batch_size", batch_size);
  vector<int64_t> dims = {time_step, batch_size};
  ge::Shape shape(dims);
  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(ge::DT_FLOAT);
  op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ContinuationIndicator, ContinuationIndicatorInferShape);
// ------------------ContinuationIndicator End-------------------------------

IMPLEMT_INFERFUNC(Poisson, PoissonInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Poisson, PoissonInfer);

IMPLEMT_INFERFUNC(Geometric, GeometricInfer) {
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(op.GetInputDesc("x").GetShape());
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());

  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(Geometric, GeometricInfer);

}  // namespace ge
