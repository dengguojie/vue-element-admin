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
    OP_LOGE(op.GetName().c_str(), "Multinomial: get attr dtype failed");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(type);
  output_desc.SetShape(shape);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(Multinomial, MultinomialInfer);

IMPLEMT_INFERFUNC(ParameterizedTruncatedNormal, ParameterizedTruncatedNormalInfer) {
  Shape unused;
  if (WithRankAtMost(op.GetInputDesc(1), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(2), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(3), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(4), 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
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
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  return op.UpdateOutputDesc("y", output_desc);
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
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  alpha_shape = op.GetInputDesc(1).GetShape();
  (void)Concatenate(shape, alpha_shape, shape);

  auto type = op.GetInputDesc("alpha").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(type);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(RandomGamma, RandomGammaInfer);

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
    OP_LOGE(op.GetName().c_str(), "RandomPoisson: get attr dtype failed");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(type);
  output_desc.SetShape(shape);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(RandomPoisson, RandomPoissonInfer);

IMPLEMT_INFERFUNC(RandomShuffle, RandomShuffleInfer) {
  TensorDesc desc = op.GetInputDesc(0);
  Shape shape = desc.GetShape();

  auto type = op.GetInputDesc("x").GetDataType();
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(type);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(RandomShuffle, RandomShuffleInfer);

IMPLEMT_INFERFUNC(RandomStandardNormal, RandomStandardNormalInfer) {
  return RandomShapeWithDataType(op, "shape", "dtype", "y");
}

INFER_FUNC_REG(RandomStandardNormal, RandomStandardNormalInfer);

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

IMPLEMT_INFERFUNC(DropOutGenMask, DropOutGenMaskInfer) {
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  Shape unused;
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  Shape shape;
  if (MakeShapeFromShapeTensor(shape_tensor, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  int64_t shape_size = shape.GetShapeSize();
  int64_t n128s = shape_size / 128;
  // align to 128
  if ((shape_size % 128) != 0) {
    n128s++;
  }
  int64_t n8s = n128s * 16;
  std::vector<int64_t> out_dims = {n8s};
  Shape outShape(out_dims);

  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(outShape);
  output_desc.SetDataType(DT_UINT8);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(DropOutGenMask, DropOutGenMaskInfer);

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
static void GetLinSpaceConstValue(const Operator& op, const Tensor& const_tensor, std::vector<int64_t>& const_data) {
  size_t size = 0;
  int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
  size = const_tensor.GetSize() / sizeof(int32_t);
  for (size_t i = 0; i < size; ++i) {
    const_data.push_back((int32_t)((*(const_data_ptr + i))));
    OP_LOGI(op.GetName().c_str(), "const data float fusion pass ====== %d", (int32_t)(*(const_data_ptr + i)));
  }
}

IMPLEMT_COMMON_INFERFUNC(LinSpaceInferShape) {
  Tensor input_num_tensor;
  if (op.GetInputConstData("num", input_num_tensor) != GRAPH_SUCCESS) {
    std::vector<int64_t> shape_vec;
    Shape shape_start = op.GetInputDesc("num").GetShape();
    for (size_t dim = 0; dim < shape_start.GetDimNum(); dim++) {
      shape_vec.push_back(-1);
    }
    DataType input_dtype = op.GetInputDesc("output").GetDataType();
    TensorDesc td = op.GetOutputDesc("output");
    td.SetShape(ge::Shape(shape_vec));
    td.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("output", td);
    return GRAPH_SUCCESS;
  }
  std::vector<int64_t> num_shape_vec;
  if (num_shape_vec.empty()) {
    OP_LOGI(op.GetName().c_str(), "num_shape_vec is empty!");
  }
  GetLinSpaceConstValue(op, input_num_tensor, num_shape_vec);
  OP_LOGI(op.GetName().c_str(), "The num is %d \n", num_shape_vec[0]);

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
    OP_LOGE(op.GetName().c_str(), "input x should not be empty.");
    return GRAPH_FAILED;
  } else if (input_dims == UNKNOWN_RANK) {
    output_y_desc.SetShape(Shape({UNKNOWN_DIM, UNKNOWN_DIM}));
    output_mask_desc.SetShape(Shape({UNKNOWN_DIM}));
  } else {
    int64_t count;
    if (ge::GRAPH_SUCCESS != op.GetAttr("count", count)) {
      OP_LOGE(op.GetName().c_str(), "get attr failed");
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
      OP_LOGE(op.GetName().c_str(), "input count must greater or equal to 0 but instead is %lld.", count);
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
}  // namespace ge
