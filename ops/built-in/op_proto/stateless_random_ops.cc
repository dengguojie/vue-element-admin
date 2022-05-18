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
 * \file stateless_random_ops.cpp
 * \brief
 */
#include "inc/stateless_random_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(StatelessMultinomial, StatelessMultinomialInfer) {
  auto logitsTensor = op.get_input_desc_logits();
  auto num_samplesTensor = op.get_input_desc_num_samples();
  auto seedTensor = op.get_input_desc_seed();

  Shape seed;
  std::string error_msg;
  if (WithRank(seedTensor, 1, seed, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[seed] must be 1, but get ",
        seedTensor.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (seed.GetDim(0) != 2) {
    error_msg = ConcatString("the value of dim[0] for input[seed] ",
        "must be 2, but the real value is", seed.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape logits_shape;
  Shape unused;
  if (WithRank(logitsTensor, 2, logits_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[logits] must be 2, but get ",
        logitsTensor.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(num_samplesTensor, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[num_samples] must be 0, but get ",
        num_samplesTensor.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  DataType dtype = op.get_input_desc_num_samples().GetDataType();
  if (dtype != DT_INT32) {
    error_msg = ConcatString("the dtype of input[num_samples] must be DT_INT32,",
        " but get ", DTypeStr(dtype), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Tensor numSamplesTensor;
  if (op.GetInputConstData("num_samples", numSamplesTensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get input[num_samples] const data fialed."));
    return GRAPH_FAILED;
  }
  int64_t numSamples;
  if (MakeDimForScalarInput(numSamplesTensor, numSamples, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeDimForScalarInput function ",
        "make dim for input[num_samples] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  (void)Matrix(logits_shape.GetDim(0), numSamples, shape);

  TensorDesc outputDesc = op.GetOutputDescByName("y");
  DataType output_type;
  if (op.GetAttr("output_dtype", output_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get attr[output_dtype] fialed."));
  }

  outputDesc.SetDataType(output_type);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatelessMultinomial, StatelessMultinomialInfer);

IMPLEMT_INFERFUNC(StatelessRandomUniformInt, StatelessRandomUniformIntInfer) {
  Shape unused;
  std::string error_msg;
  TensorDesc minval_desc = op.GetInputDesc(2);
  if (WithRank(minval_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[minval] must be 0, but get ",
        minval_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  TensorDesc maxval_desc = op.GetInputDesc(3);
  if (WithRank(maxval_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[maxval] must be 0, but get ",
        maxval_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape seed;
  TensorDesc seed_desc = op.GetInputDesc(1);
  if (WithRank(seed_desc, 1, seed, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[seed] must be 1, but get ",
        seed_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (seed.GetDim(0) != 2) {
    error_msg = ConcatString("the value of dim[0] for input[seed] must be 2, ",
        "but the real value is ", seed.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("failed to get input[shape] const data."));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeShapeFromShapeTensor function,",
        " make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDescByName("y");
  DataType output_type = op.GetInputDescByName("minval").GetDataType();
  outputDesc.SetDataType(output_type);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(StatelessRandomUniformInt, StatelessRandomUniformIntInfer);

IMPLEMT_VERIFIER(StatelessParameterizedTruncatedNormal, StatelessParameterizedTruncatedNormalVerify) {
  DataType shape_dtype = op.GetInputDescByName("shape").GetDataType();
  DataType seed_dtype = op.GetInputDescByName("seed").GetDataType();
  DataType means_dtype = op.GetInputDescByName("means").GetDataType();
  DataType stdevs_dtype = op.GetInputDescByName("stdevs").GetDataType();
  DataType min_dtype = op.GetInputDescByName("min").GetDataType();
  DataType max_dtype = op.GetInputDescByName("max").GetDataType();
  if ((means_dtype != stdevs_dtype) || (stdevs_dtype != min_dtype) || (max_dtype != min_dtype)) {
    string err_msg1 = ConcatString("Dtype of means, stdevs, min and max must be the same.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((means_dtype != DT_FLOAT16) && (means_dtype != DT_FLOAT) && (means_dtype != DT_DOUBLE)) {
    string err_msg1 = ConcatString("Dtype of means, stdevs, min and max must be one of float16, ",
                                   "float or double.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((seed_dtype != DT_INT32) && (seed_dtype != DT_INT64)) {
    string err_msg1 = ConcatString("Dtype of seed must be one of int32 and int64.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((shape_dtype != DT_INT32) && (shape_dtype != DT_INT64)) {
    string err_msg1 = ConcatString("Dtype of shape must be one of int32 and int64.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(StatelessParameterizedTruncatedNormalInferShape) {
  auto shape_shape = op.GetInputDescByName("shape").GetShape().GetDims();
  auto shape_means = op.GetInputDescByName("means").GetShape().GetDims();
  auto shape_stdevs = op.GetInputDescByName("stdevs").GetShape().GetDims();
  auto shape_min = op.GetInputDescByName("min").GetShape().GetDims();
  auto shape_max = op.GetInputDescByName("max").GetShape().GetDims();
  auto shape_seed = op.GetInputDescByName("seed").GetShape().GetDims();
  std::vector<int64_t> bcast_shape;
  int32_t dims1 = std::max(shape_means.size(), shape_stdevs.size());
  int32_t dims2 = std::max(shape_min.size(), shape_max.size());
  int32_t dims = std::max(dims1, dims2);
  (void)std::reverse(shape_means.begin(), shape_means.end());
  (void)std::reverse(shape_stdevs.begin(), shape_stdevs.end());
  (void)std::reverse(shape_min.begin(), shape_min.end());
  (void)std::reverse(shape_max.begin(), shape_max.end());
  (void)shape_means.resize(dims, 1);
  (void)shape_stdevs.resize(dims, 1);
  (void)shape_min.resize(dims, 1);
  (void)shape_max.resize(dims, 1);
  (void)std::reverse(shape_means.begin(), shape_means.end());
  (void)std::reverse(shape_stdevs.begin(), shape_stdevs.end());
  (void)std::reverse(shape_min.begin(), shape_min.end());
  (void)std::reverse(shape_max.begin(), shape_max.end());
  for (int32_t i = 0; i < dims; i++) {
    int32_t temp1 = std::max(shape_means[i], shape_stdevs[i]);
    int32_t temp2 = std::max(shape_min[i], shape_max[i]);
    int32_t temp = std::max(temp1, temp2);
    (void)bcast_shape.push_back((int64_t)temp);
    if ((shape_means[i] != bcast_shape[i] && shape_means[i] != 1) ||
        (shape_stdevs[i] != bcast_shape[i] && shape_stdevs[i] != 1) ||
        (shape_min[i] != bcast_shape[i] && shape_min[i] != 1) ||
        (shape_max[i] != bcast_shape[i] && shape_max[i] != 1)) {
      std::string err_msg = OtherErrMsg("Shapes of means, stdevs, min and max can't broadcast.");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (shape_shape[0] < dims) {
    string err_msg1 = ConcatString("Means, stdevs, min and max should not have larger rank than output.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((shape_seed[0] != 2) || (shape_seed.size() != 1)) {
    string err_msg1 = ConcatString("Seed must be a 1-D tensor with 2 elements.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  DataType y_dtype = op.GetInputDescByName("means").GetDataType();
  std::vector<int64_t> shape_dims;
  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) == GRAPH_SUCCESS) {
    if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) == GRAPH_SUCCESS) {
      for (int64_t i = 0; i < dims; ++i) {
        if (shape.GetDim(shape_shape[0] - dims + i) != bcast_shape[i]) {
          string err_msg1 = ConcatString("Shape passed in must end with broadcasted shape.");
          std::string err_msg = OtherErrMsg(err_msg1);
          AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
          return GRAPH_FAILED;
        }
      }
      (void)tensordesc_output.SetShape(Shape(shape));
    } else {
      for (int64_t i = 0; i < shape_shape[0]; i++) {
        (void)shape_dims.push_back(ge::UNKNOWN_DIM);
      }
      (void)tensordesc_output.SetShape(Shape(shape_dims));
    }
  } else {
    for (int64_t i = 0; i < shape_shape[0]; i++) {
      (void)shape_dims.push_back(ge::UNKNOWN_DIM);
    }
    (void)tensordesc_output.SetShape(Shape(shape_dims));
  }
  (void)tensordesc_output.SetDataType(y_dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(StatelessParameterizedTruncatedNormal, StatelessParameterizedTruncatedNormalVerify);
COMMON_INFER_FUNC_REG(StatelessParameterizedTruncatedNormal, StatelessParameterizedTruncatedNormalInferShape);

IMPLEMT_VERIFIER(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBoxVerify) {
  DataType image_size_dtype = op.GetInputDescByName("image_size").GetDataType();
  DataType seed_dtype = op.GetInputDescByName("seed").GetDataType();
  DataType bounding_boxes_dtype = op.GetInputDescByName("bounding_boxes").GetDataType();
  DataType min_object_covered_dtype = op.GetInputDescByName("min_object_covered").GetDataType();
  if ((image_size_dtype != DT_UINT8) && (image_size_dtype != DT_INT8) && (image_size_dtype != DT_INT16) &&
      (image_size_dtype != DT_INT32) && (image_size_dtype != DT_INT64)) {
    string err_msg1 = ConcatString("Dtype of image_size must be one of uint8, int8, int16, int32, int64.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (bounding_boxes_dtype != DT_FLOAT) {
    string err_msg1 = ConcatString("Dtype of bounding_boxes must be float.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (min_object_covered_dtype != DT_FLOAT) {
    string err_msg1 = ConcatString("Dtype of min_object_covered must be float.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((seed_dtype != DT_INT32) && (seed_dtype != DT_INT64)) {
    string err_msg1 = ConcatString("Dtype of seed must be one of int32 and int64.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBoxInfer) {
  bool judge = false;
  Shape image_size;
  judge = (WithRank(op.get_input_desc_image_size(), 1, image_size, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "Failed to call WithRank function, input[image_size] rank must be 1, "
        "got rank[", op.get_input_desc_image_size().GetShape().GetDimNum(), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape bounding_boxes;
  judge = (WithRank(op.get_input_desc_bounding_boxes(), 3, bounding_boxes, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "Failed to call WithRank function, input[bounding_boxes] rank must be 3, "
        "got rank[", op.get_input_desc_image_size().GetShape().GetDimNum(), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape min_object_covered;
  judge =
      (WithRank(op.get_input_desc_min_object_covered(), 0, min_object_covered, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "Failed to call WithRank function, input[min_object_covered] rank must "
        "be scalar, got rank[", op.get_input_desc_image_size().GetShape().GetDimNum(), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape seed;
  judge =
      (WithRank(op.get_input_desc_seed(), 1, seed, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "Failed to call WithRank function, input[seed] rank must "
        "be 1, got rank[", op.get_input_desc_image_size().GetShape().GetDimNum(), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  const int64_t image_size_dim_value = op.get_input_desc_image_size().GetShape().GetDim(0);
  const int64_t bounding_boxes_dim2_value = op.get_input_desc_bounding_boxes().GetShape().GetDim(2);
  const int64_t seed_dim_value = op.get_input_desc_seed().GetShape().GetDim(0);
  if (((image_size_dim_value != 3) && (image_size_dim_value != -1)) ||
     ((bounding_boxes_dim2_value != 4) && (bounding_boxes_dim2_value != -1)) ||
     ((seed_dim_value != 2) && (seed_dim_value != -1))) {
    std::string err_msg = ConcatString(
        "0th dim of input[image_size] must be 3 or -1, got[", image_size_dim_value,
        "], 2nd dim of input[bounding_boxes] must be 4 or -1, got[",
        bounding_boxes_dim2_value, "] and 0th dim of input[seed] must be 2 or -1, got[",
        seed_dim_value, "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc begin_desc = op.GetOutputDescByName("begin");
  begin_desc.SetShape(Shape({3}));
  begin_desc.SetDataType(op.GetInputDescByName("image_size").GetDataType());
  if (op.UpdateOutputDesc("begin", begin_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("Fail to update output[begin] desc."));
    return GRAPH_FAILED;
  }
  TensorDesc size_desc = op.GetOutputDescByName("size");
  size_desc.SetShape(Shape({3}));
  size_desc.SetDataType(op.GetInputDescByName("image_size").GetDataType());
  if (op.UpdateOutputDesc("size", size_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("Fail to update output[size] desc."));
    return GRAPH_FAILED;
  }
  TensorDesc bboxes_desc = op.GetOutputDescByName("bboxes");
  bboxes_desc.SetShape(Shape({1, 1, 4}));
  bboxes_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("bboxes", bboxes_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("Fail to update output[bboxes] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBoxVerify);
INFER_FUNC_REG(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBoxInfer);

IMPLEMT_VERIFIER(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2Verify) {
  DataType shape_dtype = op.GetInputDescByName("shape").GetDataType();
  DataType key_dtype = op.GetInputDescByName("key").GetDataType();
  DataType counter_dtype = op.GetInputDescByName("counter").GetDataType();
  DataType alg_dtype = op.GetInputDescByName("alg").GetDataType();
  if ((key_dtype != DT_UINT64) || (counter_dtype != DT_UINT64)) {
    string err_msg1 = ConcatString("Dtype of key and counter must be uint64.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (alg_dtype != DT_INT32) {
    string err_msg1 = ConcatString("Dtype of alg must be int32.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((shape_dtype != DT_INT32) && (shape_dtype != DT_INT64)) {
    string err_msg1 = ConcatString("Dtype of shape must be one of int32 and int64.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  ge::DataType dtype;
  (void)op.GetAttr("dtype", dtype);
  if ((dtype != DT_FLOAT16) && (dtype != DT_FLOAT) && (dtype != DT_DOUBLE)) {
    OP_LOGE(TbeGetName(op).c_str(),
            "The attr 'dtype' must be one of DT_FLOAT16, DT_FLOAT, DT_DOUBLE.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(StatelessTruncatedNormalV2InferShape) {
  auto shape_shape = op.GetInputDescByName("shape").GetShape().GetDims();
  auto shape_key = op.GetInputDescByName("key").GetShape().GetDims();
  auto shape_counter = op.GetInputDescByName("counter").GetShape().GetDims();
  auto shape_alg = op.GetInputDescByName("alg").GetShape();
  if ((shape_key[0] != 1) || (shape_key.size() != 1)) {
    string err_msg1 = ConcatString("Key must be a 1-D tensor with 1 elements.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((shape_counter[0] != 2) || (shape_counter.size() != 1)) {
    string err_msg1 = ConcatString("counter must be a 1-D tensor with 2 elements.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (shape_alg.GetDimNum() != 0) {
    string err_msg1 = ConcatString("Alg must be a scalar.");
    std::string err_msg = OtherErrMsg(err_msg1);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Tensor alg_tensor;
  if (op.GetInputConstData("alg", alg_tensor) == GRAPH_SUCCESS) {
    auto alg_data = reinterpret_cast<int32_t *>(alg_tensor.GetData());
    if (*alg_data != 1) {
      string err_msg1 = ConcatString("The RNG algorithm must be philox.");
      std::string err_msg = OtherErrMsg(err_msg1);
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  ge::DataType dtype;
  (void)op.GetAttr("dtype", dtype);
  std::vector<int64_t> shape_dims;
  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) == GRAPH_SUCCESS) {
    if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) == GRAPH_SUCCESS) {
      (void)tensordesc_output.SetShape(Shape(shape));
    } else {
      for (int64_t i = 0; i < shape_shape[0]; i++) {
        (void)shape_dims.push_back(ge::UNKNOWN_DIM);
      }
      (void)tensordesc_output.SetShape(Shape(shape_dims));
    }
  } else {
    for (int64_t i = 0; i < shape_shape[0]; i++) {
      (void)shape_dims.push_back(ge::UNKNOWN_DIM);
    }
    (void)tensordesc_output.SetShape(Shape(shape_dims));
  }
  (void)tensordesc_output.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2Verify);
COMMON_INFER_FUNC_REG(StatelessTruncatedNormalV2, StatelessTruncatedNormalV2InferShape);

}  // namespace ge
