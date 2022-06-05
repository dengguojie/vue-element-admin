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
#include "error_util.h"
#include "util/util.h"

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

bool JudgeRank(Operator& op){
  bool judge = false;
  Shape image_size;
  constexpr int64_t BoundingBoxes= 3;
  judge = (WithRank(op.GetInputDescByName("image_size"), 1, image_size, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "Failed to call WithRank function, input[image_size] rank must be 1, "
        "got rank[",
        op.GetInputDescByName("image_size").GetShape().GetDimNum(), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return judge;
  }
  Shape bounding_boxes;
  judge = (WithRank(op.GetInputDescByName("bounding_boxes"), BoundingBoxes, bounding_boxes, TbeGetName(op).c_str()) != 
           GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "Failed to call WithRank function, input[bounding_boxes] rank must be 3, "
        "got rank[", op.GetInputDescByName("bounding_boxes").GetShape().GetDimNum(), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return judge;
  }
  Shape min_object_covered;
  judge = (WithRank(op.GetInputDescByName("min_object_covered"), 0, min_object_covered, TbeGetName(op).c_str()) !=
           GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "Failed to call WithRank function, input[min_object_covered] rank must "
        "be scalar, got rank[", op.GetInputDescByName("min_object_covered").GetShape().GetDimNum(), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return judge;
  }
  Shape seed;
  judge = (WithRank(op.GetInputDescByName("seed"), 1, seed, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "Failed to call WithRank function, input[seed] rank must "
        "be 1, got rank[", op.GetInputDescByName("seed").GetShape().GetDimNum(), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return judge;
  }
  return judge;
}

IMPLEMT_INFERFUNC(StatelessSampleDistortedBoundingBox, StatelessSampleDistortedBoundingBoxInfer) {
  if(JudgeRank(op)){
    return GRAPH_FAILED;
  };
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("Fail to update output[begin] desc."));
    return GRAPH_FAILED;
  }
  TensorDesc size_desc = op.GetOutputDescByName("size");
  size_desc.SetShape(Shape({3}));
  size_desc.SetDataType(op.GetInputDescByName("image_size").GetDataType());
  if (op.UpdateOutputDesc("size", size_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("Fail to update output[size] desc."));
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

// ----------------StatelessRandomGammaV2InferShape Begin-------------------
IMPLEMT_INFERFUNC(StatelessRandomGammaV2, StatelessRandomGammaV2Infer) {
  TensorDesc alpha_desc = op.GetInputDescByName("alpha");
  Shape shape, seed, alpha;
  std::string error_msg;
  // check and get shape
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
  // check alpha
  if (WithRank(alpha_desc, 1, alpha, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
                             "the rank of input[alpha] must be 1, but get ",
                             alpha_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  int num = shape.GetDimNum();
  if (alpha.GetDim(0) != shape.GetDim(num - 1)) {
    error_msg = ConcatString("Shape of input[alpha] passed in must end with broadcasted shape, ",
                             "but the real value is ", alpha.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  // infer output shape and type
  TensorDesc outputDesc = op.GetOutputDescByName("y");
  DataType output_type = op.GetInputDescByName("alpha").GetDataType();
  outputDesc.SetDataType(output_type);
  outputDesc.SetShape(shape);
  (void)op.UpdateOutputDesc("y", outputDesc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(StatelessRandomGammaV2, StatelessRandomGammaV2Verify) {
  TensorDesc seed_desc = op.GetInputDescByName("seed");
  Shape seed;
  std::string error_msg;
  // check seed
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
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StatelessRandomGammaV2, StatelessRandomGammaV2Infer);

VERIFY_FUNC_REG(StatelessRandomGammaV2, StatelessRandomGammaV2Verify);
// ----------------StatelessRandomGammaV2InferShape End-------------------

IMPLEMT_INFERFUNC(StatelessRandomUniformFullInt, StatelessRandomUniformFullIntInfer) {
  std::string error_msg;

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("failed to get input[shape] const data."));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
        TbeGetName(op),
        std::string("failed to call MakeShapeFromShapeTensor function, make shape for output[y] failed."));
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDescByName("y");
  DataType output_type = DT_INT32;
  if (op.GetAttr("dtype", output_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get attr[dtype] fialed."));
  }
  outputDesc.SetDataType(output_type);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

IMPLEMT_VERIFIER(StatelessRandomUniformFullInt, StatelessRandomUniformFullIntVerify) {
  std::string error_msg;
  Shape seed;
  TensorDesc seed_desc = op.GetInputDescByName("seed");
  if (WithRank(seed_desc, 1, seed, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ", "the rank of input[seed] must be 1, but get ",
                             seed_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (seed.GetDim(0) != 2) {
    error_msg =
        ConcatString("the value of dim[0] for input[seed] must be 2, ", "but the real value is ", seed.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StatelessRandomUniformFullInt, StatelessRandomUniformFullIntInfer);
VERIFY_FUNC_REG(StatelessRandomUniformFullInt, StatelessRandomUniformFullIntVerify);
// ----------------StatelessRandomUniformFullIntInferShape End-------------------

graphStatus CheckStatelessRandomUniformFullIntV2Params(Operator& op) {
 std::string error_msg;
  // alg-start
  Shape alg;
  Tensor alg_tensor;
  if (op.GetInputConstData("alg", alg_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("failed to get input[alg] const data."));
    return GRAPH_FAILED;
  }
  int32_t* const_data_ptr = reinterpret_cast<int32_t*>(alg_tensor.GetData());
  int32_t alg_value = *(const_data_ptr);

  TensorDesc alg_desc = op.GetInputDescByName("alg");
  if (WithRank(alg_desc, 0, alg, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ", "the rank of input[alg] must be 0, but get ",
                             alg_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  // alg-end

  // counter-start
  Shape counter;
  TensorDesc counter_desc = op.GetInputDescByName("counter");
  if (WithRank(counter_desc, 1, counter, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ", "the rank of input[counter] must be 1, but get ",
                             counter_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (alg_value == INPUT_NUM2 && counter.GetDim(0) != 1) {
    error_msg = ConcatString("the value of dim[0] for input[counter] must be 1, ", "but the real value is ",
                             counter.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (alg_value == 1 && counter.GetDim(0) != INPUT_NUM2) {
    error_msg = ConcatString("the value of dim[0] for input[counter] must be 2, ", "but the real value is ",
                             counter.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(StatelessRandomUniformFullIntV2, StatelessRandomUniformFullIntV2Infer) {
  std::string error_msg;

  if (CheckStatelessRandomUniformFullIntV2Params(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  // counter-end
  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("failed to get input[shape] const data."));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeShapeFromShapeTensor function,", " make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDescByName("y");
  DataType output_type = DT_INT32;
  if (op.GetAttr("dtype", output_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get attr[dtype] fialed."));
  }
  outputDesc.SetDataType(output_type);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

IMPLEMT_VERIFIER(StatelessRandomUniformFullIntV2, StatelessRandomUniformFullIntV2Verify) {
  std::string error_msg;
  // key-start
  Shape key;
  TensorDesc key_desc = op.GetInputDescByName("key");
  if (WithRank(key_desc, 1, key, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ", "the rank of input[key] must be 1, but get ",
                             key_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (key.GetDim(0) != 1) {
    error_msg =
        ConcatString("the value of dim[0] for input[key] must be 1, ", "but the real value is ", key.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  // key-end
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StatelessRandomUniformFullIntV2, StatelessRandomUniformFullIntV2Infer);
VERIFY_FUNC_REG(StatelessRandomUniformFullIntV2, StatelessRandomUniformFullIntV2Verify);
// ----------------StatelessRandomUniformFullIntV2InferShape End-------------------
graphStatus CheckStatelessRandomUniformIntV2Params(Operator& op) {
  std::string error_msg;

  // alg-start
  Shape alg;
  Tensor alg_tensor;
  if (op.GetInputConstData("alg", alg_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("failed to get input[alg] const data."));
    return GRAPH_FAILED;
  }
  int32_t* const_data_ptr = reinterpret_cast<int32_t*>(alg_tensor.GetData());
  int32_t alg_value = *(const_data_ptr);

  TensorDesc alg_desc = op.GetInputDescByName("alg");
  if (WithRank(alg_desc, 0, alg, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ", "the rank of input[alg] must be 0, but get ",
                             alg_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  // alg-end

  // counter-start
  Shape counter;
  TensorDesc counter_desc = op.GetInputDescByName("counter");
  if (WithRank(counter_desc, 1, counter, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ", "the rank of input[counter] must be 1, but get ",
                             counter_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (alg_value == INPUT_NUM2 && counter.GetDim(0) != 1) {
    error_msg = ConcatString("the value of dim[0] for input[counter] must be 1, ", "but the real value is ",
                             counter.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (alg_value == 1 && counter.GetDim(0) != INPUT_NUM2) {
    error_msg = ConcatString("the value of dim[0] for input[counter] must be 2, ", "but the real value is ",
                             counter.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  // counter-end
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(StatelessRandomUniformIntV2, StatelessRandomUniformIntV2Infer) {
  std::string error_msg;
  if (CheckStatelessRandomUniformIntV2Params(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("failed to get input[shape] const data."));
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeShapeFromShapeTensor function,", " make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDescByName("y");
  DataType output_type = op.GetInputDescByName("minval").GetDataType();

  outputDesc.SetDataType(output_type);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}
IMPLEMT_VERIFIER(StatelessRandomUniformIntV2, StatelessRandomUniformIntV2Verify) {
  std::string error_msg;
  // key-start
  Shape key;
  TensorDesc key_desc = op.GetInputDescByName("key");
  if (WithRank(key_desc, 1, key, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ", "the rank of input[key] must be 1, but get ",
                             key_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (key.GetDim(0) != 1) {
    error_msg =
        ConcatString("the value of dim[0] for input[key] must be 1, ", "but the real value is ", key.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  // key-end

  Shape unused;
  TensorDesc minval_desc = op.GetInputDescByName("minval");
  if (WithRank(minval_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ", "the rank of input[minval] must be 0, but get ",
                             minval_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  TensorDesc maxval_desc = op.GetInputDescByName("maxval");
  if (WithRank(maxval_desc, 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ", "the rank of input[maxval] must be 0, but get ",
                             maxval_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StatelessRandomUniformIntV2, StatelessRandomUniformIntV2Infer);
VERIFY_FUNC_REG(StatelessRandomUniformIntV2, StatelessRandomUniformIntV2Verify);
// ----------------StatelessRandomUniformIntV2InferShape End-------------------

IMPLEMT_VERIFIER(StatelessRandomBinomial, StatelessRandomBinomialVerify) {
  DataType probs_dtype = op.GetInputDescByName("probs").GetDataType();
  DataType cpunts_dtype = op.GetInputDescByName("counts").GetDataType();
  if ((probs_dtype != cpunts_dtype)) {
    string err_msg1 = ConcatString("Dtype of probs and counts must be the same.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<DataType> support_list;
  support_list.push_back(DT_FLOAT16);
  support_list.push_back(DT_FLOAT);
  support_list.push_back(DT_DOUBLE);
  support_list.push_back(DT_INT32);
  support_list.push_back(DT_INT64);
  if (CheckInputDataType(op, "probs", support_list) == false) {
    return GRAPH_FAILED;
  }

  std::vector<DataType> int_support_list;
  int_support_list.push_back(DT_INT32);
  int_support_list.push_back(DT_INT64);
  if (CheckInputDataType(op, "seed", int_support_list) == false) {
    return GRAPH_FAILED;
  }
  if (CheckInputDataType(op, "shape", int_support_list) == false) {
    return GRAPH_FAILED;
  }

  ge::DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    dtype = DT_INT32;
  }
  if ((dtype != DT_INT32) && (dtype != DT_INT64) && (dtype != DT_FLOAT16) &&
      (dtype != DT_FLOAT) && (dtype != DT_DOUBLE)) {
    OP_LOGE(TbeGetName(op).c_str(),
            "The attr 'dtype' must be one of DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(StatelessRandomBinomialInferShape) {
  auto shape_shape = op.GetInputDescByName("shape").GetShape().GetDims();
  auto shape_probs = op.GetInputDescByName("probs").GetShape().GetDims();
  auto shape_counts = op.GetInputDescByName("counts").GetShape().GetDims();
  auto shape_seed = op.GetInputDescByName("seed").GetShape().GetDims();
  std::vector<int64_t> bcast_shape;
  int32_t dims = std::max(shape_probs.size(), shape_counts.size());
  (void)std::reverse(shape_probs.begin(), shape_probs.end());
  (void)std::reverse(shape_counts.begin(), shape_counts.end());
  (void)shape_probs.resize(dims, 1);
  (void)shape_counts.resize(dims, 1);
  (void)std::reverse(shape_probs.begin(), shape_probs.end());
  (void)std::reverse(shape_counts.begin(), shape_counts.end());

  for (int32_t i = 0; i < dims; i++) {
    (void)bcast_shape.push_back((int64_t)std::max(shape_probs[i], shape_counts[i]));
    CHECK(((shape_probs[i] != bcast_shape[i] && shape_probs[i] != 1) ||
           (shape_counts[i] != bcast_shape[i] && shape_counts[i] != 1)),
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
               OtherErrMsg("Shapes of probs and counts can't broadcast.")),
          return GRAPH_FAILED);
  }
  if (shape_shape[0] < dims) {
    string err_msg1 = ConcatString("Neither of probs and counts should have larger rank than output.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((shape_seed[0] != 2) || (shape_seed.size() != 1)) {
    string err_msg1 = ConcatString("Seed must be a 1-D tensor with 2 elements.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  ge::DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    dtype = DT_INT32;
  }
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
  (void)tensordesc_output.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(StatelessRandomBinomial, StatelessRandomBinomialInferShape);
VERIFY_FUNC_REG(StatelessRandomBinomial, StatelessRandomBinomialVerify);
// ----------------StatelessRandomBinomialInferShape End-------------------

IMPLEMT_VERIFIER(StatelessRandomPoisson, StatelessRandomPoissonVerify) {
  TensorDesc seed_desc = op.GetInputDescByName("seed");
  Shape seed;
  std::string error_msg;
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
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(StatelessRandomPoisson, StatelessRandomPoissonVerify);

IMPLEMT_INFERFUNC(StatelessRandomPoisson, StatelessRandomPoissonInfer) {
  Tensor shape_tensor;
  std::string error_msg;
  TensorDesc lam_desc = op.GetInputDescByName("lam");
  Shape y_shape, lam;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("failed to get input[shape] const data."));
    return GRAPH_FAILED;
  }

  if (MakeShapeFromShapeTensor(shape_tensor, y_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call MakeShapeFromShapeTensor function,",
        " make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(lam_desc, 1, lam, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
        "the rank of input[lam] must be 1, but get ",
        lam_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  int num = y_shape.GetDimNum();
  if (lam.GetDim(0) != y_shape.GetDim(num - 1)) {
    error_msg = ConcatString("the value of dim[0] for input[lam] must be last dim of input[shape], ",
        "but the real value  is ", lam.GetDim(0), " and ", y_shape.GetDim(num - 1), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  TensorDesc outputDesc = op.GetOutputDescByName("y");
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get attr[dtype] fialed."));
  }

  outputDesc.SetDataType(dtype);
  outputDesc.SetShape(y_shape);
  (void)op.UpdateOutputDesc("y", outputDesc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StatelessRandomPoisson, StatelessRandomPoissonInfer);
// ----------------StatelessRandomPoissonInferShape End-------------------

// ----------------StatelessRandomGetAlg Begin----------------------------
IMPLEMT_INFERFUNC(StatelessRandomGetAlg, StatelessRandomGetAlgInfer) {
  TensorDesc alg_desc = op.GetOutputDescByName("alg");
  std::vector<int64_t> alg_dim = {};
  Shape alg_shape(alg_dim);
  alg_desc.SetShape(alg_shape);
  alg_desc.SetDataType(DT_INT32);
  op.UpdateOutputDesc("alg", alg_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StatelessRandomGetAlg, StatelessRandomGetAlgInfer);
// ----------------StatelessRandomGetAlg End------------------------------

// ----------------StatelessRandomGetKeyCounter Begin---------------------
static constexpr int RNG_KEY_SIZE = 1;
static constexpr int RNG_MAX_COUNTER_SIZE = 2;

IMPLEMT_INFERFUNC(StatelessRandomGetKeyCounter,
                  StatelessRandomGetKeyCounterInfer) {
  TensorDesc outputDesc = op.GetOutputDescByName("key");
  outputDesc.SetShape(Shape({RNG_KEY_SIZE}));
  outputDesc.SetDataType(DT_UINT64);
  (void)op.UpdateOutputDesc("key", outputDesc);

  TensorDesc outputDesc2 = op.GetOutputDescByName("counter");
  outputDesc2.SetShape(Shape({RNG_MAX_COUNTER_SIZE}));
  outputDesc2.SetDataType(DT_UINT64);
  (void)op.UpdateOutputDesc("counter", outputDesc2);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StatelessRandomGetKeyCounter, StatelessRandomGetKeyCounterInfer);

IMPLEMT_VERIFIER(StatelessRandomGetKeyCounter,
                 StatelessRandomGetKeyCounterVerify) {
  std::string error_msg;
  TensorDesc seed_desc = op.GetInputDescByName("seed");
  const std::vector<DataType> seedSupportList = {DT_INT32, DT_INT64};
  if (!CheckInputDataType(op, "seed", seedSupportList)) {
    return GRAPH_FAILED;
  }
  Shape seed;
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
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(StatelessRandomGetKeyCounter,
                StatelessRandomGetKeyCounterVerify);
// ----------------StatelessRandomGetKeyCounter End--------------------------

// ----------------StatelessRandomGetKeyCounterAlg Begin---------------------
IMPLEMT_INFERFUNC(StatelessRandomGetKeyCounterAlg,
                  StatelessRandomGetKeyCounterAlgInfer) {
  const std::vector<DataType> seedSupportList = {DT_INT32, DT_INT64};
  if (!CheckInputDataType(op, "seed", seedSupportList)) {
    return GRAPH_FAILED;
  }

  std::string error_msg;
  Shape seed_shape;
  TensorDesc seed_desc = op.GetInputDescByName("seed");
  if (WithRank(seed_desc, 1, seed_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ",
                             "the rank of input[seed] must be 1, but get ",
                             seed_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  if (seed_shape.GetDim(0) != 2) {
    error_msg =
        ConcatString("the value of dim[0] for input[seed] must be 2, ",
                     "but the real value is ", seed_shape.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc key_desc = op.GetOutputDescByName("key");
  std::vector<int64_t> key_dim = {1};
  Shape key_shape(key_dim);
  key_desc.SetShape(key_shape);
  key_desc.SetDataType(DT_UINT64);
  op.UpdateOutputDesc("key", key_desc);

  TensorDesc counter_desc = op.GetOutputDescByName("counter");
  std::vector<int64_t> counter_dim = {2};
  Shape counter_shape(counter_dim);
  counter_desc.SetShape(counter_shape);
  counter_desc.SetDataType(DT_UINT64);
  op.UpdateOutputDesc("counter", counter_desc);

  TensorDesc alg_desc = op.GetOutputDescByName("alg");
  std::vector<int64_t> alg_dim = {};
  Shape alg_shape(alg_dim);
  alg_desc.SetShape(alg_shape);
  alg_desc.SetDataType(DT_INT32);
  op.UpdateOutputDesc("alg", alg_desc);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StatelessRandomGetKeyCounterAlg,
               StatelessRandomGetKeyCounterAlgInfer);
// ----------------StatelessRandomGetKeyCounterAlg End-----------------------

// ----------------StatelessRandomNormalV2 Begin--------------------------
IMPLEMT_INFERFUNC(StatelessRandomNormalV2, StatelessRandomNormalV2Infer) {
  std::string error_msg;
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    dtype = DT_FLOAT;
  }
  if ((dtype != DT_FLOAT16) && (dtype != DT_FLOAT) && (dtype != DT_DOUBLE)) {
    error_msg = ConcatString(
        "attr[dtype] data type should be DT_FLOAT16,  DT_FLOAT or DT_DOUBLE, "
        "got[",
        DTypeStr(dtype), "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    error_msg = "failed to get input[shape] const data.";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  if (MakeShapeFromShapeTensor(shape_tensor, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    error_msg =
        ConcatString("failed to call MakeShapeFromShapeTensor function,",
                     " make shape for output[y] failed.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetDataType(dtype);
  y_desc.SetShape(shape);
  return op.UpdateOutputDesc("y", y_desc);
}

INFER_FUNC_REG(StatelessRandomNormalV2, StatelessRandomNormalV2Infer);

IMPLEMT_VERIFIER(StatelessRandomNormalV2, StatelessRandomNormalV2Verify) {
  const std::vector<DataType> keySupportList = {DT_UINT64};
  if (!CheckInputDataType(op, "key", keySupportList)) {
    return GRAPH_FAILED;
  }

  const std::vector<DataType> counterSupportList = {DT_UINT64};
  if (!CheckInputDataType(op, "counter", counterSupportList)) {
    return GRAPH_FAILED;
  }

  const std::vector<DataType> algSupportList = {DT_INT32};
  if (!CheckInputDataType(op, "alg", algSupportList)) {
    return GRAPH_FAILED;
  }

  std::string error_msg;
  Shape key_shape = op.GetInputDescByName("key").GetShape();
  if (key_shape.GetDims().size() != 1) {
    error_msg = ConcatString("The number of dim for input[key]",
                             "must be 1, but the real value is ",
                             key_shape.GetDims().size(), ".");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape counter_shape = op.GetInputDescByName("counter").GetShape();
  if (counter_shape.GetDims().size() != 1) {
    error_msg = ConcatString("The number of dim for input[counter]",
                             "must be 1, but the real value is ",
                             counter_shape.GetDims().size(), ".");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }

  Shape alg_shape = op.GetInputDescByName("alg").GetShape();
  if (alg_shape.GetDims().size() != 0) {
    error_msg = ConcatString("The number of dim for input[alg]",
                             "must be 0, but the real value is ",
                             alg_shape.GetDims().size(), ".");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(StatelessRandomNormalV2, StatelessRandomNormalV2Verify);
// ----------------StatelessRandomNormalV2 End--------------------------

// ----------------StatelessRandomUniformV2 Begin--------------------------
IMPLEMT_INFERFUNC(StatelessRandomUniformV2, StatelessRandomUniformV2Infer) {
  std::string error_msg;
  // alg-start
  Shape alg;
  TensorDesc alg_desc = op.GetInputDescByName("alg");
  if (WithRank(alg_desc, 0, alg, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("the rank of alg must be 0, but get ", alg_desc.GetShape().GetDimNum());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  // alg-end

  // counter-start
  Shape counter;
  TensorDesc counter_desc = op.GetInputDescByName("counter");
  if (WithRank(counter_desc, 1, counter, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("the rank of counter must be 1, but get ", counter_desc.GetShape().GetDimNum());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  // counter-end
  GeShape shape;
  if (MakeShapeFromShapeTensor(op, "shape", shape,
                               TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call MakeShapeFromShapeTensor function to make shape from "
        "input[shape]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  DataType output_type = DT_FLOAT;
  op.GetAttr("dtype", output_type);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto outputDesc = op_desc->MutableOutputDesc(0);
  outputDesc->SetDataType(output_type);
  outputDesc->SetShape(shape);
  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(StatelessRandomUniformV2, StatelessRandomUniformV2Verify) {
  std::string error_msg;
  // key-start
  Shape key;
  TensorDesc key_desc = op.GetInputDescByName("key");
  if (WithRank(key_desc, 1, key, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    error_msg = ConcatString("failed to call WithRank function, ", "the rank of input[key] must be 1, but get ",
                             key_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  if (key.GetDim(0) != 1) {
    error_msg =
        ConcatString("the value of dim[0] for input[key] must be 1, ", "but the real value is ", key.GetDim(0), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
    return GRAPH_FAILED;
  }
  // key-end
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StatelessRandomUniformV2, StatelessRandomUniformV2Infer);
VERIFY_FUNC_REG(StatelessRandomUniformV2, StatelessRandomUniformV2Verify);
// ----------------StatelessRandomUniformV2 End--------------------------

}  // namespace ge
