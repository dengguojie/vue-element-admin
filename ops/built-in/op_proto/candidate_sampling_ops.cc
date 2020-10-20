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
 * \file candidate_sampling_ops.cpp
 * \brief
 */
#include <set>

#include "inc/candidate_sampling_ops.h"
#include "common/inc/op_log.h"
#include "candidate_sampling_shape_fns.h"

namespace ge {

IMPLEMT_INFERFUNC(ThreadUnsafeUnigramCandidateSampler, ThreadUnsafeUnigramCandidateSamplerInfer) {
  return CandidateSamplerShape(op);
}

INFER_FUNC_REG(ThreadUnsafeUnigramCandidateSampler, ThreadUnsafeUnigramCandidateSamplerInfer);

IMPLEMT_INFERFUNC(UniformCandidateSampler, UniformCandidateSamplerInfer) {
  return CandidateSamplerShape(op);
}

INFER_FUNC_REG(UniformCandidateSampler, UniformCandidateSamplerInfer);

IMPLEMT_INFERFUNC(FixedUnigramCandidateSampler, FixedUnigramCandidateSamplerInfer) {
  TensorDesc tensordesc_output = op.GetOutputDesc("sampled_candidates");
  TensorDesc tensordesc_output1 = op.GetOutputDesc("true_expected_count");
  TensorDesc tensordesc_output2 = op.GetOutputDesc("sampled_expected_count");
  tensordesc_output.SetDataType(DT_INT64);
  tensordesc_output1.SetDataType(DT_FLOAT);
  tensordesc_output2.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("sampled_candidates", tensordesc_output);
  (void)op.UpdateOutputDesc("true_expected_count", tensordesc_output1);
  (void)op.UpdateOutputDesc("sampled_expected_count", tensordesc_output2);
  return CandidateSamplerShape(op);
}

INFER_FUNC_REG(FixedUnigramCandidateSampler, FixedUnigramCandidateSamplerInfer);

IMPLEMT_INFERFUNC(LearnedUnigramCandidateSampler, LearnedUnigramCandidateSamplerInfer) {
  TensorDesc sampled_candidates_desc = op.GetOutputDesc("sampled_candidates");
  sampled_candidates_desc.SetDataType(DT_INT64);
  op.UpdateOutputDesc("sampled_candidates", sampled_candidates_desc);
  TensorDesc true_expected_count_desc = op.GetOutputDesc("true_expected_count");
  true_expected_count_desc.SetDataType(DT_FLOAT);
  op.UpdateOutputDesc("true_expected_count", true_expected_count_desc);
  TensorDesc expected_count_desc = op.GetOutputDesc("sampled_expected_count");
  expected_count_desc.SetDataType(DT_FLOAT);
  op.UpdateOutputDesc("sampled_expected_count", expected_count_desc);
  return CandidateSamplerShape(op);
}

INFER_FUNC_REG(LearnedUnigramCandidateSampler, LearnedUnigramCandidateSamplerInfer);

IMPLEMT_INFERFUNC(LogUniformCandidateSampler, LogUniformCandidateSamplerInfer) {
  return CandidateSamplerShape(op);
}

INFER_FUNC_REG(LogUniformCandidateSampler, LogUniformCandidateSamplerInfer);

IMPLEMT_INFERFUNC(AllCandidateSampler, AllCandidateSamplerInfer) {
  bool judge = false;
  int64_t num_true = 0;
  op.GetAttr("num_true", num_true);
  if (num_true < 1) {
    OP_LOGE(op.GetName().c_str(), "Attr num_true must >= 1.");
    return GRAPH_FAILED;
  }

  int64_t num_sampled = 0;
  op.GetAttr("num_sampled", num_sampled);
  if (num_sampled < 1) {
    OP_LOGE(op.GetName().c_str(), "Attr num_sampled must >=1.");
    return GRAPH_FAILED;
  }

  Shape true_classes;
  if (WithRank(op.GetInputDesc(0), 2, true_classes, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input true_classes must be 2-D.");
    return GRAPH_FAILED;
  }

  int64_t batch_size = op.GetInputDesc(0).GetShape().GetDim(0);

  vector<int64_t> sampled_dims;
  sampled_dims.reserve(1);
  sampled_dims.push_back(num_sampled);

  vector<int64_t> true_dims;
  true_dims.reserve(2);
  true_dims.push_back(batch_size);
  true_dims.push_back(num_true);

  TensorDesc candidate_desc = op.GetOutputDesc("sampled_candidates");
  candidate_desc.SetShape(Shape(sampled_dims));
  candidate_desc.SetDataType(DT_INT64);
  judge = (op.UpdateOutputDesc("sampled_candidates", candidate_desc) != GRAPH_SUCCESS);
  if (judge) {
    OP_LOGE(op.GetName().c_str(), "fail to update output sampled_candidates.");
    return GRAPH_FAILED;
  }

  TensorDesc true_desc = op.GetOutputDesc("true_expected_count");
  true_desc.SetShape(Shape(true_dims));
  true_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("true_expected_count", true_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output true_expected_count.");
    return GRAPH_FAILED;
  }

  TensorDesc sampled_desc = op.GetOutputDesc("sampled_expected_count");
  sampled_desc.SetShape(Shape(sampled_dims));
  sampled_desc.SetDataType(DT_FLOAT);
  judge = (op.UpdateOutputDesc("sampled_expected_count", sampled_desc) != GRAPH_SUCCESS);
  if (judge) {
    OP_LOGE(op.GetName().c_str(), "fail to update output sampled_expected_count.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AllCandidateSampler, AllCandidateSamplerInfer);

IMPLEMT_INFERFUNC(ComputeAccidentalHits, ComputeAccidentalHitsInfer) {
  int64_t num_true = 0;
  op.GetAttr("num_true", num_true);

  Shape true_classes;
  if (WithRank(op.GetInputDesc(0), 2, true_classes, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input true_classes must be 2-D.");
    return GRAPH_FAILED;
  }

  if (op.GetInputDesc(0).GetShape().GetDim(1) != num_true) {
    OP_LOGE(op.GetName().c_str(), "input true_classes dim[1] must equal to attr num_true.");
    return GRAPH_FAILED;
  }

  Shape sampled_candidates;
  if (WithRank(op.GetInputDesc(1), 1, sampled_candidates, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input sampled_candidates must be 1-D.");
    return GRAPH_FAILED;
  }

  Tensor true_classes_tensor;
  if (op.GetInputConstData("true_classes", true_classes_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fail to get the constdata of input true_classes.");
    return GRAPH_FAILED;
  }
  int64_t true_classes_size = op.GetInputDesc(0).GetShape().GetShapeSize();
  const int64_t* first_shape_data = reinterpret_cast<const int64_t*>(true_classes_tensor.GetData());

  Tensor sampled_candidates_tensor;
  if (op.GetInputConstData("sampled_candidates", sampled_candidates_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fail to get the constdata of input sampled_candidates.");
    return GRAPH_FAILED;
  }
  int64_t sampled_candidates_size = op.GetInputDesc(1).GetShape().GetShapeSize();
  std::set<int64_t> sampled_candidates_data;
  const int64_t* second_shape_data = reinterpret_cast<const int64_t*>(sampled_candidates_tensor.GetData());
  for (int64_t i = 0; i < sampled_candidates_size; i++) {
    sampled_candidates_data.insert(second_shape_data[i]);
  }

  int64_t count = 0;
  for (int64_t j = 0; j < true_classes_size; j++) {
    auto search = sampled_candidates_data.find(first_shape_data[j]);
    if (search != sampled_candidates_data.end()) {
      count++;
    }
  }
  Shape v_dims;
  if (Vector(count, v_dims) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fial to gen vDims.");
    return GRAPH_FAILED;
  }

  TensorDesc indices_desc = op.GetOutputDesc("indices");
  indices_desc.SetShape(Shape(v_dims));
  indices_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("indices", indices_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output indices.");
    return GRAPH_FAILED;
  }

  TensorDesc ids_desc = op.GetOutputDesc("ids");
  ids_desc.SetShape(Shape(v_dims));
  ids_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("ids", ids_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output ids.");
    return GRAPH_FAILED;
  }

  TensorDesc weights_desc = op.GetOutputDesc("weights");
  weights_desc.SetShape(Shape(v_dims));
  weights_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("weights", weights_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output weights.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ComputeAccidentalHits, ComputeAccidentalHitsInfer);
}  // namespace ge
