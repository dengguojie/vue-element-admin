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
#include "./util/error_util.h"
#include "util/util.h"

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
    OP_LOGE(TbeGetName(op).c_str(), "Attr num_true must >= 1.");
    return GRAPH_FAILED;
  }

  int64_t num_sampled = 0;
  op.GetAttr("num_sampled", num_sampled);
  if (num_sampled < 1) {
    OP_LOGE(TbeGetName(op).c_str(), "Attr num_sampled must >=1.");
    return GRAPH_FAILED;
  }

  Shape true_classes;
  if (WithRank(op.GetInputDesc(0), 2, true_classes, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "input true_classes must be 2-D.");
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
    OP_LOGE(TbeGetName(op).c_str(), "fail to update output sampled_candidates.");
    return GRAPH_FAILED;
  }

  TensorDesc true_desc = op.GetOutputDesc("true_expected_count");
  true_desc.SetShape(Shape(true_dims));
  true_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("true_expected_count", true_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "fail to update output true_expected_count.");
    return GRAPH_FAILED;
  }

  TensorDesc sampled_desc = op.GetOutputDesc("sampled_expected_count");
  sampled_desc.SetShape(Shape(sampled_dims));
  sampled_desc.SetDataType(DT_FLOAT);
  judge = (op.UpdateOutputDesc("sampled_expected_count", sampled_desc) != GRAPH_SUCCESS);
  if (judge) {
    OP_LOGE(TbeGetName(op).c_str(), "fail to update output sampled_expected_count.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AllCandidateSampler, AllCandidateSamplerInfer);

IMPLEMT_INFERFUNC(ComputeAccidentalHits, ComputeAccidentalHitsInfer) {
  int64_t num_true = 0;
  if (op.GetAttr("num_true", num_true) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("get attr[num_true] failed"));
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto true_classes_desc = op_desc->MutableInputDesc(0);
  op_desc->SetOpInferDepends({"true_classes", "sampled_candidates"});

  GeShape true_classes;
  if (WithRank(true_classes_desc, 2, true_classes, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(true_classes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  int64_t unused_dim = 0;
  if (WithValue(true_classes.GetDim(1), num_true, unused_dim, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithValue function, dim[1] of input[true_classes] is [",
                                       true_classes.GetDim(1), "], it is not equal to attr[", num_true, "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto sampled_candidates_desc = op_desc->MutableInputDesc(1);
  GeShape sampled_candidates;
  if (WithRank(sampled_candidates_desc, 1, sampled_candidates, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(sampled_candidates_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t count = 0;
  Tensor true_classes_tensor;
  if (op.GetInputConstData("true_classes", true_classes_tensor) !=
      GRAPH_SUCCESS) {
    count = UNKNOWN_DIM;
  }
  int64_t true_classes_size = op.GetInputDesc(0).GetShape().GetShapeSize();
  const int64_t* first_shape_data =
      reinterpret_cast<const int64_t*>(true_classes_tensor.GetData());

  Tensor sampled_candidates_tensor;
  if (op.GetInputConstData("sampled_candidates", sampled_candidates_tensor) !=
      GRAPH_SUCCESS) {
    count = UNKNOWN_DIM;
  }

  if (count != UNKNOWN_DIM) {
    std::set<int64_t> sampled_candidates_data;
    int64_t sampled_candidates_size =
        sampled_candidates_desc->GetShape().GetShapeSize();
    const int64_t* second_shape_data =
        reinterpret_cast<const int64_t*>(sampled_candidates_tensor.GetData());
    for (int64_t i = 0; i < sampled_candidates_size; i++) {
      sampled_candidates_data.insert(second_shape_data[i]);
    }

    for (int64_t j = 0; j < true_classes_size; j++) {
      auto search = sampled_candidates_data.find(first_shape_data[j]);
      if (search != sampled_candidates_data.end()) {
        count++;
      }
    }
  }

  Shape v_dims;
  Vector(count, v_dims);
  
  auto indices_desc = op_desc->MutableOutputDesc(0);
  indices_desc->SetShape(GeShape(v_dims.GetDims()));
  indices_desc->SetDataType(DT_INT32);

  auto ids_desc = op_desc->MutableOutputDesc(1);
  ids_desc->SetShape(GeShape(v_dims.GetDims()));
  ids_desc->SetDataType(DT_INT64);

  auto weights_desc = op_desc->MutableOutputDesc(2);
  weights_desc->SetShape(GeShape(v_dims.GetDims()));
  weights_desc->SetDataType(DT_FLOAT);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ComputeAccidentalHits, ComputeAccidentalHitsInfer);
}  // namespace ge