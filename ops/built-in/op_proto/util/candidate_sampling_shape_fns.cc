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
 * \file candidate_sampling_shape_fns.cpp
 * \brief
 */
#include "candidate_sampling_shape_fns.h"
#include <vector>
#include "op_log.h"
#include "error_util.h"

namespace ge {
graphStatus CandidateSamplerShape(Operator& op) {
  int64_t num_true = 0;
  op.GetAttr("num_true", num_true);
  if (num_true < 1) {
    string err_msg = ConcatString("attr[num_true] must >= 1, real value is ",
                     num_true);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t num_sampled = 0;
  op.GetAttr("num_sampled", num_sampled);
  if (num_sampled < 1) {
    string err_msg = ConcatString("attr[num_sampled] must >= 1, real value is ",
                     num_sampled);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t range_max = 0;
  op.GetAttr("range_max", range_max);
  if (range_max < 1) {
    string err_msg = ConcatString("attr[range_max] must >= 1, real value is ",
                     range_max);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape true_classes;
  if (WithRank(op.GetInputDesc(0), 2, true_classes, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    string err_msg = ConcatString("input[true_classes] must be 2-D, real rank is ",
                     true_classes.GetDimNum());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t batch_size = op.GetInputDesc(0).GetShape().GetDim(0);

  std::vector<int64_t> sampled_dims;
  sampled_dims.reserve(1);
  sampled_dims.push_back(num_sampled);

  std::vector<int64_t> true_dims;
  true_dims.reserve(2);
  true_dims.push_back(batch_size);
  true_dims.push_back(num_true);

  TensorDesc candidate_desc = op.GetOutputDesc("sampled_candidates");
  candidate_desc.SetShape(Shape(sampled_dims));
  candidate_desc.SetDataType(DT_INT64);

  TensorDesc true_desc = op.GetOutputDesc("true_expected_count");
  true_desc.SetShape(Shape(true_dims));
  true_desc.SetDataType(DT_FLOAT);

  TensorDesc sampled_desc = op.GetOutputDesc("sampled_expected_count");
  sampled_desc.SetShape(Shape(sampled_dims));
  sampled_desc.SetDataType(DT_FLOAT);

  if (op.UpdateOutputDesc("sampled_candidates", candidate_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("fail to update output[sampled_candidates] desc"));
    return GRAPH_FAILED;
  }

  if (op.UpdateOutputDesc("true_expected_count", true_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("fail to update output[true_expected_count] desc"));
    return GRAPH_FAILED;
  }

  if (op.UpdateOutputDesc("sampled_expected_count", sampled_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("fail to update output[sampled_expected_count] desc"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge