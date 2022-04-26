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
 * \file sdca_ops.cpp
 * \brief
 */
#include "inc/sdca_ops.h"
#include <utility>
#include "common/inc/op_log.h"
#include "common_shape_fns.h"
#include "util/util.h"
#include "graph/operator.h"
#include "util/error_util.h"

namespace ge {

IMPLEMT_INFERFUNC(SdcaOptimizerV2, SdcaOptimizerV2Infer) {
  int num_sparse_features;
  int num_dense_features;
  if (op.GetAttr("num_sparse_features", num_sparse_features) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get attr[num_sparse_features] failed."));
    return GRAPH_FAILED;
  }
  if (op.GetAttr("num_dense_features", num_dense_features) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get attr[num_dense_features] failed."));
    return GRAPH_FAILED;
  }

  graphStatus output_status;
  for (int64_t i = 0; i < num_sparse_features; ++i) {
    Shape shape = op.GetDynamicInputDesc("sparse_weights", i).GetShape();
    TensorDesc desc = op.GetDynamicOutputDesc("out_delta_sparse_weights", i);
    desc.SetDataType(DT_FLOAT);
    desc.SetShape(shape);
    output_status = op.UpdateDynamicOutputDesc("out_delta_sparse_weights", i, desc);
    if (output_status != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("update description for output out_delta_sparse_weights[", i, "] failed.");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  for (int64_t i = 0; i < num_dense_features; ++i) {
    Shape shape = op.GetDynamicInputDesc("dense_weights", i).GetShape();
    TensorDesc desc = op.GetDynamicOutputDesc("out_delta_dense_weights", i);
    desc.SetDataType(DT_FLOAT);
    desc.SetShape(shape);
    if (op.UpdateDynamicOutputDesc("out_delta_dense_weights", i, desc) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("update description for output out_delta_dense_weights[", i, "] failed.");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  vector<int64_t> dims;
  dims.reserve(2);
  dims.push_back(UNKNOWN_DIM);
  dims.push_back(4);
  Shape sec_shape(dims);
  TensorDesc state_tensor_desc = op.GetOutputDesc("out_example_state_data");
  state_tensor_desc.SetShape(sec_shape);
  state_tensor_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("out_example_state_data", state_tensor_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("update description for output[out_example_state_data] failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SdcaOptimizerV2, SdcaOptimizerV2Infer);

}  // namespace ge