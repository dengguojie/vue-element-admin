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
 * \file boosted_trees_ops.cpp
 * \brief
 */
#include "inc/boosted_trees_ops.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"

namespace ge {
IMPLEMT_INFERFUNC(BoostedTreesBucketize, BoostedTreesBucketizeInfer) {
  int64_t num_features;
  if (op.GetAttr("num_features", num_features) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get addr[num_features] failed"));
    return GRAPH_FAILED;
  }
  if (num_features < 0) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        GetAttrValueErrMsg("num_features", std::to_string(num_features), ">=0"));
    return GRAPH_FAILED;
  }

  for (int32_t i = 0; i < num_features; i++) {
    Shape value_shape;
    if (WithRank(op.GetInputDesc(i), 1, value_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
          ConcatString("call WithRank function failed, invalied shape",
              DebugString(op.GetInputDesc(i).GetShape().GetDims()), " of ", i,
              "th dynamic input[float_values], it should be 1D"));
      return GRAPH_FAILED;
    }

    Shape boundaries_shape;
    if (WithRank(op.GetInputDesc(i + num_features), 1, boundaries_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
          ConcatString("call WithRank function failed, invalied shape",
              DebugString(op.GetInputDesc(i + num_features).GetShape().GetDims()),
              " of ", i, "th dynamic input[boundaries_shape], it should be 1D"));
      return GRAPH_FAILED;
    }

    int64_t unused_dim = 0;
    if (Merge(value_shape.GetDim(0), op.GetInputDesc(0).GetShape().GetDim(0), unused_dim)) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
          ConcatString("call Merge function failed, each 0th dim value of dynamic",
          " input[float_values] should be equal, but ", value_shape.GetDim(0), " and ",
          op.GetInputDesc(0).GetShape().GetDim(0), " are exist"));
      return GRAPH_FAILED;
    }

    int64_t y_dim0 = value_shape.GetDim(0);
    TensorDesc y_desc = op.GetDynamicOutputDesc("y", i);
    Shape y_shape({y_dim0, 1});
    y_desc.SetShape(y_shape);
    y_desc.SetDataType(DT_INT32);
    if (op.UpdateDynamicOutputDesc("y", i, y_desc) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
          ConcatString("update ", i, "th dynamic output[y] desc failed"));
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BoostedTreesBucketize, BoostedTreesBucketizeInfer);
}  // namespace ge
