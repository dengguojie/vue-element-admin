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
 * \file data_preprocess.cc
 * \brief
 */
#include "inc/data_flow_ops.h"
#include "graph/operator.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "error_util.h"

namespace ge {
graphStatus DataPreprocGetNextCommonInfer(Operator &op) {
  std::vector<ge::DataType> output_types;
  if (op.GetAttr("output_types", output_types) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[output_types] failed"));
    return GRAPH_FAILED;
  }

  std::vector<std::vector<int64_t>> output_shapes;
  if (op.GetAttr("output_shapes", output_shapes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[output_shapes] failed"));
    return GRAPH_FAILED;
  }

  if (output_types.size() != output_shapes.size()) {
    std::string err_msg =
      "attr[output_types] and attr[output_shapes] should be the same length";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc output_desc = op.GetDynamicOutputDesc("y", i);
    Shape shape(output_shapes[i]);
    output_desc.SetShape(shape);
    output_desc.SetDataType(output_types[i]);
    graphStatus output_status = op.UpdateDynamicOutputDesc("y", i, output_desc);
    if (output_status != GRAPH_SUCCESS) {
      std::ostringstream ss;
      ss << "update output[y] index[";
      ss << i;
      ss << "] desc failed";
      std::string err_msg = ss.str();
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(PeekData, PeekDataInfer) {
  return DataPreprocGetNextCommonInfer(op);
}

INFER_FUNC_REG(PeekData, PeekDataInfer);
}  // namespace ge