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
 * \file internal_ops.cpp
 * \brief
 */
#include "inc/internal_ops.h"

#include <map>
#include <string>

#include "common/inc/op_log.h"
#include "common_shape_fns.h"
#include "util/error_util.h"

namespace ge {

graphStatus FunctionTopkV2(Operator& op) {
  uint32_t index = 0;
  TensorDesc input_desc = op.GetDynamicInputDesc("x", index);
  std::vector<int64_t> input_dims = input_desc.GetShape().GetDims();
  TensorDesc output_desc = op.GetDynamicOutputDesc("y", index);
  std::vector<int64_t> output_dims;
  if (input_dims == UNKNOWN_RANK) {
    output_dims.emplace_back(UNKNOWN_DIM);
  } else if (!input_dims.empty()) {
    int64_t last_dim = input_dims.back();
    if (last_dim == UNKNOWN_DIM) {
      output_dims.emplace_back(UNKNOWN_DIM);
    } else {
      // output shape is [2*input_last_dim], don't overflow
      output_dims.emplace_back(2 * last_dim);
    }
  } else {
    OP_LOGE(TbeGetName(op).c_str(), "Op input dim size = %zu is illegal.", input_dims.size());
    return GRAPH_FAILED;
  }
  output_desc.SetShape(Shape(output_dims));
  output_desc.SetDataType(DT_FLOAT16);
  return op.UpdateDynamicOutputDesc("y", index, output_desc);
}

IMPLEMT_INFERFUNC(AssistHelp, AssistHelpInfer) {
  const std::map<std::string, uint32_t> func_name_map = {{"topkv2", 1}};
  std::string func_name;
  if (op.GetAttr("func_name", func_name) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("get attr[func_name] failed."));
    return GRAPH_FAILED;
  }
  uint32_t func_id = 0;
  auto iter = func_name_map.find(func_name);
  if (iter != func_name_map.end()) {
    func_id = iter->second;
  }
  switch (func_id) {
    case 1:
      return FunctionTopkV2(op);
    default:
      std::string err_msg = GetAttrValueErrMsg("func_name", func_name, "topkv2");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
  }
}

INFER_FUNC_REG(AssistHelp, AssistHelpInfer);

IMPLEMT_COMMON_INFERFUNC(CacheUpdateInferShape) {
  TensorDesc out_desc = op.GetOutputDesc("x");
  out_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  if (op.UpdateOutputDesc("x", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "update output x failed.");
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "x");
}

INFER_FUNC_REG(CacheUpdate, CacheUpdateInferShape);

COMMON_INFER_FUNC_REG(InternalDataMove, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));

}  // namespace ge
