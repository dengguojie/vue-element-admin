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
 * \file resource_variable_ops_shape_fns.cpp
 * \brief
 */
#include "resource_variable_ops_shape_fns.h"
#include "graph/types.h"
#include "op_log.h"
#include "common_shape_fns.h"

namespace ge {
graphStatus CreateAssignShapeFn(Operator& op) {
  std::vector<ShapeAndType> shape_and_type;
  if (ValidateVariableResourceHandle(op, shape_and_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Validate variable resource handle failed.");
    return GRAPH_FAILED;
  }

  Shape shape = op.GetInputDesc(1).GetShape();
  Shape unused;
  if (Merge(shape_and_type[0].GetShape(), shape, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Merge input 0 and 1 shape failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
}  // namespace ge
