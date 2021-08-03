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
 * \file slice_write_ops.cpp
 * \brief
 */
#include "inc/slice_write_ops.h"

#include <vector>

#include "util/util.h"
#include "op_log.h"

namespace ge {
IMPLEMT_VERIFIER(SliceWrite, SliceWriteVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "value")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SliceWriteInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape x_shape = op_desc->MutableInputDesc("x")->GetShape();
  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  op_desc->MutableInputDesc("x")->GetShapeRange(x_shape_range);
  DataType input_dtype = op_desc->MutableInputDesc("x")->GetDataType();
  GeTensorDescPtr td = op_desc->MutableOutputDesc("x");
  td->SetShape(x_shape);
  td->SetDataType(input_dtype);
  td->SetShapeRange(x_shape_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SliceWrite, SliceWriteInferShape);
VERIFY_FUNC_REG(SliceWrite, SliceWriteVerify);
}  // namespace ge
