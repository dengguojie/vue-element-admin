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
 * \file cast.cc
 * \brief
 */
#include "cast.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"

namespace ge {
// ----------------Cast-------------------
IMPLEMT_COMMON_INFERFUNC(CastInferShape) {
  // get input desc
  TensorDesc input_desc = op.GetInputDesc("x");
  vector<int64_t> input_shape = input_desc.GetShape().GetDims();

  TensorDesc output_desc = op.GetOutputDesc("y");
  if (IsUnknown(input_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc.GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);

    output_desc.SetShape(Shape(input_shape));
    output_desc.SetOriginShape(Shape(input_shape));
    output_desc.SetShapeRange(input_range);
	op.UpdateOutputDesc("y", output_desc);
  } else {
    output_desc.SetShape(Shape(input_shape));
	op.UpdateOutputDesc("y", output_desc);
  }
  int type;
  if (op.GetAttr("dst_type", type) == GRAPH_SUCCESS) {
    output_desc.SetDataType((ge::DataType)type);
    op.UpdateOutputDesc("y", output_desc);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Cast, CastInferShape);
// --------------Cast END-----------------
}  // namespace ge
