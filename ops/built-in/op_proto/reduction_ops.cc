/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file reduction_ops.cpp
 * \brief
 */
#include "inc/reduce_ops.h"

#include <string>
#include <vector>

#include "register/register.h"
#include "common/util/error_manager/error_manager.h"

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"

namespace ge {

static bool InferReductionShape(const ge::Operator& operation, const string& input_name, ge::TensorDesc& result_desc) {
  result_desc = operation.GetInputDesc(input_name);
  auto shape = result_desc.GetShape();
  int64_t dimNum = shape.GetDimNum();
  int64_t axis = 0;
  int64_t idx = 0;

  if (ge::GRAPH_SUCCESS != operation.GetAttr("axis", axis)) {
    OP_LOGE("Reduction", "Get axis failed!");
    OpsGetAttrErrReport(operation.GetName().c_str(), "axis");
    return false;
  }

  if (axis < -dimNum || axis >= dimNum) {
    OP_LOGE("Reduction", "The range of the axis must be between %ld and %ld !", -dimNum, dimNum - 1);
    string minvalue = ConcatString(-dimNum);
    string maxvalue = ConcatString(dimNum - 1);
    string excepted_value = ConcatString("in the range of[", minvalue, ",", maxvalue, "]");
    OpsAttrValueErrReport(operation.GetName(), "axis", excepted_value, ConcatString(axis));
    return false;
  }

  if (axis < 0) {
    axis += dimNum;
  }

  for (idx = axis; idx < dimNum; idx++) {
    shape.SetDim(idx, 1);
  }
  result_desc.SetShape(shape);

  return true;
}

IMPLEMT_COMMON_INFERFUNC(ReductionInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter Reduction proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReductionShape(op, "x", result_desc)) {
    return GRAPH_FAILED;
  }

  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();

  // update output desc
  ge::TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Reduction, ReductionVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Reduction, ReductionInferShape);

VERIFY_FUNC_REG(Reduction, ReductionVerify);
}  // namespace ge
