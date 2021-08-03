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
 * \file case_condition_ops.cpp
 * \brief
 */
#include "inc/case_condition_ops.h"

#include <vector>

#include "util/util.h"
#include "op_log.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(CaseConditionInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr td = op_desc->MutableOutputDesc("y");
  std::vector<int64_t> scalar;
  td->SetShape(GeShape(scalar));
  td->SetDataType(DT_INT32);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CaseCondition, CaseConditionInferShape);
}  // namespace ge
