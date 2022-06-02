/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "common/utils/ut_op_common.h"

void CommonInferShapeOperatorWithConst(ge::Operator& op, vector<bool> input_const, vector<string> attrs,
                                       vector<vector<int64_t>> expect_shapes) {
  ATTACH_OPERATOR_TO_HOLDER_WITH_CONST(holder, op, input_const, attrs);

  std::string optype = op.GetOpType();
  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_SUCCESS);

  size_t expect_count = std::min(output_size, expect_shapes.size());
  for (size_t i = 0; i < expect_count; i++) {
    VERIFY_OUTPUT_SHAPE(holder, i, expect_shapes[i]);
  }
}

void CommonInferShapeOperatorWithConstFail(ge::Operator& op, vector<bool> input_const,
                                           vector<string> attrs) {
  ATTACH_OPERATOR_TO_HOLDER_WITH_CONST(holder, op, input_const, attrs);

  std::string optype = op.GetOpType();
  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_FAILED);
}

void CommonInferShapeOperatorWithIrNum(ge::Operator& op, vector<uint32_t> irnum,
                                       vector<string> attrs, vector<vector<int64_t>> expect_shapes) {
  ATTACH_OPERATOR_TO_HOLDER_WITH_IRNUM(holder, op, irnum, attrs);

  std::string optype = op.GetOpType();
  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_SUCCESS);

  size_t expect_count = std::min(output_size, expect_shapes.size());
  for (size_t i = 0; i < expect_count; i++) {
    VERIFY_OUTPUT_SHAPE(holder, i, expect_shapes[i]);
  }
}

void CommonInferShapeOperatorWithIrNumFail(ge::Operator& op, vector<uint32_t> irnum,
                                           vector<string> attrs) {
  ATTACH_OPERATOR_TO_HOLDER_WITH_IRNUM(holder, op, irnum, attrs);

  std::string optype = op.GetOpType();
  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_FAILED);
}

void CommonInferShapeOperator(ge::Operator& op, vector<string> attrs, std::vector<std::vector<int64_t>> expect_shapes) {
  std::string optype = op.GetOpType();

  ATTACH_OPERATOR_TO_HOLDER(holder, op, attrs);

  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_SUCCESS);

  size_t expect_count = std::min(output_size, expect_shapes.size());
  for (size_t i = 0; i < expect_count; i++) {
    VERIFY_OUTPUT_SHAPE(holder, i, expect_shapes[i]);
  }
}

void CommonInferShapeOperatorFail(ge::Operator& op, vector<string> attrs) {
  std::string optype = op.GetOpType();

  ATTACH_OPERATOR_TO_HOLDER(holder, op, attrs);

  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_FAILED);
}
