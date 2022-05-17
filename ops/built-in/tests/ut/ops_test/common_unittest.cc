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

static std::map<std::string, std::vector<std::vector<std::string>>> operator_info_map = {
  {"Add", {{"x1", "x2"}, {"y"}, {}}},
  {"Mul", {{"x1", "x2"}, {"y"}, {}}},
  {"ReduceSum", {{"x", "axes"}, {"y"}, {"keep_dims"}}},
  {"MaxPoolV3", {{"x"}, {"y"}, {"ksize", "strides", "padding_mode", "pads", "data_format", "global_pooling", "ceil_mode"}}},
  {"Flatten", {{"x"}, {"y"}, {"axis"}}},
  {"Cast", {{"x"}, {"y"}, {"dst_type"}}},
};

void CommonInferShapeOperator2(ge::Operator& op, vector<bool> input_const,
                               vector<string> attrs, vector<vector<int64_t>> expect_shapes) {
  ATTACH_OPERATOR_TO_HOLDER2(holder, op, input_const, attrs);

  std::string optype = op.GetOpType();
  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_SUCCESS);

  size_t expect_count = std::min(output_size, expect_shapes.size());
  for (size_t i = 0; i < expect_count; i++) {
    VERIFY_OUTPUT_SHAPE(holder, i, expect_shapes[i]);
  }
}

void CommonInferShapeOperatorWithIrNum(ge::Operator& op, vector<uint32_t> irnum,
                                       vector<string> attrs, vector<vector<int64_t>> expect_shapes) {
  ATTACH_OPERATOR_TO_HOLDER3(holder, op, irnum, attrs);

  std::string optype = op.GetOpType();
  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_SUCCESS);

  size_t expect_count = std::min(output_size, expect_shapes.size());
  for (size_t i = 0; i < expect_count; i++) {
    VERIFY_OUTPUT_SHAPE(holder, i, expect_shapes[i]);
  }
}

void CommonInferShapeOperator(ge::Operator& op, std::vector<std::vector<int64_t>> expect_shapes) {
  std::string optype = op.GetOpType();
  auto find_it = operator_info_map.find(optype);
  ASSERT_NE(find_it, operator_info_map.end());
  auto input_output_attr = find_it->second;

  ATTACH_OPERATOR_TO_HOLDER(holder, op, input_output_attr[2]);

  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_SUCCESS);

  size_t expect_count = std::min(output_size, expect_shapes.size());
  for (size_t i = 0; i < expect_count; i++) {
    VERIFY_OUTPUT_SHAPE(holder, i, expect_shapes[i]);
  }
}
