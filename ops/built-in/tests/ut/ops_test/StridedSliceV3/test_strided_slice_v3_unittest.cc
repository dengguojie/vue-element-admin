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

#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "common/utils/ut_op_common.h"

namespace gert_test{
class StridedSliceV3UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "strided_slice_v3 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "strided_slice_v3 TearDown" << std::endl;
  }
};

using namespace ut_util;

template <typename T>
static void RunTest(const vector<int64_t>& x_shape, const std::vector<int64_t>& expected_output_shape,
                    std::map<std::string, std::vector<T>>& value_dict) {
  auto const_dtype = ge::DT_INT32;
  if (sizeof(T) == sizeof(int64_t)) {
    const_dtype = ge::DT_INT64;
  }

  std::vector<T>& begin_value = value_dict["begin"];
  std::vector<T>& end_value = value_dict["end"];
  std::vector<T>& axes_value = value_dict["axes"];
  std::vector<int64_t> axes_shape = {};
  if (!axes_value.empty()) {
    axes_shape.push_back(static_cast<int64_t>(axes_value.size()));
  }

  std::vector<T>& strides_value = value_dict["strides"];
  std::vector<int64_t> strides_shape = {};
  if (!strides_value.empty()) {
    strides_shape.push_back(static_cast<int64_t>(strides_value.size()));
  }

  // gen StridedSliceV3 op
  auto test_op = op::StridedSliceV3("StridedSliceV3");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, x_shape, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, begin, {static_cast<int64_t>(begin_value.size())}, const_dtype,
                                          ge::FORMAT_ND, begin_value);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, end, {static_cast<int64_t>(end_value.size())}, const_dtype,
                                          ge::FORMAT_ND, end_value);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, axes, axes_shape, const_dtype, ge::FORMAT_ND, axes_value);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, strides, strides_shape, const_dtype, ge::FORMAT_ND, strides_value);
  std::vector<bool> input_const = {false, true, true, true, true};
  CommonInferShapeOperatorWithConst(test_op, input_const, {}, {expected_output_shape});
}

TEST_F(StridedSliceV3UT, normal_case) {
  std::vector<int64_t> x_shape = {9, 10, 11, 12};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {2, 3}},
      {"end", {8, 7}},
      {"axes", {2, 3}},
      {"strides", {1, 1}}};
  std::vector<int64_t> expected_output_shape = {9, 10, 6, 4};

  RunTest(x_shape, expected_output_shape, value_dict);
}

TEST_F(StridedSliceV3UT, neg_axes) {
  std::vector<int64_t> x_shape = {9, 10, 11, 12};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 0}},
      {"end", {8, 7}},
      {"axes", {-2, -1}},
      {"strides", {1, 1}}};
  std::vector<int64_t> expected_output_shape = {9, 10, 8, 7};

  RunTest(x_shape, expected_output_shape, value_dict);
}

TEST_F(StridedSliceV3UT, neg_ends) {
  std::vector<int64_t> x_shape = {9, 10, 11, 12};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 0}},
      {"end", {-2, -1}},
      {"axes", {-2, -1}},
      {"strides", {2, 3}}};
  std::vector<int64_t> expected_output_shape = {9, 10, 5, 4};

  RunTest(x_shape, expected_output_shape, value_dict);
}

TEST_F(StridedSliceV3UT, ends_out_of_range) {
  std::vector<int64_t> x_shape = {20, 10, 5};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 1}},
      {"end", {1000, 1000}},
      {"axes", {0, 1}},
      {"strides", {1, 1}}};
  std::vector<int64_t> expected_output_shape = {20, 9, 5};

  RunTest(x_shape, expected_output_shape, value_dict);
}

TEST_F(StridedSliceV3UT, empty_strides) {
  std::vector<int64_t> x_shape = {20, 10, 5, 100};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 0, 3}},
      {"end", {20, 10, 4}},
      {"axes", {0, 1, 2}},
      {"strides", {}}};
  std::vector<int64_t> expected_output_shape = {20, 10, 1, 100};

  RunTest(x_shape, expected_output_shape, value_dict);
}

TEST_F(StridedSliceV3UT, empty_axes) {
  std::vector<int64_t> x_shape = {20, 10, 5, 100};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 0, 3}},
      {"end", {20, 10, 4}},
      {"axes", {}},
      {"strides", {2, 3, 4}}};
  std::vector<int64_t> expected_output_shape = {10, 4, 1, 100};

  RunTest(x_shape, expected_output_shape, value_dict);
}

TEST_F(StridedSliceV3UT, empty_axes_and_strides) {
  std::vector<int64_t> x_shape = {20, 10, 5};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 0, 0}},
      {"end", {10,10,10}},
      {"axes", {}},
      {"strides", {}}};
  std::vector<int64_t> expected_output_shape = {10, 10, 5};

  RunTest(x_shape, expected_output_shape, value_dict);
}

TEST_F(StridedSliceV3UT, ends_out_of_int32_range) {
  std::vector<int64_t> x_shape = vector<int64_t>({10, 10, 64});
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {-1}},
      {"end", {9223372036854775807}},
      {"axes", {0}},
      {"strides", {1}}};
  // expect result info
  std::vector<int64_t> expected_output_shape = {1, 10, 64};

  RunTest(x_shape, expected_output_shape, value_dict);
}

TEST_F(StridedSliceV3UT, neg_begin) {
  std::vector<int64_t> x_shape = vector<int64_t>({28, 8, 10, 64});
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {-85}},
      {"end", {9223372036854775807}},
      {"axes", {2}},
      {"strides", {1}}};
  // expect result info
  std::vector<int64_t> expected_output_shape = {28, 8, 10, 64};

  RunTest(x_shape, expected_output_shape, value_dict);
}

TEST_F(StridedSliceV3UT, neg_start_ends_out_of_range) {
  std::vector<int64_t> x_shape = vector<int64_t>({20, 10, 5});
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {-85}},
      {"end", {9223372}},
      {"axes", {1}},
      {"strides", {1}}};
  // expect result info
  std::vector<int64_t> expected_output_shape = {20, 10, 5};

  RunTest(x_shape, expected_output_shape, value_dict);
}
}  // namespace gert_test