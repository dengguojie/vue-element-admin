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
#include "strided_slice_v3.h"
#include "register/op_tiling_registry.h"
#include "all_ops.h"
#include "test_common.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"



class StridedSliceV3UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stried_slice_v3_tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stried_slice_v3_tiling TearDown" << std::endl;
  }
};

using namespace ut_util;

template <typename T>
static void RunTest(const vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape, ge::DataType x_dtype,
                    std::map<std::string, std::vector<T>>& value_dict, std::string compile_info_str,
                    std::string expect_tiling_data) {
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
  TENSOR_INPUT_WITH_SHAPE(test_op, x, x_shape, x_dtype, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, begin, {static_cast<int64_t>(begin_value.size())}, const_dtype,
                                          ge::FORMAT_ND, begin_value);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, end, {static_cast<int64_t>(end_value.size())}, const_dtype,
                                          ge::FORMAT_ND, end_value);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, axes, axes_shape, const_dtype, ge::FORMAT_ND, axes_value);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, strides, strides_shape, const_dtype, ge::FORMAT_ND, strides_value);

  TENSOR_OUTPUT_WITH_SHAPE(test_op, y, y_shape, x_dtype, ge::FORMAT_ND, {});

  optiling::StridedSliceV3CompileInfo info;
  TILING_PARSE_JSON_TO_COMPILEINFO("StridedSliceV3", compile_info_str, info);

  std::vector<bool> input_const = {false, true, true, true, true};
  std::vector<std::string> attrs = {};
  ATTACH_OPERATOR_TO_HOLDER_CONST(holder, test_op, input_const, attrs, 2048, info);
  HOLDER_DO_TILING(holder, "StridedSliceV3", ge::GRAPH_SUCCESS);
  TILING_DATA_VERIFY_BYTYPE(holder, int64_t, expect_tiling_data);
}

TEST_F(StridedSliceV3UT, stried_slice_v3_normal_case) {
  std::string js_str =
      R"( {"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  std::string expect_tiling_data = "1 3 30 17 18 30 9 7 0 0 0 30 9 7 1 1 1 ";
  std::vector<int64_t> x_shape = {5, 6, 17, 18};
  std::vector<int64_t> y_shape = {5, 6, 9, 7};
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {0, 0}}, 
      {"end", {9, 7}}, 
      {"axes", {2, 3}}, 
      {"strides", {1, 1}}};

  RunTest(x_shape, y_shape, ge::DT_FLOAT16, value_dict, js_str, expect_tiling_data);
}

TEST_F(StridedSliceV3UT, stried_slice_v3_no_mask) {
  std::string js_str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  std::string expect_tiling_data = "1 4 4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ";
  std::vector<int64_t> x_shape = {4, 4, 4, 4};
  std::vector<int64_t> y_shape = {2, 2, 2, 2};
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {1, 1, 1, 1}}, 
      {"end", {3, 3, 3, 3}}, 
      {"strides", {1, 1, 1, 1}}, 
      {"axes", {0, 1, 2, -1}}};

  RunTest(x_shape, y_shape, ge::DT_FLOAT, value_dict, js_str, expect_tiling_data);
}

TEST_F(StridedSliceV3UT, stried_slice_v3_no_axes) {
  std::string js_str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  std::string expect_tiling_data = "1 4 4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ";
  std::vector<int64_t> x_shape = {4, 4, 4, 4};
  std::vector<int64_t> y_shape = {2, 2, 2, 2};
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {1, 1, 1, 1}}, 
      {"end", {3, 3, 3, 3}}, 
      {"axes", {}}, 
      {"strides", {1, 1, 1, 1}}};

  RunTest(x_shape, y_shape, ge::DT_FLOAT, value_dict, js_str, expect_tiling_data);
}

TEST_F(StridedSliceV3UT, stried_slice_v3_pad_head) {
  std::string js_str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  std::string expect_tiling_data = "2 2 4 64 2 32 1 16 3 48 1 1 ";
  std::vector<int64_t> x_shape = {4, 4, 4, 4};
  std::vector<int64_t> y_shape = {2, 2, 4, 4};
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {1, 1}}, 
      {"end", {3, 3}}, 
      {"axes", {0, 1}}, 
      {"strides", {1, 1}}};

  RunTest(x_shape, y_shape, ge::DT_FLOAT, value_dict, js_str, expect_tiling_data);
}

TEST_F(StridedSliceV3UT, stried_slice_v3_pad_tail) {
  std::string js_str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  std::string expect_tiling_data = "1 3 16 4 4 16 2 2 0 1 1 16 3 3 1 1 1 ";
  std::vector<int64_t> x_shape = {4, 4, 4, 4};
  std::vector<int64_t> y_shape = {4, 4, 2, 2};
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {1, 1}}, 
      {"end", {3, 3}}, 
      {"axes", {2, 3}}, 
      {"strides", {1, 1}}};

  RunTest(x_shape, y_shape, ge::DT_FLOAT, value_dict, js_str, expect_tiling_data);
}

TEST_F(StridedSliceV3UT, stried_slice_v3_no_mask_neg) {
  std::string js_str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  std::string expect_tiling_data = "1 4 4 4 4 4 2 2 2 2 1 1 1 1 3 3 3 3 1 1 1 1 ";
  std::vector<int64_t> x_shape = {4, 4, 4, 4};
  std::vector<int64_t> y_shape = {2, 2, 2, 2};
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {1, 1, 1, -3}}, 
      {"end", {3, 3, 3, -1}}, 
      {"axes", {0, 1, 2, -1}}, 
      {"strides", {}}};

  RunTest(x_shape, y_shape, ge::DT_FLOAT, value_dict, js_str, expect_tiling_data);
}

TEST_F(StridedSliceV3UT, stried_slice_v3_no_stride) {  // error
  std::string js_str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  std::string expect_tiling_data = "2 3 4 4 16 2 2 8 1 1 4 3 3 12 1 1 1 ";
  std::vector<int64_t> x_shape = {4, 4, 4, 4};
  std::vector<int64_t> y_shape = {2, 2, 2, 4};
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {1, 1, 1, -1000}}, 
      {"end", {3, 3, 3, 3000}}, 
      {"axes", {0, 1, 2, -1}}, 
      {"strides", {}}};

  RunTest(x_shape, y_shape, ge::DT_FLOAT, value_dict, js_str, expect_tiling_data);
}

TEST_F(StridedSliceV3UT, stried_slice_v3_end_out_of_range) {
  std::string js_str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  std::string expect_tiling_data = "7 1 6400 640 5760 6400 1 ";
  std::vector<int64_t> x_shape = {10, 10, 64};
  std::vector<int64_t> y_shape = {1, 10, 64};
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {-1}}, 
      {"end", {9223372036854775807}}, 
      {"axes", {0}}, 
      {"strides", {1}}};

  RunTest(x_shape, y_shape, ge::DT_FLOAT, value_dict, js_str, expect_tiling_data);
}

TEST_F(StridedSliceV3UT, stried_slice_v3_neg_input_end_out_of_range) {
  std::string js_str =
      R"({"vars": {"block_dim": 32, "begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0, "ub_size": 262144}})";

  std::string expect_tiling_data = "7 1 143360 143360 0 143360 1 ";
  std::vector<int64_t> x_shape = {28, 8, 10, 64};
  std::vector<int64_t> y_shape = {28, 8, 10, 64};
  std::map<std::string, std::vector<int64_t>> value_dict = {
      {"begin", {-85}}, 
      {"end", {9223372036854775807}}, 
      {"axes", {2}}, 
      {"strides", {}}};

  RunTest(x_shape, y_shape, ge::DT_FLOAT, value_dict, js_str, expect_tiling_data);
}