/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "array_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "op_proto_test_util.h"
#include "transformation_ops.h"

using namespace ge;
using namespace op;

class flatten : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "flatten Proto Test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "flatten Proto Test TearDown" << std::endl; }
};

TEST_F(flatten, flatten_infershape_1) {
  // set input info
  auto input_shape = vector<int64_t>({4, 3, 4});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{4, 4}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;

  // expect result
  std::vector<int64_t> expected_shape = {4, 12};

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(flatten, flatten_infershape_2) {
  // set input info
  auto input_shape = vector<int64_t>({-1, -1, -1});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{4, 4}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;

  // expect result
  std::vector<int64_t> expected_shape = {-1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_range = {{4, 4}, {12, 12}};

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(flatten, flatten_infershape_3) {
  // set input info
  auto input_shape = vector<int64_t>({-2});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{4, 4}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;

  // expect result
  std::vector<int64_t> expected_shape = {-2};

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(flatten, flatten_infershape_4) {
  // set input info
  auto input_shape = vector<int64_t>({-2});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{4, 4}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;

  // expect result
  std::vector<int64_t> expected_shape = {-2};

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);
  op.SetAttr("axis", 1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(flatten, flatten_infershape_5) {
  // set input info
  auto input_shape = vector<int64_t>({4, 3, 4});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{4, 4}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;

  // expect result
  std::vector<int64_t> expected_shape = {12, 4};

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);
  op.SetAttr("axis", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(flatten, flatten_infershape_static_shape) {
  // set input info
  auto input_shape = vector<int64_t>({4, 3, 4, 5});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{4, 4}, {3, 3}, {4, 4}, {5, 5}};
  auto test_format = ge::FORMAT_ND;

  // expect result
  std::vector<int64_t> expected_shape = {12, 20};

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);
  op.SetAttr("axis", -2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(flatten, flatten_infershape_y_range) {
  // set input info
  auto input_shape = vector<int64_t>({-1, -1, -1});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{5, 5}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;

  // expect result
  std::vector<int64_t> expected_shape = {-1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_range = {{15, 15}, {4, 4}};

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);
  op.SetAttr("axis", -1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(flatten, flatten_infershape_default_axis) {
  // set input info
  auto input_shape = vector<int64_t>({5, -1, -1});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{5, 5}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;

  // expect result
  std::vector<int64_t> expected_shape = {5, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_range = {{5, 5}, {12, 12}};

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(flatten, flatten_infershape_fail_axis_out_range) {
  // set input info
  auto input_shape = vector<int64_t>({5, -1, -1});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{5, 5}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);
  op.SetAttr("axis", -4);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(flatten, flatten_infershape_fail_axis_out_range_1) {
  // set input info
  auto input_shape = vector<int64_t>({5, -1, -1});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{5, 5}, {3, 3}, {4, 4}};
  auto test_format = ge::FORMAT_ND;

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);
  op.SetAttr("axis", 4);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(flatten, flatten_infershape_x_range_empty) {
  // set input info
  auto input_shape = vector<int64_t>({5, -1, -1});
  std::vector<std::pair<int64_t, int64_t>> input_range;
  auto test_format = ge::FORMAT_ND;

  // expect result
  std::vector<int64_t> expected_shape = {-1, -1};

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);
  op.SetAttr("axis", 2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(flatten, flatten_infershape_x_1d) {
  // set input info
  auto input_shape = vector<int64_t>({-1});
  std::vector<std::pair<int64_t, int64_t>> input_range = {{1, 30}};
  auto test_format = ge::FORMAT_ND;

  // expect result
  std::vector<int64_t> expected_shape = {-1};
  std::vector<std::pair<int64_t, int64_t>> expected_range = {{1, 30}};

  // create desc
  auto input_desc =
      create_desc_shape_range(input_shape, ge::DT_FLOAT, test_format, input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::Flatten op;
  op.UpdateInputDesc("x", input_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}