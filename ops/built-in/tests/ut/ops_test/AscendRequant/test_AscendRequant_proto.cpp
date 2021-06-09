/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_AscendRequant_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "quantize_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

class AscendRequantProto : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AscendRequantProto SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AscendRequantProto TearDown" << std::endl;
  }
};

TEST_F(AscendRequantProto, AscendRequant_proto_0) {
  auto shape_x = std::vector<int64_t>({-1, -1, -1, -1, 16});
  std::vector<std::pair<int64_t, int64_t>> range_x = {{1, 64}, {2, 8}, {17, 60}, {10, 128}, {16, 16}};
  auto format_x = ge::FORMAT_NC1HWC0;
  auto tensor_desc_x = create_desc_shape_range(shape_x, ge::DT_INT32, format_x, shape_x, format_x, range_x);

  auto shape_scale = std::vector<int64_t>({1, -1, 1, 1, 16});
  std::vector<std::pair<int64_t, int64_t>> range_scale = {{1, 1}, {1, 16}, {1, 1}, {1, 1}, {16, 16}};
  auto format_scale = ge::FORMAT_NC1HWC0;
  auto tensor_desc_scale =
      create_desc_shape_range(shape_scale, ge::DT_UINT16, format_scale, shape_scale, format_scale, range_scale);

  std::vector<int64_t> expected_shape = {-1, -1, -1, -1, 16};
  std::vector<std::pair<int64_t, int64_t>> expected_range = {{1, 64}, {2, 8}, {17, 60}, {10, 128}, {16, 16}};

  auto sqrt_mode = false;
  auto relu_flag = false;

  ge::op::AscendRequant op;
  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("scale", tensor_desc_scale);
  op.set_attr_relu_flag(relu_flag);
  auto ret = op.InferShapeAndType();

  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(AscendRequantProto, AscendRequant_proto_1) {
  auto shape_x = std::vector<int64_t>({1, -1, -1, 16});
  std::vector<std::pair<int64_t, int64_t>> range_x = {{1, 1}, {2, 8}, {10, 128}, {16, 16}};
  auto format_x = ge::FORMAT_FRACTAL_NZ;
  auto tensor_desc_x = create_desc_shape_range(shape_x, ge::DT_INT32, format_x, shape_x, format_x, range_x);

  auto shape_scale = std::vector<int64_t>({1, -1, 1, 1, 16});
  std::vector<std::pair<int64_t, int64_t>> range_scale = {{1, 1}, {1, 16}, {1, 1}, {1, 1}, {16, 16}};
  auto format_scale = ge::FORMAT_NC1HWC0;
  auto tensor_desc_scale =
      create_desc_shape_range(shape_scale, ge::DT_UINT16, format_scale, shape_scale, format_scale, range_scale);

  std::vector<int64_t> expected_shape = {1, -1, -1, 16};
  std::vector<std::pair<int64_t, int64_t>> expected_range = {{1, 1}, {2, 8}, {10, 128}, {16, 16}};

  auto sqrt_mode = false;
  auto relu_flag = false;

  ge::op::AscendRequant op;
  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("scale", tensor_desc_scale);
  op.set_attr_relu_flag(relu_flag);
  auto ret = op.InferShapeAndType();

  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}
