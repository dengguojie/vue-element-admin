/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_reducemeanwithcount_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class ReduceMeanWithCount : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ReduceMeanWithCount SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ReduceMeanWithCount TearDown" << std::endl;
  }
};

TEST_F(ReduceMeanWithCount, reducemeanwithcount_infer_shape_001) {
  ge::op::ReduceMeanWithCount op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({-1, 100, 4},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("axes", {2,});
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, 100};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ReduceMeanWithCount, reducemeanwithcount_infer_shape_002) {
  ge::op::ReduceMeanWithCount op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 25}, {1, 25}, {1, 25}, {3, 3}, {1, 1}};
  auto tensor_desc = create_desc_shape_range({-1, -1, -1, 3, 1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, -1, -1, 3, 1},
                                             ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("axes", {2,});
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, 3, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ReduceMeanWithCount, reducemeanwithcount_infer_shape_003) {
  ge::op::ReduceMeanWithCount op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("axes", {});
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ReduceMeanWithCount, reducemeanwithcount_infer_shape_004) {
  ge::op::ReduceMeanWithCount op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("axes", {});
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ReduceMeanWithCount, reducemeanwithcount_infer_shape_005) {
  ge::op::ReduceMeanWithCount op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 25}, {2, 25}, {2, 25}, {3, 3}, {1, 1}};
  auto tensor_desc = create_desc_shape_range({-1, -1, -1, 3, 1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, -1, -1, 3, 1},
                                             ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("axes", {1,});
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,1,-1,3,1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
