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
 * @file test_fifo_queue_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "candidate_sampling_ops.h"

class ComputeAccidentalHitsTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ComputeAccidentalHitsTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ComputeAccidentalHitsTest TearDown" << std::endl;
  }
};

TEST_F(ComputeAccidentalHitsTest, InferShape) {
  ge::op::ComputeAccidentalHits op;
  op.UpdateInputDesc("true_classes", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("sampled_candidates", create_desc({5}, ge::DT_INT64));
  op.SetAttr("num_true", 2);
  op.SetAttr("seed", 0);
  op.SetAttr("seed2", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto indices_desc = op.GetOutputDesc("indices");
  EXPECT_EQ(indices_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_indices_shape = {-1};
  EXPECT_EQ(indices_desc.GetShape().GetDims(), expected_indices_shape);

  auto ids_desc = op.GetOutputDesc("ids");
  EXPECT_EQ(ids_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_ids_shape = {-1};
  EXPECT_EQ(ids_desc.GetShape().GetDims(), expected_ids_shape);

  auto weights_desc = op.GetOutputDesc("weights");
  EXPECT_EQ(weights_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_weights_shape = {-1};
  EXPECT_EQ(weights_desc.GetShape().GetDims(), expected_weights_shape);
}

TEST_F(ComputeAccidentalHitsTest, InferShape2) {
  ge::op::ComputeAccidentalHits op;
  op.UpdateInputDesc("true_classes", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("sampled_candidates", create_desc({5}, ge::DT_INT64));
  op.SetAttr("seed", 0);
  op.SetAttr("seed2", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ComputeAccidentalHitsTest, InferShape3) {
  ge::op::ComputeAccidentalHits op;
  op.UpdateInputDesc("true_classes", create_desc({3, 2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("sampled_candidates", create_desc({5}, ge::DT_INT64));
  op.SetAttr("num_true", 3);
  op.SetAttr("seed", 0);
  op.SetAttr("seed2", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ComputeAccidentalHitsTest, InferShape4) {
  ge::op::ComputeAccidentalHits op;
  op.UpdateInputDesc("true_classes", create_desc({3, 2}, ge::DT_INT64));
  op.UpdateInputDesc("sampled_candidates", create_desc({5,4}, ge::DT_INT64));
  op.SetAttr("num_true", 2);
  op.SetAttr("seed", 0);
  op.SetAttr("seed2", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
