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
 * @file test_combined_non_max_suppression_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"

class CombinedNonMaxSuppression : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CombinedNonMaxSuppression SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CombinedNonMaxSuppression TearDown" << std::endl;
  }
};

TEST_F(CombinedNonMaxSuppression, CombinedNonMaxSuppression_InferShape_dynamic_1) {
  ge::op::CombinedNonMaxSuppression op;
  op.UpdateInputDesc("boxes", create_desc({8, -1, 1, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("scores", create_desc({8, -1, 90}, ge::DT_FLOAT));
  op.UpdateInputDesc("max_output_size_per_class", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("max_total_size", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("iou_threshold", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("score_threshold", create_desc({}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CombinedNonMaxSuppression, CombinedNonMaxSuppression_InferShape_dynamic_2) {
  ge::op::CombinedNonMaxSuppression op;
  op.UpdateInputDesc("boxes", create_desc({8, -2, 1, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("scores", create_desc({8, -2, 90}, ge::DT_FLOAT));
  op.UpdateInputDesc("max_output_size_per_class", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("max_total_size", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("iou_threshold", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("score_threshold", create_desc({}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
