/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_segment_max_d_d_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class segment_max_d : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "segment_max_d SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "segment_max_d TearDown" << std::endl;
  }
};

TEST_F(segment_max_d, segment_max_d_infershape_test_1) {
  ge::op::SegmentMaxD op;
  op.UpdateInputDesc("x", create_desc_shape_range({9, 10, 2, 6, 7},
                                                  ge::DT_INT32,
                                                  ge::FORMAT_ND,
                                                  {9, 10, 2, 6, 7},
                                                  ge::FORMAT_ND,
                                                  {{9,9},{10,10},{2,2},{6,6},{7,7}}
                                                  ));
  std::vector<int64_t> segment_ids = {9, 10, 2, 6, 7};
  op.SetAttr("segment_ids", segment_ids);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(segment_max_d, segment_max_d_infershape_test_2) {
  ge::op::SegmentMaxD op;
  op.UpdateInputDesc("x", create_desc_shape_range({},
                                                  ge::DT_INT32,
                                                  ge::FORMAT_ND,
                                                  {},
                                                  ge::FORMAT_ND,
                                                  {}
                                                  ));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(segment_max_d, segment_max_d_infershape_test_3) {
  ge::op::SegmentMaxD op;
  op.UpdateInputDesc("x", create_desc_shape_range({5, 10, 2, 6, 7},
                                                  ge::DT_INT32,
                                                  ge::FORMAT_ND,
                                                  {5, 10, 2, 6, 7},
                                                  ge::FORMAT_ND,
                                                  {{5,5},{10,10},{2,2},{6,6},{7,7}}
                                                  ));
  std::vector<int64_t> segment_ids = {-9, 10, 2, 6, 7};
  op.SetAttr("segment_ids", segment_ids);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(segment_max_d, segment_max_d_infershape_test_4) {
  ge::op::SegmentMaxD op;
  op.UpdateInputDesc("x", create_desc_shape_range({5, 10, 2, 6, 7},
                                                  ge::DT_INT32,
                                                  ge::FORMAT_ND,
                                                  {5, 10, 2, 6, 7},
                                                  ge::FORMAT_ND,
                                                  {{5,5},{10,10},{2,2},{6,6},{7,7}}
                                                  ));
  std::vector<int64_t> segment_ids = {0, 10, 2, 6, 7};
  op.SetAttr("segment_ids", segment_ids);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(segment_max_d, segment_max_d_infershape_test_5) {
  ge::op::SegmentMaxD op;
  op.UpdateInputDesc("x", create_desc_shape_range({5, 10, 2, 6, 7},
                                                  ge::DT_INT32,
                                                  ge::FORMAT_ND,
                                                  {5, 10, 2, 6, 7},
                                                  ge::FORMAT_ND,
                                                  {{5,5},{10,10},{2,2},{6,6},{7,7}}
                                                  ));
  std::vector<int64_t> segment_ids = {0, 1, 2, 6, 7};
  op.SetAttr("segment_ids", segment_ids);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
