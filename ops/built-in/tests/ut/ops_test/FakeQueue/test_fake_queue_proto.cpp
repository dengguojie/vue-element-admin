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
 * @file test_decode_jpeg_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include <vector>

class FakeQueue : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FakeQueue SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FakeQueue TearDown" << std::endl;
  }
};

TEST_F(FakeQueue, fake_queue_infer_shape_success) {
  ge::op::FakeQueue op;

  op.UpdateInputDesc("resource", create_desc({-1}, ge::DT_RESOURCE));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto reader_handle_desc = op.GetOutputDesc("handle");
  std::vector<int64_t> expected_reader_handle_result {2};
  EXPECT_EQ(reader_handle_desc.GetShape().GetDims(), expected_reader_handle_result);
}