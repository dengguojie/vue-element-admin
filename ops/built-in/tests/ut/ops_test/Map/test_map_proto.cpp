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
 * @file test_map_proto.cpp
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

class MapTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "MapTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MapTest TearDown" << std::endl;
  }
};

TEST_F(MapTest, map_size_infershape_test) {
  ge::op::MapSize op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MapTest, map_incomplete_size_infershape_test) {
  ge::op::MapIncompleteSize op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MapTest, map_peek_infershape_test) {
  ge::op::MapPeek op;
  std::vector<ge::DataType> dtypes{ ge::DT_FLOAT, ge::DT_FLOAT };
  op.SetAttr("dtypes", dtypes);
  op.create_dynamic_output_values(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MapTest, map_peek_infershape_test_output_failed) {
  ge::op::MapPeek op;
  std::vector<ge::DataType> dtypes{ ge::DT_FLOAT, ge::DT_FLOAT };
  op.SetAttr("dtypes", dtypes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
