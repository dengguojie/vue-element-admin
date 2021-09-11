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
 * @file test_lower_bound_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"

class LowerBoundTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "LowerBoundTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LowerBoundTest TearDown" << std::endl;
  }
};

TEST_F(LowerBoundTest, lower_bound_infershape_test) {
  ge::op::LowerBound op;
  ge:: TensorDesc tensorDesc;
  ge::Shape shape({2, 2});
  tensorDesc.SetDataType(ge::DT_FLOAT);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);
  op.UpdateInputDesc("sorted_x", tensorDesc);
  op.UpdateInputDesc("values", tensorDesc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
