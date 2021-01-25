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
 * @file test_ifft_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "graph/operator.h"
#include "op_proto_test_util.h"
#include "spectral_ops.h"

class IFFT_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "IFFT_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "IFFT_UT TearDown" << std::endl;
  }
};

TEST_F(IFFT_UT, IfftInferShape) {
  ge::op::IFFT op;
  op.UpdateInputDesc("x", create_desc({2, 2}, ge::DT_COMPLEX64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_COMPLEX64);

}
