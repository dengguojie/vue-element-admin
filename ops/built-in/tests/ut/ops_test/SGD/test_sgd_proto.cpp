/**
 * Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_dynamic_AddN_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class SGD : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SGD SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SGD TearDown" << std::endl;
  }
};

TEST_F(SGD, sgd_verfiy_parameters_dims_large_8_case) {
  ge::op::SGD op;
  op.UpdateInputDesc("parameters", create_desc({1, 1, 1, 1, 1, 1, 1, 1, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("gradient", create_desc({1, 1, 1, 1, 1, 1, 1, 1, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("learning_rate", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({1, 1, 1, 1, 1, 1, 1, 1, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("momentum", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("stat", create_desc({1, }, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);

}
