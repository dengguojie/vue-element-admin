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
 * @file test_dummySeedGenerator_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "stateless_random_ops.h"
#include "op_proto_test_util.h"

class dummy_seed_generator : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "dummy_seed_generator proto test SetUp" << std::endl;
}

  static void TearDownTestCase() {
    std::cout << "dummy_seed_generator proto test TearDown" << std::endl;
    }
};

TEST_F(dummy_seed_generator, dummy_seed_generator_infer__resource_test1) {
  ge::op::DummySeedGenerator op;

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
