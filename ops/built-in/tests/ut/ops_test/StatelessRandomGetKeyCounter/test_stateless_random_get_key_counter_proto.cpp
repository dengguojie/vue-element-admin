/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this
 file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_stateless_random_get_key_counter_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include "array_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "op_proto_test_util.h"
#include "stateless_random_ops.h"

using namespace ge;
using namespace op;

class statelessRandomGetKeyCounter : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "statelessRandomGetKeyCounter test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "statelessRandomGetKeyCounter test TearDown" << std::endl;
  }
};

TEST_F(statelessRandomGetKeyCounter,
       stateless_random_get_key_counter_infer_verrity_success) {
  ge::op::StatelessRandomGetKeyCounter op;
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expected_output_shape;
  auto output_desc = op.GetOutputDescByName("key");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT64);
  expected_output_shape = {1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  auto output_desc2 = op.GetOutputDescByName("counter");
  EXPECT_EQ(output_desc2.GetDataType(), ge::DT_UINT64);
  expected_output_shape = {2};
  EXPECT_EQ(output_desc2.GetShape().GetDims(), expected_output_shape);
}

TEST_F(statelessRandomGetKeyCounter,
       stateless_random_get_key_counter_verify_seed_shape1) {
  ge::op::StatelessRandomGetKeyCounter op;
  ge::TensorDesc tensor_desc_seed(ge::Shape({2, 5}), ge::FORMAT_ND,
                                  ge::DT_INT32);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomGetKeyCounter,
       stateless_random_get_key_counter_verify_seed_shape2) {
  ge::op::StatelessRandomGetKeyCounter op;
  ge::TensorDesc tensor_desc_seed(ge::Shape({5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomGetKeyCounter,
       stateless_random_get_key_counter_verify_seed_dtype) {
  ge::op::StatelessRandomGetKeyCounter op;
  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_DOUBLE);
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
