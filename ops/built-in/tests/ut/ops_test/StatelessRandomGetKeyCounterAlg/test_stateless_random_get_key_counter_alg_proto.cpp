/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this
 * file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_stateless_random_get_key_counter_alg_proto.cpp
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

class statelessRandomGetKeyCounterAlg : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "statelessRandomGetKeyCounterAlg test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "statelessRandomGetKeyCounterAlg TearDown" << std::endl;
  }
};

TEST_F(statelessRandomGetKeyCounterAlg,
       state_less_random_get_key_counter_alg_infer_success) {
  ge::op::StatelessRandomGetKeyCounterAlg op;

  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
  tensor_desc_seed.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expected_output_shape;
  expected_output_shape = {1};
  EXPECT_EQ(op.GetOutputDescByName("key").GetShape().GetDims(),
            expected_output_shape);
  expected_output_shape = {2};
  EXPECT_EQ(op.GetOutputDescByName("counter").GetShape().GetDims(),
            expected_output_shape);
  expected_output_shape = {};
  EXPECT_EQ(op.GetOutputDescByName("alg").GetShape().GetDims(),
            expected_output_shape);
}

TEST_F(statelessRandomGetKeyCounterAlg,
       state_less_random_get_key_counter_alg_verify_seed_type) {
  ge::op::StatelessRandomGetKeyCounterAlg op;

  ge::TensorDesc tensor_desc_seed(ge::Shape({2}), ge::FORMAT_ND, ge::DT_DOUBLE);
  tensor_desc_seed.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomGetKeyCounterAlg,
       state_less_random_get_key_counter_alg_verify_seed_shape_01) {
  ge::op::StatelessRandomGetKeyCounterAlg op;

  ge::TensorDesc tensor_desc_seed(ge::Shape({2, 3}), ge::FORMAT_ND,
                                  ge::DT_INT64);
  tensor_desc_seed.SetOriginShape(ge::Shape({2, 3}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomGetKeyCounterAlg,
       state_less_random_get_key_counter_alg_verify_seed_shape_02) {
  ge::op::StatelessRandomGetKeyCounterAlg op;

  ge::TensorDesc tensor_desc_seed(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
  tensor_desc_seed.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("seed", tensor_desc_seed);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}