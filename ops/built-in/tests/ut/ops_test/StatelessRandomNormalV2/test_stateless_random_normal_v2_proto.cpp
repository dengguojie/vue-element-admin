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
 * @file test_stateless_random_normal_v2_proto.cpp
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

class statelessRandomNormalV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "statelessRandomNormalV2 test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "statelessRandomNormalV2 TearDown" << std::endl;
  }
};

TEST_F(statelessRandomNormalV2,
       state_less_random_normal_v2_infer_and_verify_success) {
  ge::op::StatelessRandomNormalV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  op.UpdateInputDesc("key", tensor_desc_key);
  ge::TensorDesc tensor_desc_counter(ge::Shape({1}), ge::FORMAT_ND,
                                     ge::DT_UINT64);
  op.UpdateInputDesc("counter", tensor_desc_counter);
  ge::TensorDesc tensor_desc_alg(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("alg", tensor_desc_alg);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  op.SetAttr("dtype", ge::DT_FLOAT16);
  ge::TensorDesc tensor_desc_shape(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_shape.SetOriginShape(ge::Shape({3}));
  op.UpdateInputDesc("shape", tensor_desc_shape);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(op.GetOutputDescByName("y").GetShape().GetDims(),
            expected_output_shape);
}

TEST_F(statelessRandomNormalV2,
       state_less_random_normal_v2_infer_shape_no_const_data) {
  ge::op::StatelessRandomNormalV2 op;

  op.SetAttr("dtype", ge::DT_FLOAT16);
  ge::TensorDesc tensor_desc_shape(ge::Shape({5}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("shape", tensor_desc_shape);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomNormalV2,
       state_less_random_normal_v2_infer_shape_no_data) {
  ge::op::StatelessRandomNormalV2 op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomNormalV2, state_less_random_normal_v2_verify_dtype) {
  ge::op::StatelessRandomNormalV2 op;

  op.SetAttr("dtype", ge::DT_INT32);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomNormalV2, state_less_random_normal_v2_verify_key_type) {
  ge::op::StatelessRandomNormalV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("key", tensor_desc_key);
  ge::TensorDesc tensor_desc_counter(ge::Shape({1}), ge::FORMAT_ND,
                                     ge::DT_UINT64);
  op.UpdateInputDesc("counter", tensor_desc_counter);
  ge::TensorDesc tensor_desc_alg(ge::Shape({0}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("alg", tensor_desc_alg);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomNormalV2,
       state_less_random_normal_v2_verify_counter_type) {
  ge::op::StatelessRandomNormalV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({2}), ge::FORMAT_ND, ge::DT_UINT64);
  op.UpdateInputDesc("key", tensor_desc_key);
  ge::TensorDesc tensor_desc_counter(ge::Shape({1}), ge::FORMAT_ND,
                                     ge::DT_INT64);
  op.UpdateInputDesc("counter", tensor_desc_counter);
  ge::TensorDesc tensor_desc_alg(ge::Shape({0}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("alg", tensor_desc_alg);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomNormalV2, state_less_random_normal_v2_verify_alg_type) {
  ge::op::StatelessRandomNormalV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({2}), ge::FORMAT_ND, ge::DT_UINT64);
  op.UpdateInputDesc("key", tensor_desc_key);
  ge::TensorDesc tensor_desc_counter(ge::Shape({1}), ge::FORMAT_ND,
                                     ge::DT_UINT64);
  op.UpdateInputDesc("counter", tensor_desc_counter);
  ge::TensorDesc tensor_desc_alg(ge::Shape({0}), ge::FORMAT_ND, ge::DT_INT64);
  op.UpdateInputDesc("alg", tensor_desc_alg);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomNormalV2, state_less_random_normal_v2_verify_key_shape) {
  ge::op::StatelessRandomNormalV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({2, 3}), ge::FORMAT_ND,
                                 ge::DT_UINT64);
  op.UpdateInputDesc("key", tensor_desc_key);
  ge::TensorDesc tensor_desc_counter(ge::Shape({1}), ge::FORMAT_ND,
                                     ge::DT_UINT64);
  op.UpdateInputDesc("counter", tensor_desc_counter);
  ge::TensorDesc tensor_desc_alg(ge::Shape({0}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("alg", tensor_desc_alg);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomNormalV2,
       state_less_random_normal_v2_verify_counter_shape) {
  ge::op::StatelessRandomNormalV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({2}), ge::FORMAT_ND, ge::DT_UINT64);
  op.UpdateInputDesc("key", tensor_desc_key);
  ge::TensorDesc tensor_desc_counter(ge::Shape({1, 5}), ge::FORMAT_ND,
                                     ge::DT_UINT64);
  op.UpdateInputDesc("counter", tensor_desc_counter);
  ge::TensorDesc tensor_desc_alg(ge::Shape({0}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("alg", tensor_desc_alg);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(statelessRandomNormalV2, state_less_random_normal_v2_verify_alg_shape) {
  ge::op::StatelessRandomNormalV2 op;

  ge::TensorDesc tensor_desc_key(ge::Shape({1}), ge::FORMAT_ND, ge::DT_UINT64);
  op.UpdateInputDesc("key", tensor_desc_key);
  ge::TensorDesc tensor_desc_counter(ge::Shape({1}), ge::FORMAT_ND,
                                     ge::DT_UINT64);
  op.UpdateInputDesc("counter", tensor_desc_counter);
  ge::TensorDesc tensor_desc_alg(ge::Shape({0}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("alg", tensor_desc_alg);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}