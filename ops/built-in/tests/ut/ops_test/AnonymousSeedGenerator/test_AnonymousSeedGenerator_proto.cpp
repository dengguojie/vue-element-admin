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
 * @file test_AnonymousSeedGenerator_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "stateless_random_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class anonymousseedgenerator_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "anonymousseedgenerator_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "anonymousseedgenerator_infer_test TearDown" << std::endl;
  }
};

TEST_F(anonymousseedgenerator_infer_test, anonymousseedgenerator_infer_test_1) {
  //new op
  ge::op::AnonymousSeedGenerator op;
  // set input info
  ge::TensorDesc tensor_desc_seed = create_desc_shape_range({},ge::DT_INT64, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_seed2 = create_desc_shape_range({},ge::DT_INT64,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed2", tensor_desc_seed2);
  ge::TensorDesc tensor_desc_reshuffle=create_desc_shape_range({1},ge::DT_BOOL,
                                       ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("reshuffle", tensor_desc_reshuffle);
  auto states = op.VerifyAllAttr(true);
  // check result
  EXPECT_EQ(states, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(anonymousseedgenerator_infer_test, anonymousseedgenerator_infer_test_2) {
  //new op
  ge::op::AnonymousSeedGenerator op;
  // set input info
  ge::TensorDesc tensor_desc_seed = create_desc_shape_range({1},ge::DT_INT64, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_seed2 = create_desc_shape_range({2},ge::DT_INT64,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed2", tensor_desc_seed2);
  ge::TensorDesc tensor_desc_reshuffle=create_desc_shape_range({1},ge::DT_BOOL,
                                       ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("reshuffle", tensor_desc_reshuffle);
  auto states = op.VerifyAllAttr(true);
  // check result
  EXPECT_EQ(states, ge::GRAPH_FAILED);
}

TEST_F(anonymousseedgenerator_infer_test, anonymousseedgenerator_infer_test_3) {
  //new op
  ge::op::AnonymousSeedGenerator op;
  // set input info
  ge::TensorDesc tensor_desc_seed = create_desc_shape_range({},ge::DT_INT64, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_seed2 = create_desc_shape_range({1},ge::DT_INT64,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed2", tensor_desc_seed2);
  ge::TensorDesc tensor_desc_reshuffle=create_desc_shape_range({1},ge::DT_BOOL,
                                       ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("reshuffle", tensor_desc_reshuffle);
  auto states = op.VerifyAllAttr(true);
  // check result
  EXPECT_EQ(states, ge::GRAPH_FAILED);
}

TEST_F(anonymousseedgenerator_infer_test, anonymousseedgenerator_infer_test_4) {
  //new op
  ge::op::AnonymousSeedGenerator op;
  // set input info
  ge::TensorDesc tensor_desc_seed = create_desc_shape_range({},ge::DT_INT64, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_seed2 = create_desc_shape_range({},ge::DT_FLOAT,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed2", tensor_desc_seed2);
  ge::TensorDesc tensor_desc_reshuffle=create_desc_shape_range({1},ge::DT_BOOL,
                                       ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("reshuffle", tensor_desc_reshuffle);
  auto states = op.VerifyAllAttr(true);
  // check result
  EXPECT_EQ(states, ge::GRAPH_FAILED);
}

TEST_F(anonymousseedgenerator_infer_test, anonymousseedgenerator_infer_test_5) {
  //new op
  ge::op::AnonymousSeedGenerator op;
  // set input info
  ge::TensorDesc tensor_desc_seed = create_desc_shape_range({},ge::DT_INT64, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_seed2 = create_desc_shape_range({},ge::DT_INT64,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed2", tensor_desc_seed);
  ge::TensorDesc tensor_desc_reshuffle=create_desc_shape_range({1},ge::DT_INT32,
                                       ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("reshuffle", tensor_desc_reshuffle);
  auto states = op.VerifyAllAttr(true);
  // check result
  EXPECT_EQ(states, ge::GRAPH_FAILED);
}

TEST_F(anonymousseedgenerator_infer_test, anonymousseedgenerator_infer_test_6) {
  //new op
  ge::op::AnonymousSeedGenerator op;
  // set input info
  ge::TensorDesc tensor_desc_seed = create_desc_shape_range({},ge::DT_DOUBLE, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_seed2 = create_desc_shape_range({},ge::DT_INT64,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed2", tensor_desc_seed);
  ge::TensorDesc tensor_desc_reshuffle=create_desc_shape_range({1},ge::DT_INT32,
                                       ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("reshuffle", tensor_desc_reshuffle);
  auto states = op.VerifyAllAttr(true);
  // check result
  EXPECT_EQ(states, ge::GRAPH_FAILED);  
}

TEST_F(anonymousseedgenerator_infer_test, anonymousseedgenerator_infer_test_7) {
  //new op
  ge::op::AnonymousSeedGenerator op;
  // set input info
  ge::TensorDesc tensor_desc_seed = create_desc_shape_range({1,2},ge::DT_INT64, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_seed2 = create_desc_shape_range({1},ge::DT_INT64,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed2", tensor_desc_seed);
  ge::TensorDesc tensor_desc_reshuffle=create_desc_shape_range({1},ge::DT_INT32,
                                       ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("reshuffle", tensor_desc_reshuffle);
  auto states = op.VerifyAllAttr(true);
  // check result
  EXPECT_EQ(states, ge::GRAPH_FAILED);
}

TEST_F(anonymousseedgenerator_infer_test, anonymousseedgenerator_infer_test_8) {
  //new op
  ge::op::AnonymousSeedGenerator op;
  // set input info
  ge::TensorDesc tensor_desc_seed = create_desc_shape_range({},ge::DT_INT32, 
                                    ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed", tensor_desc_seed);
  ge::TensorDesc tensor_desc_seed2 = create_desc_shape_range({},ge::DT_INT32,
                                     ge::FORMAT_ND,{1},ge::FORMAT_ND,{});
  op.UpdateInputDesc("seed2", tensor_desc_seed2);
  ge::TensorDesc tensor_desc_reshuffle=create_desc_shape_range({1},ge::DT_BOOL,
                                       ge::FORMAT_ND,{},ge::FORMAT_ND,{});
  op.UpdateInputDesc("reshuffle", tensor_desc_reshuffle);
  auto states = op.VerifyAllAttr(true);
  // check result
  EXPECT_EQ(states, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}