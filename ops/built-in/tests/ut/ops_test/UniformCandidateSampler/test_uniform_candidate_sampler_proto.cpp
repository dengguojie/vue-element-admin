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
 * @file test_unbatch_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "candidate_sampling_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

class UniformCandidateSampler : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Unbatch SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Unbatch TearDown" << std::endl;
  }
};

TEST_F(UniformCandidateSampler, uniform_candidate_sampler_infer_shape01) {
  ge::op::UniformCandidateSampler op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {2, 2}};
  auto tensor_desc = create_desc_shape_range({2, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("true_classes", tensor_desc);
  op.SetAttr("num_true", 3);
  op.SetAttr("num_sampled", 3);
  op.SetAttr("range_max", 3);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(UniformCandidateSampler, uniform_candidate_sampler_infer_shape02) {
  ge::op::UniformCandidateSampler op;
  op.SetAttr("num_true", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UniformCandidateSampler, uniform_candidate_sampler_infer_shape03) {
  ge::op::UniformCandidateSampler op;
  op.SetAttr("num_true", 3);
  op.SetAttr("num_sampled", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(UniformCandidateSampler, uniform_candidate_sampler_infer_shape04) {
  ge::op::UniformCandidateSampler op;
  op.SetAttr("num_true", 3);
  op.SetAttr("num_sampled", 3);
  op.SetAttr("range_max", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}