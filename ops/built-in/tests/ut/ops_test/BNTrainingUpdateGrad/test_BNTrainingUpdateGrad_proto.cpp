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
 * @file test_BNTrainingUpdateGrad_impl_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class BNTrainingUpdateGrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BNTrainingUpdateGrad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BNTrainingUpdateGrad TearDown" << std::endl;
  }
};

TEST_F(BNTrainingUpdateGrad, bn_training_update_grad_test_1) {
  ge::op::BNTrainingUpdateGrad op;

  op.UpdateInputDesc("grads", create_desc({2,4,6,6,16}, ge::DT_FLOAT));
  op.UpdateInputDesc("x", create_desc({2,4,6,6,16}, ge::DT_FLOAT));
  op.UpdateInputDesc("batch_mean", create_desc({1,4,1,1,16}, ge::DT_FLOAT));
  op.UpdateInputDesc("batch_variance", create_desc({1,4,1,1,16}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_scale_desc = op.GetOutputDesc("diff_scale");
  auto output_offset_desc = op.GetOutputDesc("diff_offset");

  EXPECT_EQ(output_scale_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_offset_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {1,4,1,1,16};
  EXPECT_EQ(output_scale_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_offset_desc.GetShape().GetDims(), expected_output_shape);
}

