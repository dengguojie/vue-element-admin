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
 * @file test_SigmoidCrossEntropyWithLogits_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class SigmoidCrossEntropyWithLogits : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SigmoidCrossEntropyWithLogits SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SigmoidCrossEntropyWithLogits TearDown" << std::endl;
  }
};

//REG_OP(SigmoidCrossEntropyWithLogits)
//    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
//    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
//    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
//    .OP_END_FACTORY_REG(SigmoidCrossEntropyWithLogits)

TEST_F(SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogits_infer_shape_fp16) {
  ge::op::SigmoidCrossEntropyWithLogits op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 100}, {1, 10}};
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{50, 1000}, {10, 100}};
  auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, -1},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc1 = create_desc_shape_range({-1, -1},
                                              ge::DT_FLOAT16, ge::FORMAT_ND,
                                              {-1, -1},
                                              ge::FORMAT_ND, shape_range1);
  op.UpdateInputDesc("predict", tensor_desc);
  op.UpdateInputDesc("target", tensor_desc1);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("loss");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {50, 100}, {10, 10}
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
