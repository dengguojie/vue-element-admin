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
 * @file test_HardSigmoid_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class HardSigmoidTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "HardShrink test SetUp" << std::endl;
}

  static void TearDownTestCase() {
    std::cout << "HardShrink test TearDown" << std::endl;
  }
};

TEST_F(HardSigmoidTest, hard_sigmoid_test_case_1) {
  ge::op::HardSigmoid hard_sigmoid_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({2, 3, 4});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  hard_sigmoid_op.UpdateInputDesc("input_x", tensorDesc);
  
  auto ret = hard_sigmoid_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = hard_sigmoid_op.GetOutputDesc("output_y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}