/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_SmoothL1Loss_proto.cpp
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class smooth_l1_loss_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "smooth_l1_loss_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "smooth_l1_loss_v2 TearDown" << std::endl;
  }
};

TEST_F(smooth_l1_loss_v2, smooth_l1_loss_v2_infershape_test_1) {

  ge::op::SmoothL1LossV2 op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 200});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  
  op.UpdateInputDesc("predict",tensorDesc);
  op.UpdateInputDesc("label",tensorDesc);
  float sigma = 1.0;
  op.SetAttr("sigma", sigma);
  std::string reduction = "mean";
  op.SetAttr("reduction", reduction);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("loss");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(smooth_l1_loss_v2, smooth_l1_loss_v2_infershape_test_2) {

  ge::op::SmoothL1LossV2 op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 200});
  tensorDesc.SetDataType(ge::DT_FLOAT);
  tensorDesc.SetShape(shape);
  
  op.UpdateInputDesc("predict",tensorDesc);
  op.UpdateInputDesc("label",tensorDesc);
  float sigma = 1.0;
  op.SetAttr("sigma", sigma);
  std::string reduction = "none";
  op.SetAttr("reduction", reduction);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("loss");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {10, 200};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(smooth_l1_loss_v2, smooth_l1_loss_v2_infershape_test_3) {

  ge::op::SmoothL1LossV2 op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 200});
  tensorDesc.SetDataType(ge::DT_FLOAT);
  tensorDesc.SetShape(shape);
  
  op.UpdateInputDesc("predict",tensorDesc);
  op.UpdateInputDesc("label",tensorDesc);
  float sigma = 1.0;
  op.SetAttr("sigma", sigma);
  std::string reduction = "what";
  op.SetAttr("reduction", reduction);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}