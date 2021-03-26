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
 * @file test_ApplyAdamWithAmsgrad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"

class ApplyAdamWithAmsgradProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyAdamWithAmsgrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyAdamWithAmsgrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyAdamWithAmsgradProtoTest,
       apply_adam_with_amsgrad_infershape_verify_test) {
  ge::op::ApplyAdamWithAmsgrad op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("m", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("v", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("vhat", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(ApplyAdamWithAmsgradProtoTest,
       apply_adam_with_amsgrad_verify_fail_test) {
  ge::op::ApplyAdamWithAmsgrad op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("m", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("v", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("vhat", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({2, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}


class ApplyAdamWithAmsgradDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyAdamWithAmsgradD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyAdamWithAmsgradD Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyAdamWithAmsgradDProtoTest,
       apply_adam_with_amsgrad_d_infershape_verify_test) {
  ge::op::ApplyAdamWithAmsgradD op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("m", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("v", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("vhat", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.SetAttr("beta1", (float)0.01);
  op.SetAttr("beta2", (float)0.05);
  op.SetAttr("epsilon", (float)0.001);
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(var_desc.GetShape().GetDims(), expected_var_output_shape);

  auto m_desc = op.GetOutputDesc("m");
  EXPECT_EQ(m_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_m_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(m_desc.GetShape().GetDims(), expected_m_output_shape);

  auto v_desc = op.GetOutputDesc("v");
  EXPECT_EQ(v_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_v_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(v_desc.GetShape().GetDims(), expected_v_output_shape);

  auto vhat_desc = op.GetOutputDesc("vhat");
  EXPECT_EQ(vhat_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_vhat_output_shape = {16, 48, 16, 32};
  EXPECT_EQ(vhat_desc.GetShape().GetDims(), expected_vhat_output_shape);
}

TEST_F(ApplyAdamWithAmsgradDProtoTest,
       apply_adam_with_amsgrad_d_verify_fail0_test) {
  ge::op::ApplyAdamWithAmsgradD op;
  op.UpdateInputDesc("var", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("m", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("v", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("vhat", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({2, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 48, 16, 32}, ge::DT_FLOAT));
  op.SetAttr("beta1", (float)0.01);
  op.SetAttr("beta2", (float)0.05);
  op.SetAttr("epsilon", (float)0.001);
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ApplyAdamWithAmsgradDProtoTest, dynamic_apply_adam_with_amsgrad_d_test_01) {
  ge::op::ApplyAdamWithAmsgradD op;

  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 2}, {1, 7}, {1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 7, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("var", tensor_desc);
  op.UpdateInputDesc("m", tensor_desc);
  op.UpdateInputDesc("v", tensor_desc);
  op.UpdateInputDesc("vhat", tensor_desc);
  op.UpdateInputDesc("beta1_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("beta2_power", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", tensor_desc);
  op.SetAttr("beta1", (float)0.01);
  op.SetAttr("beta2", (float)0.05);
  op.SetAttr("epsilon", (float)0.001);
  op.SetAttr("use_locking", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = shape_range;

  auto output_desc_var = op.GetOutputDesc("var");
  EXPECT_EQ(output_desc_var.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_var.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc_var.GetOriginShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_var_shape_range;
  EXPECT_EQ(output_desc_var.GetShapeRange(output_var_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_var_shape_range, expected_shape_range);

  auto output_desc_m = op.GetOutputDesc("m");
  EXPECT_EQ(output_desc_m.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_m.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc_m.GetOriginShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_m_shape_range;
  EXPECT_EQ(output_desc_m.GetShapeRange(output_m_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_m_shape_range, expected_shape_range);

  auto output_desc_v = op.GetOutputDesc("v");
  EXPECT_EQ(output_desc_v.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_v.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc_v.GetOriginShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_v_shape_range;
  EXPECT_EQ(output_desc_v.GetShapeRange(output_v_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_v_shape_range, expected_shape_range);

  auto output_desc_vhat = op.GetOutputDesc("vhat");
  EXPECT_EQ(output_desc_vhat.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_vhat.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc_vhat.GetOriginShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_vhat_shape_range;
  EXPECT_EQ(output_desc_vhat.GetShapeRange(output_vhat_shape_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_vhat_shape_range, expected_shape_range);
}
