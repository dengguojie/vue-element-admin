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
 * @file test_CIoUGrad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class CIoUGradTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "CIoUGradTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CIoUGradTest TearDown" << std::endl;
  }
};

TEST_F(CIoUGradTest, CIoUGradTest_infershape_test_1) {
  ge::op::CIoUGrad op;

  op.UpdateInputDesc("dy", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("bboxes", create_desc({4, 1}, ge::DT_FLOAT));
  op.UpdateInputDesc("gtboxes", create_desc({4, 1}, ge::DT_FLOAT));
  op.UpdateInputDesc("atan_sub", create_desc({1,}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc_0 = op.GetOutputDesc("dbboxes");
  EXPECT_EQ(output_desc_0.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {4, 1};
  EXPECT_EQ(output_desc_0.GetShape().GetDims(), expected_output_shape);

  auto output_desc_1 = op.GetOutputDesc("dgtboxes");
  EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CIoUGradTest, CIoUGradTest_infershape_test_2) {
  ge::op::CIoUGrad op;

  op.UpdateInputDesc("dy", create_desc({15360,}, ge::DT_FLOAT));
  op.UpdateInputDesc("bboxes", create_desc({4, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("gtboxes", create_desc({4, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("atan_sub", create_desc({15360,}, ge::DT_FLOAT));
  op.SetAttr("is_cross", false);
  op.SetAttr("mode", "iou");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc_0 = op.GetOutputDesc("dbboxes");
  EXPECT_EQ(output_desc_0.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {4, 15360};
  EXPECT_EQ(output_desc_0.GetShape().GetDims(), expected_output_shape);
  
  auto output_desc_1 = op.GetOutputDesc("dgtboxes");
  EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CIoUGradTest, CIoUGradTest_infershape_test_3) {
  ge::op::CIoUGrad op;

  op.UpdateInputDesc("dy", create_desc({15360,}, ge::DT_FLOAT));
  op.UpdateInputDesc("bboxes", create_desc({4, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("gtboxes", create_desc({4, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("atan_sub", create_desc({15360,}, ge::DT_FLOAT));
  op.SetAttr("is_cross", false);
  op.SetAttr("mode", "iou");

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(CIoUGradTest, CIoUGradTest_infershape_test_4) {
  ge::op::CIoUGrad op;

  op.UpdateInputDesc("dy", create_desc({15360,}, ge::DT_FLOAT));
  op.UpdateInputDesc("bboxes", create_desc({5, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("gtboxes", create_desc({5, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("atan_sub", create_desc({15360,}, ge::DT_FLOAT));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CIoUGradTest, CIoUGradTest_infershape_test_5) {
  ge::op::CIoUGrad op;

  op.UpdateInputDesc("dy", create_desc({15360,}, ge::DT_FLOAT));
  op.UpdateInputDesc("bboxes", create_desc({4, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("gtboxes", create_desc({4, 15360}, ge::DT_INT32));
  op.UpdateInputDesc("atan_sub", create_desc({15360,}, ge::DT_FLOAT));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CIoUGradTest, CIoUGradTest_infershape_test_6) {
  ge::op::CIoUGrad op;

  op.UpdateInputDesc("dy", create_desc({15360,}, ge::DT_FLOAT));
  op.UpdateInputDesc("bboxes", create_desc({4, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("gtboxes", create_desc({4, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("atan_sub", create_desc({15360,}, ge::DT_FLOAT));
  op.SetAttr("is_cross", true);

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CIoUGradTest, CIoUGradTest_infershape_test_7) {
  ge::op::CIoUGrad op;

  op.UpdateInputDesc("dy", create_desc({15360,}, ge::DT_FLOAT));
  op.UpdateInputDesc("bboxes", create_desc({4, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("gtboxes", create_desc({4, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("atan_sub", create_desc({15360,}, ge::DT_FLOAT));
  op.SetAttr("mode", "iof");

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CIoUGradTest, CIoUGradTest_infershape_test_8) {
  ge::op::CIoUGrad op;

  op.UpdateInputDesc("dy", create_desc({15360,}, ge::DT_FLOAT));
  op.UpdateInputDesc("bboxes", create_desc({4, 15360}, ge::DT_FLOAT));
  op.UpdateInputDesc("gtboxes", create_desc({15360, 4}, ge::DT_FLOAT));
  op.UpdateInputDesc("atan_sub", create_desc({15360,}, ge::DT_FLOAT));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}