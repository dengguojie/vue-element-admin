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
 * @file test_string_format_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"


class Accumulator : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Accumulator Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Accumulator Proto Test TearDown" << std::endl;
  }
};

TEST_F(Accumulator, AccumulatorApplyGradient_infer_shape) {
  ge::op::AccumulatorApplyGradient op;
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  op.UpdateInputDesc("local_step", create_desc_with_ori({1}, ge::DT_INT64, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Accumulator, AccumulatorApplyGradient_infer_shape01) {
  ge::op::AccumulatorApplyGradient op;
  ge::TensorDesc tensor_desc_handle(ge::Shape({0}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  op.UpdateInputDesc("local_step", create_desc_with_ori({1}, ge::DT_INT32, ge::FORMAT_NHWC, {1}, ge::FORMAT_NHWC));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Accumulator, AccumulatorNumAccumulated_infer_shape) {
  ge::op::AccumulatorNumAccumulated op;
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Accumulator, AccumulatorSetGlobalStep_infer_shape) {
  ge::op::AccumulatorSetGlobalStep op;
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("new_global_step", create_desc_with_ori({1}, ge::DT_INT64, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Accumulator, AccumulatorTakeGradient_infer_shape) {
  ge::op::AccumulatorTakeGradient op;
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("num_required", create_desc_with_ori({1}, ge::DT_INT64, ge::FORMAT_NHWC, {2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}