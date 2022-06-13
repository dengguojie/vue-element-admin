/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_masked_select)_v2_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class MaskedSelectV2UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaskedSelectV2UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaskedSelectV2UT TearDown" << std::endl;
  }
};

TEST_F(MaskedSelectV2UT, masked_select_test_1) {
  ge::op::MaskedSelectV2 op;

  ge::TensorDesc tensor_desc;
  ge::Shape shape({3, 4, 2});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape);
  op.UpdateInputDesc("x", tensor_desc);

  auto ret = op.InferShapeAndType();
}

TEST_F(MaskedSelectV2UT, masked_select_test_2) {
  ge::op::MaskedSelectV2 op;

  ge::TensorDesc tensor_desc;
  ge::Shape shape({3, 4, 2});
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(shape);
  tensor_desc.SetOriginShape(shape);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("mask", tensor_desc);

  auto ret = op.InferShapeAndType();
}
