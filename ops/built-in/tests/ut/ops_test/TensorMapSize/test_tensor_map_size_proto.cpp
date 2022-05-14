/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this
 file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_tensor_map_size_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "graph/operator.h"
#include "map_ops.h"
#include "op_proto_test_common.h"
#include "op_proto_test_util.h"

class TENSOR_MAP_SIZE_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TENSOR_MAP_SIZE_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TENSOR_MAP_SIZE_UT TearDown" << std::endl;
  }
};

TEST_F(TENSOR_MAP_SIZE_UT, MapSizeInferShape) {
  ge::op::TensorMapSize op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  ge::InferenceContextPtr inferCtxPtr =
      std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto y_desc = op.GetOutputDescByName("size");
  EXPECT_EQ(y_desc.GetDataType(), ge::DT_INT32);
}
