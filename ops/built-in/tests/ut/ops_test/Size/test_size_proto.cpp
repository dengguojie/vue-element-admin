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
 * @file test_reshape_unknown_shape_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include <climits>
#include "op_proto_test_util.h"
#include "util.h"
#include "graph/utils/op_desc_utils.h"
#include "array_ops.h"
#include "graph/ge_tensor.h"
#include "graph/utils/graph_utils.h"

class SIZE_UT : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "SIZE_UT SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "SIZE_UT TearDown" << std::endl;
    }
};

TEST_F(SIZE_UT, InferShape_succ) {
  ge::op::Size op("Size");
  op.UpdateInputDesc("x", create_desc({8, 3, 224, 224}, ge::DT_INT32));
  op.SetAttr("dtype", 9);

  EXPECT_EQ(op.InferShapeAndType(), ge::GRAPH_SUCCESS);
  ge::TensorDesc td_y = op.GetOutputDesc("y");
  EXPECT_EQ(td_y.GetDataType(), ge::DT_INT64);
  auto dims = td_y.GetShape().GetDims();
  EXPECT_EQ(dims.size(), 0);
}