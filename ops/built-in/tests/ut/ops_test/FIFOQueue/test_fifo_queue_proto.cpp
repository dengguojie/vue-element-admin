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
 * @file test_fifo_queue_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"

class FIFOQueueTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FIFOQueueTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FIFOQueueTest TearDown" << std::endl;
  }
};

TEST_F(FIFOQueueTest, fifo_queue_infershape_test) {
    ge::op::FIFOQueue op;
    ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
    op.SetInferenceContext(inferCtxPtr);
    std::vector<ge::DataType> component_types{ ge::DT_FLOAT, ge::DT_FLOAT };
    op.SetAttr("component_types", component_types);
    std::vector<int64_t> shape{16, 16, 3};
    ge::Operator::OpListListInt elem_shapes{shape, shape};
    op.SetAttr("shapes", elem_shapes);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
