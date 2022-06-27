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
 * @file test_flow_func_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>

#include "op_proto_test_util.h"
#include "data_flow_ops.h"

class FlowFuncTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FlowFuncTest test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FlowFuncTest test TearDown" << std::endl;
  }
};

TEST_F(FlowFuncTest, flow_func_infershape_invalid_test1) {
  ge::op::FlowFunc op;
  std::vector<int64_t> shape0{2,2};
  std::vector<int64_t> shape1{3,3};
  ge::Operator::OpListListInt output_shapes{shape0, shape1};
  op.set_attr_output_shapes(output_shapes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(FlowFuncTest, flow_func_infershape_invalid_test2) {
  ge::op::FlowFunc op;
  std::vector<ge::DataType> output_types{ ge::DT_FLOAT16};
  op.set_attr_output_types(output_types);
  std::vector<int64_t> shape0{2,2};
  std::vector<int64_t> shape1{3,3};
  ge::Operator::OpListListInt output_shapes{shape0, shape1};
  op.set_attr_output_shapes(output_shapes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(FlowFuncTest, flow_func_infershape_types_test1) {
  ge::op::FlowFunc op;
  std::vector<ge::DataType> output_types{ ge::DT_FLOAT16, ge::DT_FLOAT};
  op.set_attr_output_types(output_types);
  op.create_dynamic_output_y(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(FlowFuncTest, flow_func_infershape_types_tes2) {
  ge::op::FlowFunc op;
  std::vector<ge::DataType> output_types{ ge::DT_FLOAT16, ge::DT_FLOAT};
  op.set_attr_output_types(output_types);
  std::vector<int64_t> shape0{2,2};
  std::vector<int64_t> shape1{3,3};
  ge::Operator::OpListListInt output_shapes{shape0, shape1};
  op.set_attr_output_shapes(output_shapes);
  op.create_dynamic_output_y(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}