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
 * @file test_GeluGrad_proto.cpp
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"
#include "common/utils/ut_op_common.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

class gelugrad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gelugrad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gelugrad TearDown" << std::endl;
  }
};

TEST_F(gelugrad, gelugrad_infershape_test){
  ge::op::GeluGrad op;
  
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("dy", tensor_desc);
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("y", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  float negative_slope = 0.0;
  op.SetAttr("negative_slope", negative_slope);

  auto output_desc = op.GetOutputDesc("z");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1,16},{1,16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
  
  CommonInferShapeOperator(op, {}, {expected_output_shape});
  auto output_desc1 = op.GetOutputDesc(0);
  EXPECT_EQ(output_desc1.GetShape().GetDims(), expected_output_shape);
  Runtime2TestParam param;
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
  auto output0_desc = op.GetOutputDesc(0);
  EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}
