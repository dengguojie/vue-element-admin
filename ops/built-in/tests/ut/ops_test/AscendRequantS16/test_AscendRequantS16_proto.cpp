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
 * @file test_AscendRequantS16_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "quantize_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

class ascend_requant_s16_infer_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ascend_requant_s16_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ascend_requant_s16_infer_test TearDown" << std::endl;
  }
};

TEST_F(ascend_requant_s16_infer_test, ascend_requant_s16_infer_test_1) {
  bool relu_flag = false;
  bool dual_output = false;

  // expect result
  std::vector<int64_t> expected_shape = {3, 16, 16, 64};

  // new op and do infershape
  ge::op::AscendRequantS16 op;
  op.UpdateInputDesc("x0", create_desc_with_ori({3, 16, 16, 64}, ge::DT_INT16, ge::FORMAT_FRACTAL_NZ, {3, 16, 16, 64},
                                                ge::FORMAT_FRACTAL_NZ));
  op.UpdateInputDesc("req_scale", create_desc_with_ori({1, 1, 1, 64}, ge::DT_UINT64, ge::FORMAT_FRACTAL_NZ, {3, 3, 64},
                                                       ge::FORMAT_FRACTAL_NZ));
  op.set_attr_dual_output(dual_output);
  op.set_attr_relu_flag(relu_flag);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y0");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(ascend_requant_s16_infer_test, ascend_requant_s16_infer_test_2) {
  bool relu_flag = true;
  bool dual_output = true;

  ge::op::AscendRequantS16 op;
  op.UpdateInputDesc("x0", create_desc_with_ori({1, -1, -1, -1, -1}, ge::DT_INT16, ge::FORMAT_NC1HWC0, {1, 2, 4, 4, 16},
                                                ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("req_scale", create_desc_with_ori({1, 2, 1, 1, 16}, ge::DT_UINT64, ge::FORMAT_NC1HWC0, {32, },
                                                       ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("x1", create_desc_with_ori({1, -1, 1, 1, 16}, ge::DT_INT16, ge::FORMAT_NC1HWC0, {1, 2, 1, 1, 16},
                                                ge::FORMAT_NC1HWC0));
  op.set_attr_dual_output(dual_output);
  op.set_attr_relu_flag(relu_flag);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expected_shape_y0 = {1, -1, -1, -1, -1};
  auto output_desc_y0 = op.GetOutputDesc("y0");
  EXPECT_EQ(output_desc_y0.GetShape().GetDims(), expected_shape_y0);

  std::vector<int64_t> expected_shape_y1 = {1, -1, -1, -1, -1};
  auto output_desc_y1 = op.GetOutputDesc("y1");
  EXPECT_EQ(output_desc_y1.GetShape().GetDims(), expected_shape_y1);
}
