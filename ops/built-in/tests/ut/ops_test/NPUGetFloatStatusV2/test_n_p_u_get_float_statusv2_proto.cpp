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
 * @file test_add_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "npu_loss_scale_ops.h"

class n_p_u_get_float_status_v2_infer_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NPUGetFloatStatusV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NPUGetFloatStatusV2 TearDown" << std::endl;
  }
};

TEST_F(n_p_u_get_float_status_v2_infer_test, n_p_u_get_float_status_v2_infer_test_1) {
  ge::op::NPUGetFloatStatusV2 op;
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("data");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {8};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  };