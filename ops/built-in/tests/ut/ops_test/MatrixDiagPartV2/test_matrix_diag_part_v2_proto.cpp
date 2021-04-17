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
 * @file test_matrix_diag_part_v2_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"
#include "split_combination_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

class MatrixDiagPartV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MatrixDiagPartV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MatrixDiagPartV2 TearDown" << std::endl;
  }
};

TEST_F(MatrixDiagPartV2, matrix_diag_part_v2_infer_shape) {
  ge::op::MatrixDiagPartV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range1 = {{2, 2}, {2, 2}};
  std::vector<std::pair<int64_t,int64_t>> shape_range2 = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2, 2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2, 2},
                                              ge::FORMAT_ND, shape_range1);
  auto tensor_desc2 = create_desc_shape_range({2},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {2},
                                              ge::FORMAT_ND, shape_range2);
  auto tensor_desc3 = create_desc_shape_range({},
                                              ge::DT_INT64, ge::FORMAT_ND,
                                              {},
                                              ge::FORMAT_ND, {{}}});

  op.UpdateInputDesc("input", tensor_desc1);
  op.UpdateInputDesc("k", tensor_desc2);
  op.UpdateInputDesc("padding_value", tensor_desc3);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}