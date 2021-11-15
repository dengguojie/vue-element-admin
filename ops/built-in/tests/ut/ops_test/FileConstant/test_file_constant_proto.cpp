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
 * @file test_file_constant_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"

class FileConstantTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FileConstant SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FileConstant TearDown" << std::endl;
  }
};

TEST_F(FileConstantTest, file_constant) {
  auto const0 =
      ge::op::FileConstant().set_attr_shape({32, 16, 2, 2}).set_attr_dtype(ge::DT_FLOAT).set_attr_file_id("file");
  auto ret = const0.InferShapeAndType();
  EXPECT_EQ(ret,ge::GRAPH_SUCCESS);

  auto out_var_desc = const0.GetOutputDesc("y");
  std::vector<int64_t> expected_var_output_shape = {32, 16, 2, 2};
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}
