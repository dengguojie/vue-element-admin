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
 * @file test_img_warp_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "image_ops.h"

class IMGWarp : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "IMGWarp SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "IMGWarp TearDown" << std::endl;
  }
};

TEST_F(IMGWarp, img_warp_infer_shape_ok) {
  ge::op::IMGWarp op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({1, 100, 4, 3},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1, 100, 4, 3},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("img", tensor_desc);
  op.UpdateInputDesc("warp_offset", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("warp_img");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 100, 4, 3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(IMGWarp, img_warp_infer_shape_img_failed) {
  ge::op::IMGWarp op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({100, 4, 3},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {100, 4, 3},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("img", tensor_desc);
  op.UpdateInputDesc("warp_offset", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(IMGWarp, img_warp_infer_shape_img_failed_1) {
  ge::op::IMGWarp op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  std::vector<std::pair<int64_t, int64_t>> shape_range_4 = {{1, 1}, {2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc_not_ok = create_desc_shape_range({100, 4, 3},
                                                    ge::DT_FLOAT16, ge::FORMAT_ND,
                                                    {100, 4, 3},
                                                    ge::FORMAT_ND, shape_range);
  auto tensor_desc_ok = create_desc_shape_range({1, 100, 4, 3},
                                                ge::DT_FLOAT, ge::FORMAT_ND,
                                                {1, 100, 4, 3},
                                                ge::FORMAT_ND, shape_range_4);
  op.UpdateInputDesc("img", tensor_desc_ok);
  op.UpdateInputDesc("warp_offset", tensor_desc_not_ok);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

