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
 * @file test_resize_nearest_neighbor_v2_grad_proto.cpp
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

class ResizeNearestNeighborV2Grad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResizeNearestNeighborV2Grad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResizeNearestNeighborV2Grad TearDown" << std::endl;
  }
};

TEST_F(ResizeNearestNeighborV2Grad, resize_nearest_neighbor_v2_grad_infer_shape01) {
  ge::op::ResizeNearestNeighborV2Grad op;
  std::vector<std::pair<int64_t,int64_t>> shape_grad_range = {{22, 22}, {10, 10}, {4, 4}, {1, 1}};
  std::vector<std::pair<int64_t,int64_t>> shape_size_range = {{2, 2}};
  auto tensor_grad_desc = create_desc_shape_range({22, 10, 4, 1},
                                             ge::DT_INT32, ge::FORMAT_NHWC,
                                             {22, 10, 4, 1},
                                             ge::FORMAT_NHWC, shape_grad_range);
  auto tensor_size_desc = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_size_range);
  op.UpdateInputDesc("grads", tensor_grad_desc);
  op.UpdateInputDesc("size", tensor_size_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ResizeNearestNeighborV2Grad, resize_nearest_neighbor_v2_grad_infer_shape02) {
  ge::op::ResizeNearestNeighborV2Grad op;
  auto tensor_grad_desc = create_desc_shape_range({-2},
                                             ge::DT_INT32, ge::FORMAT_NHWC,
                                             {-2},
                                             ge::FORMAT_NHWC, {{}});
  auto tensor_size_desc = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, {{2, 2}});
  op.UpdateInputDesc("grads", tensor_grad_desc);
  op.UpdateInputDesc("size", tensor_size_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ResizeNearestNeighborV2Grad, resize_nearest_neighbor_v2_grad_fail_01) {
  ge::op::ResizeNearestNeighborV2Grad op;
  std::vector<std::pair<int64_t,int64_t>> shape_grad_range = {{22, 22}, {10, 10}};
  std::vector<std::pair<int64_t,int64_t>> shape_size_range = {{2, 2}};
  auto tensor_grad_desc = create_desc_shape_range({22, 10},
                                             ge::DT_INT32, ge::FORMAT_NHWC,
                                             {22, 10},
                                             ge::FORMAT_NHWC, shape_grad_range);
  auto tensor_size_desc = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_size_range);
  op.UpdateInputDesc("grads", tensor_grad_desc);
  op.UpdateInputDesc("size", tensor_size_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeNearestNeighborV2Grad, resize_nearest_neighbor_v2_grad_fail_02) {
  ge::op::ResizeNearestNeighborV2Grad op;
  std::vector<std::pair<int64_t,int64_t>> shape_grad_range = {{22, 22}, {10, 10}, {4, 4}, {1, 1}};
  std::vector<std::pair<int64_t,int64_t>> shape_size_range = {{2, 2}, {2, 2}};
  auto tensor_grad_desc = create_desc_shape_range({22, 10, 4, 1},
                                             ge::DT_INT32, ge::FORMAT_NHWC,
                                             {22, 10, 4, 1},
                                             ge::FORMAT_NHWC, shape_grad_range);
  auto tensor_size_desc = create_desc_shape_range({2, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 2},
                                             ge::FORMAT_ND, shape_size_range);
  op.UpdateInputDesc("grads", tensor_grad_desc);
  op.UpdateInputDesc("size", tensor_size_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeNearestNeighborV2Grad, resize_nearest_neighbor_v2_grad_fail_03) {
  ge::op::ResizeNearestNeighborV2Grad op;
  std::vector<std::pair<int64_t,int64_t>> shape_grad_range = {{22, 22}, {10, 10}, {4, 4}, {1, 1}};
  std::vector<std::pair<int64_t,int64_t>> shape_size_range = {{4, 4}};
  auto tensor_grad_desc = create_desc_shape_range({22, 10, 4, 1},
                                             ge::DT_INT32, ge::FORMAT_NHWC,
                                             {22, 10, 4, 1},
                                             ge::FORMAT_NHWC, shape_grad_range);
  auto tensor_size_desc = create_desc_shape_range({4},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {4},
                                             ge::FORMAT_ND, shape_size_range);
  op.UpdateInputDesc("grads", tensor_grad_desc);
  op.UpdateInputDesc("size", tensor_size_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeNearestNeighborV2Grad, resize_nearest_neighbor_v2_grad_fail_04) {
  ge::op::ResizeNearestNeighborV2Grad op;
  std::vector<std::pair<int64_t,int64_t>> shape_grad_range = {{22, 22}, {10, 10}, {4, 4}, {1, 1}};
  std::vector<std::pair<int64_t,int64_t>> shape_size_range = {{2, 2}};
  auto tensor_grad_desc = create_desc_shape_range({22, 10, 4, 1},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {22, 10, 4, 1},
                                             ge::FORMAT_ND, shape_grad_range);
  auto tensor_size_desc = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_size_range);
  op.UpdateInputDesc("grads", tensor_grad_desc);
  op.UpdateInputDesc("size", tensor_size_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
