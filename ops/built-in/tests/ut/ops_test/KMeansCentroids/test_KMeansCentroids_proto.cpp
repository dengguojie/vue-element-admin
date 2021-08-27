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
 * @file test_KMeansCentroids_proto.cpp
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "cluster.h"

class KMeansCentroids : public testing::Test {
  protected:
    static void SetUpTestCase() {
      std::cout << "KMeansCentroids SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "KMeansCentroids TearDown" << std::endl;
    }
};

TEST_F(KMeansCentroids, k_means_centroids_infershape_test1) {
  ge::op::KMeansCentroids op;

  op.UpdateInputDesc("x", create_desc_with_ori({256, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({256, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("sum_square_y", create_desc_with_ori({1, 256}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {1, 256}, ge::FORMAT_ND));

  op.UpdateOutputDesc("segment_sum", create_desc_with_ori({256, 128}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateOutputDesc("segment_count", create_desc_with_ori({256, 1}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 1}, ge::FORMAT_ND));
  op.UpdateOutputDesc("kmean_total_sum", create_desc_with_ori({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(KMeansCentroids, k_means_centroids_infershape_test2) {
  ge::op::KMeansCentroids op;

  op.UpdateInputDesc("x", create_desc_with_ori({256, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({256, 122}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 122}, ge::FORMAT_ND));
  op.UpdateInputDesc("sum_square_y", create_desc_with_ori({1, 256}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {1, 256}, ge::FORMAT_ND));

  op.UpdateOutputDesc("segment_sum", create_desc_with_ori({256, 128}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateOutputDesc("segment_count", create_desc_with_ori({256, 1}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 1}, ge::FORMAT_ND));
  op.UpdateOutputDesc("kmean_total_sum", create_desc_with_ori({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(KMeansCentroids, k_means_centroids_infershape_test3) {
  ge::op::KMeansCentroids op;

  op.UpdateInputDesc("x", create_desc_with_ori({256, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({256, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("sum_square_y", create_desc_with_ori({1, 255}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {1, 255}, ge::FORMAT_ND));

  op.UpdateOutputDesc("segment_sum", create_desc_with_ori({256, 128}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateOutputDesc("segment_count", create_desc_with_ori({256, 1}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 1}, ge::FORMAT_ND));
  op.UpdateOutputDesc("kmean_total_sum", create_desc_with_ori({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(KMeansCentroids, k_means_centroids_infershape_test4) {
  ge::op::KMeansCentroids op;

  op.UpdateInputDesc("x", create_desc_with_ori({256, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({256, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("sum_square_y", create_desc_with_ori({1, 256}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {1, 256}, ge::FORMAT_ND));
  op.SetAttr("use_actual_distance", true);
  op.UpdateInputDesc("sum_square_x", create_desc_with_ori({255, 1}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {255, 1}, ge::FORMAT_ND));

  op.UpdateOutputDesc("segment_sum", create_desc_with_ori({256, 128}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateOutputDesc("segment_count", create_desc_with_ori({256, 1}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 1}, ge::FORMAT_ND));
  op.UpdateOutputDesc("kmean_total_sum", create_desc_with_ori({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(KMeansCentroids, k_means_centroids_infershape_test5) {
  ge::op::KMeansCentroids op;

  op.UpdateInputDesc("x", create_desc_with_ori({256, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({256, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("sum_square_y", create_desc_with_ori({1, 256}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {1, 256}, ge::FORMAT_ND));
  op.SetAttr("use_actual_distance", true);
  op.UpdateInputDesc("sum_square_x", create_desc_with_ori({256, 1}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {256, 1}, ge::FORMAT_ND));

  op.UpdateOutputDesc("segment_sum", create_desc_with_ori({256, 128}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateOutputDesc("segment_count", create_desc_with_ori({256, 1}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 1}, ge::FORMAT_ND));
  op.UpdateOutputDesc("kmean_total_sum", create_desc_with_ori({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(KMeansCentroids, k_means_centroids_infershape_test6) {
  ge::op::KMeansCentroids op;

  op.UpdateInputDesc("x", create_desc_with_ori({222, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {222, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({256, 128}, ge::DT_FLOAT, ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateInputDesc("sum_square_y", create_desc_with_ori({1, 256}, ge::DT_FLOAT,
                     ge::FORMAT_ND, {1, 256}, ge::FORMAT_ND));

  op.UpdateOutputDesc("segment_sum", create_desc_with_ori({256, 128}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 128}, ge::FORMAT_ND));
  op.UpdateOutputDesc("segment_count", create_desc_with_ori({256, 1}, ge::DT_FLOAT,
                      ge::FORMAT_ND, {256, 1}, ge::FORMAT_ND));
  op.UpdateOutputDesc("kmean_total_sum", create_desc_with_ori({1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

