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
 * @file test_embedding_rank_id_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "pad_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

class EmbeddingRankIdInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "EmbeddingRankIdInferShape SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "EmbeddingRankIdInferShape TearDown" << std::endl;
  }
};

TEST_F(EmbeddingRankIdInferShape, embedding_rank_id_infer_shape01) {
  ge::op::EmbeddingRankId op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("addr_table", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(EmbeddingRankIdInferShape, embedding_rank_id_infer_shape02) {
  ge::op::EmbeddingRankId op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {2, 2}};
  auto tensor_desc = create_desc_shape_range({2, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("addr_table", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(EmbeddingRankIdInferShape, embedding_rank_id_infer_shape03) {
  ge::op::EmbeddingRankId op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{-1, 1}, {3, 3}};
  auto tensor_desc = create_desc_shape_range({-1, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {-1, 2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("addr_table", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(EmbeddingRankIdInferShape, embedding_rank_id_infer_shape04) {
  ge::op::EmbeddingRankId op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 1}, {3, 3}};
  auto tensor_desc = create_desc_shape_range({1, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {1, 2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("addr_table", tensor_desc);
  op.UpdateInputDesc("index", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(EmbeddingRankIdInferShape, embedding_rank_id_infer_shape05) {
  ge::op::EmbeddingRankId op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 1}, {3, 3}};
  auto addr_table_tensor_desc = create_desc_shape_range({1, 2},
                                                        ge::DT_INT32, ge::FORMAT_ND,
                                                        {1, 2},
                                                        ge::FORMAT_ND, shape_range);
  auto index_tensor_desc = create_desc_shape_range({1},
                                                   ge::DT_INT32, ge::FORMAT_ND,
                                                   {1},
                                                   ge::FORMAT_ND, {{1, 1}});
  op.UpdateInputDesc("addr_table", addr_table_tensor_desc);
  op.UpdateInputDesc("index", index_tensor_desc);
  op.SetAttr("row_memory", -1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(EmbeddingRankIdInferShape, embedding_rank_id_infer_shape06) {
  ge::op::EmbeddingRankId op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 1}, {3, 3}};
  auto addr_table_tensor_desc = create_desc_shape_range({1, 2},
                                                        ge::DT_INT32, ge::FORMAT_ND,
                                                        {1, 2},
                                                        ge::FORMAT_ND, shape_range);
  auto index_tensor_desc = create_desc_shape_range({1},
                                                   ge::DT_INT32, ge::FORMAT_ND,
                                                   {1},
                                                   ge::FORMAT_ND, {{1, 1}});
  op.UpdateInputDesc("addr_table", addr_table_tensor_desc);
  op.UpdateInputDesc("index", index_tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(EmbeddingRankIdInferShape, embedding_rank_id_infer_shape07) {
  ge::op::EmbeddingRankId op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 1}, {3, 3}};
  auto addr_table_tensor_desc = create_desc_shape_range({1, 2},
                                                        ge::DT_INT32, ge::FORMAT_ND,
                                                        {1, 2},
                                                        ge::FORMAT_ND, shape_range);
  auto index_tensor_desc = create_desc_shape_range({1},
                                                   ge::DT_INT32, ge::FORMAT_ND,
                                                   {1},
                                                   ge::FORMAT_ND, {{1, 1}});
  op.UpdateInputDesc("addr_table", addr_table_tensor_desc);
  op.UpdateInputDesc("index", index_tensor_desc);
  op.SetAttr("row_memory", 3);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}