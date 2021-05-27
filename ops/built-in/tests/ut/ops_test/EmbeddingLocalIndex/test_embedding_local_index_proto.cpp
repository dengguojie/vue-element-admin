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

class EmbeddingLocalIndexTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "EmbeddingLocalIndexTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "EmbeddingLocalIndexTest TearDown" << std::endl;
  }
};

TEST_F(EmbeddingLocalIndexTest, InferShape_01) {
  ge::op::EmbeddingLocalIndex op;
  op.UpdateInputDesc("addr_table", create_desc({2, 3}, ge::DT_INT32));
  op.UpdateInputDesc("index", create_desc({8}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(EmbeddingLocalIndexTest, InferShape_02) {
  ge::op::EmbeddingLocalIndex op;
  op.UpdateInputDesc("addr_table", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("index", create_desc({8}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(EmbeddingLocalIndexTest, InferShape_03) {
  ge::op::EmbeddingLocalIndex op;
  op.UpdateInputDesc("addr_table", create_desc({2, 2}, ge::DT_INT32));
  op.UpdateInputDesc("index", create_desc({8}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(EmbeddingLocalIndexTest, InferShape_04) {
  ge::op::EmbeddingLocalIndex op;
  op.UpdateInputDesc("addr_table", create_desc({-1, 3}, ge::DT_INT32));
  op.UpdateInputDesc("index", create_desc({8}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(EmbeddingLocalIndexTest, InferShape_05) {
  ge::op::EmbeddingLocalIndex op;
  op.UpdateInputDesc("addr_table", create_desc({2, 3}, ge::DT_INT32));
  op.UpdateInputDesc("index", create_desc({2, 8}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(EmbeddingLocalIndexTest, InferShape_06) {
  ge::op::EmbeddingLocalIndex op;
  op.UpdateInputDesc("addr_table", create_desc({2, 3}, ge::DT_INT32));
  op.UpdateInputDesc("index", create_desc({8}, ge::DT_INT32));
  op.SetAttr("row_memory", -1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
