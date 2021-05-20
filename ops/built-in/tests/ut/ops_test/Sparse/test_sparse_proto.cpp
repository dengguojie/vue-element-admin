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
 * @file test_fifo_queue_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "sparse_ops.h"

class SparseTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseTest TearDown" << std::endl;
  }
};

TEST_F(SparseTest, SerializeSparseInferShape) {
  ge::op::SerializeSparse op;
  op.UpdateInputDesc("indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SparseTest, SerializeSparseInferShapeError1) {
  ge::op::SerializeSparse op;
  op.UpdateInputDesc("indices", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, SerializeSparseInferShapeError2) {
  ge::op::SerializeSparse op;
  op.UpdateInputDesc("indices", create_desc({2,3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2,3}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, SerializeSparseInferShapeError3) {
  ge::op::SerializeSparse op;
  op.UpdateInputDesc("indices", create_desc({2,3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3,3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, SerializeManySparseInferShape) {
  ge::op::SerializeManySparse op;
  op.UpdateInputDesc("indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SparseTest, SerializeManySparseInferShapeError1) {
  ge::op::SerializeManySparse op;
  op.UpdateInputDesc("indices", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, SerializeManySparseInferShapeError2) {
  ge::op::SerializeManySparse op;
  op.UpdateInputDesc("indices", create_desc({2,3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2,3}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, SerializeManySparseInferShapeError3) {
  ge::op::SerializeManySparse op;
  op.UpdateInputDesc("indices", create_desc({2,3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3,3}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, DeserializeSparseInferShape) {
  ge::op::DeserializeSparse op;
  op.UpdateInputDesc("serialized_sparse", create_desc({2, 3}, ge::DT_STRING));
  op.SetAttr("dtype", ge::DT_FLOAT16);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SparseTest, DeserializeSparseInferShapeError1) {
  ge::op::DeserializeSparse op;
  op.UpdateInputDesc("serialized_sparse", create_desc({2}, ge::DT_STRING));
  op.SetAttr("dtype", ge::DT_FLOAT16);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, DeserializeSparseInferShapeError2) {
  ge::op::DeserializeSparse op;
  op.UpdateInputDesc("serialized_sparse", create_desc({2,3}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, DeserializeManySparseInferShape) {
  ge::op::DeserializeManySparse op;
  op.UpdateInputDesc("serialized_sparse", create_desc({2, 3}, ge::DT_STRING));
  op.SetAttr("dtype", ge::DT_FLOAT16);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SparseTest, DeserializeManySparseInferShapeError1) {
  ge::op::DeserializeManySparse op;
  op.UpdateInputDesc("serialized_sparse", create_desc({2}, ge::DT_STRING));
  op.SetAttr("dtype", ge::DT_FLOAT16);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, DeserializeManySparseInferShapeError2) {
  ge::op::DeserializeManySparse op;
  op.UpdateInputDesc("serialized_sparse", create_desc({2, 3}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


TEST_F(SparseTest, AddManySparseToTensorsMapInferShape) {
  ge::op::AddManySparseToTensorsMap op;
  op.UpdateInputDesc("indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SparseTest, AddManySparseToTensorsMapInferShape2) {
  ge::op::AddManySparseToTensorsMap op;
  op.UpdateInputDesc("indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({0}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3,2}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, AddManySparseToTensorsMapInferShape3) {
  ge::op::AddManySparseToTensorsMap op;
  op.UpdateInputDesc("indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({1}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({1}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SparseTest, AddManySparseToTensorsMapInferShape4) {
  ge::op::AddManySparseToTensorsMap op;
  op.UpdateInputDesc("indices", create_desc({2, 3, 1}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({1}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({0}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, AddSparseToTensorsMapInferShape) {
  ge::op::AddSparseToTensorsMap op;
  op.UpdateInputDesc("indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SparseTest, AddSparseToTensorsMapInferShape1) {
  ge::op::AddSparseToTensorsMap op;
  op.UpdateInputDesc("indices", create_desc({2, 3, 3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2,1}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3,1}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, AddSparseToTensorsMapInferShape2) {
  ge::op::AddSparseToTensorsMap op;
  op.UpdateInputDesc("indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({2,2,3}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({3, 1}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SparseTest, AddSparseToTensorsMapInferShape3) {
  ge::op::AddSparseToTensorsMap op;
  op.UpdateInputDesc("indices", create_desc({2, 3}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({1}, ge::DT_INT64));
  op.UpdateInputDesc("shape", create_desc({2,1,3}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}