/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "array_ops.h"
#include "matrix_calculation_ops.h"
#include "op_proto_test_util.h"

class AttentionScoreTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AttentionScore Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AttentionScore Proto Test TearDown" << std::endl;
  }
};


TEST_F(AttentionScoreTest, AttentionScoreInferShapeTest_1) {
  int32_t batch_dim0 = 32;
  int32_t batch_dim1 = 16;
  int32_t seq_dim = 384;
  int32_t n_dim = 64;

  ge::op::AttentionScore op;
  op.UpdateInputDesc("query", create_desc_shape_range({batch_dim0, batch_dim1, seq_dim, n_dim},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {batch_dim0, batch_dim1, seq_dim, n_dim}, ge::FORMAT_ND,
                                                      {{batch_dim0, batch_dim0}, {batch_dim1, batch_dim1},
                                                       {seq_dim, seq_dim}, {n_dim, n_dim}}));
  op.UpdateInputDesc("key", create_desc_shape_range({batch_dim0, batch_dim1, seq_dim, n_dim},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {batch_dim0, batch_dim1, seq_dim, n_dim}, ge::FORMAT_ND,
                                                      {{batch_dim0, batch_dim0}, {batch_dim1, batch_dim1},
                                                       {seq_dim, seq_dim}, {n_dim, n_dim}}));
  op.UpdateInputDesc("value", create_desc_shape_range({batch_dim0, batch_dim1, seq_dim, n_dim},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {batch_dim0, batch_dim1, seq_dim, n_dim}, ge::FORMAT_ND,
                                                      {{batch_dim0, batch_dim0}, {batch_dim1, batch_dim1},
                                                       {seq_dim, seq_dim}, {n_dim, n_dim}}));
  op.UpdateInputDesc("padding_mask", create_desc_shape_range({batch_dim0, 1, seq_dim, seq_dim},
                                                         ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {batch_dim0, 1, seq_dim, seq_dim}, ge::FORMAT_ND,
                                                      {{batch_dim0, batch_dim0}, {1, 1},
                                                       {seq_dim, seq_dim}, {seq_dim, seq_dim}}));
  op.UpdateInputDesc("scale", create_desc_shape_range({1},
                                                         ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {1}, ge::FORMAT_ND,
                                                      {{1, 1}}));
  op.UpdateInputDesc("drop_mask", create_desc_shape_range({batch_dim0, batch_dim1, seq_dim, n_dim},
                                                      ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {batch_dim0, batch_dim1, seq_dim, n_dim}, ge::FORMAT_ND,
                                                      {{batch_dim0, batch_dim0}, {batch_dim1, batch_dim1},
                                                       {seq_dim, seq_dim}, {n_dim, n_dim}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}



