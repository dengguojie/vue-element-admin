#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "lookup_ops.h"

class MutableDenseHashTable : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MutableDenseHashTable SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MutableDenseHashTable TearDown" << std::endl;
  }
};

TEST_F(MutableDenseHashTable, MutableDenseHashTable_infer_shape_0) {
  ge::op::MutableDenseHashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MutableDenseHashTable, MutableDenseHashTable_infer_shape_1) {
  ge::op::MutableDenseHashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("empty_key", create_desc({1}, ge::DT_INT64));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MutableDenseHashTable, MutableDenseHashTable_infer_shape_2) {
  ge::op::MutableDenseHashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  std::vector<std::vector<int64_t>> value_shape = {{-1}};
  op.SetAttr("value_shape", value_shape);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
} 


TEST_F(MutableDenseHashTable, MutableDenseHashTable_infer_shape_3) {
  ge::op::MutableDenseHashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("empty_key", create_desc({1}, ge::DT_INT32));
  op.SetAttr("value_dtype", ge::DT_UINT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
} 

TEST_F(MutableDenseHashTable, MutableDenseHashTable_infer_shape_4) {
  ge::op::MutableDenseHashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("empty_key", create_desc({1}, ge::DT_INT64));
  op.SetAttr("value_dtype", ge::DT_UINT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MutableDenseHashTable, MutableDenseHashTable_infer_shape_5) {
  ge::op::MutableDenseHashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  std::vector<std::vector<int64_t>> value_shape = {{-1}};
  op.SetAttr("value_shape1", value_shape);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
} 

