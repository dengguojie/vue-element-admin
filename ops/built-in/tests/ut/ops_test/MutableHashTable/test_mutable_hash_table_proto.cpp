#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "lookup_ops.h"

class MutableHashTable : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MutableHashTable SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MutableHashTable TearDown" << std::endl;
  }
};

TEST_F(MutableHashTable, MutableHashTable_infer_shape_0) {
  ge::op::MutableHashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MutableHashTable, MutableHashTable_infer_shape_1) {
  ge::op::MutableDenseHashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.SetAttr("key_dtype", ge::DT_INT64);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MutableHashTable, MutableHashTable_infer_shape_3) {
  ge::op::MutableHashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.SetAttr("key_dtype", ge::DT_INT64);
  op.SetAttr("value_dtype", ge::DT_STRING);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MutableHashTable, MutableHashTable_infer_shape_4) {
  ge::op::MutableHashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.SetAttr("key_dtype", ge::DT_INT32);
  op.SetAttr("value_dtype", ge::DT_STRING);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
