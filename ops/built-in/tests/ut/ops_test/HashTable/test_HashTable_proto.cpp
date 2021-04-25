#include <gtest/gtest.h>

#include <iostream>

#include "lookup_ops.h"
#include "op_proto_test_util.h"

class HashTableTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "HashTableTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HashTableTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(HashTableTest, HashTableTest_key_dtype_error) {
  ge::op::HashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(HashTableTest, HashTableTest_value_dtype_error) {
  ge::op::HashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.SetAttr("key_dtype", ge::DT_FLOAT);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(HashTableTest, HashTableTest_key_value_error1) {
  ge::op::HashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.SetAttr("key_dtype", ge::DT_INT64);
  op.SetAttr("value_dtype", ge::DT_UINT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(HashTableTest, HashTableTest_key_value_error2) {
  ge::op::HashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.SetAttr("key_dtype", ge::DT_INT32);
  op.SetAttr("value_dtype", ge::DT_UINT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(HashTableTest, HashTableTest_key_value_error3) {
  ge::op::HashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.SetAttr("key_dtype", ge::DT_STRING);
  op.SetAttr("value_dtype", ge::DT_UINT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(HashTableTest, HashTableTest_success) {
  ge::op::HashTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));
  op.SetAttr("key_dtype", ge::DT_STRING);
  op.SetAttr("value_dtype", ge::DT_INT64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
