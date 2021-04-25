#include <gtest/gtest.h>

#include <iostream>

#include "lookup_ops.h"
#include "op_proto_test_util.h"

class InitializeTableTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "InitializeTableTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "InitializeTableTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(InitializeTableTest, InitializeTableTest_handle_error) {
  ge::op::InitializeTable op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(InitializeTableTest, InitializeTableTest_key_error) {
  ge::op::InitializeTable op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("keys", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("values", create_desc({}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(InitializeTableTest, InitializeTableTest_value_error) {
  ge::op::InitializeTable op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("keys", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("values", create_desc({}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(InitializeTableTest, InitializeTableTest_success) {
  ge::op::InitializeTable op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("keys", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("values", create_desc({1}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
