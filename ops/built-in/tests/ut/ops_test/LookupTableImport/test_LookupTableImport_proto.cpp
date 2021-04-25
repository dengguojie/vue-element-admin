#include <gtest/gtest.h>

#include <iostream>

#include "lookup_ops.h"
#include "op_proto_test_util.h"

class LookupTableImportTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LookupTableImportTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LookupTableImportTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(LookupTableImportTest, LookupTableImportTest_handle_error) {
  ge::op::LookupTableImport op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(LookupTableImportTest, LookupTableImportTest_keys_error) {
  ge::op::LookupTableImport op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("keys", create_desc({}, ge::DT_FLOAT16));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(LookupTableImportTest, LookupTableImportTest_key_values_merge_error) {
  ge::op::LookupTableImport op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("keys", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("values", create_desc({1, 2}, ge::DT_FLOAT16));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(LookupTableImportTest, LookupTableImportTest_success) {
  ge::op::LookupTableImport op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("keys", create_desc({1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("values", create_desc({1}, ge::DT_FLOAT16));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
