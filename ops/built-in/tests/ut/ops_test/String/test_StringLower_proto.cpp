#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "string_ops.h"

class StringLower : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StringLower SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StringLower TearDown" << std::endl;
  }
};

TEST_F(StringLower, StringLower_infershape_diff_test){
  ge::op::StringLower op;
  op.UpdateInputDesc("input", create_desc({1}, ge::DT_STRING));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_STRING);
  std::vector<int64_t> expected_output_shape = {1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}