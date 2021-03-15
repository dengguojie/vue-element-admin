#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "parsing_ops.h"

class decoderaw : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "decoderaw SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "decoderaw TearDown" << std::endl;
  }
};

TEST_F(decoderaw, decoderaw_infershape_diff_test){
  ge::op::DecodeRaw op;
  op.UpdateInputDesc("bytes", create_desc({1}, ge::DT_STRING));
  op.SetAttr("out_type", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}