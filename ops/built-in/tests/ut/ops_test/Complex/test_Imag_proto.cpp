#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"

class imag : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "imag SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "imag TearDown" << std::endl;
  }
};

TEST_F(imag, imag_infershape_diff_test){
  ge::op::Imag op;
  op.UpdateInputDesc("input", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.SetAttr("Tout", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}