#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"

class complex : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "complex SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "complex TearDown" << std::endl;
  }
};

TEST_F(complex, complex_infershape_diff_test){
  ge::op::Complex op;
  op.UpdateInputDesc("real", create_desc({4}, ge::DT_FLOAT));
  op.UpdateInputDesc("imag", create_desc({4}, ge::DT_FLOAT));
  op.SetAttr("Tout", ge::DT_COMPLEX64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("out");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_COMPLEX64);
  std::vector<int64_t> expected_output_shape = {4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(complex, complex_infershape_diff_test1){
  ge::op::Complex op;
  op.UpdateInputDesc("real", create_desc({-2}, ge::DT_FLOAT));
  op.UpdateInputDesc("imag", create_desc({3},ge::DT_FLOAT));
  op.SetAttr("Tout", ge::DT_COMPLEX64);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("out");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_COMPLEX64);
}