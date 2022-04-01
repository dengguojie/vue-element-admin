#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class celuv2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "celuv2 Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "celuv2 Proto Test TearDown" << std::endl;
  }
};

TEST_F(celuv2, celu_infershape_diff_test){
  ge::op::CeluV2 op;
  op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(celuv2, celu_infershape_same_test){
  ge::op::CeluV2 op;
  op.UpdateInputDesc("x", create_desc({1, 3, 4}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}