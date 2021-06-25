#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class ApplyAdamV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyAdamV2 Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyAdamV2 Proto Test TearDown" << std::endl;
  }
};

TEST_F(ApplyAdamV2, apply_adam_v2_infershape_test_1){
  ge::op::ApplyAdamV2 op;
  op.UpdateInputDesc("var", create_desc({3, 10, 1024}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expected_output_shape = {3, 10, 1024};

  auto output_desc_0 = op.GetOutputDesc("var");
  EXPECT_EQ(output_desc_0.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc_0.GetShape().GetDims(), expected_output_shape);

  auto output_desc_1 = op.GetOutputDesc("m");
  EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);

  auto output_desc_2 = op.GetOutputDesc("v");
  EXPECT_EQ(output_desc_2.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc_2.GetShape().GetDims(), expected_output_shape);
}
