#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class GroupNormRelu : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GroupNormRelu Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GroupNormRelu Proto Test TearDown" << std::endl;
  }
};

TEST_F(GroupNormRelu, group_norm_relu_infershape_test_1){
  ge::op::GroupNormRelu op;
  op.UpdateInputDesc("x", create_desc({8, 16, 15, 15}, ge::DT_FLOAT16));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {8, 16, 15, 15};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
