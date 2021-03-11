#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class bitwise_xor : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "bitwise_xor SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "bitwise_xor TearDown" << std::endl;
  }
};

TEST_F(bitwise_xor, bitwise_xor_infershape_diff_test){
  ge::op::BitwiseXor op;
  auto tensor_desc_x = create_desc_shape_range({-1,8,375},
                                              ge::DT_INT16, ge::FORMAT_ND,
                                              {16,8,375},
                                              ge::FORMAT_ND, {{15, 16},{8,8},{375,375}});
  op.UpdateInputDesc("x1", tensor_desc_x);
  op.UpdateInputDesc("x2", tensor_desc_x);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_y1_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_INT16);
  std::vector<int64_t> expected_output_shape = {-1, 8, 375};
  EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{15, 16},{8,8},{375,375}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(bitwise_xor, bitwise_xor_infershape_same_test){
  ge::op::BitwiseXor op;
  op.UpdateInputDesc("x1", create_desc({1, 3, 4}, ge::DT_INT16));  
  op.UpdateInputDesc("x2", create_desc({1, 3, 4}, ge::DT_INT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT16);
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
