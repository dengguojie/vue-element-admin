#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class fresnel_sin : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "fresnel_sin Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fresnel_sin Proto Test TearDown" << std::endl;
  }
};

TEST_F(fresnel_sin, fresnel_sin_infershape_f16_diff_test){
  ge::op::FresnelSin op;
  op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(fresnel_sin, fresnel_sin_infershape_f16_same_test){
  ge::op::FresnelSin op;
  op.UpdateInputDesc("x", create_desc({1, 3, 4}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(fresnel_sin, fresnel_sin_infershape_f32_diff_test){
  ge::op::FresnelSin op;
  op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(fresnel_sin, fresnel_sin_infershape_f32_same_test){
  ge::op::FresnelSin op;
  op.UpdateInputDesc("x", create_desc({1, 3, 4}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}