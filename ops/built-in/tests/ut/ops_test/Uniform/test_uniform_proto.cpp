#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"

class UniformProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Uniform Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Uniform Proto Test TearDown" << std::endl;
  }
};

TEST_F(UniformProtoTest, uniform_infershape_test_default_attr){
  ge::op::Uniform op;
  op.UpdateInputDesc("x", create_desc_with_ori({4,}, ge::DT_FLOAT, ge::FORMAT_ND, {4,}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {4, };
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(UniformProtoTest, uniform_infershape_test){
  ge::op::Uniform op;
  op.UpdateInputDesc("x", create_desc_with_ori({4,}, ge::DT_FLOAT, ge::FORMAT_ND, {4,}, ge::FORMAT_ND));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  op.SetAttr("from", 1);
  op.SetAttr("to", 2);
  
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {4, };
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

