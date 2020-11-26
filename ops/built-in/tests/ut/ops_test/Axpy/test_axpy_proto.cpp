#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class AxpyProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Axpy Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Axpy Proto Test TearDown" << std::endl;
  }
};

TEST_F(AxpyProtoTest, axpy_infershape_diff_test){
  ge::op::Axpy op;
  op.UpdateInputDesc("x1", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x2", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  
  float alpha = 1.0;
  op.SetAttr("alpha", alpha);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(AxpyProtoTest, axpy_infershape_same_test){
  ge::op::Axpy op;
  op.UpdateInputDesc("x1", create_desc({1, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x2", create_desc({4, 3, 4}, ge::DT_FLOAT16));

  float alpha = 1.0;
  op.SetAttr("alpha", alpha);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

