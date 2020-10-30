#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class fills : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "fills Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fills Proto Test TearDown" << std::endl;
  }
};

TEST_F(fills, fills_infershape_diff_test){
  ge::op::Fills op;
  op.UpdateInputDesc("x", create_desc({3, 30, 10, 16, 17}, ge::DT_FLOAT16));
  
  float value = 3.0;
  op.SetAttr("value", value);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {3, 30, 10, 16, 17};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(fills, fills_infershape_same_test){
  ge::op::Fills op;
  op.UpdateInputDesc("x", create_desc({3, 30, 10, 16, 17}, ge::DT_FLOAT16));

  float value = 3.0;
  op.SetAttr("value", value);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {3, 30, 10, 16, 17};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

