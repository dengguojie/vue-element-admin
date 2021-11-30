#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class mod : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "mod SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "mod TearDown" << std::endl;
  }
};

TEST_F(mod, mod_infershape_diff_test){
  ge::op::Mod op;
  op.UpdateInputDesc("x1", create_desc({4, 3, 4}, ge::DT_INT8));
  op.UpdateInputDesc("x2", create_desc({4, 3, 4}, ge::DT_INT8));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT8);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(mod, mod_infershape_same_test){
  ge::op::Mod op;
  op.UpdateInputDesc("x1", create_desc({1, 3, 4}, ge::DT_INT8));  
  op.UpdateInputDesc("x2", create_desc({1, 3, 4}, ge::DT_INT8));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT8);
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
