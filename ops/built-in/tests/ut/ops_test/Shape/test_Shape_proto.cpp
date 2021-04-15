#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"

class shape : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "shape Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "shape Proto Test TearDown" << std::endl;
  }
};

TEST_F(shape, shape_infershape_diff_test){
  ge::op::Shape op;
  op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_INT32));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {3};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
