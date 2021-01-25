#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"

class angle : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "angle SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "angle TearDown" << std::endl;
  }
};

TEST_F(angle, angle_infershape_diff_test) {
  ge::op::Angle op;
  op.UpdateInputDesc("input", create_desc({4}, ge::DT_FLOAT16));
  op.SetAttr("Tout", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
