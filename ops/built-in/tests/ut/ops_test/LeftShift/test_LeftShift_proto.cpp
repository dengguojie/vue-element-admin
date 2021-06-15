#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "bitwise_ops.h"
using namespace ge;
using namespace op;

class LeftShiftTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LeftShift test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LeftShift test TearDown" << std::endl;
  }
};

TEST_F(LeftShiftTest, left_shift_infershape_test_1) {
  ge::op::LeftShift op;
  op.UpdateInputDesc("x", create_desc({16, 4, 1}, ge::DT_UINT32));
  op.UpdateInputDesc("y", create_desc({1, 1, 1}, ge::DT_UINT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("z");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT32);
  std::vector<int64_t> expected_output_shape = {16, 4, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LeftShiftTest, left_shift_infershape_test_2) {
  ge::op::LeftShift op;
  op.UpdateInputDesc("x", create_desc({16, 4, 1}, ge::DT_UINT16));
  op.UpdateInputDesc("y", create_desc({1, 1, 1}, ge::DT_UINT16));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("z");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT16);
  std::vector<int64_t> expected_output_shape = {16, 4, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LeftShiftTest, left_shift_verify_test) {
  ge::op::LeftShift op;
  op.UpdateInputDesc("x", create_desc({16, 4, 1}, ge::DT_UINT16));
  op.UpdateInputDesc("y", create_desc({1, 1, 1}, ge::DT_UINT16));
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}