#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class ReductionProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ReductionProtoTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ReductionProtoTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(ReductionProtoTest, reduction_infershape_test_1) {
  ge::op::Reduction op;

  auto tensor_desc = create_desc_with_ori({1, 3, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 256}, ge::FORMAT_ND);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("axis", 2);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

