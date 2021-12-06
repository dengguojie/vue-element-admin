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

TEST_F(ReductionProtoTest, reduction_infershape_test_2) {
  ge::op::Reduction op;
  auto tensor_desc = create_desc_with_ori({1, 3, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 256}, ge::FORMAT_ND);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("axis", false);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ReductionProtoTest, reduction_infershape_test_3) {
  ge::op::Reduction op;
  auto tensor_desc = create_desc_with_ori({1, 3, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 256}, ge::FORMAT_ND);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("axis", 10);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ReductionProtoTest, reduction_infershape_test_4) {
  ge::op::Reduction op;

  auto tensor_desc = create_desc_with_ori({1, 3, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 256}, ge::FORMAT_ND);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("axis", -2);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(ReductionProtoTest, reduction_infershape_test_5) {
  ge::op::Reduction op;
  auto tensor_desc = create_desc_with_ori({1, 3, 256}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 3, 256}, ge::FORMAT_ND);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("axis", 0);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
