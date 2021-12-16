#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_detect_ops.h"

class NonMaxSuppressionV7Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NonMaxSuppressionV7 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NonMaxSuppressionV7 TearDown" << std::endl;
  }
};

TEST_F(NonMaxSuppressionV7Test, non_max_suppression_v7_test_case_1) {
  ge::op::NonMaxSuppressionV7 op;
  op.UpdateInputDesc("boxes",
                     create_desc_with_ori({2, 6, 4}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 6, 4}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("scores",
                     create_desc_with_ori({2, 1, 6}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2, 1, 6}, ge::FORMAT_NCHW));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}