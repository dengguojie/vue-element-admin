#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "split_combination_ops.h"
#include "array_ops.h"


class SplitDTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitDTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitDTest TearDown" << std::endl;
  }
};

TEST_F(SplitDTest, splitD_test_infershape_diff_test_1) {
  ge::op::SplitD op;

  op.UpdateInputDesc("x", create_desc_shape_range({-1, 32, 128}, ge::DT_INT32, ge::FORMAT_ND, {2, 32, 128}, ge::FORMAT_ND,{{1,100},{32,32},{128,128}}));
  op.SetAttr("split_dim", 1);
  op.SetAttr("num_split", 2);

  op.InferShapeAndType();
}
