#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "split_combination_ops.h"

class SplitvdTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitvdTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitvdTest TearDown" << std::endl;
  }
};

TEST_F(SplitvdTest, splitvd_test_infershape_diff_test_1) {
  ge::op::SplitVD op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({17, 32, 1, 1});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  op.UpdateInputDesc("x", tensorDesc);
  op.SetAttr("split_dim", 2);
  op.SetAttr("num_split", 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
