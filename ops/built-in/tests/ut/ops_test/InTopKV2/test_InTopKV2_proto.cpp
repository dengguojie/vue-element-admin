#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_ops.h"

class intopkv2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "intopkv2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "intopkv2 TearDown" << std::endl;
  }
};

TEST_F(intopkv2, intopkv2_infershape_test) {
  ge::op::InTopKV2 op;
  op.UpdateInputDesc("predictions", create_desc({1,1}, ge::DT_FLOAT));
  op.UpdateInputDesc("targets", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("k",create_desc({1}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("precision");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
  std::vector<int64_t> expected_output_shape = {1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
