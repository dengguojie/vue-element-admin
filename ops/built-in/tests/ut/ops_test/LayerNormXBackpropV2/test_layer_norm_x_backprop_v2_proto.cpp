#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class layer_norm_x_backprop_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "layer_norm_x_backprop_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "layer_norm_x_backprop_v2 TearDown" << std::endl;
  }
};

TEST_F(layer_norm_x_backprop_v2, layer_norm_x_backprop_v2_infershape_diff_test_1) {
  ge::op::LayerNormXBackpropV2 op;
  op.UpdateInputDesc("dy", create_desc({16,512,512}, ge::DT_FLOAT));
  op.UpdateInputDesc("x", create_desc({16,512,512}, ge::DT_FLOAT));
  op.UpdateInputDesc("variance", create_desc({16, 512,1}, ge::DT_FLOAT));
  op.UpdateInputDesc("mean", create_desc({16, 512, 1}, ge::DT_FLOAT));
  op.UpdateInputDesc("gamma", create_desc({512}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("pd_x");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {16, 512, 512};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

