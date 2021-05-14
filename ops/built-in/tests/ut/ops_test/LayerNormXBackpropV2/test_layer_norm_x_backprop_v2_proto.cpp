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
  op.UpdateInputDesc("dy", create_desc_shape_range({32,64}, ge::DT_FLOAT, ge::FORMAT_ND, {32,64}, ge::FORMAT_ND, {{32,32},{64,64}}));
  op.UpdateInputDesc("x", create_desc_shape_range({32,64}, ge::DT_FLOAT, ge::FORMAT_ND, {32,64}, ge::FORMAT_ND, {{32,32},{64,64}}));
  op.UpdateInputDesc("variance", create_desc_shape_range({32,1}, ge::DT_FLOAT, ge::FORMAT_ND, {32,1}, ge::FORMAT_ND, {{32,32},{1,1}}));
  op.UpdateInputDesc("mean", create_desc_shape_range({32,1}, ge::DT_FLOAT, ge::FORMAT_ND, {32,1}, ge::FORMAT_ND, {{32,32},{1,1}}));
  op.UpdateInputDesc("gamma", create_desc_shape_range({64}, ge::DT_FLOAT, ge::FORMAT_ND, {64}, ge::FORMAT_ND, {{64,64}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("pd_x");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {32,64};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

