#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class layer_norm_beta_gamma_backprop_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "layer_norm_beta_gamma_backprop_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "layer_norm_beta_gamma_backprop_v2 TearDown" << std::endl;
  }
};

TEST_F(layer_norm_beta_gamma_backprop_v2, layer_norm_beta_gamma_backprop_v2_infershape_diff_test_1) {
  ge::op::LayerNormBetaGammaBackpropV2 op;
  op.UpdateInputDesc("dy", create_desc_shape_range({32,64}, ge::DT_FLOAT, ge::FORMAT_ND, {32,64}, ge::FORMAT_ND, {{32,32},{64,64}}));
  op.UpdateInputDesc("res_for_gamma", create_desc_shape_range({32,64}, ge::DT_FLOAT, ge::FORMAT_ND, {32,64}, ge::FORMAT_ND, {{32,32},{64,64}}));
  op.SetAttr("shape_gamma", {64});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("pd_gamma");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {64};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

