#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include"nn_norm_ops.h"

class SoftmaxV2WithDropOutDoMaskV3DTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SoftmaxV2WithDropOutDoMaskV3D Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SoftmaxV2WithDropOutDoMaskV3D Proto Test TearDown" << std::endl;
  }
};

TEST_F(SoftmaxV2WithDropOutDoMaskV3DTest, softmax_with_dropout_do_mask_v3_test_1) {
    ge::op::SoftmaxV2WithDropOutDoMaskV3D softmax_with_dropout_do_mask_v3_op;
    ge::TensorDesc tensor_x_desc;
    ge::TensorDesc tensor_mask_desc;
    ge::Shape x_shape({4,16,32,32,16,16});
    tensor_x_desc.SetDataType(ge::DT_FLOAT16);
    tensor_x_desc.SetShape(x_shape);
    tensor_x_desc.SetFormat(ge::FORMAT_FRACTAL_NZ);

    tensor_mask_desc.SetDataType(ge::DT_UINT8);
    tensor_mask_desc.SetShape(x_shape);
    tensor_mask_desc.SetFormat(ge::FORMAT_FRACTAL_NZ);

    // update attr
    softmax_with_dropout_do_mask_v3_op.SetAttr("axes", {-1});
    softmax_with_dropout_do_mask_v3_op.SetAttr("keep_prob", 0.1f);
    // update input
    softmax_with_dropout_do_mask_v3_op.UpdateInputDesc("x", tensor_x_desc);
    softmax_with_dropout_do_mask_v3_op.UpdateInputDesc("mask", tensor_mask_desc);
    // infer
    auto ret = softmax_with_dropout_do_mask_v3_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}