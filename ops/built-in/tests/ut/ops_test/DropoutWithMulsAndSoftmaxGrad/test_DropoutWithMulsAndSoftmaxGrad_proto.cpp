#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include"nn_norm_ops.h"

class DropoutWithMulsAndSoftmaxGradTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DropoutWithMulsAndSoftmaxGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DropoutWithMulsAndSoftmaxGrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(DropoutWithMulsAndSoftmaxGradTest, drop_out_with_muls_and_softmax_grad_test_1) {
    ge::op::DropoutWithMulsAndSoftmaxGrad drop_out_with_muls_and_softmax_grad_op;
    ge::TensorDesc tensor_y_grad_desc;
    ge::TensorDesc tensor_mask_desc;
    ge::TensorDesc tensor_softmax_grad_desc;
    ge::Shape y_grad_shape({96,12,24,24,16,16});
    tensor_y_grad_desc.SetDataType(ge::DT_FLOAT16);
    tensor_y_grad_desc.SetShape(y_grad_shape);
    tensor_y_grad_desc.SetFormat(ge::FORMAT_FRACTAL_NZ);

    tensor_mask_desc.SetDataType(ge::DT_UINT8);
    tensor_mask_desc.SetShape(y_grad_shape);
    tensor_mask_desc.SetFormat(ge::FORMAT_FRACTAL_NZ);

    tensor_softmax_grad_desc.SetDataType(ge::DT_FLOAT16);
    tensor_softmax_grad_desc.SetShape(y_grad_shape);
    tensor_softmax_grad_desc.SetFormat(ge::FORMAT_FRACTAL_NZ);

    // update attr
    drop_out_with_muls_and_softmax_grad_op.SetAttr("axes", {-1});
    drop_out_with_muls_and_softmax_grad_op.SetAttr("keep_prob", 0.1f);
    drop_out_with_muls_and_softmax_grad_op.SetAttr("alpha", 0.5f);
    // update input
    drop_out_with_muls_and_softmax_grad_op.UpdateInputDesc("y_grad", tensor_y_grad_desc);
    drop_out_with_muls_and_softmax_grad_op.UpdateInputDesc("mask", tensor_mask_desc);
    drop_out_with_muls_and_softmax_grad_op.UpdateInputDesc("softmax_grad", tensor_softmax_grad_desc);
    // infer
    auto ret = drop_out_with_muls_and_softmax_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
