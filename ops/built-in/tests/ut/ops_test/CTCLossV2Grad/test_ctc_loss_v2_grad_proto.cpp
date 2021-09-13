#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "ctc_ops.h"

// ----------------CTCLossV2Grad-------------------
class CTCLossV2GradProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CTCLossV2Grad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CTCLossV2Grad Proto Test TearDown" << std::endl;
  }
};

TEST_F(CTCLossV2GradProtoTest, ctc_loss_v2_grad_infer_shape_test_1) {

    int N = 32;
    int T = 944;
    int C = 29;
    int S = 285;

    ge::op::CTCLossV2Grad rnn_op;
    
    ge::Shape WShape({N, T, 2 * S + 1});
    ge::Shape XShape({T, N, C});
    ge::Shape YShape({N, S});
    ge::Shape ZShape({N});

    ge::TensorDesc grad_out_desc;
    grad_out_desc.SetDataType(ge::DT_FLOAT);
    grad_out_desc.SetShape(ZShape);
    grad_out_desc.SetOriginShape(ZShape);
    ge::TensorDesc log_probs_desc;
    log_probs_desc.SetDataType(ge::DT_FLOAT);
    log_probs_desc.SetShape(XShape);
    log_probs_desc.SetOriginShape(XShape);
    ge::TensorDesc targets_desc;
    targets_desc.SetDataType(ge::DT_INT32);
    targets_desc.SetShape(YShape);
    targets_desc.SetOriginShape(YShape);
    ge::TensorDesc input_lengths_desc;
    input_lengths_desc.SetDataType(ge::DT_INT32);
    input_lengths_desc.SetShape(ZShape);
    input_lengths_desc.SetOriginShape(ZShape);
    ge::TensorDesc target_lengths_desc;
    target_lengths_desc.SetDataType(ge::DT_INT32);
    target_lengths_desc.SetShape(ZShape);
    target_lengths_desc.SetOriginShape(ZShape);
    ge::TensorDesc neg_log_likelihood_desc;
    neg_log_likelihood_desc.SetDataType(ge::DT_FLOAT);
    neg_log_likelihood_desc.SetShape(ZShape);
    neg_log_likelihood_desc.SetOriginShape(ZShape);
    ge::TensorDesc log_alpha_desc;
    log_alpha_desc.SetDataType(ge::DT_FLOAT);
    log_alpha_desc.SetShape(WShape);
    log_alpha_desc.SetOriginShape(WShape);

    rnn_op.UpdateInputDesc("grad_out", grad_out_desc);
    rnn_op.UpdateInputDesc("log_probs", log_probs_desc);
    rnn_op.UpdateInputDesc("targets", targets_desc);
    rnn_op.UpdateInputDesc("input_lengths", input_lengths_desc);
    rnn_op.UpdateInputDesc("target_lengths", target_lengths_desc);
    rnn_op.UpdateInputDesc("neg_log_likelihood", neg_log_likelihood_desc);
    rnn_op.UpdateInputDesc("log_alpha", log_alpha_desc);
    
    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = rnn_op.GetOutputDesc("grad");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {T, N, C};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CTCLossV2GradProtoTest, ctc_loss_v2_grad_infer_shape_test_2) {

    int N = 2560;
    int T = 26;
    int C = 37;
    int S = 18;

    ge::op::CTCLossV2Grad rnn_op;
    
    ge::Shape WShape({N, T, 2 * S + 1});
    ge::Shape XShape({T, N, C});
    ge::Shape YShape({N, S});
    ge::Shape ZShape({N});
    ge::TensorDesc grad_out_desc;
    grad_out_desc.SetDataType(ge::DT_INT32);
    grad_out_desc.SetShape(ZShape);
    ge::TensorDesc log_probs_desc;
    log_probs_desc.SetDataType(ge::DT_FLOAT);
    log_probs_desc.SetShape(XShape);
    ge::TensorDesc targets_desc;
    targets_desc.SetDataType(ge::DT_INT32);
    targets_desc.SetShape(YShape);
    ge::TensorDesc input_lengths_desc;
    input_lengths_desc.SetDataType(ge::DT_INT32);
    input_lengths_desc.SetShape(ZShape);
    ge::TensorDesc target_lengths_desc;
    target_lengths_desc.SetDataType(ge::DT_INT32);
    target_lengths_desc.SetShape(ZShape);
    ge::TensorDesc neg_log_likelihood_desc;
    neg_log_likelihood_desc.SetDataType(ge::DT_FLOAT);
    neg_log_likelihood_desc.SetShape(ZShape);
    ge::TensorDesc log_alpha_desc;
    log_alpha_desc.SetDataType(ge::DT_FLOAT);
    log_alpha_desc.SetShape(WShape);

    rnn_op.UpdateInputDesc("grad_out", grad_out_desc);
    rnn_op.UpdateInputDesc("log_probs", log_probs_desc);
    rnn_op.UpdateInputDesc("targets", targets_desc);
    rnn_op.UpdateInputDesc("input_lengths", input_lengths_desc);
    rnn_op.UpdateInputDesc("target_lengths", target_lengths_desc);
    rnn_op.UpdateInputDesc("neg_log_likelihood", neg_log_likelihood_desc);
    rnn_op.UpdateInputDesc("log_alpha", log_alpha_desc);
    
    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CTCLossV2GradProtoTest, ctc_loss_v2_grad_infer_shape_test_3) {

    int N = 2560;
    int T = 26;
    int S = 18;

    ge::op::CTCLossV2Grad rnn_op;
    
    ge::Shape WShape({N, T, 2 * S + 1});
    ge::Shape XShape({T, N});
    ge::Shape YShape({N, S});
    ge::Shape ZShape({N});

    ge::TensorDesc grad_out_desc;
    grad_out_desc.SetDataType(ge::DT_FLOAT);
    grad_out_desc.SetShape(ZShape);
    ge::TensorDesc log_probs_desc;
    log_probs_desc.SetDataType(ge::DT_FLOAT);
    log_probs_desc.SetShape(XShape);
    ge::TensorDesc targets_desc;
    targets_desc.SetDataType(ge::DT_INT32);
    targets_desc.SetShape(YShape);
    ge::TensorDesc input_lengths_desc;
    input_lengths_desc.SetDataType(ge::DT_INT32);
    input_lengths_desc.SetShape(ZShape);
    ge::TensorDesc target_lengths_desc;
    target_lengths_desc.SetDataType(ge::DT_INT32);
    target_lengths_desc.SetShape(ZShape);
    ge::TensorDesc neg_log_likelihood_desc;
    neg_log_likelihood_desc.SetDataType(ge::DT_FLOAT);
    neg_log_likelihood_desc.SetShape(ZShape);
    ge::TensorDesc log_alpha_desc;
    log_alpha_desc.SetDataType(ge::DT_FLOAT);
    log_alpha_desc.SetShape(WShape);

    rnn_op.UpdateInputDesc("grad_out", grad_out_desc);
    rnn_op.UpdateInputDesc("log_probs", log_probs_desc);
    rnn_op.UpdateInputDesc("targets", targets_desc);
    rnn_op.UpdateInputDesc("input_lengths", input_lengths_desc);
    rnn_op.UpdateInputDesc("target_lengths", target_lengths_desc);
    rnn_op.UpdateInputDesc("neg_log_likelihood", neg_log_likelihood_desc);
    rnn_op.UpdateInputDesc("log_alpha", log_alpha_desc);
    
    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CTCLossV2GradProtoTest, ctc_loss_v2_grad_infer_shape_test_4) {

    int N = 2560;
    int T = 26;
    int C = 37;
    int S = 18;

    ge::op::CTCLossV2Grad rnn_op;
    
    ge::Shape WShape({N, T, 2 * S + 1});
    ge::Shape XShape({T, N, C});
    ge::Shape YShape({N, S});
    ge::Shape ZShape({N});
    
    ge::TensorDesc grad_out_desc;
    grad_out_desc.SetDataType(ge::DT_FLOAT);
    grad_out_desc.SetShape(ZShape);
    ge::TensorDesc log_probs_desc;
    log_probs_desc.SetDataType(ge::DT_INT32);
    log_probs_desc.SetShape(XShape);
    ge::TensorDesc targets_desc;
    targets_desc.SetDataType(ge::DT_INT32);
    targets_desc.SetShape(YShape);
    ge::TensorDesc input_lengths_desc;
    input_lengths_desc.SetDataType(ge::DT_INT32);
    input_lengths_desc.SetShape(ZShape);
    ge::TensorDesc target_lengths_desc;
    target_lengths_desc.SetDataType(ge::DT_INT32);
    target_lengths_desc.SetShape(ZShape);
    ge::TensorDesc neg_log_likelihood_desc;
    neg_log_likelihood_desc.SetDataType(ge::DT_FLOAT);
    neg_log_likelihood_desc.SetShape(ZShape);
    ge::TensorDesc log_alpha_desc;
    log_alpha_desc.SetDataType(ge::DT_FLOAT);
    log_alpha_desc.SetShape(WShape);

    rnn_op.UpdateInputDesc("grad_out", grad_out_desc);
    rnn_op.UpdateInputDesc("log_probs", log_probs_desc);
    rnn_op.UpdateInputDesc("targets", targets_desc);
    rnn_op.UpdateInputDesc("input_lengths", input_lengths_desc);
    rnn_op.UpdateInputDesc("target_lengths", target_lengths_desc);
    rnn_op.UpdateInputDesc("neg_log_likelihood", neg_log_likelihood_desc);
    rnn_op.UpdateInputDesc("log_alpha", log_alpha_desc);
    
    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

