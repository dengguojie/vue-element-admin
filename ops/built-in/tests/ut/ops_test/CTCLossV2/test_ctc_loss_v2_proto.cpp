#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "ctc_ops.h"

// ----------------CTCLossV2-------------------
class CTCLossV2ProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CTCLossV2 Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CTCLossV2 Proto Test TearDown" << std::endl;
  }
};


TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test_1) {

    int N = 32;
    int T = 944;
    int C = 29;
    int S = 285;
    ge::op::CTCLossV2 rnn_op;
    
    ge::TensorDesc log_probs_desc;
    ge::Shape xShape({T, N, C});
    log_probs_desc.SetDataType(ge::DT_FLOAT);
    log_probs_desc.SetShape(xShape);
    log_probs_desc.SetOriginShape(xShape);
    
    ge::TensorDesc targets_desc;
    ge::Shape YShape({N, S});
    targets_desc.SetDataType(ge::DT_INT32);
    targets_desc.SetShape(YShape);
    targets_desc.SetOriginShape(YShape);
    
    ge::TensorDesc input_lengths_desc;
    ge::Shape ZShape({N});
    input_lengths_desc.SetDataType(ge::DT_INT32);
    input_lengths_desc.SetShape(ZShape);
    input_lengths_desc.SetOriginShape(ZShape);

    ge::TensorDesc target_lengths_desc;
    target_lengths_desc.SetDataType(ge::DT_INT32);
    target_lengths_desc.SetShape(ZShape);
    target_lengths_desc.SetOriginShape(ZShape);

    rnn_op.UpdateInputDesc("log_probs", log_probs_desc);
    rnn_op.UpdateInputDesc("targets", targets_desc);
    rnn_op.UpdateInputDesc("input_lengths", input_lengths_desc);
    rnn_op.UpdateInputDesc("target_lengths", target_lengths_desc);
    

    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc_1 = rnn_op.GetOutputDesc("log_alpha");
    EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {N, T, 2 * S + 1};
    EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);
    
    auto output_desc_2 = rnn_op.GetOutputDesc("neg_log_likelihood");
    EXPECT_EQ(output_desc_2.GetDataType(), ge::DT_FLOAT);
    expected_output_shape = {N};
    EXPECT_EQ(output_desc_2.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test_2) {

    int N = 256;
    int T = 195;
    int C = 41;
    int S = 74;
    ge::op::CTCLossV2 rnn_op;
    
    ge::TensorDesc log_probs_desc;
    ge::Shape xShape({T, N, C});
    log_probs_desc.SetDataType(ge::DT_FLOAT);
    log_probs_desc.SetShape(xShape);
    log_probs_desc.SetOriginShape(xShape);
    
    ge::TensorDesc targets_desc;
    ge::Shape YShape({N, S});
    targets_desc.SetDataType(ge::DT_INT32);
    targets_desc.SetShape(YShape);
    targets_desc.SetOriginShape(YShape);
    
    ge::TensorDesc input_lengths_desc;
    ge::Shape ZShape({N});
    input_lengths_desc.SetDataType(ge::DT_INT32);
    input_lengths_desc.SetShape(ZShape);
    input_lengths_desc.SetOriginShape(ZShape);

    ge::TensorDesc target_lengths_desc;
    target_lengths_desc.SetDataType(ge::DT_INT32);
    target_lengths_desc.SetShape(ZShape);
    target_lengths_desc.SetOriginShape(ZShape);

    rnn_op.UpdateInputDesc("log_probs", log_probs_desc);
    rnn_op.UpdateInputDesc("targets", targets_desc);
    rnn_op.UpdateInputDesc("input_lengths", input_lengths_desc);
    rnn_op.UpdateInputDesc("target_lengths", target_lengths_desc);
    

    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc_1 = rnn_op.GetOutputDesc("log_alpha");
    EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {N, T, 2 * S + 1};
    EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);
    
    auto output_desc_2 = rnn_op.GetOutputDesc("neg_log_likelihood");
    EXPECT_EQ(output_desc_2.GetDataType(), ge::DT_FLOAT);
    expected_output_shape = {N};
    EXPECT_EQ(output_desc_2.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_infer_shape_test_3) {

    int N = 2560;
    int T = 26;
    int C = 37;
    int S = 18;
    ge::op::CTCLossV2 rnn_op;
    
    ge::TensorDesc log_probs_desc;
    ge::Shape xShape({T, N, C});
    log_probs_desc.SetDataType(ge::DT_FLOAT);
    log_probs_desc.SetShape(xShape);
    log_probs_desc.SetOriginShape(xShape);
    
    ge::TensorDesc targets_desc;
    ge::Shape YShape({N, S});
    targets_desc.SetDataType(ge::DT_INT32);
    targets_desc.SetShape(YShape);
    targets_desc.SetOriginShape(YShape);
    
    ge::TensorDesc input_lengths_desc;
    ge::Shape ZShape({N});
    input_lengths_desc.SetDataType(ge::DT_INT32);
    input_lengths_desc.SetShape(ZShape);
    input_lengths_desc.SetOriginShape(ZShape);

    ge::TensorDesc target_lengths_desc;
    target_lengths_desc.SetDataType(ge::DT_INT32);
    target_lengths_desc.SetShape(ZShape);
    target_lengths_desc.SetOriginShape(ZShape);

    rnn_op.UpdateInputDesc("log_probs", log_probs_desc);
    rnn_op.UpdateInputDesc("targets", targets_desc);
    rnn_op.UpdateInputDesc("input_lengths", input_lengths_desc);
    rnn_op.UpdateInputDesc("target_lengths", target_lengths_desc);
    

    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc_1 = rnn_op.GetOutputDesc("log_alpha");
    EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {N, T, 2 * S + 1};
    EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);
    
    auto output_desc_2 = rnn_op.GetOutputDesc("neg_log_likelihood");
    EXPECT_EQ(output_desc_2.GetDataType(), ge::DT_FLOAT);
    expected_output_shape = {N};
    EXPECT_EQ(output_desc_2.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CTCLossV2ProtoTest, ctc_loss_v2_grad_infer_shape_test_4) {

    int N = 2560;
    int T = 26;
    int C = 37;
    int S = 18;
    ge::op::CTCLossV2 rnn_op;
    
    ge::TensorDesc log_probs_desc;
    ge::Shape xShape({T, N, C});
    log_probs_desc.SetDataType(ge::DT_FLOAT);
    log_probs_desc.SetShape(xShape);
    log_probs_desc.SetOriginShape(xShape);
    
    ge::TensorDesc targets_desc;
    ge::Shape YShape({N * S});
    targets_desc.SetDataType(ge::DT_INT32);
    targets_desc.SetShape(YShape);
    targets_desc.SetOriginShape(YShape);
    
    ge::TensorDesc input_lengths_desc;
    ge::Shape ZShape({N});
    input_lengths_desc.SetDataType(ge::DT_INT32);
    input_lengths_desc.SetShape(ZShape);
    input_lengths_desc.SetOriginShape(ZShape);

    ge::TensorDesc target_lengths_desc;
    target_lengths_desc.SetDataType(ge::DT_INT32);
    target_lengths_desc.SetShape(ZShape);
    target_lengths_desc.SetOriginShape(ZShape);

    rnn_op.SetAttr("blank", 0);

    rnn_op.UpdateInputDesc("log_probs", log_probs_desc);
    rnn_op.UpdateInputDesc("targets", targets_desc);
    rnn_op.UpdateInputDesc("input_lengths", input_lengths_desc);
    rnn_op.UpdateInputDesc("target_lengths", target_lengths_desc);

    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
