#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class RnnGenMaskTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "RnnGenMaskTest Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "RnnGenMaskTest Proto Test TearDown" << std::endl;
    }
};

TEST_F(RnnGenMaskTest, rnn_gen_mask_tsest_1) {
    ge::op::RnnGenMask rnn_gen_mask_op;
    ge::TensorDesc tensor_x_desc;
    ge::Shape x_shape({3});
    tensor_x_desc.SetDataType(ge::DT_INT32);
    tensor_x_desc.SetShape(x_shape);
    tensor_x_desc.SetOriginShape(x_shape);
    // update attr
    rnn_gen_mask_op.SetAttr("num_step", 2);
    rnn_gen_mask_op.SetAttr("hidden_size", 4);
    // update input
    rnn_gen_mask_op.UpdateInputDesc("seq_length", tensor_x_desc);
    // infer
    auto ret = rnn_gen_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // compare
    auto output_desc = rnn_gen_mask_op.GetOutputDesc("seq_mask");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RnnGenMaskTest, rnn_gen_mask_tsest_2) {
    ge::op::RnnGenMask rnn_gen_mask_op;
    ge::TensorDesc tensor_x_desc;
    ge::Shape x_shape({4});
    tensor_x_desc.SetDataType(ge::DT_INT32);
    tensor_x_desc.SetShape(x_shape);
    tensor_x_desc.SetOriginShape(x_shape);
    // update attr
    rnn_gen_mask_op.SetAttr("num_step", 12);
    rnn_gen_mask_op.SetAttr("hidden_size", 4);
    // update input
    rnn_gen_mask_op.UpdateInputDesc("seq_length", tensor_x_desc);
    // infer
    auto ret = rnn_gen_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // compare
    auto output_desc = rnn_gen_mask_op.GetOutputDesc("seq_mask");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {12, 4, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RnnGenMaskTest, rnn_gen_mask_tsest_3) {
    ge::op::RnnGenMask rnn_gen_mask_op;
    ge::TensorDesc tensor_x_desc;
    ge::Shape x_shape({4});
    tensor_x_desc.SetDataType(ge::DT_INT32);
    tensor_x_desc.SetShape(x_shape);
    tensor_x_desc.SetOriginShape(x_shape);
    // update attr
    rnn_gen_mask_op.SetAttr("num_step", 12);
    rnn_gen_mask_op.SetAttr("hidden_size", 34);
    // update input
    rnn_gen_mask_op.UpdateInputDesc("seq_length", tensor_x_desc);
    // infer
    auto ret = rnn_gen_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // compare
    auto output_desc = rnn_gen_mask_op.GetOutputDesc("seq_mask");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {12, 4, 34};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RnnGenMaskTest, rnn_gen_mask_tsest_4) {
    ge::op::RnnGenMask rnn_gen_mask_op;
    ge::TensorDesc tensor_x_desc;
    ge::Shape x_shape({4});
    tensor_x_desc.SetDataType(ge::DT_INT32);
    tensor_x_desc.SetShape(x_shape);
    tensor_x_desc.SetOriginShape(x_shape);
    // update attr
    rnn_gen_mask_op.SetAttr("num_step", 12);
    rnn_gen_mask_op.SetAttr("hidden_size", 16);
    // update input
    rnn_gen_mask_op.UpdateInputDesc("seq_length", tensor_x_desc);
    // infer
    auto ret = rnn_gen_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // compare
    auto output_desc = rnn_gen_mask_op.GetOutputDesc("seq_mask");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {12, 4, 16};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RnnGenMaskTest, rnn_gen_mask_tsest_5) {
    ge::op::RnnGenMask rnn_gen_mask_op;
    ge::TensorDesc tensor_x_desc;
    ge::Shape x_shape({4, 5});
    tensor_x_desc.SetDataType(ge::DT_INT32);
    tensor_x_desc.SetShape(x_shape);
    // update attr
    rnn_gen_mask_op.SetAttr("num_step", 12);
    rnn_gen_mask_op.SetAttr("hidden_size", 16);
    // update input
    rnn_gen_mask_op.UpdateInputDesc("seq_length", tensor_x_desc);
    // infer
    auto ret = rnn_gen_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RnnGenMaskTest, rnn_gen_mask_tsest_6) {
    ge::op::RnnGenMask rnn_gen_mask_op;
    ge::TensorDesc tensor_x_desc;
    ge::Shape x_shape({4});
    tensor_x_desc.SetDataType(ge::DT_INT32);
    tensor_x_desc.SetShape(x_shape);
    // update attr
    rnn_gen_mask_op.SetAttr("hidden_size", 16);
    // update input
    rnn_gen_mask_op.UpdateInputDesc("seq_length", tensor_x_desc);
    // infer
    auto ret = rnn_gen_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RnnGenMaskTest, rnn_gen_mask_tsest_7) {
    ge::op::RnnGenMask rnn_gen_mask_op;
    ge::TensorDesc tensor_x_desc;
    ge::Shape x_shape({4});
    tensor_x_desc.SetDataType(ge::DT_INT32);
    tensor_x_desc.SetShape(x_shape);
    // update attr
    rnn_gen_mask_op.SetAttr("num_step", 12);
    // update input
    rnn_gen_mask_op.UpdateInputDesc("seq_length", tensor_x_desc);
    // infer
    auto ret = rnn_gen_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}