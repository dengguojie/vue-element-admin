#include <gtest/gtest.h>
#include <vector>
#include "rnn.h"

class DynamicRnnTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_rnn test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "dynamic_rnn test TearDown" << std::endl;
    }
};

TEST_F(DynamicRnnTest, dynamic_rnn_test_case_1) {
    int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::DynamicRNN rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);

    ge::TensorDesc WDesc;
    ge::Shape wShape({inputSize+outputSize, 4*outputSize});
    WDesc.SetDataType(ge::DT_FLOAT16);
    WDesc.SetShape(wShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("w", WDesc);
    rnn_op.UpdateInputDesc("b", BDesc);

    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = rnn_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {t,batch, outputSize};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DynamicRnnTest, dynamic_rnn_test_case_2) {
    int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::DynamicRNN rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize,outputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);

    ge::TensorDesc WDesc;
    ge::Shape wShape({inputSize+outputSize, 4*outputSize});
    WDesc.SetDataType(ge::DT_FLOAT16);
    WDesc.SetShape(wShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("w", WDesc);
    rnn_op.UpdateInputDesc("b", BDesc);

    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynamicRnnTest, dynamic_rnn_test_case_3) {
    int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::DynamicRNN rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize,outputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);

    ge::TensorDesc WDesc;
    ge::Shape wShape({inputSize+outputSize, 4*outputSize});
    WDesc.SetDataType(ge::DT_FLOAT16);
    WDesc.SetShape(wShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);
	
    ge::TensorDesc SDesc;
    ge::Shape sShape({batch});
    SDesc.SetDataType(ge::DT_INT32);
    SDesc.SetShape(bShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("w", WDesc);
    rnn_op.UpdateInputDesc("b", BDesc);
	rnn_op.UpdateInputDesc("seq_length", SDesc);
	
    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
	
	auto output_desc = rnn_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {t,batch, outputSize};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DynamicRnnTest, dynamic_rnn_test_case_4) {
    int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::DynamicRNN rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize,outputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);

    ge::TensorDesc WDesc;
    ge::Shape wShape({inputSize+outputSize, 4*outputSize});
    WDesc.SetDataType(ge::DT_FLOAT16);
    WDesc.SetShape(wShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);
	
    ge::TensorDesc SDesc;
    ge::Shape sShape({2 * batch});
    SDesc.SetDataType(ge::DT_INT32);
    SDesc.SetShape(bShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("w", WDesc);
    rnn_op.UpdateInputDesc("b", BDesc);
	rnn_op.UpdateInputDesc("seq_length", SDesc);
	
    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
	 
}

TEST_F(DynamicRnnTest, dynamic_rnn_test_case_3) {
        int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::DynamicRNN rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);

    ge::TensorDesc WDesc;
    ge::Shape wShape({inputSize+outputSize, 4*outputSize});
    WDesc.SetDataType(ge::DT_FLOAT16);
    WDesc.SetShape(wShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("w", WDesc);
    rnn_op.UpdateInputDesc("b", BDesc);
    rnn_op.SetAttr("direction", "REDIRECTIONAL");

    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = rnn_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {t,batch, outputSize};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
