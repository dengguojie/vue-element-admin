#include <gtest/gtest.h>
#include <vector>
#include "rnn.h"

class DynamicRnnV2Test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_rnn_v2 test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "dynamic_rnn_v2 test TearDown" << std::endl;
    }
};

TEST_F(DynamicRnnV2Test, dynamic_rnn_test_case_1) {
    int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::DynamicRNNV2 rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);

    ge::TensorDesc WiDesc;
    ge::TensorDesc WhDesc;
    ge::Shape wiShape({inputSize, 4*outputSize});
    ge::Shape whShape({outputSize, 4*outputSize});
    WiDesc.SetDataType(ge::DT_FLOAT16);
    WhDesc.SetDataType(ge::DT_FLOAT16);
    WiDesc.SetShape(wiShape);
    WhDesc.SetShape(whShape);
    WiDesc.SetOriginShape(wiShape);
    WhDesc.SetOriginShape(whShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);
    BDesc.SetOriginShape(bShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("weight_input", WiDesc);
    rnn_op.UpdateInputDesc("weight_hidden", WhDesc);
    rnn_op.UpdateInputDesc("b", BDesc);

    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = rnn_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {t,batch, outputSize};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}