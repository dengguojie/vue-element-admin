#include <gtest/gtest.h>
#include <vector>
#include "rnn.h"

class DynamicRNNV3Test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_rnn_v3 test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_rnn_v3 test TearDown" << std::endl;
    }
};

TEST_F(DynamicRNNV3Test, dynamic_rnn_v3_test_case_1) {
    int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    int stateSize = 48;
    ge::op::DynamicRNNV3 rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);

    ge::TensorDesc WDesc;
    ge::Shape wShape({inputSize+outputSize, 4*outputSize});
    WDesc.SetDataType(ge::DT_FLOAT16);
    WDesc.SetShape(wShape);
    WDesc.SetOriginShape(wShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);
    BDesc.SetOriginShape(bShape);

    ge::TensorDesc PDesc;
    ge::Shape pShape({outputSize, stateSize});
    PDesc.SetDataType(ge::DT_FLOAT16);
    PDesc.SetShape(pShape);
    PDesc.SetOriginShape(pShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("w", WDesc);
    rnn_op.UpdateInputDesc("b", BDesc);
    rnn_op.UpdateInputDesc("project", PDesc);

    auto ret = rnn_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}