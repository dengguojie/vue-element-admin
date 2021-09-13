#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "rnn.h"

// ----------------EuclideanNorm--------------
class basic_lstm_cell : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "basic_lstm_cell SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "basic_lstm_cell TearDown" << std::endl;
    }
};

TEST_F(basic_lstm_cell, basic_lstm_cell_case) {
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::BasicLSTMCell op;

    ge::TensorDesc XDesc;
    ge::Shape xShape({batch, inputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);

    ge::TensorDesc HDesc;
    ge::Shape hShape({batch, outputSize});
    HDesc.SetDataType(ge::DT_FLOAT16);
    HDesc.SetShape(hShape);
    HDesc.SetOriginShape(hShape);

    ge::TensorDesc CDesc;
    ge::Shape cShape({batch, outputSize});
    CDesc.SetDataType(ge::DT_FLOAT16);
    CDesc.SetShape(cShape);
    CDesc.SetOriginShape(cShape);

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
    op.UpdateInputDesc("x", XDesc);
    op.UpdateInputDesc("h", HDesc);
    op.UpdateInputDesc("c", CDesc);
    op.UpdateInputDesc("w", WDesc);
    op.UpdateInputDesc("b", BDesc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("ct");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {batch, outputSize};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
