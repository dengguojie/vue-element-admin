#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "quantize_ops.h"

class dequantize : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dequantize SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dequantize TearDown" << std::endl;
    }
};

TEST_F(dequantize, dequantize_case) {
    ge::op::Dequantize op;

    op.UpdateInputDesc("x", create_desc({11, 33}, ge::DT_INT8));
    op.UpdateInputDesc("min_range", create_desc({1,}, ge::DT_FLOAT));
    op.UpdateInputDesc("min_range", create_desc({1,}, ge::DT_FLOAT));
    std::string mode = "SCALED";
    op.SetAttr("mode", mode);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {11,33};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}