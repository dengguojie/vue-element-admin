#include <gtest/gtest.h>
#include <vector>
#include "math_ops.h"
#include "op_proto_test_util.h"

class TruncTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "trunc test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "trunc test TearDown" << std::endl;
    }
};

TEST_F(TruncTest, trunc_test_case_1) {
    // [TODO] define your op here
    ge::op::Trunc trunc_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    // [TODO] update op input here
    trunc_op.UpdateInputDesc("input_x", tensorDesc);

    // [TODO] call InferShapeAndType function here
    auto ret = trunc_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = trunc_op.GetOutputDesc("output_y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
