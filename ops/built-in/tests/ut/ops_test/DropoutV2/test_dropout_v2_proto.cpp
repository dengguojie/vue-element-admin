#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"

class DropoutV2Test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dropout_v2 test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dropout_v2 test TearDown" << std::endl;
    }
};

TEST_F(DropoutV2Test, dropout_v2_test_case_1) {
    ge::op::DropoutV2 dropout_v2_op;
    ge:: TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);
    dropout_v2_op.UpdateInputDesc("x", tensorDesc);

    tensorDesc.SetDataType(ge::DT_FLOAT);
    dropout_v2_op.UpdateInputDesc("seed", tensorDesc);

    auto ret = dropout_v2_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = dropout_v2_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    auto mask_desc = dropout_v2_op.GetOutputDesc("mask");
    EXPECT_EQ(mask_desc.GetDataType(), ge::DT_FLOAT);
    expected_output_shape = {2, 3, 4};
    EXPECT_EQ(mask_desc.GetShape().GetDims(), expected_output_shape);

    auto seed_desc = dropout_v2_op.GetOutputDesc("seed");
    EXPECT_EQ(seed_desc.GetDataType(), ge::DT_FLOAT);
    expected_output_shape = {2, 3, 4};
    EXPECT_EQ(seed_desc.GetShape().GetDims(), expected_output_shape);
}

