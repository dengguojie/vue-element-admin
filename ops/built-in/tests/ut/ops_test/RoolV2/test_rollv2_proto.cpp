#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"
class RollV2Test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "rollv2 test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "rollv2 test TearDown" << std::endl;
    }
};

TEST_F(RollV2Test, rollv2_test) {
    ge::op::RollV2 rollV2_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    rollV2_op.UpdateInputDesc("input", tensorDesc);


    auto ret = rollV2_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = rollV2_op.GetOutputDesc("output");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}