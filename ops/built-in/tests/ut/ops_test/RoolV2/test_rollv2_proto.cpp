#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"
#include "../util/common_shape_fns.h"

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

TEST_F(RollV2Test, rollv2_test_input0_err) {
    ge::op::RollV2 rollV2_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape;
    (void)Scalar(shape);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    rollV2_op.UpdateInputDesc("input", tensorDesc);

    auto ret = rollV2_op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RollV2Test, rollv2_test_input1_err) {
    ge::op::RollV2 rollV2_op;

    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    rollV2_op.UpdateInputDesc("input", tensorDesc);

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2, 3, 4});
    tensorDesc1.SetDataType(ge::DT_INT32);
    tensorDesc1.SetShape(shape1);
    tensorDesc1.SetOriginShape(shape1);

    rollV2_op.UpdateInputDesc("shift", tensorDesc1);

    auto ret = rollV2_op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RollV2Test, rollv2_test_input2_err) {
    ge::op::RollV2 rollV2_op;

    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    rollV2_op.UpdateInputDesc("input", tensorDesc);

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({2});
    tensorDesc1.SetDataType(ge::DT_INT32);
    tensorDesc1.SetShape(shape1);
    tensorDesc1.SetOriginShape(shape1);
    rollV2_op.UpdateInputDesc("shift", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({2, 3, 4});
    tensorDesc2.SetDataType(ge::DT_INT32);
    tensorDesc2.SetShape(shape2);
    tensorDesc2.SetOriginShape(shape2);

    rollV2_op.UpdateInputDesc("axes", tensorDesc2);

    auto ret = rollV2_op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}