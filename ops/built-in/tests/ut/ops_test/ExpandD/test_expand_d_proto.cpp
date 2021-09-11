#include <gtest/gtest.h>
#include <vector>
#include "array_ops.h"

class ExpandDTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "expand_d test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "expand_d test TearDown" << std::endl;
    }
};

TEST_F(ExpandDTest, expand_d_test_case_1) {
    ge::op::ExpandD expand_d_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);

    expand_d_op.UpdateInputDesc("x", tensorDesc);
    expand_d_op.SetAttr("shape", {2, 2, 3, 4});

    auto ret = expand_d_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = expand_d_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ExpandDTest, expand_d_test_case_2) {
    ge::op::ExpandD expand_d_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 4, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    expand_d_op.UpdateInputDesc("x", tensorDesc);
    expand_d_op.SetAttr("shape", {2, 2, 3, 4});

    auto ret = expand_d_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ExpandDTest, expand_test_case_1) {
    ge::op::Expand expand_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape xShape({2, 2});
    tensorDesc1.SetDataType(ge::DT_INT32);
    tensorDesc1.SetShape(xShape);
    expand_op.UpdateInputDesc("x", tensorDesc1);

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
    constDesc.SetSize(1 * sizeof(int32_t));
    constTensor.SetTensorDesc(constDesc);
    int64_t constData[1] = {1};
    constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
    auto shape = ge::op::Constant().set_attr_value(constTensor);

    expand_op.set_input_shape(shape);
    auto desc = expand_op.GetInputDesc("shape");
    desc.SetDataType(ge::DT_INT32);
    expand_op.UpdateInputDesc("shape", desc);

    auto ret = expand_op.InferShapeAndType();
}

TEST_F(ExpandDTest, expand_test_case_2) {
    ge::op::Expand expand_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape xShape({2, 2});
    tensorDesc1.SetDataType(ge::DT_INT32);
    tensorDesc1.SetShape(xShape);
    expand_op.UpdateInputDesc("x", tensorDesc1);

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT64);
    constDesc.SetSize(1 * sizeof(int64_t));
    constTensor.SetTensorDesc(constDesc);
    int64_t constData[1] = {1};
    constTensor.SetData((uint8_t*)constData, 1 * sizeof(int64_t));
    auto shape = ge::op::Constant().set_attr_value(constTensor);

    expand_op.set_input_shape(shape);
    auto desc = expand_op.GetInputDesc("shape");
    desc.SetDataType(ge::DT_INT64);
    expand_op.UpdateInputDesc("shape", desc);

    auto ret = expand_op.InferShapeAndType();
}