#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "math_ops.h"

class CrossTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "cross test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "cross test TearDown" << std::endl;
    }
};


TEST_F(CrossTest, cross_test_case_1) {
    ge::op::Cross cross_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({16, 3, 64});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);
    cross_op.UpdateInputDesc("x1", tensorDesc);
    cross_op.UpdateInputDesc("x2", tensorDesc);

    auto ret = cross_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = cross_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {16, 3, 64};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CrossTest, cross_test_case_2) {
    ge::op::Cross cross_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({4, 3});
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);
    int dim_value = 1;
    cross_op.SetAttr("dim", dim_value);
    cross_op.UpdateInputDesc("x1", tensorDesc);
    cross_op.UpdateInputDesc("x2", tensorDesc);

    auto ret = cross_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = cross_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {4, 3};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CrossTest, cross_test_case_3) {
    ge::op::Cross cross_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 3});
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);
    int dim_value = 1;
    cross_op.SetAttr("dim", dim_value);
    cross_op.UpdateInputDesc("x1", tensorDesc);
    cross_op.UpdateInputDesc("x2", tensorDesc);

    auto ret = cross_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = cross_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2, 3, 3};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CrossTest, cross_test_case_4) {
    ge::op::Cross cross_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({4, 5, 3});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);

    int dim_value = 2;
    cross_op.SetAttr("dim", dim_value);
    cross_op.UpdateInputDesc("x1", tensorDesc);
    cross_op.UpdateInputDesc("x2", tensorDesc);

    auto ret = cross_op.VerifyAllAttr(true);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
