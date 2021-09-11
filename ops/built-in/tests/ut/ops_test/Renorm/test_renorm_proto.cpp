#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class RenormTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "renorm test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "renorm test TearDown" << std::endl;
    }
};

TEST_F(RenormTest, renorm_test_case_1) {
    ge::op::Renorm renorm_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({3, 3});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);
    float p_value = 2.0;
    renorm_op.SetAttr("p", p_value);
    int dim_value = 0;
    renorm_op.SetAttr("dim", p_value);
    float maxnorm_value = 3.0;
    renorm_op.SetAttr("maxnorm", p_value);

    renorm_op.UpdateInputDesc("x", tensorDesc);

    auto ret = renorm_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = renorm_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RenormTest, renorm_test_case_2) {
    ge::op::Renorm renorm_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({3, 3, 3});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);
    float p_value = 2.0;
    renorm_op.SetAttr("p", p_value);
    int dim_value = 0;
    renorm_op.SetAttr("dim", p_value);
    float maxnorm_value = 3.0;
    renorm_op.SetAttr("maxnorm", p_value);

    renorm_op.UpdateInputDesc("x", tensorDesc);

    auto ret = renorm_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = renorm_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 1, 1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RenormTest, renorm_test_case_3) {
    ge::op::Renorm renorm_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({3, 3, 3});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);
    float p_value = 1.0;
    renorm_op.SetAttr("p", p_value);
    int dim_value = 0;
    renorm_op.SetAttr("dim", p_value);
    float maxnorm_value = 3.0;
    renorm_op.SetAttr("maxnorm", p_value);

    renorm_op.UpdateInputDesc("x", tensorDesc);

    auto ret = renorm_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = renorm_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 1, 1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}