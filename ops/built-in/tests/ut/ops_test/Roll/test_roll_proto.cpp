#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"
class RollTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "roll test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "roll test TearDown" << std::endl;
    }
};

TEST_F(RollTest, roll_test_case_1) {
    ge::op::Roll roll_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    roll_op.UpdateInputDesc("x", tensorDesc);
    roll_op.SetAttr("dims", {});
    roll_op.SetAttr("shifts", {1});

    auto ret = roll_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = roll_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RollTest, roll_test_case_2) {
    ge::op::Roll roll_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    roll_op.UpdateInputDesc("x", tensorDesc);
    roll_op.SetAttr("dims", {});
    roll_op.SetAttr("shifts", {1});

    auto ret = roll_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = roll_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RollTest, roll_test_case_3) {
    ge::op::Roll roll_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_INT32);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    roll_op.UpdateInputDesc("x", tensorDesc);
    roll_op.SetAttr("dims", {});
    roll_op.SetAttr("shifts", {1});

    auto ret = roll_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = roll_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RollTest, roll_test_case_4) {
    ge::op::Roll roll_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_UINT32);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    roll_op.UpdateInputDesc("x", tensorDesc);
    roll_op.SetAttr("dims", {});
    roll_op.SetAttr("shifts", {1});

    auto ret = roll_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = roll_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT32);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RollTest, roll_test_case_5) {
    ge::op::Roll roll_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_INT8);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    roll_op.UpdateInputDesc("x", tensorDesc);
    roll_op.SetAttr("dims", {});
    roll_op.SetAttr("shifts", {1});

    auto ret = roll_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = roll_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT8);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RollTest, roll_test_case_6) {
    ge::op::Roll roll_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_UINT8);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    roll_op.UpdateInputDesc("x", tensorDesc);
    roll_op.SetAttr("dims", {});
    roll_op.SetAttr("shifts", {1});

    auto ret = roll_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = roll_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT8);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RollTest, rollv2_test) {
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