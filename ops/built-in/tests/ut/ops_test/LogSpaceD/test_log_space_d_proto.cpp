#include <gtest/gtest.h>
#include <vector>
#include "selection_ops.h"
#include "op_proto_test_util.h"

class LogSpaceDTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "log_space_d test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "log_space_d test TearDown" << std::endl;
    }
};

TEST_F(LogSpaceDTest, log_space_d_test_case_1) {
    ge::op::LogSpaceD log_space_d_op;

    std::int64_t attr_value = 0;
    log_space_d_op.SetAttr("dtype", attr_value);

    ge::TensorDesc tensorDesc;
    ge::Shape shape({2});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    log_space_d_op.UpdateInputDesc("assist", tensorDesc);

    auto ret = log_space_d_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = log_space_d_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LogSpaceDTest, log_space_d_test_case_2) {
    ge::op::LogSpaceD log_space_d_op;

    std::int64_t attr_value = 1;
    log_space_d_op.SetAttr("dtype", attr_value);

    ge::TensorDesc tensorDesc;
    ge::Shape shape({2});
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    log_space_d_op.UpdateInputDesc("assist", tensorDesc);

    auto ret = log_space_d_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = log_space_d_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
