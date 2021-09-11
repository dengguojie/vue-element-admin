#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "math_ops.h"

class CdistTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "cdist test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "cdist test TearDown" << std::endl;
    }
};

TEST_F(CdistTest, cdist_test_case_1) {
    ge::op::Cdist cdist_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({2, 3, 4, 5});
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    cdist_op.UpdateInputDesc("x1", tensor_desc);
    cdist_op.UpdateInputDesc("x2", tensor_desc);

    auto ret = cdist_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = cdist_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
