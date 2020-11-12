#include <gtest/gtest.h>
#include <iostream>
#include "math_ops.h"
#include "op_proto_test_util.h"


class ActsULQ : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ActsULQ Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ActsULQ Proto Test TearDown" << std::endl;
    }
};

TEST_F(ActsULQ, test_acts_ulq_float16)
{
    ge::op::ActsULQ op;
    op.UpdateInputDesc("x", create_desc({32, 3, 5, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("clamp_min", create_desc({1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("clamp_max", create_desc({1}, ge::DT_FLOAT16));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output = op.GetOutputDesc("y");
    EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {32, 3, 5, 5};
    EXPECT_EQ(output.GetShape().GetDims(), expected_output_shape);
}
