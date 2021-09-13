#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "math_ops.h"
#include "op_proto_test_util.h"

class PdistTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "pdist test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "pdist test TearDown" << std::endl;
    }
};

TEST_F(PdistTest, pdist_test_case_1){
    ge::op::Pdist op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({10, 100});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);
    op.UpdateInputDesc("x", tensorDesc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {45};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
