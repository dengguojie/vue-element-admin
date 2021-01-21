#include <gtest/gtest.h>
#include <vector>
#include "selection_ops.h"
#include "op_proto_test_util.h"

class IndexFillDdTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "index_fill_d test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "index_fill_d test TearDown" << std::endl;
    }
};

TEST_F(IndexFillDdTest, index_fill_d_test_case_1) {
    // [TODO] define your op here
    ge::op::IndexFillD index_fill_d_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);

    // [TODO] update op input here
    index_fill_d_op.UpdateInputDesc("x", tensorDesc);
    index_fill_d_op.UpdateInputDesc("assist1", tensorDesc);
    index_fill_d_op.UpdateInputDesc("assist2", tensorDesc);

    // [TODO] call InferShapeAndType function here
    auto ret = index_fill_d_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = index_fill_d_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
