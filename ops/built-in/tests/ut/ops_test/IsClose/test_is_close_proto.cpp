#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"


class IsCloseTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "is_close test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "is_close test TearDown" << std::endl;
    }
};

TEST_F(IsCloseTest, is_close_test_case_1) {
    // define your op here
    ge::op::IsClose is_close_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    // update op input here
    is_close_op.UpdateInputDesc("x1", tensorDesc);
    is_close_op.UpdateInputDesc("x2", tensorDesc);

    // call InferShapeAndType function here
    auto ret = is_close_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // compare dtype and shape of op output
    auto output_desc = is_close_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(IsCloseTest, is_close_test_case_2) {
    // define your op here
    ge::op::IsClose is_close_op;
    ge::TensorDesc XtensorDesc;
    ge::TensorDesc YtensorDesc;
    ge::Shape Xshape({32});
    ge::Shape Yshape({32,32});
    XtensorDesc.SetDataType(ge::DT_FLOAT16);
    XtensorDesc.SetShape(Xshape);
    XtensorDesc.SetOriginShape(Xshape);
    YtensorDesc.SetDataType(ge::DT_FLOAT16);
    YtensorDesc.SetShape(Yshape);
    YtensorDesc.SetOriginShape(Yshape);

    // update op input here
    is_close_op.UpdateInputDesc("x1", XtensorDesc);
    is_close_op.UpdateInputDesc("x2", YtensorDesc);

    // call InferShapeAndType function here
    auto ret = is_close_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // compare dtype and shape of op output
    auto output_desc = is_close_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_BOOL);
    std::vector<int64_t> expected_output_shape = {32,32};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
