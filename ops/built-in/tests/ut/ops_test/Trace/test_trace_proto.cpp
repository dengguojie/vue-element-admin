#include <vector>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class TraceTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "trace test setup" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "trace test teardown" <<std::endl;
    }
};

TEST_F(TraceTest, trace_test_case_1)
{
    ge::op::Trace trace_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    trace_op.UpdateInputDesc("x", tensorDesc);

    auto ret = trace_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = trace_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1,};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(TraceTest, trace_test_case_2)
{
    ge::op::Trace trace_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({123, 456});
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    trace_op.UpdateInputDesc("x", tensorDesc);

    auto ret = trace_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = trace_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {1,};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(TraceTest, trace_test_case_3)
{
    ge::op::Trace trace_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({123, 456, 789});
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    trace_op.UpdateInputDesc("x", tensorDesc);

    auto ret = trace_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(TraceTest, trace_test_case_4)
{
    ge::op::Trace trace_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({123, 456});
    tensorDesc.SetDataType(ge::DT_INT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    trace_op.UpdateInputDesc("x", tensorDesc);

    auto ret = trace_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}