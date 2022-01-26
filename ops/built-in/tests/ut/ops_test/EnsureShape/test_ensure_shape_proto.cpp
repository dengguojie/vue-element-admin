#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "array_ops.h"
class EnsureShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "EnsureShape test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "EnsureShape test TearDown" << std::endl;
    }
};

TEST_F(EnsureShapeTest, ensure_shape_test) {
    ge::op::EnsureShape ensure_shape_op;
    ge::TensorDesc tensorDesc;
    ge::Shape shape({2, 3, 4});
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetShape(shape);
    tensorDesc.SetOriginShape(shape);

    ensure_shape_op.UpdateInputDesc("input", tensorDesc);


    auto ret = ensure_shape_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = ensure_shape_op.GetOutputDesc("output");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}