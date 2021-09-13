#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "quantize_ops.h"

class QuantizeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "quantize test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "quantize test TearDown" << std::endl;
    }
};

TEST_F(QuantizeTest, quantize_test_case_1) {
    ge::op::Quantize quantize_op;
    ge::TensorDesc tensor_input;
    ge::Shape shape({3, 4, 5, 6});
    tensor_input.SetDataType(ge::DT_FLOAT16);
    tensor_input.SetShape(shape);

    ge::TensorDesc tensor_scale;
    ge::Shape shape2({1});
    tensor_input.SetDataType(ge::DT_FLOAT);
    tensor_input.SetShape(shape2);

    ge::TensorDesc tensor_point;
    tensor_point.SetDataType(ge::DT_UINT8);
    tensor_point.SetShape(shape2);

    tensor_input.SetOriginShape(shape2);
    tensor_point.SetOriginShape(shape2);

    quantize_op.UpdateInputDesc("x", tensor_input);
    quantize_op.UpdateInputDesc("scales", tensor_scale);
    quantize_op.UpdateInputDesc("zero_points", tensor_point);
    quantize_op.SetAttr("dtype", "torch.qint8");

    auto ret = quantize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = quantize_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT8);
    std::vector<int64_t> expected_output_shape = {1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(QuantizeTest, quantize_test_case_2) {
    ge::op::Quantize quantize_op;
    ge::TensorDesc tensor_input;
    ge::Shape shape({3, 4, 5, 6});
    tensor_input.SetDataType(ge::DT_FLOAT16);
    tensor_input.SetShape(shape);

    ge::TensorDesc tensor_scale;
    ge::Shape shape2({1});
    tensor_input.SetDataType(ge::DT_FLOAT);
    tensor_input.SetShape(shape2);

    ge::TensorDesc tensor_point;
    tensor_point.SetDataType(ge::DT_UINT8);
    tensor_point.SetShape(shape2);

    tensor_input.SetOriginShape(shape2);
    tensor_point.SetOriginShape(shape2);

    quantize_op.UpdateInputDesc("x", tensor_input);
    quantize_op.UpdateInputDesc("scales", tensor_scale);
    quantize_op.UpdateInputDesc("zero_points", tensor_point);
    quantize_op.SetAttr("dtype", "torch.quint8");

    auto ret = quantize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = quantize_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT8);
    std::vector<int64_t> expected_output_shape = {1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(QuantizeTest, quantize_test_case_3) {
    ge::op::Quantize quantize_op;
    ge::TensorDesc tensor_input;
    ge::Shape shape({3, 4, 5, 6});
    tensor_input.SetDataType(ge::DT_FLOAT16);
    tensor_input.SetShape(shape);

    ge::TensorDesc tensor_scale;
    ge::Shape shape2({1});
    tensor_input.SetDataType(ge::DT_FLOAT);
    tensor_input.SetShape(shape2);

    ge::TensorDesc tensor_point;
    tensor_point.SetDataType(ge::DT_UINT8);
    tensor_point.SetShape(shape2);

    tensor_input.SetOriginShape(shape2);
    tensor_point.SetOriginShape(shape2);

    quantize_op.UpdateInputDesc("x", tensor_input);
    quantize_op.UpdateInputDesc("scales", tensor_scale);
    quantize_op.UpdateInputDesc("zero_points", tensor_point);
    quantize_op.SetAttr("dtype", "torch.qint32");

    auto ret = quantize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = quantize_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
    std::vector<int64_t> expected_output_shape = {1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}