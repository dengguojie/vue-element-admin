#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "quantize_ops.h"

class AscendWeightQuantTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ascendweightquant test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "ascendweightquant test TearDown" << std::endl;
    }
};

TEST_F(AscendWeightQuantTest, ascendweightquant_test_case_1) {
    ge::op::AscendWeightQuant ascendweightquant_op;
    ge::TensorDesc tensor_input;
    ge::Shape shape({3, 4, 16, 32});
    tensor_input.SetDataType(ge::DT_INT8);
    tensor_input.SetShape(shape);

    ge::TensorDesc tensor_offset;
    ge::Shape shape2({3, 4, 16, 32});
    tensor_offset.SetDataType(ge::DT_INT8);
    tensor_offset.SetShape(shape2);

    ascendweightquant_op.UpdateInputDesc("x", tensor_input);
    ascendweightquant_op.UpdateInputDesc("offset", tensor_offset);
    ascendweightquant_op.SetAttr("dst_type", ge::DT_INT8);
    auto ret = ascendweightquant_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = ascendweightquant_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT8);
}

TEST_F(AscendWeightQuantTest, ascendweightquant_test_case_2) {
    ge::op::AscendWeightQuant ascendweightquant_op;
    ge::TensorDesc tensor_input;
    ge::Shape shape({3, 4, 16, 32});
    tensor_input.SetDataType(ge::DT_INT8);
    tensor_input.SetShape(shape);

    ge::TensorDesc tensor_offset;
    ge::Shape shape2({3, 4, 16, 32});
    tensor_offset.SetDataType(ge::DT_INT8);
    tensor_offset.SetShape(shape2);

    ascendweightquant_op.UpdateInputDesc("x", tensor_input);
    ascendweightquant_op.UpdateInputDesc("offset", tensor_offset);
    ascendweightquant_op.SetAttr("dst_type", ge::DT_INT8);

    auto ret = ascendweightquant_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = ascendweightquant_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT8);
}

TEST_F(AscendWeightQuantTest, ascendweightquant_test_case_3) {
    ge::op::AscendWeightQuant ascendweightquant_op;
    ascendweightquant_op.UpdateInputDesc("x",create_desc_with_ori(
        {3, 4, 16, 32}, ge::DT_INT8, ge::FORMAT_FRACTAL_Z,
        {3, 4, 16, 32}, ge::FORMAT_FRACTAL_Z));

    ascendweightquant_op.UpdateInputDesc("offset",create_desc_with_ori(
        {3, 4, 16, 32}, ge::DT_INT8, ge::FORMAT_FRACTAL_Z,
        {3, 4, 16, 32}, ge::FORMAT_FRACTAL_Z));
    
    ascendweightquant_op.UpdateOutputDesc("y",create_desc_with_ori(
        {3, 4, 16, 32}, ge::DT_INT8, ge::FORMAT_FRACTAL_Z,
        {3, 4, 16, 32}, ge::FORMAT_FRACTAL_Z));

    ascendweightquant_op.SetAttr("dst_type", ge::DT_INT8);

    auto ret = ascendweightquant_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

