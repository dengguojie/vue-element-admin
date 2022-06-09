#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "math_ops.h"


class SignBitsPackProtoTest : public testing::Test {
  protected:
    static void SetUpTestCase() {
        std::cout << "SignBitsPack Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "SignBitsPack Proto Test TearDown" << std::endl;
    }
};

TEST_F(SignBitsPackProtoTest, sign_bits_pack_0) {
    ge::op::SignBitsPack op;
    ge::TensorDesc x_desc;
    ge::Shape x_shape({2448 * 8});
    x_desc.SetDataType(ge::DT_FLOAT);
    x_desc.SetShape(x_shape);
    x_desc.SetOriginShape(x_shape);
    x_desc.SetFormat(ge::FORMAT_ND);

    op.UpdateInputDesc("x", x_desc);

    op.SetAttr("size", 2);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_output_shape = {2, 1224};
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT8);
    EXPECT_EQ(output_desc.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SignBitsPackProtoTest, sign_bits_pack_1) {
    ge::op::SignBitsPack op;
    ge::TensorDesc x_desc;
    ge::Shape x_shape({-1});
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, -1}};
    x_desc.SetDataType(ge::DT_FLOAT16);
    x_desc.SetShape(x_shape);
    x_desc.SetOriginShape(x_shape);
    x_desc.SetShapeRange(shape_range);
    x_desc.SetFormat(ge::FORMAT_ND);

    op.UpdateInputDesc("x", x_desc);
    
    op.SetAttr("size", 1);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_output_shape = {1, -1};
    std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1, 1}, {1, -1}};
    auto output_desc = op.GetOutputDesc("y");

    std::vector<std::pair<int64_t,int64_t>> output_shape_range;
    output_desc.GetShapeRange(output_shape_range);

    EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT8);
    EXPECT_EQ(output_desc.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_shape_range, expected_shape_range);
}