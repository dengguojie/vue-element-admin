#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"


class RotatedBoxDecodeProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RotatedBoxDecode Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RotatedBoxDecode Proto Test TearDown" << std::endl;
  }
};


TEST_F(RotatedBoxDecodeProtoTest, rotated_box_decode_0) {
    ge::op::RotatedBoxDecode op;

    int B = 2;
    int N = 8;

    ge::TensorDesc anchor_desc;
    ge::Shape anchor_shape({B, 5, N});
    anchor_desc.SetDataType(ge::DT_FLOAT);
    anchor_desc.SetShape(anchor_shape);
    anchor_desc.SetOriginShape(anchor_shape);
    anchor_desc.SetFormat(ge::FORMAT_ND);
    
    ge::TensorDesc gt_desc;
    ge::Shape gt_shape({B, 5, N});
    gt_desc.SetDataType(ge::DT_FLOAT);
    gt_desc.SetShape(gt_shape);
    gt_desc.SetOriginShape(gt_shape);
    gt_desc.SetFormat(ge::FORMAT_ND);

    op.UpdateInputDesc("anchor_box", anchor_desc);
    op.UpdateInputDesc("deltas", gt_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_output_shape = {B, 5, N};
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_desc.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RotatedBoxDecodeProtoTest, rotated_box_decode_1) {
    ge::op::RotatedBoxDecode op;

    int B = 2;
    int N = 12345;

    ge::TensorDesc anchor_desc;
    ge::Shape anchor_shape({B, 5, N});
    anchor_desc.SetDataType(ge::DT_FLOAT);
    anchor_desc.SetShape(anchor_shape);
    anchor_desc.SetOriginShape(anchor_shape);
    anchor_desc.SetFormat(ge::FORMAT_ND);
    
    ge::TensorDesc gt_desc;
    ge::Shape gt_shape({B, 5, N});
    gt_desc.SetDataType(ge::DT_FLOAT);
    gt_desc.SetShape(gt_shape);
    gt_desc.SetOriginShape(gt_shape);
    gt_desc.SetFormat(ge::FORMAT_ND);

    op.UpdateInputDesc("anchor_box", anchor_desc);
    op.UpdateInputDesc("deltas", gt_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_output_shape = {B, 5, N};
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_desc.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RotatedBoxDecodeProtoTest, rotated_box_decode_2) {
    ge::op::RotatedBoxDecode op;

    int B = 2;
    int N = 16;

    ge::TensorDesc anchor_desc;
    ge::Shape anchor_shape({B, 5, N});
    anchor_desc.SetDataType(ge::DT_FLOAT16);
    anchor_desc.SetShape(anchor_shape);
    anchor_desc.SetOriginShape(anchor_shape);
    anchor_desc.SetFormat(ge::FORMAT_ND);
    
    ge::TensorDesc gt_desc;
    ge::Shape gt_shape({B, 5, N});
    gt_desc.SetDataType(ge::DT_FLOAT16);
    gt_desc.SetShape(gt_shape);
    gt_desc.SetOriginShape(gt_shape);
    gt_desc.SetFormat(ge::FORMAT_ND);

    op.UpdateInputDesc("anchor_box", anchor_desc);
    op.UpdateInputDesc("deltas", gt_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_output_shape = {B, 5, N};
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(output_desc.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RotatedBoxDecodeProtoTest, rotated_box_decode_3) {
    ge::op::RotatedBoxDecode op;

    int B = 2;
    int N = 12345;

    ge::TensorDesc anchor_desc;
    ge::Shape anchor_shape({B, 5, N});
    anchor_desc.SetDataType(ge::DT_FLOAT16);
    anchor_desc.SetShape(anchor_shape);
    anchor_desc.SetOriginShape(anchor_shape);
    anchor_desc.SetFormat(ge::FORMAT_ND);
    
    ge::TensorDesc gt_desc;
    ge::Shape gt_shape({B, 5, N});
    gt_desc.SetDataType(ge::DT_FLOAT16);
    gt_desc.SetShape(gt_shape);
    gt_desc.SetOriginShape(gt_shape);
    gt_desc.SetFormat(ge::FORMAT_ND);

    op.UpdateInputDesc("anchor_box", anchor_desc);
    op.UpdateInputDesc("deltas", gt_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_output_shape = {B, 5, N};
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(output_desc.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}