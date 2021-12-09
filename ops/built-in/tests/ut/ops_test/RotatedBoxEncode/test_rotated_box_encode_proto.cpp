#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"


class RotatedBoxEncodeProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RotatedBoxEncode Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RotatedBoxEncode Proto Test TearDown" << std::endl;
  }
};


TEST_F(RotatedBoxEncodeProtoTest, rotated_box_encode_0) {
    ge::op::RotatedBoxEncode op;

    int B = 2;
    int N = 944;

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
    op.UpdateInputDesc("gt_box", gt_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_output_shape = {B, 5, N};
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_desc.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}