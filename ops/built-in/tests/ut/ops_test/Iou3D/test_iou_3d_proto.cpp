#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"


class Iou3DProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Iou3D Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Iou3D Proto Test TearDown" << std::endl;
  }
};


TEST_F(Iou3DProtoTest, iou_3d_proto_test_with_shape_32_7_944_32_7_29) {
    ge::op::Iou3D op;

    int B = 32;
    int N = 944;
    int K = 29;

    ge::TensorDesc boxes_desc;
    ge::Shape xShape({B, 7, N});
    boxes_desc.SetDataType(ge::DT_FLOAT);
    boxes_desc.SetShape(xShape);
    boxes_desc.SetOriginShape(xShape);

    ge::TensorDesc query_boxes_desc;
    ge::Shape YShape({B, 7, K});
    query_boxes_desc.SetDataType(ge::DT_FLOAT);
    query_boxes_desc.SetShape(YShape);
    query_boxes_desc.SetOriginShape(YShape);

    op.UpdateInputDesc("bboxes", boxes_desc);
    op.UpdateInputDesc("gtboxes", query_boxes_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("iou");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {B, N, K};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(Iou3DProtoTest, iou_3d_proto_test_with_shape_1_7_944_1_7_29) {
    ge::op::Iou3D op;

    int B = 1;
    int N = 944;
    int K = 29;

    ge::TensorDesc boxes_desc;
    ge::Shape xShape({B, 7, N});
    boxes_desc.SetDataType(ge::DT_FLOAT);
    boxes_desc.SetShape(xShape);
    boxes_desc.SetOriginShape(xShape);

    ge::TensorDesc query_boxes_desc;
    ge::Shape YShape({B, 7, K});
    query_boxes_desc.SetDataType(ge::DT_FLOAT);
    query_boxes_desc.SetShape(YShape);
    query_boxes_desc.SetOriginShape(YShape);

    op.UpdateInputDesc("bboxes", boxes_desc);
    op.UpdateInputDesc("gtboxes", query_boxes_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("iou");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {B, N, K};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(Iou3DProtoTest, iou_3d_proto_test_failed_shape_mulit_batch) {
    ge::op::Iou3D op;

    int B = 32;
    int N = 944;
    int K = 29;

    ge::TensorDesc boxes_desc;
    ge::Shape xShape({B, 7, N});
    boxes_desc.SetDataType(ge::DT_FLOAT);
    boxes_desc.SetShape(xShape);
    boxes_desc.SetOriginShape(xShape);

    ge::TensorDesc query_boxes_desc;
    ge::Shape YShape({2 * B, 7, K});
    query_boxes_desc.SetDataType(ge::DT_FLOAT);
    query_boxes_desc.SetShape(YShape);
    query_boxes_desc.SetOriginShape(YShape);

    op.UpdateInputDesc("bboxes", boxes_desc);
    op.UpdateInputDesc("gtboxes", query_boxes_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(Iou3DProtoTest, iou_3d_proto_test_failed_shape_single_batch) {
    ge::op::Iou3D op;

    int B = 1;
    int N = 944;
    int K = 29;

    ge::TensorDesc boxes_desc;
    ge::Shape xShape({B, 7, N});
    boxes_desc.SetDataType(ge::DT_FLOAT);
    boxes_desc.SetShape(xShape);
    boxes_desc.SetOriginShape(xShape);

    ge::TensorDesc query_boxes_desc;
    ge::Shape YShape({2 * B, 7, K});
    query_boxes_desc.SetDataType(ge::DT_FLOAT);
    query_boxes_desc.SetShape(YShape);
    query_boxes_desc.SetOriginShape(YShape);

    op.UpdateInputDesc("bboxes", boxes_desc);
    op.UpdateInputDesc("gtboxes", query_boxes_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
