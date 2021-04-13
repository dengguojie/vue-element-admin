#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"

class CropAndResizeGradImage : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CropAndResizeGradImage SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CropAndResizeGradImage TearDown" << std::endl;
  }
};

TEST_F(CropAndResizeGradImage, CropAndResizeGradImage_infershape_test01){
  ge::op::CropAndResizeGradImage op;
  op.UpdateInputDesc("grads", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("image_size", create_desc_with_ori(
      {4}, ge::DT_INT32, ge::FORMAT_ND,
      {4}, ge::FORMAT_ND));
  op.SetAttr("T", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1,-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CropAndResizeGradImage, CropAndResizeGradImage_infershape_test02){
  ge::op::CropAndResizeGradImage op;
  op.UpdateInputDesc("grads", create_desc_with_ori(
      {2, 3, 4}, ge::DT_FLOAT, ge::FORMAT_NHWC,
      {2, 3, 4}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("image_size", create_desc_with_ori(
      {4}, ge::DT_INT32, ge::FORMAT_ND,
      {4}, ge::FORMAT_ND));
  op.SetAttr("T", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResizeGradImage, CropAndResizeGradImage_infershape_test03){
  ge::op::CropAndResizeGradImage op;
  op.UpdateInputDesc("grads", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("image_size", create_desc_with_ori(
      {4}, ge::DT_INT32, ge::FORMAT_ND,
      {4}, ge::FORMAT_ND));
  op.SetAttr("T", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResizeGradImage, CropAndResizeGradImage_infershape_test04){
  ge::op::CropAndResizeGradImage op;
  op.UpdateInputDesc("grads", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("image_size", create_desc_with_ori(
      {2, 2}, ge::DT_INT32, ge::FORMAT_ND,
      {2, 2}, ge::FORMAT_ND));
  op.SetAttr("T", ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResizeGradImage, CropAndResizeGradImage_infershape_test05){
  ge::op::CropAndResizeGradImage op;
  op.UpdateInputDesc("grads", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("image_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}