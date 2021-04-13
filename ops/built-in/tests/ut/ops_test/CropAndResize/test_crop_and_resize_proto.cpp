#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"

class CropAndResize : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CropAndResize SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CropAndResize TearDown" << std::endl;
  }
};

TEST_F(CropAndResize, CropAndResize_infershape_test01){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2,-1, -1, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CropAndResize, CropAndResize_infershape_test02){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_test03){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_test04){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {}, ge::DT_INT32, ge::FORMAT_ND,
      {}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_test05){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {1}, ge::DT_INT32, ge::FORMAT_ND,
      {1}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_test06){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {2, 3, 4, 2}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
      {2, 3, 4, 2}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {2, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {2, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2, 2}, ge::DT_INT32, ge::FORMAT_ND,
      {2, 2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CropAndResize, CropAndResize_infershape_test07){
  ge::op::CropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori(
      {3, 2, 3, 2}, ge::DT_FLOAT16, ge::FORMAT_HWCN,
      {3, 2, 3, 2}, ge::FORMAT_HWCN));
  op.UpdateInputDesc("boxes", create_desc_with_ori(
      {3, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
      {3, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori(
      {3}, ge::DT_INT32, ge::FORMAT_ND,
      {3}, ge::FORMAT_ND));
  op.UpdateInputDesc("crop_size", create_desc_with_ori(
      {2}, ge::DT_INT32, ge::FORMAT_ND,
      {2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}