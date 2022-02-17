#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "image_ops.h"
class imageprojectivetransform : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "imageprojectivetransform SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "imageprojectivetransform TearDown" << std::endl;
  }
};

TEST_F(imageprojectivetransform, imageprojectivetransform_input_images_ok_test){
  ge::op::ImageProjectiveTransform op;
  op.UpdateInputDesc("images", create_desc({1,2,3,4}, ge::DT_UINT8));
  op.UpdateInputDesc("transforms", create_desc({1,8}, ge::DT_FLOAT));
  op.UpdateInputDesc("output_shape", create_desc({2}, ge::DT_INT32));
  op.SetAttr("interpolation",  "NEAREST");
  op.SetAttr("fill_mode",  "CONSTANT");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(imageprojectivetransform, imageprojectivetransform_input_images_ERR_test){
  ge::op::ImageProjectiveTransform op;
  op.UpdateInputDesc("images", create_desc({1,2,3}, ge::DT_UINT8));
  op.UpdateInputDesc("transforms", create_desc({1,8}, ge::DT_FLOAT));
  op.UpdateInputDesc("output_shape", create_desc({2}, ge::DT_INT32));
  op.SetAttr("interpolation",  "NEAREST");
  op.SetAttr("fill_mode",  "CONSTANT");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(imageprojectivetransform, imageprojectivetransform_outputshape_isnot1D_ERR_test){
  ge::op::ImageProjectiveTransform op;
  op.UpdateInputDesc("images", create_desc({1,2,3,4}, ge::DT_UINT8));
  op.UpdateInputDesc("transforms", create_desc({1,8}, ge::DT_FLOAT));
  op.UpdateInputDesc("output_shape", create_desc({2,2}, ge::DT_INT32));
  op.SetAttr("interpolation",  "NEAREST");
  op.SetAttr("fill_mode",  "CONSTANT");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(imageprojectivetransform, imageprojectivetransform_outputshape_dim0isnot2_ERR_test){
  ge::op::ImageProjectiveTransform op;
  op.UpdateInputDesc("images", create_desc({1,2,3,4}, ge::DT_UINT8));
  op.UpdateInputDesc("transforms", create_desc({1,8}, ge::DT_FLOAT));
  op.UpdateInputDesc("output_shape", create_desc({3}, ge::DT_INT32));
  op.SetAttr("interpolation",  "NEAREST");
  op.SetAttr("fill_mode",  "CONSTANT");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(imageprojectivetransform, imageprojectivetransform_outputshape_interpolation_null){
  ge::op::ImageProjectiveTransform op;
  op.UpdateInputDesc("images", create_desc({1,2,3,4}, ge::DT_UINT8));
  op.UpdateInputDesc("transforms", create_desc({1,8}, ge::DT_FLOAT));
  op.UpdateInputDesc("output_shape", create_desc({2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}