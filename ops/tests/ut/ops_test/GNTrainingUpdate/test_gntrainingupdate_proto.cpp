#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class gn_training_update : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gn_training_update SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gn_training_update TearDown" << std::endl;
  }
};


TEST_F(gn_training_update, gn_training_update_infershape_diff_test_1) {
  ge::op::GNTrainingUpdate op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 64, 16}, 
    ge::FORMAT_NCHW));
  op.UpdateInputDesc("sum", create_desc_with_ori({4, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{4, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW));
  op.UpdateInputDesc("square_sum", create_desc_with_ori({4, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{4, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW)); 

  op.UpdateInputDesc("scale", create_desc_with_ori({1, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{1, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW));
  op.UpdateInputDesc("offset", create_desc_with_ori({1, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{1, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW));

  op.UpdateInputDesc("mean", create_desc_with_ori({4, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{4, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW));
  op.UpdateInputDesc("variance", create_desc_with_ori({4, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{4, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW));

  op.SetAttr("num_groups", (int)2);
  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_y_output_shape = {4, 64, 64, 16};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("batch_mean");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_mean_output_shape = {4, 2, 1, 1, 1};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);

  auto out_batch_variance_desc = op.GetOutputDesc("batch_variance");
  EXPECT_EQ(out_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_variance_output_shape = {4, 2, 1, 1, 1};
  EXPECT_EQ(out_batch_variance_desc.GetShape().GetDims(), expected_batch_variance_output_shape);

}


TEST_F(gn_training_update, gn_training_update_infershape_diff_test_2) {
  ge::op::GNTrainingUpdate op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NCHW,{4, 64, 64, 16}, 
    ge::FORMAT_NCHW));
  op.UpdateInputDesc("sum", create_desc_with_ori({4, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{4, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW));
  op.UpdateInputDesc("square_sum", create_desc_with_ori({4, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{4, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW)); 

  op.UpdateInputDesc("scale", create_desc_with_ori({1, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{1, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW));
  op.UpdateInputDesc("offset", create_desc_with_ori({1, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{1, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW));

  op.UpdateInputDesc("mean", create_desc_with_ori({4, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{4, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW));
  op.UpdateInputDesc("variance", create_desc_with_ori({4, 2, 1, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW,{4, 2, 1, 1, 1}, 
    ge::FORMAT_NCHW));

  op.SetAttr("num_groups", (int)2);
  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_y_output_shape = {4, 64, 64, 16};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("batch_mean");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_mean_output_shape = {4, 2, 1, 1, 1};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);

  auto out_batch_variance_desc = op.GetOutputDesc("batch_variance");
  EXPECT_EQ(out_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_variance_output_shape = {4, 2, 1, 1, 1};
  EXPECT_EQ(out_batch_variance_desc.GetShape().GetDims(), expected_batch_variance_output_shape);

}


TEST_F(gn_training_update, gn_training_update_infershape_diff_test_3) {
  ge::op::GNTrainingUpdate op;

  op.UpdateInputDesc("x", create_desc_with_ori({8, 11, 3, 24}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 11, 3, 24}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("sum", create_desc_with_ori({8, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("square_sum", create_desc_with_ori({8, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC)); 

  op.UpdateInputDesc("scale", create_desc_with_ori({1, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{1, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("offset", create_desc_with_ori({1, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{1, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC));

  op.UpdateInputDesc("mean", create_desc_with_ori({8, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("variance", create_desc_with_ori({8, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC));

  op.SetAttr("num_groups", (int)4);
  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_y_output_shape = {8, 11, 3, 24};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("batch_mean");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_mean_output_shape = {8, 1, 1, 4, 1};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);

  auto out_batch_variance_desc = op.GetOutputDesc("batch_variance");
  EXPECT_EQ(out_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_variance_output_shape = {8, 1, 1, 4, 1};
  EXPECT_EQ(out_batch_variance_desc.GetShape().GetDims(), expected_batch_variance_output_shape);

}

TEST_F(gn_training_update, gn_training_update_infershape_diff_test_4) {
  ge::op::GNTrainingUpdate op;

  op.UpdateInputDesc("x", create_desc_with_ori({8, 11, 3, 24}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{8, 11, 3, 24}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("sum", create_desc_with_ori({8, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("square_sum", create_desc_with_ori({8, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC)); 

  op.UpdateInputDesc("scale", create_desc_with_ori({1, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{1, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("offset", create_desc_with_ori({1, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{1, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC));

  op.UpdateInputDesc("mean", create_desc_with_ori({8, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("variance", create_desc_with_ori({8, 1, 1, 4, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 4, 1}, 
    ge::FORMAT_NHWC));

  op.SetAttr("num_groups", (int)4);
  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_y_output_shape = {8, 11, 3, 24};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("batch_mean");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_mean_output_shape = {8, 1, 1, 4, 1};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);

  auto out_batch_variance_desc = op.GetOutputDesc("batch_variance");
  EXPECT_EQ(out_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_variance_output_shape = {8, 1, 1, 4, 1};
  EXPECT_EQ(out_batch_variance_desc.GetShape().GetDims(), expected_batch_variance_output_shape);

}

TEST_F(gn_training_update, gn_training_update_infershape_diff_test_5) {
  ge::op::GNTrainingUpdate op;

  op.UpdateInputDesc("x", create_desc_with_ori({8, 11, 3, 24}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{8, 11, 3, 24}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("sum", create_desc_with_ori({8, 1, 1, 2, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 2, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("square_sum", create_desc_with_ori({8, 1, 1, 2, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 2, 1}, 
    ge::FORMAT_NHWC)); 

  op.UpdateInputDesc("scale", create_desc_with_ori({1, 1, 1, 2, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{1, 1, 1, 2, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("offset", create_desc_with_ori({1, 1, 1, 2, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{1, 1, 1, 2, 1}, 
    ge::FORMAT_NHWC));

  op.UpdateInputDesc("mean", create_desc_with_ori({8, 1, 1, 2, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 2, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateInputDesc("variance", create_desc_with_ori({8, 1, 1, 2, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{8, 1, 1, 2, 1}, 
    ge::FORMAT_NHWC));

  op.SetAttr("num_groups", (int)2);
  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_y_output_shape = {8, 11, 3, 24};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("batch_mean");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_mean_output_shape = {8, 1, 1, 2, 1};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);

  auto out_batch_variance_desc = op.GetOutputDesc("batch_variance");
  EXPECT_EQ(out_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_variance_output_shape = {8, 1, 1, 2, 1};
  EXPECT_EQ(out_batch_variance_desc.GetShape().GetDims(), expected_batch_variance_output_shape);

}