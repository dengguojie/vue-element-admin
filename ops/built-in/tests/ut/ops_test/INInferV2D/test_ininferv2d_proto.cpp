#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class in_infer_v2d : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "in_infer_v2d SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "in_infer_v2d TearDown" << std::endl;
  }
};


TEST_F(in_infer_v2d, in_infer_v2d_infershape_diff_test_1) {
  ge::op::INInferV2D op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 1, 64, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("gamma", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("beta", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("mean", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance_sqrt", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));

  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_y_output_shape = {4, 1, 64, 64, 16};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("batch_mean");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_mean_output_shape = {4, 1, 1, 1, 16};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);

  auto out_batch_variance_desc = op.GetOutputDesc("batch_variance");
  EXPECT_EQ(out_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_variance_output_shape = {4, 1, 1, 1, 16};
  EXPECT_EQ(out_batch_variance_desc.GetShape().GetDims(), expected_batch_variance_output_shape);

}

TEST_F(in_infer_v2d, in_infer_v2d_infershape_diff_test_2) {
  ge::op::INInferV2D op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 100, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 1, 100, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("gamma", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("beta", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("mean", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance_sqrt", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));

  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_y_output_shape = {4, 1, 100, 64, 16};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("batch_mean");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_mean_output_shape = {4, 1, 1, 1, 16};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);

  auto out_batch_variance_desc = op.GetOutputDesc("batch_variance");
  EXPECT_EQ(out_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_variance_output_shape = {4, 1, 1, 1, 16};
  EXPECT_EQ(out_batch_variance_desc.GetShape().GetDims(), expected_batch_variance_output_shape);

}

TEST_F(in_infer_v2d, in_infer_v2d_infershape_diff_test_3) {
  ge::op::INInferV2D op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 100, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 100, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("gamma", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("beta", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("mean", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance_sqrt", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  
  op.UpdateOutputDesc("y", create_desc_with_ori({4, 1, 100, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 100, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateOutputDesc("batch_mean", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateOutputDesc("batch_variance", create_desc_with_ori({4, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));

  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_y_output_shape = {4, 1, 100, 64, 16};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("batch_mean");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_mean_output_shape = {4, 1, 1, 1, 16};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);

  auto out_batch_variance_desc = op.GetOutputDesc("batch_variance");
  EXPECT_EQ(out_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_variance_output_shape = {4, 1, 1, 1, 16};
  EXPECT_EQ(out_batch_variance_desc.GetShape().GetDims(), expected_batch_variance_output_shape);
}

TEST_F(in_infer_v2d, in_infer_v2d_infershape_diff_test_4) {
  ge::op::INInferV2D op;

  op.UpdateInputDesc("x", create_desc_with_ori({5, 1, 100, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 100, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("gamma", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("beta", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("mean", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance_sqrt", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
    
  op.UpdateOutputDesc("y", create_desc_with_ori({5, 1, 100, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 100, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateOutputDesc("batch_mean", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateOutputDesc("batch_variance", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));

  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_y_output_shape = {5, 1, 100, 64, 16};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("batch_mean");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_mean_output_shape = {5, 1, 1, 1, 16};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);

  auto out_batch_variance_desc = op.GetOutputDesc("batch_variance");
  EXPECT_EQ(out_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_variance_output_shape = {5, 1, 1, 1, 16};
  EXPECT_EQ(out_batch_variance_desc.GetShape().GetDims(), expected_batch_variance_output_shape);
}

TEST_F(in_infer_v2d, in_infer_v2d_infershape_diff_test_5) {
  ge::op::INInferV2D op;

  op.UpdateInputDesc("x", create_desc_with_ori({5, 3, 1, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("gamma", create_desc_with_ori({5, 3, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("beta", create_desc_with_ori({5, 3, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("mean", create_desc_with_ori({5, 3, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateInputDesc("variance_sqrt", create_desc_with_ori({5, 1, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
    
  op.UpdateOutputDesc("y", create_desc_with_ori({5, 3, 1, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 64, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateOutputDesc("batch_mean", create_desc_with_ori({5, 3, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));
  op.UpdateOutputDesc("batch_variance", create_desc_with_ori({5, 3, 1, 1, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 3, 1, 1, 16}, 
    ge::FORMAT_NC1HWC0));

  op.SetAttr("epsilon", (float)0.00001);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_y_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_y_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_y_output_shape = {5, 3, 1, 64, 16};
  EXPECT_EQ(out_y_desc.GetShape().GetDims(), expected_y_output_shape);

  auto out_batch_mean_desc = op.GetOutputDesc("batch_mean");
  EXPECT_EQ(out_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_mean_output_shape = {5, 3, 1, 1, 16};
  EXPECT_EQ(out_batch_mean_desc.GetShape().GetDims(), expected_batch_mean_output_shape);

  auto out_batch_variance_desc = op.GetOutputDesc("batch_variance");
  EXPECT_EQ(out_batch_variance_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_batch_variance_output_shape = {5, 3, 1, 1, 16};
  EXPECT_EQ(out_batch_variance_desc.GetShape().GetDims(), expected_batch_variance_output_shape);
}