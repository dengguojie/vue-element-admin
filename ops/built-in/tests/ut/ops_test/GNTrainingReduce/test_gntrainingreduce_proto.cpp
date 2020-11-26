#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class gn_training_reduce : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gn_training_reduce SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gn_training_reduce TearDown" << std::endl;
  }
};


TEST_F(gn_training_reduce, gn_training_reduce_infershape_diff_test_1) {
  ge::op::GNTrainingReduce op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{4, 64, 64, 16}, 
    ge::FORMAT_NCHW));

  op.SetAttr("num_groups", (int)2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {4, 2, 1, 1, 1};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);

  auto out_square_sum_desc = op.GetOutputDesc("square_sum");
  EXPECT_EQ(out_square_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_square_sum_output_shape = {4, 2, 1, 1, 1};
  EXPECT_EQ(out_square_sum_desc.GetShape().GetDims(), expected_square_sum_output_shape);

}


TEST_F(gn_training_reduce, gn_training_reduce_infershape_diff_test_2) {
  ge::op::GNTrainingReduce op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NCHW,{4, 64, 64, 16}, 
    ge::FORMAT_NCHW));

  op.SetAttr("num_groups", (int)2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {4, 2, 1, 1, 1};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);

  auto out_square_sum_desc = op.GetOutputDesc("square_sum");
  EXPECT_EQ(out_square_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_square_sum_output_shape = {4, 2, 1, 1, 1};
  EXPECT_EQ(out_square_sum_desc.GetShape().GetDims(), expected_square_sum_output_shape);

}


TEST_F(gn_training_reduce, gn_training_reduce_infershape_diff_test_3) {
  ge::op::GNTrainingReduce op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NHWC,{4, 64, 64, 16}, 
    ge::FORMAT_NHWC));

  op.UpdateOutputDesc("sum", create_desc_with_ori({4, 1, 1, 2, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{4, 1, 1, 2, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateOutputDesc("square_sum", create_desc_with_ori({4, 1, 1, 2, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{4, 1, 1, 2, 1}, 
    ge::FORMAT_NHWC));

  op.SetAttr("num_groups", (int)2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {4, 1, 1, 2, 1};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);

  auto out_square_sum_desc = op.GetOutputDesc("square_sum");
  EXPECT_EQ(out_square_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_square_sum_output_shape = {4, 1, 1, 2, 1};
  EXPECT_EQ(out_square_sum_desc.GetShape().GetDims(), expected_square_sum_output_shape);

}

TEST_F(gn_training_reduce, gn_training_reduce_infershape_diff_test_4) {
  ge::op::GNTrainingReduce op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16}, 
    ge::FORMAT_NHWC));

  op.UpdateOutputDesc("sum", create_desc_with_ori({4, 1, 1, 2, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{4, 1, 1, 2, 1}, 
    ge::FORMAT_NHWC));
  op.UpdateOutputDesc("square_sum", create_desc_with_ori({4, 1, 1, 2, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC,{4, 1, 1, 2, 1}, 
    ge::FORMAT_NHWC));

  op.SetAttr("num_groups", (int)2);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {4, 1, 1, 2, 1};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);

  auto out_square_sum_desc = op.GetOutputDesc("square_sum");
  EXPECT_EQ(out_square_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_square_sum_output_shape = {4, 1, 1, 2, 1};
  EXPECT_EQ(out_square_sum_desc.GetShape().GetDims(), expected_square_sum_output_shape);

}
