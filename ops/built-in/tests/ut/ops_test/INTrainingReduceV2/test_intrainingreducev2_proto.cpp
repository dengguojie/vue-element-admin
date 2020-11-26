#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class in_training_reduce_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "in_training_reduce_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "in_training_reduce_v2 TearDown" << std::endl;
  }
};


TEST_F(in_training_reduce_v2, in_training_reduce_v2_infershape_diff_test_1) {
  ge::op::INTrainingReduceV2 op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{4, 1, 64, 64, 16}, 
    ge::FORMAT_NC1HWC0));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {4, 1, 1, 1, 16};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);

  auto out_square_sum_desc = op.GetOutputDesc("square_sum");
  EXPECT_EQ(out_square_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_square_sum_output_shape = {4, 1, 1, 1, 16};
  EXPECT_EQ(out_square_sum_desc.GetShape().GetDims(), expected_square_sum_output_shape);

}

TEST_F(in_training_reduce_v2, in_training_reduce_v2_infershape_diff_test_2) {
  ge::op::INTrainingReduceV2 op;

  op.UpdateInputDesc("x", create_desc_with_ori({4, 1, 64, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{4, 1, 64, 64, 16}, 
    ge::FORMAT_NC1HWC0));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {4, 1, 1, 1, 16};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);

  auto out_square_sum_desc = op.GetOutputDesc("square_sum");
  EXPECT_EQ(out_square_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_square_sum_output_shape = {4, 1, 1, 1, 16};
  EXPECT_EQ(out_square_sum_desc.GetShape().GetDims(), expected_square_sum_output_shape);

}

TEST_F(in_training_reduce_v2, in_training_reduce_v2_infershape_diff_test_3) {
  ge::op::INTrainingReduceV2 op;

  op.UpdateInputDesc("x", create_desc_with_ori({5, 1, 64, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 1, 64, 64, 16}, 
    ge::FORMAT_NC1HWC0));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {5, 1, 1, 1, 16};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);

  auto out_square_sum_desc = op.GetOutputDesc("square_sum");
  EXPECT_EQ(out_square_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_square_sum_output_shape = {5, 1, 1, 1, 16};
  EXPECT_EQ(out_square_sum_desc.GetShape().GetDims(), expected_square_sum_output_shape);

}


TEST_F(in_training_reduce_v2, in_training_reduce_v2_infershape_diff_test_4) {
  ge::op::INTrainingReduceV2 op;

  op.UpdateInputDesc("x", create_desc_with_ori({5, 1, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,{5, 1, 64, 64, 16}, 
    ge::FORMAT_NC1HWC0));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {5, 1, 1, 1, 16};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);

  auto out_square_sum_desc = op.GetOutputDesc("square_sum");
  EXPECT_EQ(out_square_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_square_sum_output_shape = {5, 1, 1, 1, 16};
  EXPECT_EQ(out_square_sum_desc.GetShape().GetDims(), expected_square_sum_output_shape);

}

TEST_F(in_training_reduce_v2, in_training_reduce_v2_infershape_diff_test_5) {
  ge::op::INTrainingReduceV2 op;

  op.UpdateInputDesc("x", create_desc_with_ori({5, 2, 30, 64, 16}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0,{5, 2, 30, 64, 16}, 
    ge::FORMAT_NC1HWC0));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto out_sum_desc = op.GetOutputDesc("sum");
  EXPECT_EQ(out_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_sum_output_shape = {5, 2, 1, 1, 16};
  EXPECT_EQ(out_sum_desc.GetShape().GetDims(), expected_sum_output_shape);

  auto out_square_sum_desc = op.GetOutputDesc("square_sum");
  EXPECT_EQ(out_square_sum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_square_sum_output_shape = {5, 2, 1, 1, 16};
  EXPECT_EQ(out_square_sum_desc.GetShape().GetDims(), expected_square_sum_output_shape);

}
