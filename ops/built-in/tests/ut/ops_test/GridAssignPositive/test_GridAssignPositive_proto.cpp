#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class GridAssignPositive : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GridAssignPositive Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GridAssignPositive Proto Test TearDown" << std::endl;
  }
};

TEST_F(GridAssignPositive, grid_assign_positive_infershape_test_1){
  ge::op::GridAssignPositive op;
  op.UpdateInputDesc("assigned_gt_inds", create_desc({6300, }, ge::DT_FLOAT16));
  op.UpdateInputDesc("overlaps", create_desc({128, 6300, }, ge::DT_FLOAT16));
  op.UpdateInputDesc("box_responsible_flags", create_desc({6300, }, ge::DT_UINT8));
  op.UpdateInputDesc("max_overlaps", create_desc({6300, }, ge::DT_FLOAT16));
  op.UpdateInputDesc("argmax_overlaps", create_desc({6300, }, ge::DT_INT32));
  op.UpdateInputDesc("gt_max_overlaps", create_desc({128, }, ge::DT_FLOAT16));
  op.UpdateInputDesc("gt_argmax_overlaps", create_desc({128, }, ge::DT_INT32));
  op.UpdateInputDesc("num_gts", create_desc({1, }, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("assigned_gt_inds_pos");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {6300, };
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GridAssignPositive, grid_assign_positive_verify_test_1) {
  ge::op::GridAssignPositive op;
  op.UpdateInputDesc("assigned_gt_inds", create_desc_with_ori({6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                              {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("overlaps", create_desc_with_ori({128, 6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {128, 6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("box_responsible_flags", create_desc_with_ori({6300, }, ge::DT_UINT8, ge::FORMAT_ND,
                                                                   {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("max_overlaps", create_desc_with_ori({6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                          {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("argmax_overlaps", create_desc_with_ori({6300, }, ge::DT_INT32, ge::FORMAT_ND,
                                                             {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("gt_max_overlaps", create_desc_with_ori({128, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                             {128, }, ge::FORMAT_ND));
  op.UpdateInputDesc("gt_argmax_overlaps", create_desc_with_ori({128, }, ge::DT_INT32, ge::FORMAT_ND,
                                                                {128, }, ge::FORMAT_ND));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(GridAssignPositive, grid_assign_positive_verify_test_2) {
  ge::op::GridAssignPositive op;
  op.UpdateInputDesc("assigned_gt_inds", create_desc_with_ori({6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                              {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("overlaps", create_desc_with_ori({6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("box_responsible_flags", create_desc_with_ori({6300, }, ge::DT_UINT8, ge::FORMAT_ND,
                                                                   {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("max_overlaps", create_desc_with_ori({6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                          {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("argmax_overlaps", create_desc_with_ori({6300, }, ge::DT_INT32, ge::FORMAT_ND,
                                                             {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("gt_max_overlaps", create_desc_with_ori({128, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                             {128, }, ge::FORMAT_ND));
  op.UpdateInputDesc("gt_argmax_overlaps", create_desc_with_ori({128, }, ge::DT_INT32, ge::FORMAT_ND,
                                                                {128, }, ge::FORMAT_ND));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GridAssignPositive, grid_assign_positive_verify_test_3) {
  ge::op::GridAssignPositive op;
  op.UpdateInputDesc("assigned_gt_inds", create_desc_with_ori({6301, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                              {6301, }, ge::FORMAT_ND));
  op.UpdateInputDesc("overlaps", create_desc_with_ori({128, 6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {128, 6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("box_responsible_flags", create_desc_with_ori({6300, }, ge::DT_UINT8, ge::FORMAT_ND,
                                                                   {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("max_overlaps", create_desc_with_ori({6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                          {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("argmax_overlaps", create_desc_with_ori({6300, }, ge::DT_INT32, ge::FORMAT_ND,
                                                             {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("gt_max_overlaps", create_desc_with_ori({128, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                             {128, }, ge::FORMAT_ND));
  op.UpdateInputDesc("gt_argmax_overlaps", create_desc_with_ori({128, }, ge::DT_INT32, ge::FORMAT_ND,
                                                                {128, }, ge::FORMAT_ND));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GridAssignPositive, grid_assign_positive_verify_test_4) {
  ge::op::GridAssignPositive op;
  op.UpdateInputDesc("assigned_gt_inds", create_desc_with_ori({6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                              {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("overlaps", create_desc_with_ori({128, 6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {128, 6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("box_responsible_flags", create_desc_with_ori({6300, }, ge::DT_UINT8, ge::FORMAT_ND,
                                                                   {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("max_overlaps", create_desc_with_ori({6300, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                          {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("argmax_overlaps", create_desc_with_ori({6300, }, ge::DT_INT32, ge::FORMAT_ND,
                                                             {6300, }, ge::FORMAT_ND));
  op.UpdateInputDesc("gt_max_overlaps", create_desc_with_ori({128, }, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                             {128, }, ge::FORMAT_ND));
  op.UpdateInputDesc("gt_argmax_overlaps", create_desc_with_ori({127, }, ge::DT_INT32, ge::FORMAT_ND,
                                                                {127, }, ge::FORMAT_ND));
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}