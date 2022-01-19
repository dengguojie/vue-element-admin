#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"

class DynamicPartition : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DynamicPartition SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DynamicPartition TearDown" << std::endl;
  }
};

TEST_F(DynamicPartition, DynamicPartition_infershape_test_1) {
  ge::op::DynamicPartition op;
  op.UpdateInputDesc("x", create_desc_with_ori({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
  op.UpdateInputDesc("partitions", create_desc_with_ori({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
  op.SetAttr("num_partitions", 2);
  op.create_dynamic_output_y(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetDynamicOutputDesc("y", 0);
}

TEST_F(DynamicPartition, DynamicPartition_infershape_test_2) {
  ge::op::DynamicPartition op;
  op.UpdateInputDesc("x", create_desc_with_ori({3}, ge::DT_INT32, ge::FORMAT_ND, {3}, ge::FORMAT_ND));
  op.UpdateInputDesc("partitions", create_desc_with_ori({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
  op.SetAttr("num_partitions", 2);
  op.create_dynamic_output_y(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynamicPartition, DynamicPartition_infershape_test_5) {
  ge::op::DynamicPartition op;
  op.UpdateInputDesc("x", create_desc_with_ori({5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
  op.UpdateInputDesc("partitions", create_desc_with_ori({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
  op.SetAttr("num_partitions", 2);
  op.create_dynamic_output_y(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynamicPartition, DynamicStitch_infershape_test_3) {
  ge::op::DynamicPartition op;
  op.UpdateInputDesc("x", create_desc_with_ori({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
  op.UpdateInputDesc("partitions", create_desc_with_ori({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
  op.SetAttr("num_partitions", 2);
  op.create_dynamic_output_y(1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynamicPartition, DynamicPartition_infershape_test_4) {
  ge::op::DynamicPartition op;
  op.UpdateInputDesc("x", create_desc_with_ori({3, 4}, ge::DT_INT32, ge::FORMAT_ND, {3, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("partitions", create_desc_with_ori({2, 4}, ge::DT_INT32, ge::FORMAT_ND, {2, 4}, ge::FORMAT_ND));
  op.SetAttr("num_partitions", 2);
  op.create_dynamic_output_y(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(DynamicPartition, DynamicPartition_infershape_partitions_unknown_shape) {
  ge::op::DynamicPartition op;
  op.UpdateInputDesc("x", create_desc_with_ori({4}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
  op.UpdateInputDesc("partitions", create_desc_with_ori({-1}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::FORMAT_ND));
  op.SetAttr("num_partitions", 2);
  op.create_dynamic_output_y(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetDynamicOutputDesc("y", 0);
}