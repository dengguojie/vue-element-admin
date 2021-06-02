#include <gtest/gtest.h>

#include <iostream>
#include <numeric>

#include "array_ops.h"
#include "image_ops.h"
#include "op_proto_test_util.h"

class grid_sampler3d : public testing::Test {
    protected:
    static void SetUpTestCase() { std::cout << "grid_sampler3d SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "grid_sampler3d TearDown" << std::endl; }
};

TEST_F(grid_sampler3d, grid_sampler3d_fp16_test) {
  ge::op::GridSampler3D op;
  std::vector<int64_t> out_shape = {1, 2, 2, 3, 4};

  ge::DataType dtype = ge::DT_FLOAT16;
  ge::Format format = ge::FORMAT_ND;
  op.UpdateInputDesc("grid", create_desc_with_ori({1, 2, 3, 4, 3}, dtype, format, {1, 2, 3, 4, 3}, format));
  op.UpdateInputDesc("x", create_desc_with_ori({1, 2, 2, 3, 2}, dtype, format, {1, 2, 2, 3, 2}, format));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto diff_desc = op.get_output_desc_y();

  EXPECT_EQ(diff_desc.GetDataType(), dtype);
  EXPECT_EQ(diff_desc.GetShape().GetDims(), out_shape);
}

TEST_F(grid_sampler3d, grid_sampler3d_fp32_test) {
  ge::op::GridSampler3D op;
  std::vector<int64_t> out_shape = {11, 12, 12, 13, 14};

  ge::DataType dtype = ge::DT_FLOAT;
  ge::Format format = ge::FORMAT_ND;
  op.UpdateInputDesc("grid", create_desc_with_ori({11, 12, 13, 14, 3}, dtype, format, {11, 12, 13, 14, 3}, format));
  op.UpdateInputDesc("x", create_desc_with_ori({11, 12, 12, 13, 12}, dtype, format, {11, 12, 12, 13, 12}, format));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto diff_desc = op.get_output_desc_y();

  EXPECT_EQ(diff_desc.GetDataType(), dtype);
  EXPECT_EQ(diff_desc.GetShape().GetDims(), out_shape);
}

TEST_F(grid_sampler3d, grid_sampler3d_fail_x) {
  ge::op::GridSampler3D op;
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::Format format = ge::FORMAT_ND;
  op.UpdateInputDesc("grid", create_desc_with_ori({11, 2, 5, 4, 3}, dtype, format, {11, 2, 5, 4, 3}, format));
  op.UpdateInputDesc("x", create_desc_with_ori({11, 2, 3}, dtype, format, {11, 2, 3}, format));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(grid_sampler3d, grid_sampler3d_fail_grid) {
  ge::op::GridSampler3D op;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::Format format = ge::FORMAT_ND;
  op.UpdateInputDesc("grid", create_desc_with_ori({1, 2, 3}, dtype, format, {1, 2, 3}, format));
  op.UpdateInputDesc("x", create_desc_with_ori({1, 2, 3, 4, 5}, dtype, format, {1, 2, 3, 4, 5}, format));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(grid_sampler3d, grid_sampler3d_fail_grid_last_dim) {
  ge::op::GridSampler3D op;

  ge::DataType dtype = ge::DT_FLOAT16;
  ge::Format format = ge::FORMAT_ND;
  op.UpdateInputDesc("grid", create_desc_with_ori({21, 22, 23, 24, 5}, dtype, format, {21, 22, 23, 24, 5}, format));
  op.UpdateInputDesc("x", create_desc_with_ori({21, 3, 22, 23, 24}, dtype, format, {21, 3, 22, 23, 24}, format));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}