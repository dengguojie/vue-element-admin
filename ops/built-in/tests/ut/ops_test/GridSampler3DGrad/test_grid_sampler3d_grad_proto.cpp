#include <gtest/gtest.h>

#include <iostream>
#include <numeric>

#include "array_ops.h"
#include "image_ops.h"
#include "op_proto_test_util.h"

class grid_sampler3d_grad : public testing::Test {
    protected:
    static void SetUpTestCase() { std::cout << "grid_sampler3d_grad SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "grid_sampler3d_grad TearDown" << std::endl; }
};

TEST_F(grid_sampler3d_grad, grid_sampler3d_grad_fp16) {
  ge::op::GridSampler3DGrad op;
  int n = 3;
  int c = 4;
  int d = 3;
  int h_in = 4;
  int w_in = 5;
  int h_out = 5;
  int w_out = 6;

  ge::DataType dtype = ge::DT_FLOAT16;
  ge::Format format = ge::FORMAT_ND;
  op.UpdateInputDesc("grad", create_desc_with_ori({n, c, d, h_out, w_out}, dtype, format, {n, c, d, h_out, w_out}, format));
  op.UpdateInputDesc("x", create_desc_with_ori({n, c, d, h_in, w_in}, dtype, format, {n, c, d, h_in, w_in}, format));
  op.UpdateInputDesc("grid", create_desc_with_ori({n, d, h_out, w_out, 3}, dtype, format, {n, d, h_out, w_out, 3}, format));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto dx_desc = op.get_output_desc_dx();
  EXPECT_EQ(dx_desc.GetDataType(), dtype);
  std::vector<int64_t> out_dx_shape = {n, c, d, h_in, w_in};
  EXPECT_EQ(dx_desc.GetShape().GetDims(), out_dx_shape);

  auto dgrid_desc = op.get_output_desc_dgrid();
  EXPECT_EQ(dgrid_desc.GetDataType(), dtype);
  std::vector<int64_t> out_dgrid_shape = {n, d, h_out, w_out, 3};
  EXPECT_EQ(dgrid_desc.GetShape().GetDims(), out_dgrid_shape);
}

TEST_F(grid_sampler3d_grad, grid_sampler3d_grad_fp32) {
  ge::op::GridSampler3DGrad op;
  int n = 13;
  int c = 24;
  int d = 43;
  int h_in = 54;
  int w_in = 25;
  int h_out = 15;
  int w_out = 16;

  ge::DataType dtype = ge::DT_FLOAT;
  ge::Format format = ge::FORMAT_ND;
  op.UpdateInputDesc("grad", create_desc_with_ori({n, c, d, h_out, w_out}, dtype, format, {n, c, d, h_out, w_out}, format));
  op.UpdateInputDesc("x", create_desc_with_ori({n, c, d, h_in, w_in}, dtype, format, {n, c, d, h_in, w_in}, format));
  op.UpdateInputDesc("grid", create_desc_with_ori({n, d, h_out, w_out, 3}, dtype, format, {n, d, h_out, w_out, 3}, format));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto dx_desc = op.get_output_desc_dx();
  EXPECT_EQ(dx_desc.GetDataType(), dtype);
  std::vector<int64_t> out_dx_shape = {n, c, d, h_in, w_in};
  EXPECT_EQ(dx_desc.GetShape().GetDims(), out_dx_shape);

  auto dgrid_desc = op.get_output_desc_dgrid();
  EXPECT_EQ(dgrid_desc.GetDataType(), dtype);
  std::vector<int64_t> out_dgrid_shape = {n, d, h_out, w_out, 3};
  EXPECT_EQ(dgrid_desc.GetShape().GetDims(), out_dgrid_shape);
}

TEST_F(grid_sampler3d_grad, grid_sampler3d_grad_fail_x) {
  ge::op::GridSampler3DGrad op;
  int n = 1;
  int c = 2;
  int d = 3;
  int h_in = 4;
  int w_in = 25;
  int h_out = 15;
  int w_out = 16;

  ge::DataType dtype = ge::DT_FLOAT;
  ge::Format format = ge::FORMAT_ND;
  op.UpdateInputDesc("grad", create_desc_with_ori({n, c, d, h_out, w_out}, dtype, format, {n, c, d, h_out, w_out}, format));
  op.UpdateInputDesc("x", create_desc_with_ori({n, c, d, h_in}, dtype, format, {n, c, d, h_in}, format));
  op.UpdateInputDesc("grid", create_desc_with_ori({n, d, h_out, w_out, 3}, dtype, format, {n, d, h_out, w_out, 3}, format));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(grid_sampler3d_grad, grid_sampler3d_grad_fail_grad) {
  ge::op::GridSampler3DGrad op;
  int n = 13;
  int c = 24;
  int d = 43;
  int h_in = 54;
  int w_in = 25;
  int h_out = 15;
  int w_out = 16;

  ge::DataType dtype = ge::DT_FLOAT16;
  ge::Format format = ge::FORMAT_ND;
  op.UpdateInputDesc("grad", create_desc_with_ori({n, c, h_out, w_out}, dtype, format, {n, c, h_out, w_out}, format));
  op.UpdateInputDesc("x", create_desc_with_ori({n, c, d, h_in, w_in}, dtype, format, {n, c, d, h_in, w_in}, format));
  op.UpdateInputDesc("grid", create_desc_with_ori({n, d, h_out, w_out, 3}, dtype, format, {n, d, h_out, w_out, 3}, format));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(grid_sampler3d_grad, grid_sampler3d_grad_fail_grid) {
  ge::op::GridSampler3DGrad op;
  int n = 13;
  int c = 24;
  int d = 43;
  int h_in = 54;
  int w_in = 25;
  int h_out = 15;
  int w_out = 16;

  ge::DataType dtype = ge::DT_FLOAT16;
  ge::Format format = ge::FORMAT_ND;
  op.UpdateInputDesc("grad", create_desc_with_ori({n, c, h_out, w_out}, dtype, format, {n, c, h_out, w_out}, format));
  op.UpdateInputDesc("x", create_desc_with_ori({n, c, d, h_in, w_in}, dtype, format, {n, c, d, h_in, w_in}, format));
  op.UpdateInputDesc("grid", create_desc_with_ori({n, d, h_out, w_out}, dtype, format, {n, d, h_out, w_out}, format));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}