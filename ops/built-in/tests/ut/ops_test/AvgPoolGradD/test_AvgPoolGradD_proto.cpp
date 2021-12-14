#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_pooling_ops.h"


class AvgPoolGradDTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "avg_pool_grad_d test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "avg_pool_grad_d test TearDown" << std::endl;
    }
};

TEST_F(AvgPoolGradDTest, avg_pool_grad_d_test_case_1) {
    ge::op::AvgPoolGradD op;
    ge::TensorDesc tensordesc_input_grad;
    ge::Shape input_grad_shape({10, 1, 19, 16});
    tensordesc_input_grad.SetDataType(ge::DT_FLOAT16);
    tensordesc_input_grad.SetShape(input_grad_shape);

    ge::TensorDesc tensordesc_mean;
    ge::Shape mean_shape({10, 1, 19, 16});
    tensordesc_mean.SetDataType(ge::DT_FLOAT16);
    tensordesc_mean.SetShape(mean_shape);

    ge::TensorDesc tensordesc_kernel;
    ge::Shape kernel_shape({1, 1, 1, 16});
    tensordesc_kernel.SetDataType(ge::DT_FLOAT16);
    tensordesc_kernel.SetShape(kernel_shape);

    op.UpdateInputDesc("input_grad", tensordesc_input_grad);
    op.UpdateInputDesc("mean_matrix", tensordesc_mean);
    op.UpdateInputDesc("kernel_matrix", tensordesc_kernel);

    std::string padding = "VALID";
    std::string data_format = "NHWC";
    op.SetAttr("orig_input_shape", {10, 8, 148, 16});
    op.SetAttr("ksize", {1, 1, 1, 1});
    op.SetAttr("strides",{1, 8, 8, 1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("out_grad");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {10, 8, 148, 16};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(AvgPoolGradDTest, avg_pool_grad_d_test_case_2) {
    ge::op::AvgPoolGradD op;
    ge::TensorDesc tensordesc_input_grad;
    ge::Shape input_grad_shape({6, 16, 2, 2});
    tensordesc_input_grad.SetDataType(ge::DT_FLOAT16);
    tensordesc_input_grad.SetShape(input_grad_shape);

    ge::TensorDesc tensordesc_mean;
    ge::Shape mean_shape({6, 16, 2, 2});
    tensordesc_mean.SetDataType(ge::DT_FLOAT16);
    tensordesc_mean.SetShape(mean_shape);

    ge::TensorDesc tensordesc_kernel;
    ge::Shape kernel_shape({1, 16, 3, 3});
    tensordesc_kernel.SetDataType(ge::DT_FLOAT16);
    tensordesc_kernel.SetShape(kernel_shape);

    op.UpdateInputDesc("input_grad", tensordesc_input_grad);
    op.UpdateInputDesc("mean_matrix", tensordesc_mean);
    op.UpdateInputDesc("kernel_matrix", tensordesc_kernel);

    std::string padding = "SAME";
    std::string data_format = "NCHW";
    op.SetAttr("orig_input_shape", {6, 16, 4, 4});
    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides",{1, 1, 2, 2});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("out_grad");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {6, 16, 4, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(AvgPoolGradDTest, VerifyAvgPoolGradD_001) {
  ge::op::AvgPoolGradD op;
  op.SetAttr("orig_input_shape", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolGradDTest, VerifyAvgPoolGradD_002) {
  ge::op::AvgPoolGradD op;
  std::vector<int64_t> orig_input_size = {1, 2, 4};
  op.SetAttr("orig_input_shape", orig_input_size);
  op.SetAttr("ksize", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolGradDTest, VerifyAvgPoolGradD_003) {
  ge::op::AvgPoolGradD op;
  std::vector<int64_t> orig_input_size = {1, 2, 4};
  std::vector<int64_t> ksize = {1, 2, 3};
  op.SetAttr("orig_input_shape", orig_input_size);
  op.SetAttr("ksize", ksize);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolGradDTest, VerifyAvgPoolGradD_004) {
  ge::op::AvgPoolGradD op;
  std::vector<int64_t> orig_input_size = {1, 2, 4};
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  op.SetAttr("orig_input_shape", orig_input_size);
  op.SetAttr("ksize", ksize);
  op.SetAttr("ksize", {});

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolGradDTest, VerifyAvgPoolGradD_005) {
  ge::op::AvgPoolGradD op;
  std::vector<int64_t> orig_input_size = {1, 2, 4};
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2};
  op.SetAttr("orig_input_shape", orig_input_size);
  op.SetAttr("ksize", ksize);
  op.SetAttr("ksize", strides);
  op.SetAttr("padding", strides);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolGradDTest, VerifyAvgPoolGradD_006) {
  ge::op::AvgPoolGradD op;
  std::vector<int64_t> orig_input_size = {1, 2, 4};
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2};
  op.SetAttr("orig_input_shape", orig_input_size);
  op.SetAttr("ksize", ksize);
  op.SetAttr("ksize", strides);
  op.SetAttr("padding", "error");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolGradDTest, VerifyAvgPoolGradD_007) {
  ge::op::AvgPoolGradD op;
  std::vector<int64_t> orig_input_size = {1, 2, 4};
  std::vector<int64_t> ksize = {1, 2, 3, 4};
  std::vector<int64_t> strides = {1, 2};
  op.SetAttr("orig_input_shape", orig_input_size);
  op.SetAttr("ksize", ksize);
  op.SetAttr("ksize", strides);
  op.SetAttr("padding", "SAME");
  op.SetAttr("data_format", "ND");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}