#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class AxpyWithSoftmaxAndDropOutDoMaskTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AxpyWithSoftmaxAndDropOutDoMask Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AxpyWithSoftmaxAndDropOutDoMask Proto Test TearDown" << std::endl;
  }
};

TEST_F(AxpyWithSoftmaxAndDropOutDoMaskTest, axpy_with_softmax_and_drop_out_do_mask_test_1) {
    ge::op::AxpyWithSoftmaxAndDropOutDoMask axpy_with_softmax_and_drop_out_do_mask_op;
    axpy_with_softmax_and_drop_out_do_mask_op.UpdateInputDesc("x1", create_desc({96,12,24,24,16,16}, ge::DT_FLOAT16));

    auto ret = axpy_with_softmax_and_drop_out_do_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc_y1 = axpy_with_softmax_and_drop_out_do_mask_op.GetOutputDescByName("y1");
    auto output_desc_y2 = axpy_with_softmax_and_drop_out_do_mask_op.GetOutputDescByName("y2");
    EXPECT_EQ(output_desc_y1.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(output_desc_y2.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {96,12,24,24,16,16};
    EXPECT_EQ(output_desc_y1.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_desc_y2.GetShape().GetDims(), expected_output_shape);
}

TEST_F(AxpyWithSoftmaxAndDropOutDoMaskTest, axpy_with_softmax_and_drop_out_do_mask_test_2) {
    ge::op::AxpyWithSoftmaxAndDropOutDoMask axpy_with_softmax_and_drop_out_do_mask_op;
    axpy_with_softmax_and_drop_out_do_mask_op.UpdateInputDesc("x1", create_desc({96,12,32,32,16,16}, ge::DT_FLOAT16));

    auto ret = axpy_with_softmax_and_drop_out_do_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc_y1 = axpy_with_softmax_and_drop_out_do_mask_op.GetOutputDescByName("y1");
    auto output_desc_y2 = axpy_with_softmax_and_drop_out_do_mask_op.GetOutputDescByName("y2");
    EXPECT_EQ(output_desc_y1.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(output_desc_y2.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {96,12,32,32,16,16};
    EXPECT_EQ(output_desc_y1.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_desc_y2.GetShape().GetDims(), expected_output_shape);
}

TEST_F(AxpyWithSoftmaxAndDropOutDoMaskTest, axpy_with_softmax_and_drop_out_do_mask_test_3) {
    ge::op::AxpyWithSoftmaxAndDropOutDoMask axpy_with_softmax_and_drop_out_do_mask_op;
    axpy_with_softmax_and_drop_out_do_mask_op.UpdateInputDesc("x1", create_desc({96,12,16,16,16,16}, ge::DT_FLOAT16));

    auto ret = axpy_with_softmax_and_drop_out_do_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc_y1 = axpy_with_softmax_and_drop_out_do_mask_op.GetOutputDescByName("y1");
    auto output_desc_y2 = axpy_with_softmax_and_drop_out_do_mask_op.GetOutputDescByName("y2");
    EXPECT_EQ(output_desc_y1.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(output_desc_y2.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {96,12,16,16,16,16};
    EXPECT_EQ(output_desc_y1.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_desc_y2.GetShape().GetDims(), expected_output_shape);
}
