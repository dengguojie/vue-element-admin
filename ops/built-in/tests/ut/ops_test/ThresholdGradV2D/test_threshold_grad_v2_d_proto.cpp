#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class threshold_grad_v2_d : public testing::Test {
protected:
	static void SetUpTestCase() {
		std::cout << "threshold_grad_v2_d SetUp" << std::endl;
	}

	static void TearDownTestCase() {
    std::cout << "threshold_grad_v2_d Test TearDown" << std::endl;
  }
};

TEST_F(threshold_grad_v2_d, cce_threshold_grad_v2_d_diff_shape_success){
	ge::op::ThresholdGradV2D op;
	op.UpdateInputDesc("gradients", create_desc({1, 2, 3, 5}, ge::DT_FLOAT16));
	op.UpdateInputDesc("features", create_desc({1, 2, 3, 1}, ge::DT_FLOAT16));
	op.SetAttr("threshold", 1.2f);
	auto status = op.VerifyAllAttr(true);
	EXPECT_EQ(status, ge::GRAPH_SUCCESS);
	auto ret = op.InferShapeAndType();
	EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
	auto output_desc = op.GetOutputDesc("backprops");
	EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
	std::vector<int64_t> expected_output_shape = {1, 2, 3, 5};
	EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(threshold_grad_v2_d, cce_threshold_grad_v2_d_float32){
	ge::op::ThresholdGradV2D op;
	op.UpdateInputDesc("gradients", create_desc({1, 2, 3, 5}, ge::DT_FLOAT));
	op.UpdateInputDesc("features", create_desc({1, 2, 3, 5}, ge::DT_FLOAT));
	op.SetAttr("threshold", 1.2f);
	auto status = op.VerifyAllAttr(true);
	EXPECT_EQ(status, ge::GRAPH_SUCCESS);
	auto ret = op.InferShapeAndType();
	EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
	auto output_desc = op.GetOutputDesc("backprops");
	EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
	std::vector<int64_t> expected_output_shape = {1, 2, 3, 5};
	EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
TEST_F(threshold_grad_v2_d, cce_threshold_grad_v2_d_verify_fail){
	ge::op::ThresholdGradV2D op;
	op.UpdateInputDesc("gradients", create_desc({1, 2, 3, 5}, ge::DT_FLOAT16));
	op.UpdateInputDesc("features", create_desc({1, 2, 3, 5}, ge::DT_FLOAT));
	op.SetAttr("threshold", 1.2f);
	auto status = op.VerifyAllAttr(true);
	EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(threshold_grad_v2_d, cce_threshold_grad_v2_d_infer_shape_fail){
	ge::op::ThresholdGradV2D op;
	op.UpdateInputDesc("gradients", create_desc({1, 2, 3, 5}, ge::DT_FLOAT16));
	op.UpdateInputDesc("features", create_desc({1, 2, 3, 2}, ge::DT_FLOAT16));
	op.SetAttr("threshold", 1.2f);
	auto ret = op.InferShapeAndType();
	EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
