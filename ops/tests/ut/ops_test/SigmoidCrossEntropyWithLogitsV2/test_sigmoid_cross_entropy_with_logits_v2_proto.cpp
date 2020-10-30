#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class sigmoid_cross_entropy_with_logits_v2 : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "sigmoid_cross_entropy_with_logits_v2 SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "sigmoid_cross_entropy_with_logits_v2 TearDown" << std::endl;
    }
};

TEST_F(sigmoid_cross_entropy_with_logits_v2, sigmoid_cross_entropy_with_logits_v2_infershape_test_none) {
    ge::op::SigmoidCrossEntropyWithLogitsV2 op;
    op.UpdateInputDesc("predict", create_desc({128,128}, ge::DT_FLOAT16));
    
    std::string reduction = "none";
    op.SetAttr("reduction", reduction);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("loss");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {128,128};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(sigmoid_cross_entropy_with_logits_v2, sigmoid_cross_entropy_with_logits_v2_infershape_test_mean) {
    ge::op::SigmoidCrossEntropyWithLogitsV2 op;
    op.UpdateInputDesc("predict", create_desc({128,128}, ge::DT_FLOAT16));
    
    std::string reduction = "mean";
    op.SetAttr("reduction", reduction);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("loss");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape;
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(sigmoid_cross_entropy_with_logits_v2, sigmoid_cross_entropy_with_logits_v2_infershape_test_sum) {
    ge::op::SigmoidCrossEntropyWithLogitsV2 op;
    op.UpdateInputDesc("predict", create_desc({128,128}, ge::DT_FLOAT16));
    
    std::string reduction = "sum";
    op.SetAttr("reduction", reduction);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("loss");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape;
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

