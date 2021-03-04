#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class sigmoidcrossentropywithlogitsv2 : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "SigmoidCrossEntropyWithLogitsV2 SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SigmoidCrossEntropyWithLogitsV2 TearDown" << std::endl;
    }
};

TEST_F(sigmoidcrossentropywithlogitsv2, sigmoidcrossentropywithlogitsv2_infershape) {
    ge::op::SigmoidCrossEntropyWithLogitsV2 op;
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};

    auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, 16},
                                             ge::FORMAT_ND, shape_range);
    
    std::string reduction = "none";
    op.UpdateInputDesc("predict", tensor_desc);
    op.UpdateInputDesc("target", tensor_desc);
    op.UpdateInputDesc("weight", tensor_desc);
    op.UpdateInputDesc("pos_weight", tensor_desc);
    op.SetAttr("reduction", reduction);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("loss");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

    std::vector<int64_t> expected_output_shape = {-1, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<std::pair<int64_t,int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1,16},{1,16}};
    EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(sigmoidcrossentropywithlogitsv2, sigmoidcrossentropywithlogitsv2_infershape1) {
    ge::op::SigmoidCrossEntropyWithLogitsV2 op;
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};

    auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, 16},
                                             ge::FORMAT_ND, shape_range);
    
    std::string reduction = "mean";
    op.UpdateInputDesc("predict", tensor_desc);
    op.UpdateInputDesc("target", tensor_desc);
    op.UpdateInputDesc("weight", tensor_desc);
    op.UpdateInputDesc("pos_weight", tensor_desc);
    op.SetAttr("reduction", reduction);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("loss");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

    std::vector<int64_t> expected_output_shape = {};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

