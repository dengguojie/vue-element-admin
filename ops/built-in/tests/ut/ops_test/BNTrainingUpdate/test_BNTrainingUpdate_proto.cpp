#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class BNTrainingUpdate : public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout << "BNTrainingUpdate Proto Test SetUp" << std::endl;
        }

        static void TearDownTestCase(){
            std::cout << "BNTrainingUpdate Proto Test TearDown" << std::endl;
        }
};

TEST_F(BNTrainingUpdate, BNTrainingUpate_infershape_diff_test){
    ge::op::BNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({2,4,6,6,16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("sum", create_desc({1,4,1,1,16}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({1,4,1,1,16}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({1,4,1,1,16}, ge::DT_FLOAT));
    op.UpdateInputDesc("offset", create_desc({1,4,1,1,16}, ge::DT_FLOAT));
    op.UpdateInputDesc("mean", create_desc({1,4,1,1,16}, ge::DT_FLOAT));
    op.UpdateInputDesc("variance", create_desc({1,4,1,1,16}, ge::DT_FLOAT));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc("y");
    auto output_mean_desc = op.GetOutputDesc("mean");
    auto output_variance_desc = op.GetOutputDesc("variance");
    auto output_batch_mean_desc = op.GetOutputDesc("batch_mean");
    auto output_batch_variance_desc = op.GetOutputDesc("batch_variance");

    EXPECT_EQ(output_y_desc.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(output_mean_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_variance_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_batch_mean_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_batch_variance_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape1 = {2, 4, 6, 6, 16};
    std::vector<int64_t> expected_output_shape2 = {1, 4, 1, 1, 16};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape1);
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_output_shape2);
    EXPECT_EQ(output_variance_desc.GetShape().GetDims(), expected_output_shape2);
    EXPECT_EQ(output_batch_mean_desc.GetShape().GetDims(), expected_output_shape2);
    EXPECT_EQ(output_batch_variance_desc.GetShape().GetDims(), expected_output_shape2);
}