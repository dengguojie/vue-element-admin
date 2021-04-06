#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class mse_loss:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"mse_loss Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"mse_loss Proto Test TearDown"<<std::endl;
        }
};


TEST_F(mse_loss, mse_loss_infershape_diff_test){
    ge::op::MseLoss op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{15, 16},{8,8},{375,375}};
    auto tensor_desc = create_desc_shape_range({-1,8,375},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {16,8,375},
                                                ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("predict", tensor_desc);
    op.UpdateInputDesc("label", tensor_desc);
    op.SetAttr("reduction", "mean");
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
    EXPECT_EQ(output_shape_range, expected_shape_range);
}
