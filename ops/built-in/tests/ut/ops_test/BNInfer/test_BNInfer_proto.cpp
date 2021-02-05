#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class bn_infer:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"bn_infer Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"bn_infer Proto Test TearDown"<<std::endl;
        }
};


TEST_F(bn_infer,bn_infer_infershape_diff_test){
    ge::op::BNInfer op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 10}, {1, 1}, {1, 1}, {1, 1},{16, 16}};
    auto tensor_desc = create_desc_shape_range({-1, 1, 1, 1, 16},
                                                ge::DT_FLOAT16, ge::FORMAT_NC1HWC0,
                                                {-1, 1, 1, 1, 16},
                                                ge::FORMAT_NC1HWC0, shape_range);
    op.UpdateInputDesc("x", tensor_desc);
    op.UpdateInputDesc("scale", tensor_desc);
    op.UpdateInputDesc("offset", tensor_desc);
    op.UpdateInputDesc("mean", tensor_desc);
    op.UpdateInputDesc("variance", tensor_desc);
    op.SetAttr("epsilon", (float)0.0001);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1, 1, 1, 1, 16};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1, 10}, {1, 1}, {1, 1}, {1, 1},{16, 16}};
    EXPECT_EQ(output_shape_range, expected_shape_range);
}

