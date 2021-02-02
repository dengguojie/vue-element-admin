#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "pad_ops.h"

class broadcast_to:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"broadcast_to Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"broadcast_to Proto Test TearDown"<<std::endl;
        }
};
TEST_F(broadcast_to,broadcast_to_infershape_diff_test){
    ge::op::BroadcastTo op;
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{1,10}};
    auto tensor_desc = create_desc_shape_range({-1},
                                                ge::DT_FLOAT16,ge::FORMAT_ND,
                                                {3},
                                                ge::FORMAT_ND,shape_range);
    auto input_shape_desc = create_desc_shape_range({1},
                                                ge::DT_FLOAT16,ge::FORMAT_ND,
                                                {1},
                                                ge::FORMAT_ND,{{1, 1}});                                            
    op.UpdateInputDesc("x",tensor_desc);
    op.UpdateInputDesc("shape",input_shape_desc);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret,ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(),ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(),expected_output_shape);
    std::vector<std::pair<int64_t,int64_t>> output_shape_range;
    EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range),ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1,-1}};
    EXPECT_EQ(output_shape_range,expected_shape_range);
}

