#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "pad_ops.h"

class broadcast_to_D:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"broadcast_to_D Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"broadcast_to_D Proto Test TearDown"<<std::endl;
        }
};
TEST_F(broadcast_to_D,broadcast_to_D_infershape_diff_test){
    ge::op::BroadcastToD op;
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{1,10}};
    auto tensor_desc = create_desc_shape_range({-1},
                                                ge::DT_FLOAT16,ge::FORMAT_ND,
                                                {3},
                                                ge::FORMAT_ND,shape_range);
    std::vector<int64_t> input_shape = {1, 2, 4, 6, 1, 1, 1, 1, 1};
    op.SetAttr("shape", input_shape);                                        
    op.UpdateInputDesc("x",tensor_desc);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret,ge::GRAPH_FAILED);
}

