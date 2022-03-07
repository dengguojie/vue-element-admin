#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class seluGrad:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"seluGrad Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"seluGrad Proto Test TearDown"<<std::endl;
        }
};

TEST_F(seluGrad,seluGrad_test_1){
    ge::op::SeluGrad op;
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{2,100}};
    auto tensor_desc = create_desc_shape_range({-1},
                                                ge::DT_FLOAT16,ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND,shape_range);
    op.UpdateInputDesc("gradients",tensor_desc);
    op.UpdateInputDesc("outputs",tensor_desc);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret,ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(),ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(),expected_output_shape);
    std::vector<std::pair<int64_t,int64_t>> output_shape_range;
    EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range),ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{2,100},};
    EXPECT_EQ(output_shape_range,expected_shape_range);
}
TEST_F(seluGrad,seluGrad_test_2){
    ge::op::SeluGrad op;
    std::vector<std::pair<int64_t,int64_t>> shape_range = {{2,100}};
    auto tensor_desc = create_desc_shape_range({-2},
                                                ge::DT_FLOAT16,ge::FORMAT_ND,
                                                {-2},
                                                ge::FORMAT_ND,shape_range);
    op.UpdateInputDesc("gradients",tensor_desc);
    op.UpdateInputDesc("outputs",tensor_desc);
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret,ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(),ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-2};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(),expected_output_shape);
    std::vector<std::pair<int64_t,int64_t>> output_shape_range;
    EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range),ge::GRAPH_SUCCESS);
}
