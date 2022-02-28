#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"

class expand:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"expand Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"expand Proto Test TearDown"<<std::endl;
        }
};

TEST_F(expand, expand_infershape_diff_test){
    ge::op::Expand op;
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

TEST_F(expand, expand_infershape_const) {
    ge::op::Expand expand_op;

    ge::TensorDesc tensorDesc1;
    ge::Shape xShape({2, 2});
    tensorDesc1.SetDataType(ge::DT_INT32);
    tensorDesc1.SetShape(xShape);
    expand_op.UpdateInputDesc("x", tensorDesc1);

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape({2, 2}), ge::FORMAT_ND, ge::DT_INT32);
    constDesc.SetSize(4 * sizeof(int32_t));
    constTensor.SetTensorDesc(constDesc);
    int64_t constData[4] = {1, 1, 1, 1};
    constTensor.SetData((uint8_t *)constData, 4 * sizeof(int32_t));
    auto shape = ge::op::Constant().set_attr_value(constTensor);

    expand_op.set_input_shape(shape);
    auto desc = expand_op.GetInputDescByName("shape");
    desc.SetDataType(ge::DT_INT32);
    expand_op.UpdateInputDesc("shape", desc);

    auto ret = expand_op.InferShapeAndType();
}

