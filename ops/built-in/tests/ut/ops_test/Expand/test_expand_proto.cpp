#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "common/utils/ut_op_common.h"

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

TEST_F(expand, expand_infershape_const_1) {
    ge::op::Expand expand_op;

    ge::TensorDesc x_desc;
    ge::Shape xShape({2, 4});
    x_desc.SetDataType(ge::DT_FLOAT);
    x_desc.SetShape(xShape);
    x_desc.SetOriginShape(xShape);

    ge::TensorDesc shape_desc;
    ge::Shape yShape({4});
    shape_desc.SetDataType(ge::DT_INT32);
    shape_desc.SetShape(yShape);
    shape_desc.SetOriginShape(yShape);

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
    constDesc.SetSize(4 * sizeof(int32_t));
    constTensor.SetTensorDesc(constDesc);
    int32_t constData[4] = {1, 1, 1, 1};
    constTensor.SetData((uint8_t *)constData, 4 * sizeof(int32_t));
    auto shape = ge::op::Constant().set_attr_value(constTensor);

    expand_op.set_input_shape(shape);

    expand_op.UpdateInputDesc("x", x_desc);
    expand_op.UpdateInputDesc("shape", shape_desc);

    vector<bool> input_const = {false, true};
    std::vector<vector<int64_t>> expect_shapes_vector = {{1, 1, 2, 4}};

    CommonInferShapeOperatorWithConst(expand_op, input_const, {}, expect_shapes_vector);
}

TEST_F(expand, expand_infershape_const_2) {
    ge::op::Expand expand_op;

    ge::TensorDesc x_desc;
    ge::Shape xShape({1, 1, 1, 1});
    x_desc.SetDataType(ge::DT_FLOAT);
    x_desc.SetShape(xShape);
    x_desc.SetOriginShape(xShape);

    ge::TensorDesc shape_desc;
    ge::Shape yShape({2});
    shape_desc.SetDataType(ge::DT_INT64);
    shape_desc.SetShape(yShape);
    shape_desc.SetOriginShape(yShape);

    ge::Tensor constTensor;
    ge::TensorDesc constDesc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
    constDesc.SetSize(2 * sizeof(int64_t));
    constTensor.SetTensorDesc(constDesc);
    int64_t constData[2] = {2, 4};
    constTensor.SetData((uint8_t *)constData, 2 * sizeof(int64_t));
    auto shape = ge::op::Constant().set_attr_value(constTensor);

    expand_op.set_input_shape(shape);

    expand_op.UpdateInputDesc("x", x_desc);
    expand_op.UpdateInputDesc("shape", shape_desc);

    vector<bool> input_const = {false, true};
    std::vector<vector<int64_t>> expect_shapes_vector = {{1, 1, 2, 4}};

    CommonInferShapeOperatorWithConst(expand_op, input_const, {}, expect_shapes_vector);
}


