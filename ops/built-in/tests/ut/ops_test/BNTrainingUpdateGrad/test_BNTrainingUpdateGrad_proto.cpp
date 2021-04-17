#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class BNTrainingUpdateGrad : public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout << "BNTrainingUpdateGrad Proto Test SetUp" << std::endl;
        }

        static void TearDownTestCase(){
            std::cout << "BNTrainingUpdateGrad Proto Test TearDown" << std::endl;
        }
};

TEST_F(BNTrainingUpdateGrad, BNTrainingUpdateGrad_infershape_diff_test){
    ge::op::BNTrainingUpdateGrad op;
    std::vector<std::pair<int64_t,int64_t>> shape_batch_mean_range = {{1,1}, {1,64}, {1,1}, {1,1}, {16,16}};

    auto tensor_desc_batch_mean = create_desc_shape_range({1, -1, 1, 1, 16},
                                                          ge::DT_FLOAT, ge::FORMAT_NC1HWC0,
                                                          {1, -1, 1, 1, 16},
                                                          ge::FORMAT_NC1HWC0, shape_batch_mean_range);
    op.UpdateInputDesc("batch_mean", tensor_desc_batch_mean);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_scale_desc = op.GetOutputDesc("scale");
    auto output_offset_desc = op.GetOutputDesc("offset");

    EXPECT_EQ(output_scale_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_offset_desc.GetDataType(), ge::DT_FLOAT);

    std::vector<int64_t> expected_output_shape = {1, -1, 1, 1, 16};
    EXPECT_EQ(output_scale_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_offset_desc.GetShape().GetDims(), expected_output_shape);
    
    std::vector<std::pair<int64_t,int64_t>> output_scale_shape_range;
    std::vector<std::pair<int64_t,int64_t>> output_offset_shape_range;

    EXPECT_EQ(output_scale_desc.GetShapeRange(output_scale_shape_range), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_offset_desc.GetShapeRange(output_offset_shape_range), ge::GRAPH_SUCCESS);

    std::vector<std::pair<int64_t,int64_t>> expected_output_shape_range = {
      {1, 1},
      {1, 64},
      {1, 1},
      {1, 1},
      {16, 16}
    };

    EXPECT_EQ(output_scale_shape_range, expected_output_shape_range);
    EXPECT_EQ(output_offset_shape_range, expected_output_shape_range);

}