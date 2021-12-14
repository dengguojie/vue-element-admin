#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class maximum_grad:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"maximum_grad Proto Test SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"maximum_grad Proto Test TearDown"<<std::endl;
        }
};


TEST_F(maximum_grad,maximum_grad_infershape_diff_test){
    ge::op::MaximumGrad op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
    auto tensor_desc = create_desc_shape_range({-1},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {64},
                                                ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("grads", tensor_desc);
    op.UpdateInputDesc("x1", tensor_desc);
    op.UpdateInputDesc("x2", tensor_desc);
    op.SetAttr("grad_x", true);
    op.SetAttr("grad_y", true);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y1");
    EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{2, 100},};
    EXPECT_EQ(output_shape_range, expected_shape_range);
    auto output_y2_desc = op.GetOutputDesc("y2");
    EXPECT_EQ(output_y2_desc.GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_y2_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(maximum_grad, InfershapeMaximumGrad_001) {
  ge::op::MaximumGrad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grads", tensor_desc);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.SetAttr("grad_x", "error");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(maximum_grad, InfershapeMaximumGrad_002) {
  ge::op::MaximumGrad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grads", tensor_desc);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.SetAttr("grad_x", true);
  op.SetAttr("grad_y", "error");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(maximum_grad, InfershapeMaximumGrad_003) {
  ge::op::MaximumGrad op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
  auto tensor_desc = create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("grads", tensor_desc);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);
  op.SetAttr("grad_x", false);
  op.SetAttr("grad_y", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}