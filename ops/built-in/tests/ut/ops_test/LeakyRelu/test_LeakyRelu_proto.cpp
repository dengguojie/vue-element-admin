#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class LeakyRelu : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LeakyRelu Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LeakyRelu Proto Test TearDown" << std::endl;
  }
};

TEST_F(LeakyRelu, leaky_relu_infershape_test1){
  ge::op::LeakyRelu op;
  
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  float negative_slope = 0.0;
  op.SetAttr("negative_slope", negative_slope);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1,16},{1,16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(LeakyRelu, leaky_relu_infershape_test2){
  ge::op::LeakyRelu op;
  
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {16, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  float negative_slope = 0.0;
  op.SetAttr("negative_slope", negative_slope);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1,16},{1,16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(LeakyRelu, leaky_relu_infershape_test3){
  ge::op::LeakyRelu op;
  
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_DOUBLE, ge::FORMAT_ND,
                                             {16, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  float negative_slope = 0.0;
  op.SetAttr("negative_slope", negative_slope);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_DOUBLE);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1,16},{1,16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
