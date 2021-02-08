#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class ScaleTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScaleTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScaleTest TearDown" << std::endl;
  }
};

TEST_F(ScaleTest, scale_test_infershape_test_1) {
  ge::op::Scale op;
  // set x input shape
  ge::TensorDesc xTensorDesc;
  ge::Shape xShape({2,2,3});
  xTensorDesc.SetDataType(ge::DT_FLOAT16);
  xTensorDesc.SetShape(xShape);

  // set scale input shape
  ge::TensorDesc scaleTensorDesc;
  ge::Shape scaleShape({2,2,3});
  scaleTensorDesc.SetDataType(ge::DT_FLOAT16);
  scaleTensorDesc.SetShape(scaleShape);

  // set bias input shape
  ge::TensorDesc biasTensorDesc;
  ge::Shape biasShape({2,2,3});
  biasTensorDesc.SetDataType(ge::DT_FLOAT16);
  biasTensorDesc.SetShape(biasShape);

  op.UpdateInputDesc("x", xTensorDesc);
  op.UpdateInputDesc("scale", scaleTensorDesc);
  op.UpdateInputDesc("bias", biasTensorDesc);

  op.SetAttr("scale_from_blob", true);
  op.SetAttr("axis", 0);
  op.SetAttr("num_axes", 3);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ScaleTest, scale_test_infershape_test_2) {
  ge::op::Scale op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 10},{3,10},{4,10}};
  auto tensor_desc_x = create_desc_shape_range({-1,-1,-1},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {2,3,4},
                                               ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("scale", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("axis", 0);
  op.SetAttr("num_axes", 3);
  op.SetAttr("bias_from_blob", true);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,-1,-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  // std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  // EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  // std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    // {2, 10},{3,10},{4,10}
  // };
  
  // EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ScaleTest, scale_test_infershape_test_3) {
  ge::op::Scale op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 10},{3,10},{4,10}};
  auto tensor_desc_x = create_desc_shape_range({-1,-1,-1},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {2,3,4},
                                               ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc_x);
  op.UpdateInputDesc("scale", tensor_desc_x);
  op.UpdateInputDesc("bias", tensor_desc_x);
  op.SetAttr("axis", 0);
  op.SetAttr("num_axes", 0);
  op.SetAttr("bias_from_blob", true);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,-1,-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  // std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  // EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  // std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    // {2, 10},{3,10},{4,10}
  // };
  
  // EXPECT_EQ(output_shape_range, expected_shape_range);
}