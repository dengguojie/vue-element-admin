#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"

class BiasAddGradProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BiasAddGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BiasAddGrad Proto Test TearDown" << std::endl;
  }
};

TEST_F(BiasAddGradProtoTest, BiasAddGrad_infershape_diff_test){
  ge::op::BiasAddGrad op;
  op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT16));
  op.SetAttr("data_format", "NHWC");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(BiasAddGradProtoTest, BiasAddGrad_infershape_same_test){
  ge::op::BiasAddGrad op;
  op.UpdateInputDesc("x", create_desc({5, 3, 4}, ge::DT_FLOAT16));

  op.SetAttr("data_format", "NHWC");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(BiasAddGradProtoTest, BiasAddGrad_infershape_dynamic_test){
  ge::op::BiasAddGrad op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{10, 10}, {2, 100}, {4, 4}, {5, 5}};
  auto tensor_desc = create_desc_shape_range({-1, -1, 4, 5},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {10, 12, 4, 5},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc);

  op.SetAttr("data_format", "NCHW");
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 100},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);

}

