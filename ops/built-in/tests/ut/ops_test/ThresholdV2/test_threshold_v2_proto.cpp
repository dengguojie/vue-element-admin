#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "nonlinear_fuc_ops.h"

class ThresholdV2Test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "threshold_v2 test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "threshold_v2 test TearDown" << std::endl;
  }
};

TEST_F(ThresholdV2Test, threshold_v2_test_case_f16) {

  ge::op::ThresholdV2 threshold_v2_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({2, 3, 4});
  
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(ge::FORMAT_ND);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ge::FORMAT_ND);
  
  threshold_v2_op.UpdateInputDesc("x", tensorDesc);

  auto ret = threshold_v2_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc = threshold_v2_op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  
  std::vector<int64_t> expected_output_shape = {2, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ThresholdV2Test, threshold_v2_test_case_f32) {

  ge::op::ThresholdV2 threshold_v2_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({-1, -1});
  ge::Shape ori_shape({3, 4});
  
  tensorDesc.SetDataType(ge::DT_FLOAT);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(ge::FORMAT_ND);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ge::FORMAT_ND);
  tensorDesc.SetShapeRange({{3, 3}, {4, 4}});
  
  threshold_v2_op.UpdateInputDesc("x", tensorDesc);
  
  auto ret = threshold_v2_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc = threshold_v2_op.GetOutputDescByName("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}

TEST_F(ThresholdV2Test, threshold_v2_test_case_verify) {

  ge::op::ThresholdV2 threshold_v2_op;
  ge::TensorDesc tensorDesc;
  
  tensorDesc.SetDataType(ge::DT_FLOAT);
  threshold_v2_op.UpdateInputDesc("x", tensorDesc);

  tensorDesc.SetDataType(ge::DT_FLOAT16);
  threshold_v2_op.UpdateInputDesc("threshold", tensorDesc);
  
  auto ret = threshold_v2_op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}