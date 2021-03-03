#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "nn_ops.h"

class FusedBatchNormV2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FusedBatchNormV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FusedBatchNormV2 TearDown" << std::endl;
  }
};

TEST_F(FusedBatchNormV2, FusedBatchNormV2_infer_shape_1) {
  ge::op::FusedBatchNormV2 op; 
  op.SetAttr("is_training", true);
  op.SetAttr("data_format", "NCHW");                                                                                                                                                                                  
  op.UpdateInputDesc("x", create_desc({2, 64, 224, 224}, ge::DT_FLOAT));
  op.UpdateInputDesc("scale", create_desc({64},ge::DT_FLOAT));
  op.UpdateInputDesc("offset", create_desc({64},ge::DT_FLOAT)); 
  op.UpdateInputDesc("mean", create_desc({64},ge::DT_FLOAT));
  op.UpdateInputDesc("variance", create_desc({64},ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2, 64, 224, 224};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}