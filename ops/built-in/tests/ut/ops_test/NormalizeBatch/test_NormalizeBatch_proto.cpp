#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_norm_ops.h"

class NormalizeBatch : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NormalizeBatch Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NormalizeBatch Proto Test TearDown" << std::endl;
  }
};

TEST_F(NormalizeBatch, normalize_batch_infershape_test_1){
  ge::op::NormalizeBatch op;
  op.UpdateInputDesc("input_x", create_desc({3, 10, 1024}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("output_y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {3, 10, 1024};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(NormalizeBatch, normalize_batch_infershape_test_2){
  ge::op::NormalizeBatch op;
  op.UpdateInputDesc("input_x", create_desc({3, 10}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}