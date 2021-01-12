#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_detect_ops.h"

class NonMaxSuppressionV7Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NonMaxSuppressionV7 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NonMaxSuppressionV7 TearDown" << std::endl;
  }
};

TEST_F(NonMaxSuppressionV7Test, non_max_suppression_v7_test_case_1) {
//  int64_t batchSize = 1;
//  int64_t outputNum = 100;
  ge::op::BatchMultiClassNonMaxSuppression op;
  op.UpdateInputDesc("boxes", create_desc_with_ori({2, 6, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 6, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("scores", create_desc_with_ori({2, 1, 6}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 1, 6}, ge::FORMAT_ND));


  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

//  auto out_var_desc = op.GetOutputDesc("selected_indices");
//  std::vector<int64_t> expected_var_output_shape = {12, 3};
//  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_INT32);
//  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);


}

