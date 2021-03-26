#include <gtest/gtest.h>

#include <iostream>

#include "array_ops.h"
#include "op_proto_test_util.h"
#include "transformation_ops.h"
#include "nn_pooling_ops.h"


class SubSampleLabelsTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SubSampleLabelsTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SubSampleLabelsTest TearDown" << std::endl;
  }
};

TEST_F(SubSampleLabelsTest, sub_sample_test_case_1) {
  ge::op::SubSampleLabels op;

    op.UpdateInputDesc(
        "labels", create_desc_with_ori({41153}, ge::DT_INT32, ge::FORMAT_ND,
                                      {41153}, ge::FORMAT_ND));
    op.UpdateInputDesc(
        "shuffle_matrix", create_desc_with_ori({41153}, ge::DT_INT32, ge::FORMAT_ND,
                                      {41153}, ge::FORMAT_ND));


    // run case
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

   // test output shape
    auto out_var_desc = op.GetOutputDesc("y");
    std::vector<int64_t> expected_var_output_shape = {41153};
    EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);


}
