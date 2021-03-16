#include <gtest/gtest.h>

#include <iostream>

#include "array_ops.h"
#include "op_proto_test_util.h"
#include "transformation_ops.h"
#include "rnn.h"

class EmbeddingBagTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "EmbeddingBagTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "EmbeddingBagTest TearDown" << std::endl;
  }
};

TEST_F(EmbeddingBagTest, embedding_bag_test_case_1) {
  ge::op::EmbeddingBag op;

    op.UpdateInputDesc(
        "weight", create_desc_with_ori({100,3}, ge::DT_FLOAT, ge::FORMAT_ND,
                                       {100,3}, ge::FORMAT_ND));
    op.UpdateInputDesc(
        "indices", create_desc_with_ori({10}, ge::DT_INT32, ge::FORMAT_ND,
                                      {10}, ge::FORMAT_ND));
    op.UpdateInputDesc(
        "offsets", create_desc_with_ori({4}, ge::DT_INT32, ge::FORMAT_ND,
                                      {4}, ge::FORMAT_ND));

    // include_last_offset attr
    op.SetAttr("include_last_offset", false);

    // run case
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

   // test output shape
    auto out_var_desc = op.GetOutputDesc("y");
    std::vector<int64_t> expected_var_output_shape = {4,3};
    EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);


}
