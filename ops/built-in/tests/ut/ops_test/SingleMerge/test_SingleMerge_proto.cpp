#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_ops.h"

class SingleMerge : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SingleMerge Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SingleMerge Proto Test TearDown" << std::endl;
  }
};

TEST_F(SingleMerge, single_merge_infershape_test_1){
  ge::op::SingleMerge op;
  op.UpdateInputDesc("input_proposal", create_desc({2, 1808432, 8}, ge::DT_FLOAT16));
  op.SetAttr("k_num", 1808412);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expected_output_shape = {1808412};
  auto output_desc_0 = op.GetOutputDesc("output_data");
  EXPECT_EQ(output_desc_0.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_0.GetShape().GetDims(), expected_output_shape);

  auto output_desc_1 = op.GetOutputDesc("output_index");
  EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);

}
