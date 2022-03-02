#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_ops.h"

class MultiMerge : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MultiMerge Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MultiMerge Proto Test TearDown" << std::endl;
  }
};

TEST_F(MultiMerge, multi_merge_infershape_test_1){
  ge::op::MultiMerge op;
  op.UpdateInputDesc("input_proposal", create_desc({32, 3277584, 8}, ge::DT_FLOAT16));
  op.SetAttr("k_num", 1808412);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("output_proposal");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {8, 1808432, 8};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  auto output_desc_1 = op.GetOutputDesc("output_index");
  EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_index_shape = {1, };
  EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_index_shape);
}

TEST_F(MultiMerge, multi_merge_infershape_test_2){
  ge::op::MultiMerge op;
  op.UpdateInputDesc("input_proposal", create_desc({2, 1808432, 8}, ge::DT_FLOAT16));
  op.SetAttr("k_num", 1808412);
  op.SetAttr("include_index", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expected_output_shape = {1808412};
  auto output_desc_0 = op.GetOutputDesc("output_proposal");
  EXPECT_EQ(output_desc_0.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_0.GetShape().GetDims(), expected_output_shape);

  auto output_desc_1 = op.GetOutputDesc("output_index");
  EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);

}
