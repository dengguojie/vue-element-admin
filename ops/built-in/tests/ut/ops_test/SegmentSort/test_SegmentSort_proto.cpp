#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_ops.h"

class SegmentSort : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SegmentSort Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SegmentSort Proto Test TearDown" << std::endl;
  }
};

TEST_F(SegmentSort, segment_sort_infershape_test_1){
  ge::op::SegmentSort op;
  op.UpdateInputDesc("input_data", create_desc({104857600,}, ge::DT_FLOAT16));
  op.UpdateInputDesc("input_index", create_desc({2048,}, ge::DT_FLOAT16));
  op.SetAttr("k_num", 1808412);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("output_proposal");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {32, 3277584, 8};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
