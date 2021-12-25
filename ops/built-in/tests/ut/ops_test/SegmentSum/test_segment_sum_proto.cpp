#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"

class segment_sum : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "segment_sum SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "segment_sum TearDown" << std::endl;
  }
};

TEST_F(segment_sum, segment_sum_infer_shape_test_1) {
  ge::op::SegmentSum op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{9, 9},{10, 10}};
  auto tensor_x_desc = create_desc_shape_range({-1, 10},
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               {9, 10},
                                               ge::FORMAT_ND, shape_range);
  auto tensor_segment_ids_desc = create_desc_shape_range({9},
                                                         ge::DT_INT32, ge::FORMAT_ND,
                                                         {9},
                                                         ge::FORMAT_ND, {{9, 9}});
  op.UpdateInputDesc("x", tensor_x_desc);
  op.UpdateInputDesc("segment_ids", tensor_segment_ids_desc);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, 10};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
      {0, -1}, {10, 10},
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}