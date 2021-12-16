#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

// ----------------Sort-------------------
class SortProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Sort Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Sort Proto Test TearDown" << std::endl;
  }
};


TEST_F(SortProtoTest, sort_infer_shape_test) {
  ge::op::Sort op;

  op.UpdateInputDesc("x", create_desc_shape_range({10,10,10,32},
                    ge::DT_FLOAT16, ge::FORMAT_ND, {10,10,10,32},
                     ge::FORMAT_ND, {{10,10},{10,10},{10,10},{32,32}}));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//  get sorted output shape
  auto sorted_output_desc = op.GetOutputDesc("y1");
  EXPECT_EQ(sorted_output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {10,10,10,32};
  EXPECT_EQ(sorted_output_desc.GetShape().GetDims(), expected_output_shape);

//  check sorted output range
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{10,10},{10,10},{10,10},{32,32}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  sorted_output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);

//  get indices output shape
  auto indices_output_desc = op.GetOutputDesc("y2");
  EXPECT_EQ(indices_output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_indices_shape = {10,10,10,32};
  EXPECT_EQ(indices_output_desc.GetShape().GetDims(), expected_indices_shape);

//  check indices range
  std::vector<std::pair<int64_t, int64_t>> expected_indices_shape_range = {{10,10},{10,10},{10,10},{32,32}};
  std::vector<std::pair<int64_t, int64_t>> indices_shape_range;
  indices_output_desc.GetShapeRange(indices_shape_range);
  EXPECT_EQ(indices_shape_range, expected_indices_shape_range);
}
