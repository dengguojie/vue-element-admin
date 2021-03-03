#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class pow : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "pow SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "pow TearDown" << std::endl;
  }
};

TEST_F(pow, pow_infershape_diff_test) {
  ge::op::Pow op;
  op.UpdateInputDesc("x1", create_desc({4, 3, 4}, ge::DT_INT8));
  op.UpdateInputDesc("x2", create_desc({4, 3, 4}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT8);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(pow, pow_infershape_same_test) {
  ge::op::Pow op;
  op.UpdateInputDesc("x1", create_desc({1, 3, 4}, ge::DT_INT8));
  op.UpdateInputDesc("x2", create_desc({1, 3, 4}, ge::DT_INT8));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT8);
  std::vector<int64_t> expected_output_shape = {1, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(pow, pow_dynamic_infershape) {
  ge::op::Pow op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16}, {1, 16}};

  auto tensor_desc = create_desc_shape_range({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {16, 16},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x1", tensor_desc);
  op.UpdateInputDesc("x2", tensor_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1,16},{1,16}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
