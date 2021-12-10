#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "reduce_ops.h"

class ReduceLogSumExp : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ReduceLogSumExp SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ReduceLogSumExp TearDown" << std::endl;
  }
};

TEST_F(ReduceLogSumExp, reduce_log_sum_exp_infer_shape_float16_int32_true) {
  ge::op::ReduceLogSumExp op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({-1, 100, 4},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  auto axes_desc = create_desc_shape_range({2,},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {2,},
                                           ge::FORMAT_ND, {{2, 2},});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    {0, 2},
    {0, 200},
    {0, 8}
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(ReduceLogSumExp, reduce_log_sum_exp_infer_shape_float32_int64_false) {
  ge::op::ReduceLogSumExp op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}, {100, 200}, {4, 8}};
  auto tensor_desc = create_desc_shape_range({-1, 100, 4},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {2, 100, 4},
                                             ge::FORMAT_ND, shape_range);
  auto axes_desc = create_desc_shape_range({2,},
                                           ge::DT_INT64, ge::FORMAT_ND,
                                           {2,},
                                           ge::FORMAT_ND, {{2, 2},});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("axes", axes_desc);
  op.SetAttr("keep_dims", false);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {-1, };
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
    {2, 200}
  };
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

