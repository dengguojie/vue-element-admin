#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "randomdsa_ops.h"
using namespace ge;
using namespace op;

class DsaRandomTruncatedNormalTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DsaRandomTruncatedNormal test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DsaRandomTruncatedNormal test TearDown" << std::endl;
  }
};

TEST_F(DsaRandomTruncatedNormalTest, infer_shape_00) {
  ge::op::DSARandomTruncatedNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DsaRandomTruncatedNormalTest, infer_shape_01) {
  ge::op::DSARandomTruncatedNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({10, 10}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DsaRandomTruncatedNormalTest, infer_shape_02) {
  ge::op::DSARandomTruncatedNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({10, 10}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DsaRandomTruncatedNormalTest, infer_shape_03) {
  ge::op::DSARandomTruncatedNormal op;
  op.UpdateInputDesc("count", create_desc({10, 10}, ge::DT_INT64));
  op.UpdateInputDesc("mean", create_desc({1}, ge::DT_FLOAT));
  op.UpdateInputDesc("stdev", create_desc({1}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DsaRandomTruncatedNormalTest, infer_shape_04) {
  ge::op::DSARandomTruncatedNormal op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}};
  auto input_desc = create_desc_shape_range({2}, ge::DT_INT64, ge::FORMAT_NCHW,
                                            {2}, ge::FORMAT_NCHW, range_x1);
  op.UpdateInputDesc("count", input_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("out");
  std::vector<int64_t> expect_output_shape = {2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expect_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(DsaRandomTruncatedNormalTest, infer_shape_05) {
  ge::op::DSARandomTruncatedNormal op;
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{30, 30}};
  auto input_desc = create_desc_shape_range({-1}, ge::DT_INT64, ge::FORMAT_NCHW,
                                            {-1}, ge::FORMAT_NCHW, range_x1);
  op.UpdateInputDesc("count", input_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("out");
  std::vector<int64_t> expect_output_shape = {-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expect_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{30,30}};
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

