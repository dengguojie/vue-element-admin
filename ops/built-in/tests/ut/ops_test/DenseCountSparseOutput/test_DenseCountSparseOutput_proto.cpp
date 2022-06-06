#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"

class DenseCountSparseOutputTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DenseCountSparseOutputTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DenseCountSparseOutputTest TearDown" << std::endl;
  }
};

//success: case1
TEST_F(DenseCountSparseOutputTest, SUCCESS_1) {
  ge::op::DenseCountSparseOutput op;
  op.UpdateInputDesc("values", create_desc({2, 2}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({2, 2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto indices_desc = op.GetOutputDesc("output_indices");
  EXPECT_EQ(indices_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_indices_shape = {-1, 2};
  EXPECT_EQ(indices_desc.GetShape().GetDims(), expected_indices_shape);

  auto values_desc = op.GetOutputDesc("output_values");
  EXPECT_EQ(values_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_values_shape = {-1};
  EXPECT_EQ(values_desc.GetShape().GetDims(), expected_values_shape);

  auto shape_desc = op.GetOutputDesc("output_dense_shape");
  EXPECT_EQ(shape_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_shape = {2};
  EXPECT_EQ(shape_desc.GetShape().GetDims(), expected_shape);
}

//success: case2
TEST_F(DenseCountSparseOutputTest, SUCCESS_2) {
  ge::op::DenseCountSparseOutput op;
  op.UpdateInputDesc("values", create_desc({2, 2}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto indices_desc = op.GetOutputDesc("output_indices");
  EXPECT_EQ(indices_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_indices_shape = {-1, 2};
  EXPECT_EQ(indices_desc.GetShape().GetDims(), expected_indices_shape);

  auto values_desc = op.GetOutputDesc("output_values");
  EXPECT_EQ(values_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_values_shape = {-1};
  EXPECT_EQ(values_desc.GetShape().GetDims(), expected_values_shape);

  auto shape_desc = op.GetOutputDesc("output_dense_shape");
  EXPECT_EQ(shape_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_shape = {2};
  EXPECT_EQ(shape_desc.GetShape().GetDims(), expected_shape);
}

//error: values shape over
TEST_F(DenseCountSparseOutputTest, FAIL_1) {
  ge::op::DenseCountSparseOutput op;
  op.UpdateInputDesc("values", create_desc({2, 2, 2}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({2, 2, 2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error: values shape less
TEST_F(DenseCountSparseOutputTest, FAIL_2) {
  ge::op::DenseCountSparseOutput op;
  op.UpdateInputDesc("values", create_desc({}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error: weights shape
TEST_F(DenseCountSparseOutputTest, FAIL_3) {
  ge::op::DenseCountSparseOutput op;
  op.UpdateInputDesc("values", create_desc({2, 2}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error: values dtype
TEST_F(DenseCountSparseOutputTest, FAIL_4) {
  ge::op::DenseCountSparseOutput op;
  op.UpdateInputDesc("values", create_desc({2, 2}, ge::DT_FLOAT));
  op.UpdateInputDesc("weights", create_desc({2, 2}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error: weights dtype
TEST_F(DenseCountSparseOutputTest, FAIL_5) {
  ge::op::DenseCountSparseOutput op;
  op.UpdateInputDesc("values", create_desc({2, 2}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({2, 2}, ge::DT_COMPLEX64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
