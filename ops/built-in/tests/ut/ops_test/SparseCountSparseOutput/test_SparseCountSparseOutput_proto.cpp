#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "math_ops.h"

class SparseCountSparseOutputTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseCountSparseOutputTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseCountSparseOutputTest TearDown" << std::endl;
  }
};

//success: case1
TEST_F(SparseCountSparseOutputTest, SUCCESS_1) {
  ge::op::SparseCountSparseOutput op;
  op.UpdateInputDesc("indices", create_desc({4, 2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({4}, ge::DT_INT64));
  op.UpdateInputDesc("dense_shape", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({4}, ge::DT_INT64));

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
TEST_F(SparseCountSparseOutputTest, SUCCESS_2) {
  ge::op::SparseCountSparseOutput op;
  op.UpdateInputDesc("indices", create_desc({4, 2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({4}, ge::DT_INT64));
  op.UpdateInputDesc("dense_shape", create_desc({2}, ge::DT_INT64));
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

//error: indices shape
TEST_F(SparseCountSparseOutputTest, FAIL_1) {
  ge::op::SparseCountSparseOutput op;
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({4}, ge::DT_INT64));
  op.UpdateInputDesc("dense_shape", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error: indices dtype
TEST_F(SparseCountSparseOutputTest, FAIL_2) {
  ge::op::SparseCountSparseOutput op;
  op.UpdateInputDesc("indices", create_desc({4, 2}, ge::DT_FLOAT));
  op.UpdateInputDesc("values", create_desc({4}, ge::DT_INT64));
  op.UpdateInputDesc("dense_shape", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({4}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error: values dtype
TEST_F(SparseCountSparseOutputTest, FAIL_3) {
  ge::op::SparseCountSparseOutput op;
  op.UpdateInputDesc("indices", create_desc({4, 2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({4}, ge::DT_FLOAT));
  op.UpdateInputDesc("dense_shape", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({4}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error: dense_shape dtype
TEST_F(SparseCountSparseOutputTest, FAIL_4) {
  ge::op::SparseCountSparseOutput op;
  op.UpdateInputDesc("indices", create_desc({4, 2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({4}, ge::DT_INT64));
  op.UpdateInputDesc("dense_shape", create_desc({2}, ge::DT_FLOAT));
  op.UpdateInputDesc("weights", create_desc({4}, ge::DT_INT64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//error: weighs dtype
TEST_F(SparseCountSparseOutputTest, FAIL_5) {
  ge::op::SparseCountSparseOutput op;
  op.UpdateInputDesc("indices", create_desc({4, 2}, ge::DT_INT64));
  op.UpdateInputDesc("values", create_desc({4}, ge::DT_INT64));
  op.UpdateInputDesc("dense_shape", create_desc({2}, ge::DT_INT64));
  op.UpdateInputDesc("weights", create_desc({4}, ge::DT_COMPLEX64));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
