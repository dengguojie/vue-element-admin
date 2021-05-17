#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "linalg_ops.h"

class LogMatrixDeterminantTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "log_matrix_determinant_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "log_matrix_determinant_test TearDown" << std::endl;
  }
};

TEST_F(LogMatrixDeterminantTest, log_matrix_determinant_test_1) {
  //new op
  ge::op::LogMatrixDeterminant op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("x", tensor_desc_x);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(LogMatrixDeterminantTest, log_matrix_determinant_test_2) {
  //new op
  ge::op::LogMatrixDeterminant op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 2, 3}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("x", tensor_desc_x);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}