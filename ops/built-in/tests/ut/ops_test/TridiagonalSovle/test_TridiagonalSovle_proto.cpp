#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "linalg_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class tridiagonal_solve_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "tridiagonal_solve_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tridiagonal_solve_infer_test TearDown" << std::endl;
  }
};

// input dimension is const
TEST_F(tridiagonal_solve_infer_test, tridiagonal_solve_infer_test_1) {
  //new op
  ge::op::TridiagonalSolve op;
  // set input info
  ge::TensorDesc tensor_desc_diagonals(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("diagonals", tensor_desc_diagonals);
  ge::TensorDesc tensor_desc_rhs(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input dimension is const
TEST_F(tridiagonal_solve_infer_test, tridiagonal_solve_infer_test_2) {
  //new op
  ge::op::TridiagonalSolve op;
  // set input info
  ge::TensorDesc tensor_desc_diagonals(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("diagonals", tensor_desc_diagonals);
  ge::TensorDesc tensor_desc_rhs(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input dimension is const
TEST_F(tridiagonal_solve_infer_test, tridiagonal_solve_infer_test_3) {
  //new op
  ge::op::TridiagonalSolve op;
  // set input info
  ge::TensorDesc tensor_desc_diagonals(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("diagonals", tensor_desc_diagonals);
  ge::TensorDesc tensor_desc_rhs(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// input dimension is const
TEST_F(tridiagonal_solve_infer_test, tridiagonal_solve_infer_test_4) {
  //new op
  ge::op::TridiagonalSolve op;
  // set input info
  ge::TensorDesc tensor_desc_diagonals(ge::Shape({2, 8, 7}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("diagonals", tensor_desc_diagonals);
  ge::TensorDesc tensor_desc_rhs(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("rhs", tensor_desc_rhs);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}