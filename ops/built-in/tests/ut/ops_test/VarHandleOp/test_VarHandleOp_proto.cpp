#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "resource_variable_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class var_handle_op_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "var_handle_op_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "var_handle_op_infer_test TearDown" << std::endl;
  }
};

TEST_F(var_handle_op_infer_test, var_handle_op_infer_test_2) {
  //new op
  ge::op::VarHandleOp op;
  // set input info
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}