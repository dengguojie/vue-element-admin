#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "math_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class zeta_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "zeta_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "zeta_infer_test TearDown" << std::endl;
  }
};

// input dimension is const
TEST_F(zeta_infer_test, zeta_infer_test_1) {
  //new op
  ge::op::Zeta op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_x.SetOriginShape(ge::Shape({1, 5}));
  op.UpdateInputDesc("x", tensor_desc_x);
  ge::TensorDesc tensor_desc_q(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_q.SetOriginShape(ge::Shape({1, 5}));
  op.UpdateInputDesc("q", tensor_desc_q);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}