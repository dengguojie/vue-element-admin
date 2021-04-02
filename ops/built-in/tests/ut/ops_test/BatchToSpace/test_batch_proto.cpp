#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "transformation_ops.h"
#include "batch_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class batch_infer_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "batch_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "batch_infer_test TearDown" << std::endl;
  }
};

TEST_F(batch_infer_test, batch_infer_test_1) {


  // new op and do infershape
  ge::op::Batch op;
  op.create_dynamic_input_x_tensors(1);
  op.UpdateDynamicInputDesc("x_tensors", 0, create_desc({}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
