#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "rnn.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class non_zero_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "non_zero_fusion SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "non_zero_fusion TearDown" << std::endl;
    }
};

TEST_F(non_zero_fusion_test, non_zero_fusion_test_1) {

  //x
  auto dataX = op::Data("x");
  std::vector<int64_t> dataXVec{1000, 21128};
  ge::Shape dataXShape(dataXVec);
  ge::TensorDesc dataXDesc(dataXShape, FORMAT_ND, DT_FLOAT);
  dataX.update_input_desc_x(dataXDesc);
  dataX.update_output_desc_y(dataXDesc);
  
  auto nonZeroOp = op::NonZero("NonZero");
  nonZeroOp.set_input_x(dataX)
             .set_attr_dtype(DT_INT64);

  auto castOp = op::Cast("Cast");
  castOp.set_input_x(nonZeroOp)
             .set_attr_dst_type(DT_INT32);

  std::vector<Operator> inputs{dataX};

  std::vector<Operator> outputs{castOp};
  ge::Graph graph("non_zero_fusion_test_1");
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr computeGraphPtr = ge::GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(computeGraphPtr);

  fe::FusionPassTestUtils::RunGraphFusionPass("NonZeroFusionPass", fe::BUILT_IN_GRAPH_PASS, *computeGraphPtr);

  bool findNonZero = false;
  for (auto node: computeGraphPtr->GetAllNodes()) {
    if (node->GetType() == "NonZero") {
      findNonZero = true;
    }
  }

  EXPECT_EQ(findNonZero, true);
}
