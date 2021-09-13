#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "random_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;
class continuation_indicator_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "continuation_indicator_fusion_test SetUp" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "continuation_indicator_fusion_test TearDown" << std::endl;
  }
};

TEST_F(continuation_indicator_fusion_test, continuation_indicator_fusion_test_ok1) {
  ge::Graph graph("continuation_indicator_fusion_test_ok1");
  int64_t time_step = 10;
  int64_t batch_size = 10;
  auto continuation_indicator = op::ContinuationIndicator("continuation_indicator");
  continuation_indicator.set_attr_time_step(time_step);
  continuation_indicator.set_attr_batch_size(batch_size);
  std::vector<Operator> inputs{continuation_indicator};
  std::vector<Operator> outputs{continuation_indicator};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ContinuationIndicatorFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool find_node = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ContinuationIndicator") {
      find_node = true;
    }
  }
  EXPECT_EQ(find_node, false);
}
