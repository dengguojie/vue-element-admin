#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "transformation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class flatten_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "flattenv2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "flattenv2 TearDown" << std::endl;
  }
};

TEST_F(flatten_fusion_test, flatten_fusion_test_1) {
  ge::Graph graph("flatten_fusion_test_1");
  // set input data 1
  auto flatten_input_data = op::Data("flatten_input_data");
  std::vector<int64_t> dims{10, 11, 20, 16};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, FORMAT_NHWC, DT_FLOAT16);
  flatten_input_data.update_input_desc_x(tensorDesc);
  flatten_input_data.update_output_desc_y(tensorDesc);

  // set fusion
  auto flatten_op = op::FlattenV2("flatten").set_input_x(flatten_input_data);
  // set graph and res
  std::vector<Operator> inputs{flatten_input_data};
  std::vector<Operator> outputs{flatten_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("FlattenV2Pass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findFlatten = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "FlattenV2") {
      findFlatten = true;
      break;
    }
  }
  EXPECT_EQ(findFlatten, false);
}

