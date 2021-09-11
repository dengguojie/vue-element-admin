//
// Created by c30002892 on 2020/9/5.
//

#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class remove_node_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "inplace_add TearDown" << std::endl;
    }

};

TEST_F(remove_node_fusion_pass_test, remove_node_fusion_pass_test_1) {

  ge::Graph graph("remove_node_fusion_pass_test_1");
  auto input_data = op::Data("input_data");
  std::vector<int64_t> dims{1, 0};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape);
  tensorDesc.SetOriginShape(shape);
  input_data.update_input_desc_x(tensorDesc);
  input_data.update_output_desc_y(tensorDesc);

  auto input_data_indices = op::Data("input_data_indices");
  std::vector<int64_t> dims_indices{1, 0};
  ge::Shape shape_indices(dims_indices);
  ge::TensorDesc tensorDescIndices(shape);
  tensorDescIndices.SetOriginShape(shape);
  input_data_indices.update_input_desc_x(tensorDescIndices);
  input_data_indices.update_output_desc_y(tensorDescIndices);

  auto gather_nd_op = op::GatherNd("gather_nd_0");
  gather_nd_op.set_input_x(input_data);
  gather_nd_op.set_input_indices(input_data_indices);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(gather_nd_op);

  std::vector<Operator> inputs{input_data, input_data_indices};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ARemoveNodeFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
  std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
  fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
  fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
  EXPECT_EQ(graphFusionInfoMap["ARemoveNodeFusionPass"].GetMatchTimes(), 1);
  EXPECT_EQ(graphFusionInfoMap["ARemoveNodeFusionPass"].GetEffectTimes(), 1);
}
