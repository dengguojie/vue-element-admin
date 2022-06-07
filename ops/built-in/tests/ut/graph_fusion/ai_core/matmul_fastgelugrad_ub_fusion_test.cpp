#include "gtest/gtest.h"
#include "array_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"
#include "fusion_pass_test_utils.h"
#include "nonlinear_fuc_ops.h"
#define private public
#define protected public

using namespace ge;
using namespace op;

class matmul_fastgelugrad_ub_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "matmul_fastgelugrad_ub_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "matmul_fastgelugrad_ub_fusion_test TearDown" << std::endl;
  }
};

namespace fe {
Status RunBufferFusionPassMatmulConfusiontransposeFusion(string fusionPassName, BufferFusionPassType passType,
                                         ge::ComputeGraphPtr &compute_graph_ptr) {
  std::map<string, BufferFusionPassRegistry::CreateFn> createFns =
      BufferFusionPassRegistry::GetInstance().GetCreateFnByType(passType);
  const auto &iter = createFns.find(fusionPassName);
  if (iter != createFns.end()) {
    if (passType == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
      auto BufferFusionPassBasePtr =
          std::unique_ptr<BufferFusionPassBase>(dynamic_cast<BufferFusionPassBase *>(iter->second()));
      if (BufferFusionPassBasePtr == nullptr) {
        return FAILED;
      }
      ge::ComputeGraph::Vistor<ge::NodePtr> nodes = compute_graph_ptr->GetAllNodes();
      std::vector<ge::NodePtr> Node_v(nodes.begin(), nodes.end());

      BufferFusionPassBasePtr->SetName(fusionPassName);
      vector<BufferFusionPattern *> patterns = BufferFusionPassBasePtr->DefinePatterns();

      for (auto pattern : patterns) {
        std::vector<BufferFusionOpDesc *> node_descs = pattern->GetOpDescs();

        vector<ge::NodePtr> matmul_nodes;
        vector<ge::NodePtr> elemwise_nodes;
        vector<ge::NodePtr> input_nodes;
        vector<ge::NodePtr> output_nodes;
        for (auto node : nodes) {
          auto op_desc = node->GetOpDesc();
          if (op_desc->GetType() == "MatMulV2" || op_desc->GetType() == "MatMul") {
            matmul_nodes.push_back(node);
          } else if (op_desc->GetType() == "FastGeluGrad") {
            elemwise_nodes.push_back(node);
          }
        }

        BufferFusionMapping mapping;
        for (auto node_desc : node_descs) {
          if (node_desc->desc_name == "matmul") {
            mapping[node_desc] = matmul_nodes;
          }
          if (node_desc->desc_name == "eltwise") {
            mapping[node_desc] = elemwise_nodes;
          }
          if (node_desc->desc_name == "otherInput1") {
            mapping[node_desc] = input_nodes;
          }
          if (node_desc->desc_name == "otherInput2") {
            mapping[node_desc] = input_nodes;
          }
          if (node_desc->desc_name == "OUTPUT1") {
            mapping[node_desc] = output_nodes;
          }
          if (node_desc->desc_name == "OUTPUT2") {
            mapping[node_desc] = output_nodes;
          }
        }
        vector<ge::NodePtr> fusion_nodes;
        BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
        if (fusion_nodes.empty()) {
          return FAILED;
        }
      }
      return SUCCESS;
    }
  }

  return FAILED;
}
}  // namespace fe

TEST_F(matmul_fastgelugrad_ub_fusion_test, matmul_fastgelugrad_ub_fusion_test_1) {
  ge::Graph graph("matmul_fastgelugrad_ub_fusion_test_1");
  auto X1Data = op::Data("x1");
  std::vector<int64_t> dims_x1{16, 16};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
  X1Data.update_input_desc_x(tensorDescX1);
  X1Data.update_output_desc_y(tensorDescX1);

  auto X2Data = op::Data("x2");
  std::vector<int64_t> dims_x2{16, 16};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
  X2Data.update_input_desc_x(tensorDescX2);
  X2Data.update_output_desc_y(tensorDescX2);

  auto X3Data = op::Data("x3");
  std::vector<int64_t> dims_x3{16, 16};
  ge::Shape shape_x3(dims_x3);
  ge::TensorDesc tensorDescX3(shape_x3, FORMAT_ND, DT_FLOAT16);
  X3Data.update_input_desc_x(tensorDescX3);
  X3Data.update_output_desc_y(tensorDescX3);

  auto bmOP = op::MatMulV2("matmul");
  bmOP.set_input_x1(X1Data);
  bmOP.set_input_x2(X2Data);

  auto fastgelugrad_op = op::FastGeluGrad("matmul_transpose");
  fastgelugrad_op.set_input_dy(bmOP);
  fastgelugrad_op.set_input_x(X3Data);

  std::vector<Operator> inputs{X1Data, X2Data, X3Data};
  std::vector<Operator> outputs{fastgelugrad_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  Status res = fe::RunBufferFusionPassMatmulConfusiontransposeFusion("MatmulFastGelugradUbFusion",
                                                     fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
  EXPECT_EQ(res, fe::SUCCESS);
}
