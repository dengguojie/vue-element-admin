#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"

using namespace ge;
using namespace op;

class matmul_confusiontranspose_ub_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "matmul_confusiontranspose_ub_fusion_test SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "matmul_confusiontranspose_ub_fusion_test TearDown" << std::endl;
  }
};

namespace fe {
Status RunBufferFusionPassBase(string fusionPassName, BufferFusionPassType passType, ge::ComputeGraph &computeGraph) {
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
      BufferFusionPassBasePtr->SetName(fusionPassName);
      vector<BufferFusionPattern *> patterns = BufferFusionPassBasePtr->DefinePatterns();

      std::vector<BufferFusionOpDesc *> desc = patterns[0]->GetOpDescs();

      ge::ComputeGraph::Vistor<ge::NodePtr> NodePtrs = computeGraph.GetAllNodes();
      std::vector<ge::NodePtr> Node_v(NodePtrs.begin(), NodePtrs.end());

      vector<ge::NodePtr> ctNodes;
      vector<ge::NodePtr> mmNodes;
      for (auto i : NodePtrs) {
        auto opDesc = i->GetOpDesc();
        if (opDesc->GetType() == "MatMul") {
          std::cout << "111222" << std::endl;
          mmNodes.push_back(i);
        }
        if (opDesc->GetType() == "ConfusionTransposeD") {
          ctNodes.push_back(i);
        }
      }

      BufferFusionMapping mapping;
      for (auto i : desc) {
        std::cout << "!!!!!!" << i->desc_name << std::endl;
        if (i->desc_name == "matmul") {
          mapping[i] = mmNodes;
        }
        if (i->desc_name == "matmul_transpose") {
          mapping[i] = ctNodes;
        }
      }
      vector<ge::NodePtr> fusion_nodes;
      BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
      return SUCCESS;
    }
  }

  return FAILED;
}
}  // namespace fe
TEST_F(matmul_confusiontranspose_ub_fusion_test, matmul_confusiontranspose_ub_fusion_test_1) {
  ge::Graph graph("matmul_confusiontranspose_ub_fusion_test_1");

  // create matmul
  auto nz_shape_x1 = vector<int64_t>({4, 4, 16, 16});
  ge::TensorDesc desc_x1(ge::Shape(nz_shape_x1), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_x1 = op::Data("data_x1");
  data_x1.update_input_desc_x(desc_x1);
  data_x1.update_output_desc_y(desc_x1);

  auto nz_shape_x2 = vector<int64_t>({4, 4, 16, 16});
  ge::TensorDesc desc_x2(ge::Shape(nz_shape_x2), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_x2 = op::Data("data_x2");
  data_x2.update_input_desc_x(desc_x2);
  data_x2.update_output_desc_y(desc_x2);

  auto nz_shape_y_mm = vector<int64_t>({4, 4, 16, 16});

  auto matmul = op::MatMul("MatMul").set_input_x1(data_x1).set_input_x2(data_x2);
  TensorDesc matmul_input_desc_x1(ge::Shape(nz_shape_x1), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  TensorDesc matmul_input_desc_x2(ge::Shape(nz_shape_x2), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  TensorDesc matmul_output_desc_y(ge::Shape(nz_shape_y_mm), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  matmul.update_input_desc_x1(matmul_input_desc_x1);
  matmul.update_input_desc_x2(matmul_input_desc_x2);
  matmul.update_output_desc_y(matmul_output_desc_y);
  std::cout << "********" << std::endl;

  // create confusiontranspose
  auto nz_shape_y_ct = vector<int64_t>({64, 64});
  auto confusiontranspose = op::ConfusionTransposeD("ConfusionTransposeD")
                                .set_input_x(matmul)
                                .set_attr_perm({0, 2, 1, 3})
                                .set_attr_shape({64, 64})
                                .set_attr_transpose_first(true);
  TensorDesc ct_input_desc_x(ge::Shape(nz_shape_y_mm), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  TensorDesc ct_output_desc_y(ge::Shape(nz_shape_y_ct), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  confusiontranspose.update_input_desc_x(ct_input_desc_x);
  confusiontranspose.update_output_desc_y(ct_output_desc_y);

  std::vector<Operator> inputs{data_x1, data_x2};
  std::vector<Operator> outputs{confusiontranspose};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  std::cout << "!!!!!!!!!!!!!!!!!!!!!" << std::endl;
  fe::RunBufferFusionPassBase("MatmulConfusiontransposeUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                          *compute_graph_ptr);

  bool find_matmul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMul") {
      find_matmul = true;
    }
  }
  EXPECT_EQ(find_matmul, true);
}
