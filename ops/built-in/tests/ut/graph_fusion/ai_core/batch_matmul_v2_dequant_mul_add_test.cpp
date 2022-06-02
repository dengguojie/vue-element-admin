#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "matrix_calculation_ops.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "quantize_ops.h"
#include "fusion_pass_test_utils.h"
#include "buffer_fusion/ub_fusion/ai_core/matmul/batch_matmul_v2_dequant_mul_add.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class batch_matmul_v2_dequant_mul_add_fusion_test : public testing::Test {
  protected:
    static void SetUpTestCase() {
        std::cout << "batch_matmul_v2_dequant_mul_add_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "batch_matmul_v2_dequant_mul_add_fusion_test TearDown" << std::endl;
    }
};

namespace fe {
Status RunBufferFusionPassBmmV2DequantMulAddFusion(string fusionPassName, BufferFusionPassType passType,
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
        vector<ge::NodePtr> dequant_nodes;
        vector<ge::NodePtr> add_nodes;
        vector<ge::NodePtr> mul_nodes;
        for (auto node : nodes) {
          auto op_desc = node->GetOpDesc();
          if (op_desc->GetType() == "BatchMatMulV2" || op_desc->GetType() == "MatMulV2" ||
              op_desc->GetType() == "BatchMatMul" || op_desc->GetType() == "MatMul") {
            matmul_nodes.push_back(node);
          }
          else if (op_desc->GetType() == "AscendDequant") {
            dequant_nodes.push_back(node);
          }
          else if (op_desc->GetType() == "Mul") {
            mul_nodes.push_back(node);
          }
          else if (op_desc->GetType() == "Add") {
            add_nodes.push_back(node);
          }
        }
        BufferFusionMapping mapping;
        for (auto node_desc : node_descs) {
          if (node_desc->desc_name == "matmul") {
            mapping[node_desc] = matmul_nodes;
          }
          if (node_desc->desc_name == "dequant") {
            mapping[node_desc] = dequant_nodes;
          }
          if (node_desc->desc_name == "eltwise1") {
            mapping[node_desc] = mul_nodes;
          }
          if (node_desc->desc_name == "eltwise2") {
            mapping[node_desc] = add_nodes;
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
TEST_F(batch_matmul_v2_dequant_mul_add_fusion_test, batch_matmul_v2_dequant_mul_add_fusion_test_1) {
  ge::Graph graph("batch_matmul_v2_dequant_mul_add_fusion_test_1");
  auto X1Data = op::Data("x1");
  std::vector<int64_t> dims_x1{2, 2,2,16, 16};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensorDescX1(shape_x1, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  X1Data.update_input_desc_x(tensorDescX1);
  X1Data.update_output_desc_y(tensorDescX1);

  auto X2Data = op::Data("x2");
  std::vector<int64_t> dims_x2{2, 2,2,16, 16};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensorDescX2(shape_x2, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  X2Data.update_input_desc_x(tensorDescX2);
  X2Data.update_output_desc_y(tensorDescX2);

  auto AddData = op::Data("add");
  std::vector<int64_t> dims_add{2, 2,2,16, 16};
  ge::Shape shape_add(dims_add);
  ge::TensorDesc tensorDescAdd(shape_add, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  AddData.update_input_desc_x(tensorDescAdd);
  AddData.update_output_desc_y(tensorDescAdd);

  auto MulData = op::Data("mul");
  std::vector<int64_t> dims_mul{2, 2,2,16, 16};
  ge::Shape shape_mul(dims_mul);
  ge::TensorDesc tensorDescMul(shape_mul, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  MulData.update_input_desc_x(tensorDescMul);
  MulData.update_output_desc_y(tensorDescMul);

  auto bmOP = op::BatchMatMulV2("matmul");
  bmOP.set_input_x1(X1Data);
  bmOP.set_input_x2(X2Data);
  bmOP.update_output_desc_y(ge::TensorDesc(ge::Shape({2, 2,2,16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  float deq_scale = 1.0;
  ge::Tensor scale_tensor(tensorDescMul, reinterpret_cast<uint8_t *>(&deq_scale), sizeof(float));
  auto const_op = op::Const("deq_scale").set_attr_value(scale_tensor);
  auto dequant_op = op::AscendDequant("dequant");
  dequant_op.set_input_x(bmOP).set_input_deq_scale(const_op);
  dequant_op.update_output_desc_y(ge::TensorDesc(ge::Shape({2, 2,2,16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  auto mul_op = op::Mul("Mul");
  mul_op.set_input_x1(dequant_op);
  mul_op.set_input_x2(MulData);
  mul_op.update_output_desc_y(ge::TensorDesc(ge::Shape({2, 2,2,16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));
  auto add_op = op::Add("Add");
  add_op.set_input_x1(mul_op);
  add_op.set_input_x2(AddData);
  add_op.update_output_desc_y(ge::TensorDesc(ge::Shape({2, 2,2,16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  std::vector<Operator> inputs{X1Data, X2Data, const_op, MulData, AddData};
  std::vector<Operator> outputs{add_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  Status ret = fe::RunBufferFusionPassBmmV2DequantMulAddFusion("BatchMatmulV2DequantMulAddFusionPass",
                                       fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}