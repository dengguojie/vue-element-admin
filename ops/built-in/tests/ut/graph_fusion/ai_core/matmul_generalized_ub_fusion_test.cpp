#include <iostream>

#include "array_ops.h"
#include "buffer_fusion/ub_fusion/ai_core/matmul/matmul_generalized_ub_fusion.h"
#include "common/inc/op_log.h"
#include "common/lx_fusion_func.h"
#include "common/lxfusion_json_util.h"
#include "elewise_calculation_ops.h"
#include "fusion_pass_test_slice_utils.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "inc/common/op_slice_info.h"
#include "matrix_calculation_ops.h"
#include "nn_pooling_ops.h"
#include "nonlinear_fuc_ops.h"
#include "quantize_ops.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

#define private public
#define protected public

using namespace fe;
using namespace ge;
using namespace op;

class matmul_generalized_ub_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "matmul_generalized_ub_fusion_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "matmul_generalized_ub_fusion_test TearDown" << std::endl; }
};

namespace fe {
Status RunBufferFusionPassBmmAddFusion(string fusionPassName, BufferFusionPassType passType,
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
        vector<ge::NodePtr> adjacent_matmul_nodes;
        vector<ge::NodePtr> elemwise_0_nodes;
        vector<ge::NodePtr> elemwise_1_nodes;
        vector<ge::NodePtr> termination_nodes;
        for (auto node : nodes) {
          auto op_desc = node->GetOpDesc();
          if (op_desc->GetType() == "BatchMatMulV2" || op_desc->GetType() == "MatMulV2" ||
              op_desc->GetType() == "BatchMatMul" || op_desc->GetType() == "MatMul") {
            matmul_nodes.push_back(node);
          } else if (op_desc->GetType() == "Add") {
            elemwise_0_nodes.push_back(node);
          }
        }
        BufferFusionMapping mapping;
        for (auto node_desc : node_descs) {
          if (node_desc->desc_name == "matmul") {
            mapping[node_desc] = matmul_nodes;
          }
          if (node_desc->desc_name == "elemwise_0") {
            mapping[node_desc] = elemwise_0_nodes;
          }
        }

        vector<ge::NodePtr> fusion_nodes;
        BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
        if (fusion_nodes.empty()) {
          return FAILED;
        }
        for (const auto node : fusion_nodes) {
          if (!((node->GetType() == "Add" && node->GetName() == "Add") ||
                (node->GetType() == "BatchMatMulV2" && node->GetName() == "matmul"))) {
            return FAILED;
          }
        }
      }
      return SUCCESS;
    }
  }

  return FAILED;
}

class BatchMatmulAddUbFusion : public MatMulGeneralizedUbFusion {
 public:
  explicit BatchMatmulAddUbFusion() { match_patterns = {MatchPattern{}.SetMM(type_matmul).SetElemwise0({{"Add"}})}; }
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    return HelpGetFusionNodes(mapping, fusionNodes, match_patterns);
  }

 private:
  vector<MatchPattern> match_patterns;
};

Status RunBufferFusionPassBmmDeqFastGeluQuantFusion(string fusionPassName, BufferFusionPassType passType,
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

        vector<ge::NodePtr> transdata_before_nodes;
        vector<ge::NodePtr> matmul_nodes;
        vector<ge::NodePtr> adjacent_matmul_nodes;
        vector<ge::NodePtr> elemwise_0_nodes;
        vector<ge::NodePtr> elemwise_1_nodes;
        vector<ge::NodePtr> termination_nodes;
        for (auto node : nodes) {
          auto op_desc = node->GetOpDesc();
          if (op_desc->GetType() == "BatchMatMulV2" || op_desc->GetType() == "MatMulV2" ||
              op_desc->GetType() == "BatchMatMul" || op_desc->GetType() == "MatMul") {
            matmul_nodes.push_back(node);
          } else if (op_desc->GetType() == "AscendDequant") {
            adjacent_matmul_nodes.push_back(node);
          } else if (op_desc->GetType() == "FastGeluV2") {
            elemwise_0_nodes.push_back(node);
          } else if (op_desc->GetType() == "AscendQuant") {
            termination_nodes.push_back(node);
          }
        }

        BufferFusionMapping mapping;
        for (auto node_desc : node_descs) {
          if (node_desc->desc_name == "transdata_before") {
            mapping[node_desc] = transdata_before_nodes;
          }
          if (node_desc->desc_name == "matmul") {
            mapping[node_desc] = matmul_nodes;
          }
          if (node_desc->desc_name == "adjacent_matmul") {
            mapping[node_desc] = adjacent_matmul_nodes;
          }
          if (node_desc->desc_name == "elemwise_0") {
            mapping[node_desc] = elemwise_0_nodes;
          }
          if (node_desc->desc_name == "elemwise_1") {
            mapping[node_desc] = elemwise_1_nodes;
          }
          if (node_desc->desc_name == "termination") {
            mapping[node_desc] = termination_nodes;
          }
        }
        vector<ge::NodePtr> fusion_nodes;
        BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
        if (fusion_nodes.empty()) {
          return FAILED;
        }
        for (const auto node : fusion_nodes) {
          if (!((node->GetType() == "AscendQuant" && node->GetName() == "quant") ||
                (node->GetType() == "FastGeluV2" && node->GetName() == "elemwise") ||
                (node->GetType() == "AscendDequant" && node->GetName() == "dequant") ||
                (node->GetType() == "BatchMatMulV2" && node->GetName() == "batchmatmul"))) {
            return FAILED;
          }
        }
      }
      return SUCCESS;
    }
  }
  return FAILED;
}

class BatchMatmulDeqFastGeluQuantUbFusion : public MatMulGeneralizedUbFusion {
 public:
  explicit BatchMatmulDeqFastGeluQuantUbFusion() {
    match_patterns = {MatchPattern{}
                          .SetMM(type_matmul)
                          .SetAdjacentMM({"AscendDequant"})
                          .SetElemwise0({{"FastGeluV2"}})
                          .SetTermination({"AscendQuant"})};
  }
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    return HelpGetFusionNodes(mapping, fusionNodes, match_patterns);
  }

 private:
  vector<MatchPattern> match_patterns;
};

Status ConstructFusionMappingOfMulSigmoidMul(string fusionPassName, BufferFusionPassType passType,
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

        vector<ge::NodePtr> transdata_before_nodes;
        vector<ge::NodePtr> matmul_nodes;
        vector<ge::NodePtr> adjacent_matmul_nodes;
        vector<ge::NodePtr> elemwise_0_nodes;
        vector<ge::NodePtr> elemwise_1_nodes;
        for (auto node : nodes) {
          auto op_desc = node->GetOpDesc();
          if (op_desc->GetType() == "BatchMatMulV2" || op_desc->GetType() == "MatMulV2" ||
              op_desc->GetType() == "BatchMatMul" || op_desc->GetType() == "MatMul") {
            matmul_nodes.push_back(node);
          } else if (op_desc->GetName() == "mul0") {
            elemwise_0_nodes.push_back(node);
          } else if (op_desc->GetName() == "mul1") {
            elemwise_1_nodes.push_back(node);
          } else if (op_desc->GetType() == "Sigmoid") {
            elemwise_0_nodes.push_back(node);
          }
        }

        BufferFusionMapping mapping;
        for (auto node_desc : node_descs) {
          if (node_desc->desc_name == "matmul") {
            mapping[node_desc] = matmul_nodes;
          }
          if (node_desc->desc_name == "elemwise_0") {
            mapping[node_desc] = elemwise_0_nodes;
          }
          if (node_desc->desc_name == "elemwise_1") {
            mapping[node_desc] = elemwise_1_nodes;
          }
        }
        vector<ge::NodePtr> fusion_nodes;
        BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
        if (fusion_nodes.empty()) {
          return FAILED;
        }
        for (const auto node : fusion_nodes) {
          if (!((node->GetType() == "Mul" && node->GetName() == "mul0") ||
                (node->GetType() == "Mul" && node->GetName() == "mul1") ||
                (node->GetType() == "Sigmoid" && node->GetName() == "sigmoid") ||
                (node->GetType() == "BatchMatMulV2" && node->GetName() == "BatchMatMulV2"))) {
            return FAILED;
          }
        }
      }
      return SUCCESS;
    }
  }
  return FAILED;
}

class BatchMatmulMulSigmoidMulUbFusion : public MatMulGeneralizedUbFusion {
 public:
  explicit BatchMatmulMulSigmoidMulUbFusion() {
    match_patterns = {MatchPattern{}.SetMM(type_matmul).SetElemwise0({{"Mul"}, {"Sigmoid"}}).SetElemwise1({{"Mul"}})};
  }
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    return HelpGetFusionNodes(mapping, fusionNodes, match_patterns);
  }

 private:
  vector<MatchPattern> match_patterns;
};
}  // namespace fe

op::Data CreateData(const string &name, const ge::Shape &shape, const ge::Format &format, const ge::DataType &dtype) {
  ge::TensorDesc desc(shape, format, dtype);
  auto data = op::Data(name);
  data.update_input_desc_x(desc);
  data.update_output_desc_y(desc);
  return data;
}

TEST_F(matmul_generalized_ub_fusion_test, matmul_generalized_ub_fusion_test_01) {
  ge::Graph graph("matmul_generalized_ub_fusion_test_01");
  REGISTER_BUFFER_FUSION_PASS("BatchMatmulAddUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                              fe::BatchMatmulAddUbFusion);

  auto X1Data = op::Data("x1");
  std::vector<int64_t> dims_x1{2, 2, 2, 16, 16};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensorDescX1(shape_x1, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  X1Data.update_input_desc_x(tensorDescX1);
  X1Data.update_output_desc_y(tensorDescX1);

  auto X2Data = op::Data("x2");
  std::vector<int64_t> dims_x2{2, 2, 2, 16, 16};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensorDescX2(shape_x2, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  X2Data.update_input_desc_x(tensorDescX2);
  X2Data.update_output_desc_y(tensorDescX2);

  auto AddData = op::Data("add");
  std::vector<int64_t> dims_add{2, 2, 2, 16, 16};
  ge::Shape shape_add(dims_add);
  ge::TensorDesc tensorDescAdd(shape_add, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  AddData.update_input_desc_x(tensorDescAdd);
  AddData.update_output_desc_y(tensorDescAdd);

  auto bmOP = op::BatchMatMulV2("matmul");
  bmOP.set_input_x1(X1Data);
  bmOP.set_input_x2(X2Data);
  bmOP.update_output_desc_y(ge::TensorDesc(ge::Shape({2, 2, 2, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  auto add_op = op::Add("Add");
  add_op.set_input_x1(bmOP);
  add_op.set_input_x2(AddData);
  add_op.update_output_desc_y(ge::TensorDesc(ge::Shape({2, 2, 2, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  std::vector<Operator> inputs{X1Data, X2Data, AddData};
  std::vector<Operator> outputs{add_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  vector<AxisSplitMap> asm_mm{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})},  // batch
                         {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})},  // m
                         {CreateOutputSplitInfo(0, {2})}),
      CreateAxisSplitMap({CreateInputSplitInfo(1, {0})},  // n
                         {CreateOutputSplitInfo(0, {1})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_mm, {"BatchMatMulV2"}));
  vector<AxisSplitMap> asm_add{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1}), CreateInputSplitInfo(1, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2}), CreateInputSplitInfo(1, {2})}, {CreateOutputSplitInfo(0, {2})}),
  };

  EXPECT_TRUE(SetSplitMapToNodeByName(compute_graph_ptr, asm_add, "Add"));
  Status ret = fe::RunBufferFusionPassBmmAddFusion("BatchMatmulAddUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                                   compute_graph_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(matmul_generalized_ub_fusion_test, matmul_generalized_ub_fusion_test_02) {
  ge::Graph graph("matmul_generalized_ub_fusion_test_02");
  REGISTER_BUFFER_FUSION_PASS("BatchMatmulDeqFastGeluQuantUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                              fe::BatchMatmulDeqFastGeluQuantUbFusion);
  auto X1Data = op::Data("x1");
  std::vector<int64_t> dims_x1{8, 733, 73};
  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_FLOAT16);
  X1Data.update_input_desc_x(tensorDescX1);
  X1Data.update_output_desc_y(tensorDescX1);

  auto X2Data = op::Data("x2");
  std::vector<int64_t> dims_x2{73, 987};
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc tensorDescX2(shape_x2, FORMAT_ND, DT_FLOAT16);
  X2Data.update_input_desc_x(tensorDescX2);
  X2Data.update_output_desc_y(tensorDescX2);

  auto bmOP = op::BatchMatMulV2("batchmatmul");
  bmOP.set_input_x1(X1Data);
  bmOP.set_input_x2(X2Data);

  auto dequant_op = op::AscendDequant("dequant");
  dequant_op.set_input_x(bmOP);

  auto fastgeluv2_op = op::FastGeluV2("elemwise");
  fastgeluv2_op.set_input_x(dequant_op);

  auto quant_op = op::AscendQuant("quant");
  quant_op.set_input_x(fastgeluv2_op);

  std::vector<Operator> inputs{X1Data, X2Data};
  std::vector<Operator> outputs{quant_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  Status res = fe::RunBufferFusionPassBmmDeqFastGeluQuantFusion(
      "BatchMatmulDeqFastGeluQuantUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
  EXPECT_EQ(res, fe::SUCCESS);
}

TEST_F(matmul_generalized_ub_fusion_test, matmul_generalized_ub_fusion_test_03) {
  ge::Graph graph(this->test_info_->name());
  REGISTER_BUFFER_FUSION_PASS("BatchMatmulMulSigmoidMulUbFusion", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                              fe::BatchMatmulMulSigmoidMulUbFusion);
  // step1: Construct Graph
  auto data_a = CreateData("data_a", ge::Shape({77, 32, 7, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  auto data_b = CreateData("data_b", ge::Shape({128, 32, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);

  auto batch_matmul_op = op::BatchMatMulV2("BatchMatMulV2")
                             .set_input_x1(data_a)
                             .set_input_x2(data_b)
                             .set_attr_adj_x1(false)
                             .set_attr_adj_x2(false);
  batch_matmul_op.update_output_desc_y(
      ge::TensorDesc(ge::Shape({77, 7, 128, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));

  auto const_add = op::Const("scalar_for_mul0");
  Tensor const_add_tensor;
  float *const_add_tensor_value = new float[1];
  for (int i = 0; i < 1; i++) {
    *(const_add_tensor_value + i) = 1.7;
  }
  const_add_tensor.SetData((uint8_t *)const_add_tensor_value, 1 * 4);
  std::vector<int64_t> dims_add{1};
  ge::Shape shape_add(dims_add);
  ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NHWC, DT_FLOAT);
  const_add_tensor.SetTensorDesc(tensorDescAdd);
  const_add.set_attr_value(const_add_tensor);

  auto cast_op = op::Cast("input");                     // Note: op name used in construct BufferFusionMapping
  cast_op.set_input_x(const_add).set_attr_dst_type(1);  // dst_type 1: float16
  auto mul0 = op::Mul("mul0").set_input_x1(batch_matmul_op).set_input_x2(cast_op);
  mul0.update_output_desc_y(ge::TensorDesc(ge::Shape({77, 7, 128, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));
  auto sigmoid = op::Sigmoid("sigmoid").set_input_x(mul0);
  sigmoid.update_output_desc_y(ge::TensorDesc(ge::Shape({77, 7, 128, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));
  auto mul1 = op::Mul("mul1").set_input_x1(batch_matmul_op).set_input_x2(sigmoid);
  mul1.update_output_desc_y(ge::TensorDesc(ge::Shape({77, 7, 128, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16));
  std::vector<Operator> inputs{data_a, data_b};
  std::vector<Operator> outputs{mul1};

  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  vector<AxisSplitMap> asm_mm{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})},  // batch
                         {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})},  // m
                         {CreateOutputSplitInfo(0, {2})}),
      CreateAxisSplitMap({CreateInputSplitInfo(1, {0})},  // n
                         {CreateOutputSplitInfo(0, {1})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_mm, {"BatchMatMulV2"}));
  vector<AxisSplitMap> asm_mul0{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})}, {CreateOutputSplitInfo(0, {2})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByName(compute_graph_ptr, asm_mul0, "mul0"));
  vector<AxisSplitMap> asm_sigmoid{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2})}, {CreateOutputSplitInfo(0, {2})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByType(compute_graph_ptr, asm_sigmoid, {"Sigmoid"}));
  vector<AxisSplitMap> asm_mul1{
      CreateAxisSplitMap({CreateInputSplitInfo(0, {0}), CreateInputSplitInfo(1, {0})}, {CreateOutputSplitInfo(0, {0})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {1}), CreateInputSplitInfo(1, {1})}, {CreateOutputSplitInfo(0, {1})}),
      CreateAxisSplitMap({CreateInputSplitInfo(0, {2}), CreateInputSplitInfo(1, {2})}, {CreateOutputSplitInfo(0, {2})}),
  };
  EXPECT_TRUE(SetSplitMapToNodeByName(compute_graph_ptr, asm_mul1, "mul1"));

  // auto mapping = ConstructFusionMappingOfMulSigmoidMul(compute_graph_ptr);
  Status res = fe::ConstructFusionMappingOfMulSigmoidMul("BatchMatmulMulSigmoidMulUbFusion",
                                                         fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
  EXPECT_EQ(res, fe::SUCCESS);
}
