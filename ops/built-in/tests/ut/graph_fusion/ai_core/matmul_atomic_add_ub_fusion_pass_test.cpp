#include "array_ops.h"
#include "framework/common/types.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "matrix_calculation_ops.h"

#define private public
#define protected public
#include "buffer_fusion/ub_fusion/ai_core/matmul/matmul_atomic_add_ub_fusion.h"
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class matmul_atomic_add_ub_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "matmul_atomic_add_ub_fusion_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "matmul_atomic_add_ub_fusion_test TearDown" << std::endl; }
};

namespace fe {
Status RunBufferFusionPassMatMulAtomicAdd(string fusionPassName, BufferFusionPassType passType,
                                          ge::ComputeGraphPtr &compute_graph_ptr) {
  std::map<string, BufferFusionPassRegistry::CreateFn> createFns =
      BufferFusionPassRegistry::GetInstance().GetCreateFnByType(passType);
  const auto &iter = createFns.find(fusionPassName);
  if (iter != createFns.end()) {
    if (passType == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
      auto BufferFusionPassBasePtr =
          std::unique_ptr<MatmulAtomicAddUbFusion>(dynamic_cast<MatmulAtomicAddUbFusion *>(iter->second()));
      if (BufferFusionPassBasePtr == nullptr) {
        return FAILED;
      }
      ge::ComputeGraph::Vistor<ge::NodePtr> nodes = compute_graph_ptr->GetAllNodes();
      std::vector<ge::NodePtr> Node_v(nodes.begin(), nodes.end());

      BufferFusionPassBasePtr->SetName(fusionPassName);
      vector<BufferFusionPattern *> patterns = BufferFusionPassBasePtr->DefinePatterns();

      for (auto pattern : patterns) {
        std::vector<BufferFusionOpDesc *> node_descs = pattern->GetOpDescs();

        vector<ge::NodePtr> cube_nodes;
        for (auto node : nodes) {
          auto op_desc = node->GetOpDesc();
          if (op_desc->GetType() == "MatMul" || op_desc->GetType() == "MatMulV2") {
            cube_nodes.push_back(node);
          }
        }

        BufferFusionMapping mapping;
        for (auto node_desc : node_descs) {
          if (node_desc->desc_name == "matmul") {
            mapping[node_desc] = cube_nodes;
          }
        }
        vector<ge::NodePtr> fusion_nodes;
        BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
      }
      return SUCCESS;
    }
  }

  return FAILED;
}
}  // namespace fe

TEST_F(matmul_atomic_add_ub_fusion_test, matmul_atomic_add_test_1) {
  ge::Graph graph("matmul_atomic_add_test_1");
  std::vector<int64_t> dims_x1{32, 16384};
  std::vector<int64_t> dims_x2{16284, 64};
  std::vector<int64_t> dims_y{32, 64};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(shape_x1);
  desc_x1.SetOriginFormat(FORMAT_ND);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(shape_x2);
  desc_x2.SetOriginFormat(FORMAT_ND);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, FORMAT_ND, ge::DT_FLOAT16);
  desc_y.SetOriginShape(shape_y);
  desc_y.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc x1_desc(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  x1_desc.SetOriginShape(shape_x1);
  auto x1_data = op::Data("x1_data");
  x1_data.update_input_desc_x(x1_desc);
  x1_data.update_output_desc_y(x1_desc);

  ge::TensorDesc x2_desc(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  x2_desc.SetOriginShape(shape_x2);
  auto x2_data = op::Data("x2_data");
  x2_data.update_input_desc_x(x2_desc);
  x2_data.update_output_desc_y(x2_desc);

  auto matmul_op = op::MatMul("matmul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_transpose_x1(false)
                       .set_attr_transpose_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 32;
  opti_compilation_info.soc_version = "Ascend910A";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  Status ret = fe::RunBufferFusionPassMatMulAtomicAdd("MatmulAtomicAddUbFusion",
                                                      fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}

TEST_F(matmul_atomic_add_ub_fusion_test, matmul_atomic_add_test_2) {
  ge::Graph graph("matmul_atomic_add_test_2");

  std::vector<int64_t> dims_x1{32, 16};
  std::vector<int64_t> dims_x2{16, 64};
  std::vector<int64_t> dims_y{32, 64};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(shape_x1);
  desc_x1.SetOriginFormat(FORMAT_ND);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(shape_x2);
  desc_x2.SetOriginFormat(FORMAT_ND);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, FORMAT_ND, ge::DT_FLOAT16);
  desc_y.SetOriginShape(shape_y);
  desc_y.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc x1_desc(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  x1_desc.SetOriginShape(shape_x1);
  auto x1_data = op::Data("x1_data");
  x1_data.update_input_desc_x(x1_desc);
  x1_data.update_output_desc_y(x1_desc);

  ge::TensorDesc x2_desc(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  x2_desc.SetOriginShape(shape_x2);
  auto x2_data = op::Data("x2_data");
  x2_data.update_input_desc_x(x2_desc);
  x2_data.update_output_desc_y(x2_desc);

  auto matmul_op = op::MatMul("matmul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_transpose_x1(false)
                       .set_attr_transpose_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 32;
  opti_compilation_info.soc_version = "Ascend910A";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  Status ret = fe::RunBufferFusionPassMatMulAtomicAdd("MatmulAtomicAddUbFusion",
                                                      fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}

TEST_F(matmul_atomic_add_ub_fusion_test, matmul_atomic_add_test_3) {
  ge::Graph graph("matmul_atomic_add_test_3");

  std::vector<int64_t> dims_x1{32, 16384};
  std::vector<int64_t> dims_x2{16284, 64};
  std::vector<int64_t> dims_y{32, 64};

  ge::Shape shape_x1(dims_x1);
  ge::TensorDesc desc_x1(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  desc_x1.SetOriginShape(shape_x1);
  desc_x1.SetOriginFormat(FORMAT_ND);
  ge::Shape shape_x2(dims_x2);
  ge::TensorDesc desc_x2(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  desc_x2.SetOriginShape(shape_x2);
  desc_x2.SetOriginFormat(FORMAT_ND);
  ge::Shape shape_y(dims_y);
  ge::TensorDesc desc_y(shape_y, FORMAT_ND, ge::DT_FLOAT16);
  desc_y.SetOriginShape(shape_y);
  desc_y.SetOriginFormat(FORMAT_ND);

  ge::TensorDesc x1_desc(shape_x1, FORMAT_ND, ge::DT_FLOAT16);
  x1_desc.SetOriginShape(shape_x1);
  auto x1_data = op::Data("x1_data");
  x1_data.update_input_desc_x(x1_desc);
  x1_data.update_output_desc_y(x1_desc);

  ge::TensorDesc x2_desc(shape_x2, FORMAT_ND, ge::DT_FLOAT16);
  x2_desc.SetOriginShape(shape_x2);
  auto x2_data = op::Data("x2_data");
  x2_data.update_input_desc_x(x2_desc);
  x2_data.update_output_desc_y(x2_desc);

  auto matmul_op = op::MatMul("matmul")
                       .set_input_x1(x1_data)
                       .set_input_x2(x2_data)
                       .set_attr_transpose_x1(false)
                       .set_attr_transpose_x2(false);
  matmul_op.update_input_desc_x1(desc_x1);
  matmul_op.update_input_desc_x2(desc_x2);
  matmul_op.update_output_desc_y(desc_y);

  std::vector<Operator> inputs{x1_data, x2_data};
  std::vector<Operator> outputs{matmul_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  // set soc_version
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 32;
  opti_compilation_info.soc_version = "Ascend910A";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  Status ret = fe::RunBufferFusionPassMatMulAtomicAdd("MatmulAtomicAddUbFusion",
                                                      fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, compute_graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}