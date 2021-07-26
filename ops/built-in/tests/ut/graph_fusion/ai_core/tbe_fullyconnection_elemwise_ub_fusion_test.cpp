#include "gtest/gtest.h"
#include "array_ops.h"
#include "nonlinear_fuc_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "matrix_calculation_ops.h"
#include "common/lx_fusion_func.h"
#define private public
#define protected public
#include "inc/common/op_slice_info.h"
#include "common/lxfusion_json_util.h"
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"
#include "common/inc/op_log.h"


using namespace ge;
using namespace op;

class tbe_fullyconnection_elemwise_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "tbe_fullyconnection_elemwise_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "tbe_fullyconnection_elemwise_fusion_test TearDown" << std::endl;
    }
};

namespace fe {
  static Status RunBufferFusionPass(string fusionPassName, BufferFusionPassType passType,
    ge::ComputeGraphPtr& compute_graph_ptr) {
      std::map<string, BufferFusionPassRegistry::CreateFn> createFns =
          BufferFusionPassRegistry::GetInstance().GetCreateFnByType(passType);
      const auto &iter = createFns.find(fusionPassName);
      if (iter != createFns.end()) {
          if (passType == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
              auto BufferFusionPassFcPtr = std::unique_ptr<BufferFusionPassBase>(
                      dynamic_cast<BufferFusionPassBase *>(iter->second()));
              if (BufferFusionPassFcPtr == nullptr) {
                  return FAILED;
              }

              ge::ComputeGraph::Vistor<ge::NodePtr> NodePtrs = compute_graph_ptr->GetAllNodes();
              std::vector<ge::NodePtr> Node_v(NodePtrs.begin(), NodePtrs.end());

              BufferFusionPassFcPtr->SetName(fusionPassName);
              vector<BufferFusionPattern*> patterns = BufferFusionPassFcPtr->DefinePatterns();
              for (auto pattern : patterns) {
                std::vector<BufferFusionOpDesc *> desc = pattern->GetOpDescs();
                vector<ge::NodePtr> elemNodes;
                vector<ge::NodePtr> matmulNodes;
                for (auto i : NodePtrs) {
                  auto opDesc = i->GetOpDesc();
                  if (opDesc->GetType() == "BatchMatMul") {
                    matmulNodes.push_back(i);
                  }
                  if (opDesc->GetType() == "FastGelu") {
                    opDesc->MutableInputDesc(0)->SetShape(ge::GeShape({16, 512, 4096}));
                    elemNodes.push_back(i);
                  }
                  if (opDesc->GetType() == "FastGeluGrad") {
                    opDesc->MutableInputDesc(0)->SetShape(ge::GeShape({16, 512, 4096}));
                    elemNodes.push_back(i);
                  }
                }

                BufferFusionMapping mapping;
                for (auto i : desc) {
                  if (i->desc_name == "FullyConnection/MatMul/BatchMatmul") {
                    mapping[i] = matmulNodes;
                  }
                  if (i->desc_name == "eltwise2") {
                    mapping[i] = elemNodes;
                  }
                }

                vector<ge::NodePtr> fusion_nodes;
                BufferFusionPassFcPtr->GetFusionNodes(mapping, fusion_nodes);
              }
              return SUCCESS;
          }
      }
      return FAILED;
  }
}
TEST_F(tbe_fullyconnection_elemwise_fusion_test, tbe_fullyconnection_elemwise_fusion_test_1) {
    ge::Graph graph("tbe_fullyconnection_elemwise_fusion_test_1");

    ge::TensorDesc a_desc(ge::Shape({16, 512, 1024}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto data_a = op::Data("data_a");
    data_a.update_input_desc_x(a_desc);
    data_a.update_output_desc_y(a_desc);

    ge::TensorDesc b_desc(ge::Shape({16, 1024, 4096}), ge::FORMAT_NHWC, DT_FLOAT16);
    auto data_b = op::Data("data_b");
    data_b.update_input_desc_x(b_desc);
    data_b.update_output_desc_y(b_desc);

    auto batch_matmul_op = op::BatchMatMul("BatchMatMul")
        .set_input_x1(data_a)
        .set_input_x2(data_b)
        .set_attr_adj_x1(false)
        .set_attr_adj_x2(false);

    auto fastgelu_op = op::FastGelu("FastGelu")
        .set_input_x(batch_matmul_op);

    std::vector<Operator> inputs{data_a, data_b};
    std::vector<Operator> outputs{fastgelu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    Status res = fe::RunBufferFusionPass("TbeFullyconnectionElemwiseDequantFusionPass",
                                           fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                           compute_graph_ptr);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(tbe_fullyconnection_elemwise_fusion_test, tbe_fullyconnection_elemwise_fusion_test_2) {
    ge::Graph graph("tbe_fullyconnection_elemwise_fusion_test_2");

    ge::TensorDesc a_desc(ge::Shape({16, 512, 1024}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto data_a = op::Data("data_a");
    data_a.update_input_desc_x(a_desc);
    data_a.update_output_desc_y(a_desc);

    ge::TensorDesc b_desc(ge::Shape({16, 1024, 4096}), ge::FORMAT_NHWC, DT_FLOAT16);
    auto data_b = op::Data("data_b");
    data_b.update_input_desc_x(b_desc);
    data_b.update_output_desc_y(b_desc);

    ge::TensorDesc c_desc(ge::Shape({16, 512, 4096}), ge::FORMAT_NHWC, DT_FLOAT16);
    auto data_c = op::Data("data_c");
    data_c.update_input_desc_x(c_desc);
    data_c.update_output_desc_y(c_desc);

    auto batch_matmul_op = op::BatchMatMul("BatchMatMul")
        .set_input_x1(data_a)
        .set_input_x2(data_b)
        .set_attr_adj_x1(false)
        .set_attr_adj_x2(false);

    auto fastgelugrad_op = op::FastGeluGrad("FastGeluGrad")
        .set_input_dy(data_c)
        .set_input_x(batch_matmul_op);

    std::vector<Operator> inputs{data_a, data_b, data_c};
    std::vector<Operator> outputs{fastgelugrad_op};

    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    Status res = fe::RunBufferFusionPass("TbeFullyconnectionElemwiseDequantFusionPass",
                                           fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                           compute_graph_ptr);
    EXPECT_EQ(res, SUCCESS);
}
