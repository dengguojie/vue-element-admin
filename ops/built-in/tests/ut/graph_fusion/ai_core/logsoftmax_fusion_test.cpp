#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class logsoftmax_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "logsoftmax_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "logsoftmax_fusion_test TearDown" << std::endl;
  }
};

TEST_F(logsoftmax_fusion_test, logsoftmax_fusion_test_1) {
    ge::Graph graph("logsoftmax_fusion_test_1");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{8, 2000, 29};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto softmax_op = op::LogSoftmaxV2("logsoftmaxv2_0");
    softmax_op.set_input_logits(in_input_x_data)
        .set_attr_axes({-1});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(softmax_op);

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LogSoftmaxFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSoftmax = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "LogSoftmaxV2") {
            findSoftmax = true;
        }
    }
    EXPECT_EQ(findSoftmax, true);
}


TEST_F(logsoftmax_fusion_test, logsoftmax_fusion_test_2) {
    ge::Graph graph("logsoftmax_fusion_test_2");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{8, 2000, 29};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto softmax_op = op::LogSoftmaxV2("logsoftmaxv2_0");
    softmax_op.set_input_logits(in_input_x_data)
        .set_attr_axes({-1});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(softmax_op);

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 2;
    opti_compilation_info.soc_version = "Ascend910A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LogSoftmaxFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    fe::PlatformInfoManager::Instance().platform_info_map_.clear();

    bool findSoftmax = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "LogSoftmaxV2") {
            findSoftmax = true;
        }
    }
    EXPECT_EQ(findSoftmax, true);
}

TEST_F(logsoftmax_fusion_test, logsoftmax_fusion_test_3) {
    ge::Graph graph("logsoftmax_fusion_test_3");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{8, 64, 64, 64};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto softmax_op = op::LogSoftmaxV2("logsoftmaxv2_0");
    softmax_op.set_input_logits(in_input_x_data)
        .set_attr_axes({-1});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(softmax_op);

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "Ascend910A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LogSoftmaxFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    fe::PlatformInfoManager::Instance().platform_info_map_.clear();

    bool findSoftmax = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "LogSoftmaxV2") {
            findSoftmax = true;
        }
    }
    EXPECT_EQ(findSoftmax, true);
}