#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "selection_ops.h"
#define private public
#include "fusion_pass_test_utils.h"
#include "common/util/platform_info.h"
using namespace ge;
using namespace op;

class topk_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(topk_fusion_test, diag_fusion_test_1) {
    ge::Graph graph("topk_fusion_test_1");
    auto topk_input_data = op::Data("topk_input_data");
    std::vector<int64_t> dims{3, 32};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape);
    topk_input_data.update_input_desc_x(tensorDesc);
    topk_input_data.update_output_desc_y(tensorDesc);

    int32_t *k_data = new int32_t[1];
    k_data[0]=1;
    TensorDesc k_desc(ge::Shape({1}),FORMAT_ND, DT_INT32);
    Tensor k_tensor(k_desc, (uint8_t *)k_data,sizeof(int32_t));
    auto k_const = op::Const().set_attr_value(k_tensor);

    auto topk_op = op::TopK("topk_0");
    topk_op.set_input_x(topk_input_data);
    topk_op.set_input_k(k_const);
    topk_op.SetAttr("dim", -1);
    topk_op.SetAttr("sorted",true);
    topk_op.SetAttr("largest", true);
    std::vector<Operator> inputs{topk_input_data};
    std::vector<Operator> outputs{topk_op};
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
    fe::FusionPassTestUtils::RunGraphFusionPass("TopKFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findTopKD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TopKD") {
            findTopKD = true;
            break;
        }
    }
    EXPECT_EQ(findTopKD, true);
}

TEST_F(topk_fusion_test, diag_fusion_test_2) {
    ge::Graph graph("topk_fusion_test_1");
    auto topk_input_data = op::Data("topk_input_data");
    std::vector<int64_t> dims{3, 32};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape);
    topk_input_data.update_input_desc_x(tensorDesc);
    topk_input_data.update_output_desc_y(tensorDesc);

    int64_t *k_data = new int64_t[1];
    k_data[0]=1;
    TensorDesc k_desc(ge::Shape({1}),FORMAT_ND, DT_INT64);
    Tensor k_tensor(k_desc, (uint8_t *)k_data,sizeof(int64_t));
    auto k_const = op::Const().set_attr_value(k_tensor);

    auto topk_op = op::TopK("topk_0");
    topk_op.set_input_x(topk_input_data);
    topk_op.set_input_k(k_const);
    topk_op.SetAttr("dim", -1);
    topk_op.SetAttr("sorted",true);
    topk_op.SetAttr("largest", true);
    std::vector<Operator> inputs{topk_input_data};
    std::vector<Operator> outputs{topk_op};
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
    fe::FusionPassTestUtils::RunGraphFusionPass("TopKFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findTopKD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TopKD") {
            findTopKD = true;
            break;
        }
    }
    EXPECT_EQ(findTopKD, true);
}

TEST_F(topk_fusion_test, diag_fusion_test_3) {
    ge::Graph graph("topk_fusion_test_1");
    auto topk_input_data = op::Data("topk_input_data");
    std::vector<int64_t> dims{1000};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape);
    topk_input_data.update_input_desc_x(tensorDesc);
    topk_input_data.update_output_desc_y(tensorDesc);

    int64_t *k_data = new int64_t[1];
    k_data[0]=1000;
    TensorDesc k_desc(ge::Shape({1}),FORMAT_ND, DT_INT64);
    Tensor k_tensor(k_desc, (uint8_t *)k_data,sizeof(int64_t));
    auto k_const = op::Const().set_attr_value(k_tensor);

    auto topk_op = op::TopK("topk_0");
    topk_op.set_input_x(topk_input_data);
    topk_op.set_input_k(k_const);
    topk_op.SetAttr("dim", -1);
    topk_op.SetAttr("sorted",true);
    topk_op.SetAttr("largest", true);
    std::vector<Operator> inputs{topk_input_data};
    std::vector<Operator> outputs{topk_op};
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
    fe::FusionPassTestUtils::RunGraphFusionPass("TopKFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findTopKD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TopKD") {
            findTopKD = true;
        }
    }
    EXPECT_EQ(findTopKD, true);
}

TEST_F(topk_fusion_test, diag_fusion_test_4) {
    ge::Graph graph("topk_fusion_test_4");
    auto topk_input_data = op::Data("topk_input_data");
    std::vector<int64_t> dims{100000};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, FORMAT_ND,  DT_FLOAT16);
    topk_input_data.update_input_desc_x(tensorDesc);
    topk_input_data.update_output_desc_y(tensorDesc);

    int64_t *k_data = new int64_t[1];
    k_data[0]=100000;
    TensorDesc k_desc(ge::Shape({1}),FORMAT_ND, DT_INT64);
    Tensor k_tensor(k_desc, (uint8_t *)k_data,sizeof(int64_t));
    auto k_const = op::Const().set_attr_value(k_tensor);

    auto topk_op = op::TopK("topk_0");
    topk_op.set_input_x(topk_input_data);
    topk_op.set_input_k(k_const);
    topk_op.SetAttr("dim", -1);
    topk_op.SetAttr("sorted",true);
    topk_op.SetAttr("largest", true);
    std::vector<Operator> inputs{topk_input_data};
    std::vector<Operator> outputs{topk_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.ai_core_spec.cube_vector_split = true;
    opti_compilation_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TopKFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findTopKD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SegmentSort") {
            findTopKD = true;
        }
    }
    EXPECT_EQ(findTopKD, true);
}

TEST_F(topk_fusion_test, diag_fusion_test_5) {
    ge::Graph graph("topk_fusion_test_5");
    auto topk_input_data = op::Data("topk_input_data");
    std::vector<int64_t> dims{100000};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_FLOAT);
    topk_input_data.update_input_desc_x(tensorDesc);
    topk_input_data.update_output_desc_y(tensorDesc);

    int64_t *k_data = new int64_t[1];
    k_data[0]=100000;
    TensorDesc k_desc(ge::Shape({1}),FORMAT_ND, DT_INT64);
    Tensor k_tensor(k_desc, (uint8_t *)k_data,sizeof(int64_t));
    auto k_const = op::Const().set_attr_value(k_tensor);

    auto topk_op = op::TopK("topk_0");
    topk_op.set_input_x(topk_input_data);
    topk_op.set_input_k(k_const);
    topk_op.SetAttr("dim", -1);
    topk_op.SetAttr("sorted",true);
    topk_op.SetAttr("largest", true);
    std::vector<Operator> inputs{topk_input_data};
    std::vector<Operator> outputs{topk_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.ai_core_spec.cube_vector_split = true;
    opti_compilation_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TopKFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findTopKD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SegmentSort") {
            findTopKD = true;
        }
    }
    EXPECT_EQ(findTopKD, true);
}

TEST_F(topk_fusion_test, diag_fusion_test_6) {
    ge::Graph graph("topk_fusion_test_6");
    auto topk_input_data = op::Data("topk_input_data");
    std::vector<int64_t> dims{100000};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, FORMAT_ND, DT_INT32);
    topk_input_data.update_input_desc_x(tensorDesc);
    topk_input_data.update_output_desc_y(tensorDesc);

    int64_t *k_data = new int64_t[1];
    k_data[0]=100000;
    TensorDesc k_desc(ge::Shape({1}),FORMAT_ND, DT_INT64);
    Tensor k_tensor(k_desc, (uint8_t *)k_data,sizeof(int64_t));
    auto k_const = op::Const().set_attr_value(k_tensor);

    auto topk_op = op::TopK("topk_0");
    topk_op.set_input_x(topk_input_data);
    topk_op.set_input_k(k_const);
    topk_op.SetAttr("dim", -1);
    topk_op.SetAttr("sorted",true);
    topk_op.SetAttr("largest", true);
    std::vector<Operator> inputs{topk_input_data};
    std::vector<Operator> outputs{topk_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.ai_core_spec.cube_vector_split = true;
    opti_compilation_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TopKFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findTopKD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SegmentSort") {
            findTopKD = true;
        }
    }
    EXPECT_EQ(findTopKD, false);
}

TEST_F(topk_fusion_test, diag_fusion_test_7) {
    ge::Graph graph("topk_fusion_test_7");
    auto topk_input_data = op::Data("topk_input_data");
    std::vector<int64_t> dims{3, 256, 1600};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape);
    topk_input_data.update_input_desc_x(tensorDesc);
    topk_input_data.update_output_desc_y(tensorDesc);

    int32_t *k_data = new int32_t[1];
    k_data[0]=64;
    TensorDesc k_desc(ge::Shape({1}),FORMAT_ND, DT_INT32);
    Tensor k_tensor(k_desc, (uint8_t *)k_data,sizeof(int32_t));
    auto k_const = op::Const().set_attr_value(k_tensor);

    auto topk_op = op::TopK("topk_0");
    topk_op.set_input_x(topk_input_data);
    topk_op.set_input_k(k_const);
    topk_op.SetAttr("dim", -2);
    topk_op.SetAttr("sorted",true);
    topk_op.SetAttr("largest", true);
    std::vector<Operator> inputs{topk_input_data};
    std::vector<Operator> outputs{topk_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TopKFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool findTopKD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TopKD") {
            findTopKD = true;
            break;
        }
    }
    EXPECT_EQ(findTopKD, true);
}
