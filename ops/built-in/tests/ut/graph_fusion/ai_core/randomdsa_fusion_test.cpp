#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "../../../../op_proto/inc/random_ops.h"
#define private public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class randomdsa_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "randomdsa_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "randomdsa_fusion_test TearDown" << std::endl;
    }
};

TEST_F(randomdsa_fusion_test, randomdsa_fusion_test_1) {
    ge::Graph graph("randomdsa_fusion_test_1");
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_info;
    opti_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_info);
    // DropOutGenMask 
    auto xData = op::Data("xData");
    auto obj_probdata = op::Data("obj_probdata");
    std::vector<int64_t> dims_x{1024};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_INT64);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    std::vector<int64_t> dims1{1};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
    obj_probdata.update_input_desc_x(tensorDesc1);
    obj_probdata.update_output_desc_y(tensorDesc1);

    auto dropout_gen = op::DropOutGenMask("DropOutGenMask1");
    dropout_gen.set_input_shape(xData);
    dropout_gen.set_input_prob(obj_probdata);
    std::vector<Operator> inputs{xData,obj_probdata};
    std::vector<Operator> outputs{dropout_gen};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RandomDsaFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool finddsa = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DSAGenBitMask") {
            finddsa = true;
        }
    }
    EXPECT_EQ(finddsa, true);
}

TEST_F(randomdsa_fusion_test, randomdsa_fusion_test_2) {
    ge::Graph graph("randomdsa_fusion_test_1");
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_info;
    opti_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_info);

    // DropOutGenMask
    auto xData = op::Data("xData");
    auto mindata = op::Data("min_data");
    auto maxdata = op::Data("max_data");
    std::vector<int64_t> dims_x{1024};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_INT64);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    std::vector<int64_t> dims1{1};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT);
    mindata.update_input_desc_x(tensorDesc1);
    mindata.update_output_desc_y(tensorDesc1);

    std::vector<int64_t> dims2{1};
    ge::Shape shape2(dims2);
    ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT);
    maxdata.update_input_desc_x(tensorDesc2);
    maxdata.update_output_desc_y(tensorDesc2);

    auto randomuniform = op::RandomUniformInt("RandomUniformInt1");
    randomuniform.set_input_shape(xData);
    randomuniform.set_input_min(mindata);
    randomuniform.set_input_max(maxdata);
    std::vector<Operator> inputs{xData,mindata,maxdata};
    std::vector<Operator> outputs{randomuniform};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RandomDsaFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool finddsa = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DSARandomUniform") {
            finddsa = true;
        }
    }
    EXPECT_EQ(finddsa, true);
}

TEST_F(randomdsa_fusion_test, randomdsa_fusion_test_3) {
    ge::Graph graph("randomdsa_fusion_test_3");
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_info;
    opti_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_info);

    // DropOutGenMask
    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{1024};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_INT64);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto randomstandard = op::RandomStandardNormal("RandomStandardNormal1");
    randomstandard.set_input_shape(xData);
    std::vector<Operator> inputs{xData};
    std::vector<Operator> outputs{randomstandard};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RandomDsaFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool finddsa = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DSARandomNormal") {
            finddsa = true;
        }
    }
    EXPECT_EQ(finddsa, true);
}

TEST_F(randomdsa_fusion_test, randomdsa_fusion_test_4) {
    ge::Graph graph("randomdsa_fusion_test_4");
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_info;
    opti_info.soc_version = "Ascend920A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_info);

    // DropOutGenMask
    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{1024};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_INT32);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto truncatenormal = op::TruncatedNormal("TruncatedNormal1");
    truncatenormal.set_input_shape(xData);
    std::vector<Operator> inputs{xData};
    std::vector<Operator> outputs{truncatenormal};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    //fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RandomDsaFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool finddsa = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "DSARandomTruncatedNormal") {
            finddsa = true;
        }
    }
    EXPECT_EQ(finddsa, true);
}
