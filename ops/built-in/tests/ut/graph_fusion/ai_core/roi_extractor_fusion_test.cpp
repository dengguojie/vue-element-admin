#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "nn_detect_ops.h"
#define private public
#include "fusion_pass_test_utils.h"
#include "common/util/platform_info.h"
using namespace ge;
using namespace op;

class roi_extractor_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "roi_extractor SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "roi_extractor TearDown" << std::endl;
    }
};

TEST_F(roi_extractor_fusion_test, roi_extractor_fusion_test_1) {
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 8;
    platform_info.soc_info.vector_core_cnt = 7;
    opti_compilation_info.soc_version = "Ascend710";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend710"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    ge::Graph graph("roi_extractor_fusion_test_1");

    std::vector<int64_t> dims_input0{1, 256, 128, 128};
    ge::Shape shape_input0(dims_input0);
    ge::TensorDesc tensordesc_input0(shape_input0, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto input0 = op::Data("input0");
    input0.update_input_desc_x(tensordesc_input0);
    input0.update_output_desc_y(tensordesc_input0);

    std::vector<int64_t> dims_input1{1000, 5};
    ge::Shape shape_input1(dims_input1);
    ge::TensorDesc tensordesc_input1(shape_input1, ge::FORMAT_ND, ge::DT_FLOAT16);
    auto input1 = op::Data("input1");
    input1.update_input_desc_x(tensordesc_input1);
    input1.update_output_desc_y(tensordesc_input1);

    std::vector<int64_t> dims_output{1000, 256, 7, 7};
    ge::Shape shape_output(dims_output);
    ge::TensorDesc tensordesc_output(shape_output, ge::FORMAT_NHWC, ge::DT_FLOAT16);

    auto roi_extractor = op::RoiExtractor("RoiExtractor");
    roi_extractor.create_dynamic_input_features(4);
    for (int64_t n = 0; n < 4; n++) {
        roi_extractor.set_dynamic_input_features(n, input0);
    }

    roi_extractor.set_input_rois(input1);

    std::vector<Operator> inputs{input0, input1};
    std::vector<Operator> outputs{roi_extractor};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RoiExtractorFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_balance_rois = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BalanceRois") {
            find_balance_rois = true;
        }
    }
    EXPECT_EQ(find_balance_rois, true);
}
