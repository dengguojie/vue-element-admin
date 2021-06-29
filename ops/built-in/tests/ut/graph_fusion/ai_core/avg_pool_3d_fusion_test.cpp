#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class avg_pool_3d_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "avg_pool_3d_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "avg_pool_3d_fusion_test TearDown" << std::endl;
    }
};

TEST_F(avg_pool_3d_fusion_test, avg_pool_3d_fusion_invalid_c) {
    ge::Graph graph("avg_pool_3d_fusion_invalid_c");
    auto avg_pool_3d_input_data = op::Data("avg_pool_3d_input_data");
    std::vector<int64_t> dims{32, 28, 28, 28, -1};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    avg_pool_3d_input_data.update_input_desc_x(tensorDesc);
    avg_pool_3d_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_3d_op = op::AvgPool3D("avgpool3d_0");
    avg_pool_3d_op.set_input_x(avg_pool_3d_input_data);
    avg_pool_3d_op.set_attr_ksize({1, 1, 1, 1, 1});
    avg_pool_3d_op.set_attr_strides({1, 2, 2, 2, 1});
    avg_pool_3d_op.set_attr_pads({0, 0, 0, 0, 0, 0,});
    avg_pool_3d_op.set_attr_ceil_mode(false);
    avg_pool_3d_op.set_attr_count_include_pad(false);
    avg_pool_3d_op.set_attr_divisor_override(0);
    avg_pool_3d_op.set_attr_data_format("NDHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_3d_op);
    std::vector<Operator> inputs{avg_pool_3d_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPool3DFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool match = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool3D") {
            match = true;
        }
    }
    EXPECT_EQ(match, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPool3DFusionPass"].GetMatchTimes(), 0);
    EXPECT_EQ(graphFusionInfoMap["AvgPool3DFusionPass"].GetEffectTimes(), 0);
}

TEST_F(avg_pool_3d_fusion_test, avg_pool_3d_fusion_static_case0) {
    ge::Graph graph("avg_pool_3d_fusion_static_case0");
    auto avg_pool_3d_input_data = op::Data("avg_pool_3d_input_data");
    std::vector<int64_t> dims{1, 28, 28, 28, 1};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    avg_pool_3d_input_data.update_input_desc_x(tensorDesc);
    avg_pool_3d_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_3d_op = op::AvgPool3D("avgpool3d_0");
    avg_pool_3d_op.set_input_x(avg_pool_3d_input_data);
    avg_pool_3d_op.set_attr_ksize({1, 1, 1, 1, 1});
    avg_pool_3d_op.set_attr_strides({1, 2, 2, 2, 1});
    avg_pool_3d_op.set_attr_pads({0, 0, 0, 0, 0, 0,});
    avg_pool_3d_op.set_attr_ceil_mode(false);
    avg_pool_3d_op.set_attr_count_include_pad(false);
    avg_pool_3d_op.set_attr_divisor_override(0);
    avg_pool_3d_op.set_attr_data_format("NDHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_3d_op);
    std::vector<Operator> inputs{avg_pool_3d_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPool3DFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool match = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool3DD") {
            match = true;
        }
    }
    EXPECT_EQ(match, true);
}

TEST_F(avg_pool_3d_fusion_test, avg_pool_3d_fusion_dynamic_case0) {
    ge::Graph graph("avg_pool_3d_fusion_dynamic_case0");
    auto avg_pool_3d_input_data = op::Data("avg_pool_3d_input_data");
    std::vector<int64_t> dims{-1, 28, 28, 28, 1};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    avg_pool_3d_input_data.update_input_desc_x(tensorDesc);
    avg_pool_3d_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_3d_op = op::AvgPool3D("avgpool3d_0");
    avg_pool_3d_op.set_input_x(avg_pool_3d_input_data);
    avg_pool_3d_op.set_attr_ksize({1, 1, 1, 1, 1});
    avg_pool_3d_op.set_attr_strides({1, 2, 2, 2, 1});
    avg_pool_3d_op.set_attr_pads({0, 0, 0, 0, 0, 0,});
    avg_pool_3d_op.set_attr_ceil_mode(false);
    avg_pool_3d_op.set_attr_count_include_pad(false);
    avg_pool_3d_op.set_attr_divisor_override(0);
    avg_pool_3d_op.set_attr_data_format("NDHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_3d_op);
    std::vector<Operator> inputs{avg_pool_3d_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPool3DFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool match = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool3D") {
            match = true;
        }
    }
    EXPECT_EQ(match, true);
}
