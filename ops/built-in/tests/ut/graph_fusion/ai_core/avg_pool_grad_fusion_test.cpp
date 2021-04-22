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

class avg_pool_grad_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "avg_pool_grad_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "avg_pool_grad_fusion_test TearDown" << std::endl;
    }
};

TEST_F(avg_pool_grad_fusion_test, avg_pool_grad_fusion_test_1) {
    ge::Graph graph("avg_pool_grad_fusion_test_1");
    auto avg_pool_grad_input_data = op::Data("avg_pool_grad_input_data");
    std::vector<int64_t> dims{32, 28, 28, 22};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    avg_pool_grad_input_data.update_input_desc_x(tensorDesc);
    avg_pool_grad_input_data.update_output_desc_y(tensorDesc);

    auto avg_pool_grad_input_ori_shape_data = op::Data("avg_pool_grad_input_ori_shape_data");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NHWC, ge::DT_INT32);
    avg_pool_grad_input_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    avg_pool_grad_input_ori_shape_data.update_output_desc_y(ori_tensorDesc);

    auto avg_pool_grad_op = op::AvgPoolGrad("avgpoolgrad_0");
    avg_pool_grad_op.set_input_orig_input_shape(avg_pool_grad_input_ori_shape_data);
    avg_pool_grad_op.set_input_input_grad(avg_pool_grad_input_data);
    avg_pool_grad_op.set_attr_ksize({1, 1, 1, 1});
    avg_pool_grad_op.set_attr_strides({1, 1, 1, 1});
    avg_pool_grad_op.set_attr_padding("VALID");
    avg_pool_grad_op.set_attr_data_format("NHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_grad_op);
    std::vector<Operator> inputs{avg_pool_grad_input_ori_shape_data, avg_pool_grad_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool match = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolGrad") {
            match = true;
        }
    }
    EXPECT_EQ(match, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolGradFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolGradFusionPass"].GetEffectTimes(), 1);
}

TEST_F(avg_pool_grad_fusion_test, avg_pool_grad_fusion_test_2) {
    ge::Graph graph("avg_pool_grad_fusion_test_1");
    auto avg_pool_grad_input_data = op::Data("avg_pool_grad_input_data");
    std::vector<int64_t> dims{32, 28, 28, 22};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    avg_pool_grad_input_data.update_input_desc_x(tensorDesc);
    avg_pool_grad_input_data.update_output_desc_y(tensorDesc);

    auto avg_pool_grad_input_ori_shape_data = op::Data("avg_pool_grad_input_ori_shape_data");
    std::vector<int64_t> ori_dims{-1};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NHWC, ge::DT_INT32);
    avg_pool_grad_input_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    avg_pool_grad_input_ori_shape_data.update_output_desc_y(ori_tensorDesc);

    auto avg_pool_grad_op = op::AvgPoolGrad("avgpoolgrad_0");
    avg_pool_grad_op.set_input_orig_input_shape(avg_pool_grad_input_ori_shape_data);
    avg_pool_grad_op.set_input_input_grad(avg_pool_grad_input_data);
    avg_pool_grad_op.set_attr_ksize({1, 1, 1, 1});
    avg_pool_grad_op.set_attr_strides({1, 1, 1, 1});
    avg_pool_grad_op.set_attr_padding("VALID");
    avg_pool_grad_op.set_attr_data_format("NHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_grad_op);
    std::vector<Operator> inputs{avg_pool_grad_input_ori_shape_data, avg_pool_grad_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool match = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolGrad") {
            match = true;
        }
    }
    EXPECT_EQ(match, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolGradFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolGradFusionPass"].GetEffectTimes(), 0);
}
//avg_pool_grad_input_data shape = 1 and [0]th = -2
TEST_F(avg_pool_grad_fusion_test, avg_pool_grad_fusion_test_3) {
    ge::Graph graph("avg_pool_grad_fusion_test_1");
    auto avg_pool_grad_input_data = op::Data("avg_pool_grad_input_data");
    std::vector<int64_t> dims{-2};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    avg_pool_grad_input_data.update_input_desc_x(tensorDesc);
    avg_pool_grad_input_data.update_output_desc_y(tensorDesc);

    auto avg_pool_grad_input_ori_shape_data = op::Data("avg_pool_grad_input_ori_shape_data");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NHWC, ge::DT_INT32);
    avg_pool_grad_input_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    avg_pool_grad_input_ori_shape_data.update_output_desc_y(ori_tensorDesc);

    auto avg_pool_grad_op = op::AvgPoolGrad("avgpoolgrad_0");
    avg_pool_grad_op.set_input_orig_input_shape(avg_pool_grad_input_ori_shape_data);
    avg_pool_grad_op.set_input_input_grad(avg_pool_grad_input_data);
    avg_pool_grad_op.set_attr_ksize({1, 1, 1, 1});
    avg_pool_grad_op.set_attr_strides({1, 1, 1, 1});
    avg_pool_grad_op.set_attr_padding("VALID");
    avg_pool_grad_op.set_attr_data_format("NHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_grad_op);
    std::vector<Operator> inputs{avg_pool_grad_input_ori_shape_data, avg_pool_grad_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool match = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPoolGrad") {
            match = true;
        }
    }
    EXPECT_EQ(match, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolGradFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolGradFusionPass"].GetEffectTimes(), 0);
}
