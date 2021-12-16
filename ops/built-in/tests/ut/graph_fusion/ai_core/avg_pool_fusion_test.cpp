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
#include "fusion_pass_test_utils.h"
#include "quantize_ops.h"

using namespace ge;
using namespace op;

class avg_pool_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "avg_pool_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "avg_pool_fusion_test TearDown" << std::endl;
    }
};

TEST_F(avg_pool_fusion_test, avg_pool_fusion_test_1) {
    ge::Graph graph("avg_pool_fusion_test_1");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{32, 28, 28, 22};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(avg_pool_input_data);
    avg_pool_op.set_attr_ksize({1, 2, 2, 1});
    avg_pool_op.set_attr_strides({1, 2, 2, 1});
    avg_pool_op.set_attr_padding("VALID");
    avg_pool_op.set_attr_data_format("NHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
            avgPoolMatch = true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetEffectTimes(), 1);
}

TEST_F(avg_pool_fusion_test, avg_pool_fusion_test_2) {
    ge::Graph graph("avg_pool_fusion_test_2");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{32, 14, 14, 88};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(avg_pool_input_data);
    avg_pool_op.set_attr_ksize({1, 2, 2, 1});
    avg_pool_op.set_attr_strides({1, 1, 1, 1});
    avg_pool_op.set_attr_padding("SAME");
    avg_pool_op.set_attr_data_format("NHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
                avgPoolMatch= true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetEffectTimes(), 1);
}

TEST_F(avg_pool_fusion_test, avg_pool_fusion_test_3) {
    ge::Graph graph("avg_pool_fusion_test_3");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{2, 28, 28, 64};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_INT8);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(avg_pool_input_data);
    avg_pool_op.set_attr_ksize({1, 2, 2, 1});
    avg_pool_op.set_attr_strides({1, 2, 2, 1});
    avg_pool_op.set_attr_padding("VALID");
    avg_pool_op.set_attr_data_format("NHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
            avgPoolMatch = true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
}

TEST_F(avg_pool_fusion_test, avg_pool_fusion_test_4) {
    ge::Graph graph("avg_pool_fusion_test_4");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{10, 16, 16, 128};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_INT8);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(avg_pool_input_data);
    avg_pool_op.set_attr_ksize({1, 2, 2, 1});
    avg_pool_op.set_attr_strides({1, 1, 1, 1});
    avg_pool_op.set_attr_padding("SAME");
    avg_pool_op.set_attr_data_format("NHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
                avgPoolMatch= true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
}

// dynamic -2
TEST_F(avg_pool_fusion_test, avg_pool_fusion_dynamic_rank) {
    ge::Graph graph("avg_pool_fusion_dynamic_rank");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{-2};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(avg_pool_input_data);
    avg_pool_op.set_attr_ksize({1, 1, 2, 2});
    avg_pool_op.set_attr_strides({1, 1, 2, 2});
    avg_pool_op.set_attr_padding("VALID");
    avg_pool_op.set_attr_data_format("NHW");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
            avgPoolMatch = true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetEffectTimes(), 0);
}

// dynamic nhw
TEST_F(avg_pool_fusion_test, avg_pool_fusion_dynamic_nhw) {
    ge::Graph graph("avg_pool_fusion_dynamic_nhw");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{-1, 22, -1, -1};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(avg_pool_input_data);
    avg_pool_op.set_attr_ksize({1, 1, 2, 2});
    avg_pool_op.set_attr_strides({1, 1, 2, 2});
    avg_pool_op.set_attr_padding("VALID");
    avg_pool_op.set_attr_data_format("NCHW");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
            avgPoolMatch = true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetEffectTimes(), 1);
}

// dynamic n, stride> 63
TEST_F(avg_pool_fusion_test, avg_pool_fusion_dynamic_n) {
    ge::Graph graph("avg_pool_fusion_dynamic_n");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{-1, 22, 128, 128};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(avg_pool_input_data);
    avg_pool_op.set_attr_ksize({1, 1, 2, 2});
    avg_pool_op.set_attr_strides({1, 1, 64, 2});
    avg_pool_op.set_attr_padding("VALID");
    avg_pool_op.set_attr_data_format("NCHW");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
            avgPoolMatch = true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetEffectTimes(), 0);
}

// dynamic c
TEST_F(avg_pool_fusion_test, avg_pool_fusion_dynamic_c) {
    ge::Graph graph("avg_pool_fusion_dynamic_c");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{32, 14, 14, -1};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(avg_pool_input_data);
    avg_pool_op.set_attr_ksize({1, 2, 2, 1});
    avg_pool_op.set_attr_strides({1, 1, 1, 1});
    avg_pool_op.set_attr_padding("SAME");
    avg_pool_op.set_attr_data_format("NHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
                avgPoolMatch= true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetEffectTimes(), 0);
}

// kh mul kw great 255
TEST_F(avg_pool_fusion_test, avg_pool_fusion_dynamic_kh_mul_kw_great_255) {
    ge::Graph graph("avg_pool_fusion_dynamic_kh_mul_kw_great_255");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{-1, 22, -1, -1};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(avg_pool_input_data);
    avg_pool_op.set_attr_ksize({1, 1, 16, 16});
    avg_pool_op.set_attr_strides({1, 1, 2, 2});
    avg_pool_op.set_attr_padding("VALID");
    avg_pool_op.set_attr_data_format("NCHW");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
            avgPoolMatch = true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetEffectTimes(),0);
}

TEST_F(avg_pool_fusion_test, avg_pool_fusion_quant_test_2) {
    ge::Graph graph("avg_pool_fusion_test_1");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{32, 22, 28, 28};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);

    float scale = 1.0;
    float offset = 0.0;
    auto quant_op = op::AscendQuant("quant_op_0");
    quant_op.set_input_x(avg_pool_input_data)
            .set_attr_scale(scale)
            .set_attr_offset(offset);

    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(quant_op);
    avg_pool_op.set_attr_ksize({1, 2, 2, 1});
    avg_pool_op.set_attr_strides({1, 2, 2, 1});
    avg_pool_op.set_attr_padding("SAME");
    avg_pool_op.set_attr_data_format("NHWC");

    float deq_scale = 1.0;
    ge::Tensor scale_tensor(tensorDesc, reinterpret_cast<uint8_t*>(&deq_scale), sizeof(float));
    auto const_op = op::Const("deq_scale").set_attr_value(scale_tensor);
    auto deq_op = op::AscendDequant("deq_op_0");
    deq_op.set_input_x(avg_pool_op)
          .set_input_deq_scale(const_op)
          .set_attr_dtype(DT_FLOAT16);

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(deq_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
            avgPoolMatch = true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetEffectTimes(), 1);
}

TEST_F(avg_pool_fusion_test, avg_pool_fusion_quant_test_1) {
    ge::Graph graph("avg_pool_fusion_test_1");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{32, 28, 28, 22};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);

    float scale = 1.0;
    float offset = 0.0;
    auto quant_op = op::AscendQuant("quant_op_0");
    quant_op.set_input_x(avg_pool_input_data)
            .set_attr_scale(scale)
            .set_attr_offset(offset);

    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(quant_op);
    avg_pool_op.set_attr_ksize({1, 2, 2, 1});
    avg_pool_op.set_attr_strides({1, 2, 2, 1});
    avg_pool_op.set_attr_padding("VALID");
    avg_pool_op.set_attr_data_format("NHWC");

    float deq_scale = 1.0;
    ge::Tensor scale_tensor(tensorDesc, reinterpret_cast<uint8_t*>(&deq_scale), sizeof(float));
    auto const_op = op::Const("deq_scale").set_attr_value(scale_tensor);
    auto deq_op = op::AscendDequant("deq_op_0");
    deq_op.set_input_x(avg_pool_op)
          .set_input_deq_scale(const_op)
          .set_attr_dtype(DT_FLOAT16);

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(deq_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
            avgPoolMatch = true;
        }
    }
    EXPECT_EQ(avgPoolMatch, true);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetEffectTimes(), 1);
}

TEST_F(avg_pool_fusion_test, avg_pool_invalid_kernel_test) {
    ge::Graph graph("avg_pool_invalid_kernel_test");
    auto avg_pool_input_data = op::Data("avg_pool_input_data");
    std::vector<int64_t> dims{32, 28, 28, 22};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    avg_pool_input_data.update_input_desc_x(tensorDesc);
    avg_pool_input_data.update_output_desc_y(tensorDesc);
    auto avg_pool_op = op::AvgPool("avgpool_0");
    avg_pool_op.set_input_x(avg_pool_input_data);
    avg_pool_op.set_attr_ksize({1, 1, 1, 1});
    avg_pool_op.set_attr_strides({1, 1, 1, 1});
    avg_pool_op.set_attr_padding("VALID");
    avg_pool_op.set_attr_data_format("NHWC");
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(avg_pool_op);
    std::vector<Operator> inputs{avg_pool_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool avgPoolMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AvgPool") {
            avgPoolMatch = true;
        }
    }
    EXPECT_EQ(avgPoolMatch, false);
    std::map<std::string, fe::FusionInfo> graphFusionInfoMap;
    std::map<std::string, fe::FusionInfo> bufferFusionInfoMap;
    fe::FusionStatisticRecorder &fusionStatisticInst = fe::FusionStatisticRecorder::Instance();
    fusionStatisticInst.GetAndClearFusionInfo("0_0", graphFusionInfoMap, bufferFusionInfoMap);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetMatchTimes(), 1);
    EXPECT_EQ(graphFusionInfoMap["AvgPoolFusionPass"].GetEffectTimes(), 1);
}
