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
#include "nn_norm_ops.h"
#include "fp16_t.hpp"
#include "reduce_ops.h"

using namespace ge;
using namespace op;

class reducemax_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(reducemax_fusion_test, reducemax_fusion_test_1) {
    ge::Graph graph("reducemax_fusion_test_1");
    auto reducemax_input_data = op::Data("reducemax_input_data");
    std::vector<int64_t> dims{64,96,3,3};
    ge::Shape shape(dims);
    std::vector<int64_t> dims1{64,96};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    reducemax_input_data.update_input_desc_x(tensorDesc);
    reducemax_input_data.update_output_desc_y(tensorDesc1);
    auto reducemax_op = op::ReduceMaxD("ReduceMaxD");
    reducemax_op.set_input_x(reducemax_input_data);
    reducemax_op.set_attr_axes({2,-1});
    reducemax_op.set_attr_keep_dims(false);
   
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(reducemax_op);
    std::vector<Operator> inputs{reducemax_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ReduceMaxDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool reducemax = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ReduceMaxD") {
            reducemax = true;
        }
    }
    EXPECT_EQ(reducemax, true);
}
TEST_F(reducemax_fusion_test, reducemax_fusion_test_2) {
    ge::Graph graph("reducemax_fusion_test_2");
    auto reducemax_input_data = op::Data("reducemax_input_data");
    std::vector<int64_t> dims{64,96,3,3};
    ge::Shape shape(dims);
    std::vector<int64_t> dims1{3,3};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    reducemax_input_data.update_input_desc_x(tensorDesc);
    reducemax_input_data.update_output_desc_y(tensorDesc1);
    auto reducemax_op = op::ReduceMaxD("ReduceMaxD");
    reducemax_op.set_input_x(reducemax_input_data);
    reducemax_op.set_attr_axes({0,1});
    reducemax_op.set_attr_keep_dims(false);
   
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(reducemax_op);
    std::vector<Operator> inputs{reducemax_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ReduceMaxDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool reducemax = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ReduceMaxD") {
            reducemax = true;
        }
    }
    EXPECT_EQ(reducemax, true);
}
TEST_F(reducemax_fusion_test, reducemax_fusion_test_3) {
    ge::Graph graph("reducemax_fusion_test_3");
    auto reducemax_input_data = op::Data("reducemax_input_data");
    std::vector<int64_t> dims{64,96,3,3};
    ge::Shape shape(dims);
    std::vector<int64_t> dims1{96,3};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    reducemax_input_data.update_input_desc_x(tensorDesc);
    reducemax_input_data.update_output_desc_y(tensorDesc1);
    auto reducemax_op = op::ReduceMaxD("ReduceMaxD");
    reducemax_op.set_input_x(reducemax_input_data);
    reducemax_op.set_attr_axes({0,2});
    reducemax_op.set_attr_keep_dims(false);
   
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(reducemax_op);
    std::vector<Operator> inputs{reducemax_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ReduceMaxDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool reducemax = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ReduceMaxD") {
            reducemax = true;
        }
    }
    EXPECT_EQ(reducemax, true);
}

