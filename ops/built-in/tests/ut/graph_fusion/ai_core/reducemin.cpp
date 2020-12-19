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

class reducemin_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(reducemin_fusion_test, reducemin_fusion_test_1) {
    ge::Graph graph("reducemin_fusion_test_1");
    auto reducemin_input_data = op::Data("reducemin_input_data");
    std::vector<int64_t> dims{64,96,3,3};
    ge::Shape shape(dims);
    std::vector<int64_t> dims1{64,96};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    reducemin_input_data.update_input_desc_x(tensorDesc);
    reducemin_input_data.update_output_desc_y(tensorDesc1);
    auto reducemin_op = op::ReduceMinD("ReduceMinD");
    reducemin_op.set_input_x(reducemin_input_data);
    reducemin_op.set_attr_axes({2,-1});
    reducemin_op.set_attr_keep_dims(false);
   
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(reducemin_op);
    std::vector<Operator> inputs{reducemin_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ReduceMinDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool reducemin = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ReduceMinD") {
            reducemin = true;
        }
    }
    EXPECT_EQ(reducemin, true);
}
TEST_F(reducemin_fusion_test, reducemin_fusion_test_2) {
    ge::Graph graph("reducemin_fusion_test_2");
    auto reducemin_input_data = op::Data("reducemin_input_data");
    std::vector<int64_t> dims{64,96,3,3};
    ge::Shape shape(dims);
    std::vector<int64_t> dims1{1,1,3,3};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    reducemin_input_data.update_input_desc_x(tensorDesc);
    reducemin_input_data.update_output_desc_y(tensorDesc1);
    auto reducemin_op = op::ReduceMinD("ReduceMinD");
    reducemin_op.set_input_x(reducemin_input_data);
    reducemin_op.set_attr_axes({0,1});
    reducemin_op.set_attr_keep_dims(true);
   
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(reducemin_op);
    std::vector<Operator> inputs{reducemin_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ReduceMinDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool reducemin = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ReduceMinD") {
            reducemin = true;
        }
    }
    EXPECT_EQ(reducemin, true);
}
TEST_F(reducemin_fusion_test, reducemin_fusion_test_3) {
    ge::Graph graph("reducemin_fusion_test_3");
    auto reducemin_input_data = op::Data("reducemin_input_data");
    std::vector<int64_t> dims{64,96,3,3};
    ge::Shape shape(dims);
    std::vector<int64_t> dims1{1,96,1,3};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    reducemin_input_data.update_input_desc_x(tensorDesc);
    reducemin_input_data.update_output_desc_y(tensorDesc1);
    auto reducemin_op = op::ReduceMinD("ReduceMinD");
    reducemin_op.set_input_x(reducemin_input_data);
    reducemin_op.set_attr_axes({0,2});
    reducemin_op.set_attr_keep_dims(true);
   
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(reducemin_op);
    std::vector<Operator> inputs{reducemin_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ReduceMinDFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool reducemin = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ReduceMinD") {
            reducemin = true;
        }
    }
    EXPECT_EQ(reducemin, true);
}


