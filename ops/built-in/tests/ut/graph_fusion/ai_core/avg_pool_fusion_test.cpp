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

using namespace ge;
using namespace op;

class avg_pool_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

// TEST_F(avg_pool_fusion_test, avg_pool_fusion_test_1) {
//     ge::Graph graph("avg_pool_fusion_test_1");
//     auto avg_pool_input_data = op::Data("avg_pool_input_data");
//     std::vector<int64_t> dims{32, 28, 28, 22};
//     ge::Shape shape(dims);
//     ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
//     avg_pool_input_data.update_input_desc_x(tensorDesc);
//     avg_pool_input_data.update_output_desc_y(tensorDesc);
//     auto avg_pool_op = op::AvgPool("avgpool_0");
//     avg_pool_op.set_input_x(avg_pool_input_data);
//     avg_pool_op.set_attr_ksize({1, 1, 1, 1});
//     avg_pool_op.set_attr_strides({1, 2, 2, 1});
//     avg_pool_op.set_attr_padding("VALID");
//     avg_pool_op.set_attr_data_format("NHWC");
//     auto end_op = op::Square("end_op_0");
//     end_op.set_input_x(avg_pool_op);
//     std::vector<Operator> inputs{avg_pool_input_data};
//     std::vector<Operator> outputs{end_op};
//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
//     bool avgPoolMatch = false;
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "AvgPool") {
//             avgPoolMatch = true;
//         }
//     }
//     EXPECT_EQ(avgPoolMatch, true);
// }
// 
// TEST_F(avg_pool_fusion_test, avg_pool_fusion_test_2) {
//     ge::Graph graph("avg_pool_fusion_test_2");
//     auto avg_pool_input_data = op::Data("avg_pool_input_data");
//     std::vector<int64_t> dims{32, 14, 14, 88};
//     ge::Shape shape(dims);
//     ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
//     avg_pool_input_data.update_input_desc_x(tensorDesc);
//     avg_pool_input_data.update_output_desc_y(tensorDesc);
//     auto avg_pool_op = op::AvgPool("avgpool_0");
//     avg_pool_op.set_input_x(avg_pool_input_data);
//     avg_pool_op.set_attr_ksize({1, 2, 2, 1});
//     avg_pool_op.set_attr_strides({1, 1, 1, 1});
//     avg_pool_op.set_attr_padding("SAME");
//     avg_pool_op.set_attr_data_format("NHWC");
//     auto end_op = op::Square("end_op_0");
//     end_op.set_input_x(avg_pool_op);
//     std::vector<Operator> inputs{avg_pool_input_data};
//     std::vector<Operator> outputs{end_op};
//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
//     bool avgPoolMatch = false;
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "AvgPool") {
//                 avgPoolMatch= true;
//         }
//     }
//     EXPECT_EQ(avgPoolMatch, true);
// }
// 
// TEST_F(avg_pool_fusion_test, avg_pool_fusion_test_3) {
//     ge::Graph graph("avg_pool_fusion_test_3");
//     auto avg_pool_input_data = op::Data("avg_pool_input_data");
//     std::vector<int64_t> dims{2, 28, 28, 64};
//     ge::Shape shape(dims);
//     ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_INT8);
//     avg_pool_input_data.update_input_desc_x(tensorDesc);
//     avg_pool_input_data.update_output_desc_y(tensorDesc);
//     auto avg_pool_op = op::AvgPool("avgpool_0");
//     avg_pool_op.set_input_x(avg_pool_input_data);
//     avg_pool_op.set_attr_ksize({1, 1, 1, 1});
//     avg_pool_op.set_attr_strides({1, 2, 2, 1});
//     avg_pool_op.set_attr_padding("VALID");
//     avg_pool_op.set_attr_data_format("NHWC");
//     auto end_op = op::Square("end_op_0");
//     end_op.set_input_x(avg_pool_op);
//     std::vector<Operator> inputs{avg_pool_input_data};
//     std::vector<Operator> outputs{end_op};
//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
//     bool avgPoolMatch = false;
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "AvgPool") {
//             avgPoolMatch = true;
//         }
//     }
//     EXPECT_EQ(avgPoolMatch, true);
// }
// 
// TEST_F(avg_pool_fusion_test, avg_pool_fusion_test_4) {
//     ge::Graph graph("avg_pool_fusion_test_4");
//     auto avg_pool_input_data = op::Data("avg_pool_input_data");
//     std::vector<int64_t> dims{10, 16, 16, 128};
//     ge::Shape shape(dims);
//     ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_INT8);
//     avg_pool_input_data.update_input_desc_x(tensorDesc);
//     avg_pool_input_data.update_output_desc_y(tensorDesc);
//     auto avg_pool_op = op::AvgPool("avgpool_0");
//     avg_pool_op.set_input_x(avg_pool_input_data);
//     avg_pool_op.set_attr_ksize({1, 2, 2, 1});
//     avg_pool_op.set_attr_strides({1, 1, 1, 1});
//     avg_pool_op.set_attr_padding("SAME");
//     avg_pool_op.set_attr_data_format("NHWC");
//     auto end_op = op::Square("end_op_0");
//     end_op.set_input_x(avg_pool_op);
//     std::vector<Operator> inputs{avg_pool_input_data};
//     std::vector<Operator> outputs{end_op};
//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
//     bool avgPoolMatch = false;
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "AvgPool") {
//                 avgPoolMatch= true;
//         }
//     }
//     EXPECT_EQ(avgPoolMatch, true);
// }
