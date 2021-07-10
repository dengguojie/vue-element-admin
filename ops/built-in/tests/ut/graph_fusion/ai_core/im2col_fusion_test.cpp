#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class im2col_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "im2col_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "im2col_fusion_test TearDown" << std::endl;
  }
};

TEST_F(im2col_fusion_test, im2col_fusion_test_1) {
    ge::Graph graph("im2col_fusion_test_1");

    auto in_input_x_data = op::Data("diag_input_data");
    std::vector<int64_t> dims{4, 8, 900, 3};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    in_input_x_data.update_input_desc_x(tensorDesc);
    in_input_x_data.update_output_desc_y(tensorDesc);

    auto im2col_op = op::Im2col("im2col_0");
    im2col_op.set_input_x(in_input_x_data)
             .set_attr_ksizes({2,2})
             .set_attr_strides({1,1})
             .set_attr_dilations({1,1});

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(im2col_op);

    std::vector<Operator> inputs{in_input_x_data};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Im2colFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findIm2col = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Im2col") {
            findIm2col = true;
        }
    }
    EXPECT_EQ(findIm2col, true);
}

// TEST_F(im2col_fusion_test, im2col_fusion_test_2) {
//     ge::Graph graph("im2col_fusion_test_2");

//     auto in_input_x_data = op::Data("diag_input_data");
//     std::vector<int64_t> dims{1, 4, 8, 900, 3};
//     ge::Shape shape(dims);
//     ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
//     in_input_x_data.update_input_desc_x(tensorDesc);
//     in_input_x_data.update_output_desc_y(tensorDesc);

//     auto im2col_op = op::Im2col("im2col_0");
//     im2col_op.set_input_x(in_input_x_data)
//         .set_attr_axes({-1});

//     auto end_op = op::Square("end_op_0");
//     end_op.set_input_x(im2col_op);

//     std::vector<Operator> inputs{in_input_x_data};
//     std::vector<Operator> outputs{end_op};

//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("Im2colFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

//     bool findIm2col = false;
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "Im2col") {
//             findIm2col = true;
//         }
//     }
//     EXPECT_EQ(findIm2col, true);
// }

// TEST_F(im2col_fusion_test, im2col_fusion_test_3) {
//     ge::Graph graph("im2col_fusion_test_3");

//     auto in_input_x_data = op::Data("diag_input_data");
//     std::vector<int64_t> dims{1, 8732, 21};
//     ge::Shape shape(dims);
//     ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
//     in_input_x_data.update_input_desc_x(tensorDesc);
//     in_input_x_data.update_output_desc_y(tensorDesc);

//     auto im2col_op = op::Im2col("im2col_0");
//     im2col_op.set_input_x(in_input_x_data)
//         .set_attr_axes({-1});

//     auto end_op = op::Square("end_op_0");
//     end_op.set_input_x(im2col_op);

//     std::vector<Operator> inputs{in_input_x_data};
//     std::vector<Operator> outputs{end_op};

//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("Im2colFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

//     bool findIm2col = false;
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "Im2col") {
//             findIm2col = true;
//         }
//     }
//     EXPECT_EQ(findIm2col, true);
// }

// TEST_F(im2col_fusion_test, im2col_fusion_test_4) {
//     ge::Graph graph("im2col_fusion_test_4");

//     auto in_input_x_data = op::Data("diag_input_data");
//     std::vector<int64_t> dims{8, 8732, 81};
//     ge::Shape shape(dims);
//     ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
//     in_input_x_data.update_input_desc_x(tensorDesc);
//     in_input_x_data.update_output_desc_y(tensorDesc);

//     auto im2col_op = op::Im2col("im2col_0");
//     im2col_op.set_input_x(in_input_x_data)
//         .set_attr_axes({-1});

//     auto end_op = op::Square("end_op_0");
//     end_op.set_input_x(im2col_op);

//     std::vector<Operator> inputs{in_input_x_data};
//     std::vector<Operator> outputs{end_op};

//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("Im2colFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

//     bool findIm2col = false;
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "Im2col") {
//             findIm2col = true;
//         }
//     }
//     EXPECT_EQ(findIm2col, true);
// }

// TEST_F(im2col_fusion_test, im2col_fusion_test_5) {
//     ge::Graph graph("im2col_fusion_test_5");

//     auto in_input_x_data = op::Data("diag_input_data");
//     std::vector<int64_t> dims{2, 8732, 81};
//     ge::Shape shape(dims);
//     ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
//     in_input_x_data.update_input_desc_x(tensorDesc);
//     in_input_x_data.update_output_desc_y(tensorDesc);

//     auto im2col_op = op::Im2col("im2col_0");
//     im2col_op.set_input_x(in_input_x_data)
//         .set_attr_axes({-1});

//     auto end_op = op::Square("end_op_0");
//     end_op.set_input_x(im2col_op);

//     std::vector<Operator> inputs{in_input_x_data};
//     std::vector<Operator> outputs{end_op};

//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("Im2colFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

//     bool findIm2col = false;
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "Im2col") {
//             findIm2col = true;
//         }
//     }
//     EXPECT_EQ(findIm2col, true);
// }