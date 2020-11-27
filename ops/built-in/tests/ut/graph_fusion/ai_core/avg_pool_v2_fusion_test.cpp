#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class avg_pool_v2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "avg_pool_v2_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "avg_pool_v2_fusion_test TearDown" << std::endl;
    }
};

TEST_F(avg_pool_v2_fusion_test, avg_pool_v2_fusion_test_1) {
    ge::Graph graph("avg_pool_v2_fusion_test_1");

    auto input_x_data = op::Data("input_x_data");
    std::vector<int64_t> dims_x{1, 16, 3, 3};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    input_x_data.update_input_desc_x(tensorDescX);
    input_x_data.update_output_desc_y(tensorDescX);

    auto avg_pool_v2_op = op::AvgPoolV2("avg_pool_v2_0");
    avg_pool_v2_op.set_input_x(input_x_data)
                  .set_attr_exclusive(false)
                  .set_attr_ksize({1,1,2,2})
                  .set_attr_strides({1,1,2,2});

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(avg_pool_v2_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMul = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 16, 2, 2};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findMul = true;
        }
        if (node->GetType() == "AvgPoolV2") {
             auto inputDesc = node->GetOpDesc()->GetInputDesc(1);
             std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
             if (dims == expectShape) {
                 shapeMatch = true;
             }
        }
    }
    EXPECT_EQ(findMul, false);
    EXPECT_EQ(shapeMatch, true);
}

TEST_F(avg_pool_v2_fusion_test, avg_pool_v2_fusion_test_2) {
    ge::Graph graph("avg_pool_v2_fusion_test_2");

    auto input_x_data = op::Data("input_x_data");
    std::vector<int64_t> dims_x{1, 16, 3, 3};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    input_x_data.update_input_desc_x(tensorDescX);
    input_x_data.update_output_desc_y(tensorDescX);

    auto avg_pool_v2_op = op::AvgPoolV2("avg_pool_v2_0");
    avg_pool_v2_op.set_input_x(input_x_data)
                  .set_attr_padding_mode("VALID")
                  .set_attr_ksize({1,1,2,2})
                  .set_attr_strides({1,1,2,2});

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(avg_pool_v2_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMul = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{1, 16, 2, 2};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findMul = true;
        }
        if (node->GetType() == "AvgPoolV2") {
             auto inputDesc = node->GetOpDesc()->GetInputDesc(1);
             std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
             if (dims == expectShape) {
                 shapeMatch = true;
             }
        }
    }
    EXPECT_EQ(findMul, false);
    EXPECT_EQ(shapeMatch, true);
}

// TEST_F(avg_pool_v2_fusion_test, avg_pool_v2_fusion_test_3) {
//     ge::Graph graph("avg_pool_v2_fusion_test_3");
// 
//     auto input_x_data = op::Data("input_x_data");
//     std::vector<int64_t> dims_x{1, 16, 3, 3};
//     ge::Shape shape_x(dims_x);
//     ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
//     input_x_data.update_input_desc_x(tensorDescX);
//     input_x_data.update_output_desc_y(tensorDescX);
// 
//     auto avg_pool_v2_op = op::AvgPoolV2("avg_pool_v2_0");
//     avg_pool_v2_op.set_input_x(input_x_data)
//                   .set_attr_padding_mode("SAME")
//                   .set_attr_ksize({1,1,2,2})
//                   .set_attr_strides({1,1,2,2});
// 
//     auto relu_op = op::Relu("relu_op");
//     relu_op.set_input_x(avg_pool_v2_op);
// 
//     std::vector<Operator> inputs{input_x_data};
//     std::vector<Operator> outputs{relu_op};
// 
//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
// 
//     bool findMul = false;
//     bool shapeMatch = false;
//     bool shapeMulMatch = false;
//     vector<int64_t> expectShape{1, 16, 2, 2};
//     vector<int64_t> expectMulShape{1, 1, 2, 2, 16};
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "Mul") {
//             findMul = true;
//             auto mulInputDesc = node->GetOpDesc()->GetInputDesc(1);
//             std::vector<int64_t> dimsMul = mulInputDesc.GetShape().GetDims();
//             if (dimsMul == expectMulShape) {
//                 shapeMulMatch = true;
//             }
//         }
//         if (node->GetType() == "AvgPoolV2") {
//              auto inputDesc = node->GetOpDesc()->GetInputDesc(1);
//              std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
//              if (dims == expectShape) {
//                  shapeMatch = true;
//              }
//         }
//     }
//     EXPECT_EQ(findMul, true);
//     EXPECT_EQ(shapeMatch, true);
// }

TEST_F(avg_pool_v2_fusion_test, avg_pool_v2_fusion_test_4) {
    ge::Graph graph("avg_pool_v2_fusion_test_4");

    auto input_x_data = op::Data("input_x_data");
    std::vector<int64_t> dims_x{1, 16, 3, 3};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    input_x_data.update_input_desc_x(tensorDescX);
    input_x_data.update_output_desc_y(tensorDescX);

    auto avg_pool_v2_op = op::AvgPoolV2("avg_pool_v2_0");
    avg_pool_v2_op.set_input_x(input_x_data)
                  .set_attr_global_pooling(true)
                  .set_attr_ksize({1,1,2,2})
                  .set_attr_strides({1,1,2,2});

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(avg_pool_v2_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMul = false;
    bool shapeMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findMul = true;
        }
        if (node->GetType() == "AvgPoolV2") {
             auto num = node->GetOpDesc()->GetInputsSize();
             if (num == 1) {
                 shapeMatch = true;
             }
        }
    }
    EXPECT_EQ(findMul, false);
    EXPECT_EQ(shapeMatch, true);
}

TEST_F(avg_pool_v2_fusion_test, avg_pool_v2_fusion_test_5) {
    ge::Graph graph("avg_pool_v2_fusion_test_5");

    auto input_x_data = op::Data("input_x_data");
    std::vector<int64_t> dims_x{1, 16, 3, 3};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    input_x_data.update_input_desc_x(tensorDescX);
    input_x_data.update_output_desc_y(tensorDescX);

    auto avg_pool_v2_op = op::AvgPoolV2("avg_pool_v2_0");
    avg_pool_v2_op.set_input_x(input_x_data)
                  .set_attr_ksize({1,1,3,3})
                  .set_attr_strides({1,1,2,2});

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(avg_pool_v2_op);

    std::vector<Operator> inputs{input_x_data};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMul = false;
    bool shapeMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findMul = true;
        }
        if (node->GetType() == "AvgPoolV2") {
             auto num = node->GetOpDesc()->GetInputsSize();
             if (num == 1) {
                 shapeMatch = true;
             }
        }
    }
    EXPECT_EQ(findMul, false);
    EXPECT_EQ(shapeMatch, true);
}

// TEST_F(avg_pool_v2_fusion_test, avg_pool_v2_fusion_test_6) {
//     ge::Graph graph("avg_pool_v2_fusion_test_6");
// 
//     auto input_x_data = op::Data("input_x_data");
//     std::vector<int64_t> dims_x{1, 16, 3, 3};
//     ge::Shape shape_x(dims_x);
//     ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
//     input_x_data.update_input_desc_x(tensorDescX);
//     input_x_data.update_output_desc_y(tensorDescX);
// 
//     auto avg_pool_v2_op = op::AvgPoolV2("avg_pool_v2_0");
//     avg_pool_v2_op.set_input_x(input_x_data)
//                   .set_attr_pads({1,1,1,1})
//                   .set_attr_ksize({1,1,2,2})
//                   .set_attr_strides({1,1,2,2});
// 
//     auto relu_op = op::Relu("relu_op");
//     relu_op.set_input_x(avg_pool_v2_op);
// 
//     std::vector<Operator> inputs{input_x_data};
//     std::vector<Operator> outputs{relu_op};
// 
//     graph.SetInputs(inputs).SetOutputs(outputs);
//     ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
//     fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
//     fe::FusionPassTestUtils::RunGraphFusionPass("AvgPoolV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
// 
//     bool findMul = false;
//     bool shapeMatch = false;
//     bool shapeMulMatch = false;
//     vector<int64_t> expectShape{1, 16, 2, 2};
//     vector<int64_t> expectMulShape{1, 1, 2, 2, 16};
//     for (auto node: compute_graph_ptr->GetAllNodes()) {
//         if (node->GetType() == "Mul") {
//             findMul = true;
//             auto mulInputDesc = node->GetOpDesc()->GetInputDesc(1);
//             std::vector<int64_t> dimsMul = mulInputDesc.GetShape().GetDims();
//             if (dimsMul == expectMulShape) {
//                 shapeMulMatch = true;
//             }
//         }
//         if (node->GetType() == "AvgPoolV2") {
//              auto inputDesc = node->GetOpDesc()->GetInputDesc(1);
//              std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
//              if (dims == expectShape) {
//                  shapeMatch = true;
//              }
//         }
//     }
//     EXPECT_EQ(findMul, true);
//     EXPECT_EQ(shapeMatch, true);
// }
