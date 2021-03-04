#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class pow_2_square_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "pow_2_square_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "pow_2_square_fusion_test TearDown" << std::endl;
    }
};

TEST_F(pow_2_square_fusion_test, pow_2_square_fusion_test_1) {
    ge::Graph graph("pow_2_square_fusion_test_1");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{1024, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    auto yData = op::Data("yData");
    std::vector<int64_t> dims_y{1024, 4};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, FORMAT_ND,  DT_FLOAT);
    yData.update_input_desc_x(tensorDescY);
    yData.update_output_desc_y(tensorDescY);

    auto powOp = op::Pow("Pow_1");
    powOp.set_input_x1(xData);
    powOp.set_input_x2(yData);

    std::vector<Operator> inputs{xData, yData};
    std::vector<Operator> outputs{powOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Pow2SquareFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSquare = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Square") {
            findSquare = true;
        }
    }
    EXPECT_EQ(findSquare, false);
}

TEST_F(pow_2_square_fusion_test, pow_2_square_fusion_test_2) {
    ge::Graph graph("pow_2_square_fusion_test_2");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{1024, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    float *y_data = new float[1]{2.0};
    TensorDesc y_desc(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
    Tensor y_tensor(y_desc);
    y_tensor.SetData((uint8_t *)y_data, sizeof(float));
    delete[] y_data;
    auto y_const = op::Const("y_const").set_attr_value(y_tensor);
    auto powOp = op::Pow("Pow_2");
    powOp.set_input_x1(xData);
    powOp.set_input_x2(y_const);

    std::vector<Operator> inputs{xData, y_const};
    std::vector<Operator> outputs{powOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Pow2SquareFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSquare = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Square") {
            findSquare = true;
        }
    }
    EXPECT_EQ(findSquare, true);
}

TEST_F(pow_2_square_fusion_test, pow_2_square_fusion_test_3) {
    ge::Graph graph("pow_2_square_fusion_test_3");

    auto xData = op::Data("xData");
    std::vector<int64_t> dims_x{1024, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT);
    xData.update_input_desc_x(tensorDescX);
    xData.update_output_desc_y(tensorDescX);

    float *y_data = new float[1]{3.0};
    TensorDesc y_desc(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
    Tensor y_tensor(y_desc);
    y_tensor.SetData((uint8_t *)y_data, sizeof(float));
    delete[] y_data;
    auto y_const = op::Const("y_const").set_attr_value(y_tensor);
    auto powOp = op::Pow("Pow_3");
    powOp.set_input_x1(xData);
    powOp.set_input_x2(y_const);

    std::vector<Operator> inputs{xData, y_const};
    std::vector<Operator> outputs{powOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("Pow2SquareFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findSquare = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Square") {
            findSquare = true;
        }
    }
    EXPECT_EQ(findSquare, false);
}