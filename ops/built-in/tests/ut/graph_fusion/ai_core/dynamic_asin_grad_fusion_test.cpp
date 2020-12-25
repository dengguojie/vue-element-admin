#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;
using namespace std;

class dynamic_asin_grad_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_asin_grad SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_asin_grad TearDown" << std::endl;
    }
};

TEST_F(dynamic_asin_grad_fusion_test, dynamic_asin_grad_fusion_test_1) {
    ge::Graph graph("dynamic_asin_grad_1");

    // init input shape and range
    ge::Shape y_shape(std::vector<int64_t>{-1, -1});
    std::vector<std::pair<int64_t, int64_t>> input_y_range;
    input_y_range.push_back(std::pair<int64_t, int64_t>{1, 2});
    input_y_range.push_back(std::pair<int64_t, int64_t>{2, 3});
    
    ge::TensorDesc input_y_tensor_desc(y_shape, FORMAT_ND, DT_FLOAT16);
    ge::TensorDesc input_dy_tensor_desc(y_shape, FORMAT_ND, DT_FLOAT16);
    input_y_tensor_desc.SetShapeRange(input_y_range);
    input_dy_tensor_desc.SetShapeRange(input_y_range);

    auto y_data = op::Data("y_data");
    y_data.update_input_desc_x(input_y_tensor_desc);
    y_data.update_output_desc_y(input_y_tensor_desc);

    auto dy_data = op::Data("dy_data");
    dy_data.update_input_desc_x(input_dy_tensor_desc);
    dy_data.update_output_desc_y(input_dy_tensor_desc);

    auto output_op = op::AsinGrad("AsinGrad_1");
    output_op.set_input_y(y_data);
    output_op.set_input_dy(dy_data);

    std::vector<Operator> inputs{y_data, dy_data};
    std::vector<Operator> outputs{output_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AsinGrad", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool is_shape_match = false;
    bool is_range_match = false;
    vector<int64_t> expect_shape{-1, -1};
    std::vector<std::pair<int64_t, int64_t>> expect_range = input_y_range;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "AsinGrad") {
            auto output_desc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> output_dims = output_desc.GetShape().GetDims();
            if (output_dims == expect_shape) {
                is_shape_match = true;
            }
            std::vector<std::pair<int64_t, int64_t>> output_range;
            output_desc.GetShapeRange(output_range);
            if (output_range == expect_range) {
                is_range_match = true;
            }
        }
    }

    EXPECT_EQ(is_shape_match, true);
    EXPECT_EQ(is_range_match, true);
}
