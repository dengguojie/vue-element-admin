#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "reduce_ops.h"
#include "pad_ops.h"

using namespace ge;
using namespace op;

class pad_v2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "pad_v2_fusion SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "pad_v2_fusion TearDown" << std::endl;
    }
};

TEST_F(pad_v2_fusion_test, pad_v2_fusion_test_1) {
    ge::Graph graph("pad_v2_fusion_test_1");

    std::vector<int64_t> dims_input0{1, 256, 38, 8};
    ge::Shape shape_input0(dims_input0);
    ge::TensorDesc tensordesc_input0(shape_input0, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto input0 = op::Data("input0");
    input0.update_input_desc_x(tensordesc_input0);
    input0.update_output_desc_y(tensordesc_input0);

    std::vector<int64_t> dims_input1{-1, 2};
    ge::Shape shape_input1(dims_input1);
    ge::TensorDesc tensordesc_input1(shape_input1, ge::FORMAT_NHWC, ge::DT_INT32);
    auto input1 = op::Data("input1");
    input1.update_input_desc_x(tensordesc_input1);
    input1.update_output_desc_y(tensordesc_input1);

    std::vector<int64_t> dims_input2{1};
    ge::Shape shape_input2(dims_input2);
    ge::TensorDesc tensordesc_input2(shape_input2, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto input2 = op::Data("input2");
    input2.update_input_desc_x(tensordesc_input2);
    input2.update_output_desc_y(tensordesc_input2);


    std::vector<int64_t> dims_output{1, 256, 40, 10};
    ge::Shape shape_output(dims_output);
    ge::TensorDesc tensordesc_output(shape_output, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    auto pad_v2 = op::PadV2("PadV2");
    pad_v2.update_input_desc_x(tensordesc_input0);
    pad_v2.update_input_desc_paddings(tensordesc_input1);
    pad_v2.update_input_desc_constant_values(tensordesc_input2);
    pad_v2.update_output_desc_y(tensordesc_output);
    
    pad_v2.set_input_x(input0)
          .set_input_paddings(input1)
          .set_input_constant_values(input2);
    
    std::vector<Operator> inputs{input0, input1, input2};
    std::vector<Operator> outputs{pad_v2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("PadV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_pad_v2_mean = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "PadV2") {
            find_pad_v2_mean = true;
        }
    }
    EXPECT_EQ(find_pad_v2_mean, true);
}
