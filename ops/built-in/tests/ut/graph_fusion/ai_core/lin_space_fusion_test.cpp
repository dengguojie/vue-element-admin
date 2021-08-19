#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "random_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class lin_space_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "lin_space_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "lin_space_fusion_test TearDown" << std::endl;
    }
};

TEST_F(lin_space_fusion_test, lin_space_fusion_test_1) {
    ge::Graph graph("lin_space_fusion_test_1");
    auto lin_space_input_data = op::Data("lin_space_input_data");
    std::vector<int64_t> dims{128};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
    lin_space_input_data.update_input_desc_x(tensorDesc);
    lin_space_input_data.update_output_desc_y(tensorDesc);

    auto lin_space_input_data_1 = op::Data("lin_space_input_data_1");
    lin_space_input_data_1.update_input_desc_x(tensorDesc);
    lin_space_input_data_1.update_output_desc_y(tensorDesc);

    auto multiples_shape = ge::Shape({1});
    TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
    Tensor num_tensor(desc_input_size_1);
    uint32_t *nums_tensor_value = new uint32_t[1]{4};
    num_tensor.SetData((uint8_t *) nums_tensor_value, 1 * sizeof(uint32_t));

    auto num = op::Const("num").set_attr_value(num_tensor);

    auto linspace_op = op::LinSpace("linspace_op_0");
    linspace_op.set_input_start(lin_space_input_data);
    linspace_op.set_input_stop(lin_space_input_data_1);
    linspace_op.set_input_num(num);

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(linspace_op);
    std::vector<Operator> inputs{lin_space_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("LinSpaceFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool linspace_match = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "LinSpaceD") {
            linspace_match = true;
        }
    }
    EXPECT_EQ(linspace_match, true);
}
