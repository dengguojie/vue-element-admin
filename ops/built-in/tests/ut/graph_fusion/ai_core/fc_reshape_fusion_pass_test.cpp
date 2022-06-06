#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "math_ops.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class fc_reshape_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "fc_reshape_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "fc_reshape_fusion_pass_test TearDown" << std::endl;
    }
};

// fusion, reshape -> fc ====> fc
TEST_F(fc_reshape_fusion_pass_test, fc_reshape_fusion_pass_test_1) {
    ge::Graph graph("fc_reshape_fusion_pass_test_1");

    auto X1Data = op::Data("x1");
    std::vector<int64_t> dims_x1{304, 324};
    ge::Shape shape_x1(dims_x1);
    ge::TensorDesc ReshapeInputTensorDesc(shape_x1, FORMAT_ND, DT_FLOAT16);
    X1Data.update_input_desc_x(ReshapeInputTensorDesc);

    std::vector<int64_t> reshape_output_dims{304, 324, 1, 1};
    ge::Shape reshape_output_shape(reshape_output_dims);
    ge::TensorDesc ReshapeOutputTensorDesc(reshape_output_shape, FORMAT_NCHW, DT_FLOAT16);

    auto reshape_op = op::Reshape("reshape");
    reshape_op.set_input_x(X1Data);
    reshape_op.update_input_desc_x(ReshapeInputTensorDesc);
    reshape_op.update_output_desc_y(ReshapeOutputTensorDesc);

    auto fc_weight_data = op::Data("fc_weight_data");
    std::vector<int64_t> weight_dims{304, 324, 1, 1};
    ge::Shape weight_shape(weight_dims);
    ge::TensorDesc weghtInputTensorDesc(weight_shape, FORMAT_FRACTAL_NZ, DT_INT8);
    fc_weight_data.update_input_desc_x(weghtInputTensorDesc);
    fc_weight_data.update_output_desc_y(weghtInputTensorDesc);
    auto fc_op = op::FullyConnection("fc").set_input_x(reshape_op)
                                          .set_input_w(fc_weight_data)
                                          .set_attr_num_output(1);

    std::vector<Operator> inputs{X1Data};
    std::vector<Operator> outputs{fc_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
     fe::FusionPassTestUtils::RunGraphFusionPass("AFullyConnectionReshapePass",
        fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    std::map<std::string, uint32_t> expected = {{"FullyConnection", 1}};
    std::map<std::string, uint32_t> actual;
    for (auto node : compute_graph_ptr->GetAllNodes()) {
        if (node->GetType().compare("FullyConnection") == 0) {
            actual[node->GetType()]++;
        }
    }
    ASSERT_EQ(expected, actual);
}

