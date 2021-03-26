#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_detect_ops.h"
#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class sub_sample_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "sub_sample_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "sub_sample_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(sub_sample_fusion_pass_test, sub_sample_fusion_pass_test_1) {
    ge::Graph graph("sub_sample_fusion_pass_test_1");

    auto labels_data = op::Data("labels_data");
    std::vector<int64_t> dims_x{41153};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensor_desc_x(shape_x, FORMAT_ND,  DT_INT32);
    labels_data.update_input_desc_x(tensor_desc_x);
    labels_data.update_output_desc_y(tensor_desc_x);

    // get input data
    auto sub_sample_op = op::SubSample("SubSample");
    sub_sample_op.set_input_labels(labels_data);


    // init input and output data
    std::vector<Operator> inputs{labels_data};
    std::vector<Operator> outputs{sub_sample_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SubSamplePass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findNonSubSampleLabels = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "SubSampleLabels") {
            findNonSubSampleLabels = true;
        }
    }
    EXPECT_EQ(findNonSubSampleLabels, true);
}
