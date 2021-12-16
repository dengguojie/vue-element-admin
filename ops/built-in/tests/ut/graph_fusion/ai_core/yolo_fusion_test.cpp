#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nn_detect_ops.h"

using namespace ge;
using namespace op;

class yolo_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "yolo_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "yolo_fusion_test TearDown" << std::endl;
    }
};

TEST_F(yolo_fusion_test, yolo_fusion_test_1) {
    ge::Graph graph("yolo_fusion_test_1");
    auto xdata = op::Data("x");
    std::vector<int64_t> dims{10, 10, 10, 10, 10};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc0(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    xdata.update_input_desc_x(tensorDesc0);
    xdata.update_output_desc_y(tensorDesc0);
    auto yolo_op = op::Yolo("Yolo_1");
    yolo_op.set_input_x(xdata);
    std::vector<Operator> inputs{xdata};
    std::vector<Operator> outputs{yolo_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("YoloPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findYolo = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Yolo") {
            findYolo = true;
        }
    }
    EXPECT_EQ(findYolo, true);
}
