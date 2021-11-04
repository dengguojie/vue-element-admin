#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class reduce_mean_var_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "reduce_mean_var_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "reduce_mean_var_fusion_test TearDown" << std::endl;
    }
};

TEST_F(reduce_mean_var_fusion_test, reduce_mean_var_fusion_test_1) {
    ge::Graph graph("reduce_mean_var_fusion_test_1");

    std::vector<int64_t> axes;
    axes.push_back(1);
    axes.push_back(2);
    axes.push_back(3);
    auto data0 = op::Data().set_attr_index(0);
    auto mean0 = op::ReduceMeanD("mean0")
                        .set_input_x(data0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);
    auto squared0 = op::SquaredDifference("squared0")
                        .set_input_x1(mean0)
                        .set_input_x2(data0);
    auto mean1 = op::ReduceMeanD("mean1")
                        .set_input_x(squared0)
                        .set_attr_axes(axes)
                        .set_attr_keep_dims(true);

    std::vector<int64_t> data0_vec{4, 224, 224, 160, 32};
    ge::Shape data0_shape(data0_vec);
    ge::TensorDesc data0_desc(data0_shape, FORMAT_NDHWC,  DT_FLOAT);
    data0.update_input_desc_x(data0_desc);
    data0.update_output_desc_y(data0_desc);
    mean0.update_input_desc_x(data0_desc);
    std::vector<Operator> inputs{data0};
    std::vector<Operator> outputs{mean0};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZReduceMeanVarianceFusionPass",
                                                fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findOp = false;
    bool shapeMatch = false;
    vector<int64_t> expectShape{4, 224, 224, 160, 32};
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ReduceMeanVariance") {
            findOp = true;
            auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
            std::vector<int64_t> dims = inputDesc.GetShape().GetDims();
            if (dims == expectShape) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(findOp, true);
    EXPECT_EQ(shapeMatch, true);
}





