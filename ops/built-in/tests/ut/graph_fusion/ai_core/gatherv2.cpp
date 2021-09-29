#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class gatherv2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(gatherv2_fusion_test, gatherv2_fusion_test_1) {
    ge::Graph graph("gatherv2_fusion_test_1");
    std::vector<int64_t> dims{3, 32};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, FORMAT_NHWC, DT_INT64);

    auto cast_input_data = op::Data("cast_input_data");
    cast_input_data.update_input_desc_x(tensorDesc);
    cast_input_data.update_output_desc_y(tensorDesc);

    auto cast_op = op::Cast("cast");
    cast_op.set_input_x(cast_input_data)
           .set_attr_dst_type(3);

    std::vector<int64_t> grad_dims{3, 32};
    ge::Shape grad_shape(grad_dims);
    ge::TensorDesc tensorDesc2(grad_shape, FORMAT_NHWC, DT_FLOAT16);
    ge::TensorDesc tensorDesc3(grad_shape, FORMAT_NHWC, DT_INT64);
    auto gatherv2_data = op::Data("gatherv2_data");
    gatherv2_data.update_input_desc_x(tensorDesc2);
    gatherv2_data.update_output_desc_y(tensorDesc2);

    auto axis_data = op::Data("axis_data");
    axis_data.update_input_desc_x(tensorDesc3);
    axis_data.update_output_desc_y(tensorDesc3);

    auto gatherV2_op = op::GatherV2("gatherv2");
    gatherV2_op.set_input_x(gatherv2_data)
               .set_input_indices(cast_op)
	       .set_input_axis(axis_data)
	       .set_attr_batch_dims(0);

    std::vector<Operator> inputs{cast_input_data, gatherv2_data, axis_data};
    std::vector<Operator> outputs{gatherV2_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrGatherV2Fusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

}
