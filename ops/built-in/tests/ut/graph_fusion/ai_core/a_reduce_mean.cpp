#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "elewise_calculation_ops.h"
#include "state_ops.h"
#include "nn_batch_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "reduce_ops.h"
using namespace ge;
using namespace op;

class a_reduce_mean_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "a_reduce_mean_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "a_reduce_mean_fusion_test TearDown" << std::endl;
    }
};

TEST_F(a_reduce_mean_fusion_test, a_reduce_mean_fusion_test_1) {
    ge::Graph graph("a_reduce_mean_fusion_test_1");

    auto reduce_mean_data = op::Data("reduce_mean_data");
    std::vector<int64_t> dims_x{};
    ge::Tensor begin_tensor;
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT);
    reduce_mean_data.update_input_desc_x(tensorDescX);
    reduce_mean_data.update_output_desc_y(tensorDescX);

    auto axis_shape = ge::Shape({1});
    TensorDesc desc_input_size_1(axis_shape, FORMAT_ND, DT_INT32);
    Tensor axis_tensor(desc_input_size_1);
    uint32_t *axis_tensor_value = new uint32_t[1]{0};
    axis_tensor.SetData((uint8_t *) axis_tensor_value, 1 * sizeof(uint32_t));
    auto begin = op::Constant().set_attr_value(axis_tensor);

    delete []axis_tensor_value;

    auto reduce_sum_ptr = op::ReduceMean("reduce_mean");
    reduce_sum_ptr.set_input_x(reduce_mean_data).set_input_axes(begin).set_attr_keep_dims(false);

 
    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(reduce_sum_ptr);

    std::vector<Operator> inputs{reduce_mean_data, begin};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AReduceMeanFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_reduce_mean = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Relu") {
            find_reduce_mean = true;
        }
    }
    EXPECT_EQ(find_reduce_mean, true);
}
