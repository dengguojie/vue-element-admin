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

class a_reduce_all_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "a_reduce_all_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "a_reduce_all_fusion_test TearDown" << std::endl;
    }
};

TEST_F(a_reduce_all_fusion_test, a_reduce_all_fusion_test_1) {
    ge::Graph graph("a_reduce_all_fusion_test_1");

    auto reduce_all_data = op::Data("reduce_all_data");
    std::vector<int64_t> dims_x{1, 32, 34, 3};
    ge::Tensor begin_tensor;
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT);
    reduce_all_data.update_input_desc_x(tensorDescX);
    reduce_all_data.update_output_desc_y(tensorDescX);

    int32_t begin_size = tensorDescX.GetShape().GetShapeSize();
    tensorDescX.SetSize(begin_size * sizeof(int32_t));
    begin_tensor.SetTensorDesc(tensorDescX);
    int32_t* begin_data = nullptr;
    begin_data = new int32_t[begin_size];
    *(begin_data + 0) = 0;
    begin_tensor.SetData((uint8_t*)begin_data, begin_size * sizeof(int32_t));
    auto begin = op::Constant().set_attr_value(begin_tensor);

    delete []begin_data;

    auto reduce_all_ptr = op::ReduceAll("reduce_all");
    reduce_all_ptr.set_input_x(reduce_all_data).set_input_axes(begin).set_attr_keep_dims(true);

 
    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(reduce_all_ptr);

    std::vector<Operator> inputs{reduce_all_data, begin};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AReduceAllFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_reduce_all = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Relu") {
            find_reduce_all = true;
        }
    }
    EXPECT_EQ(find_reduce_all, true);
}

TEST_F(a_reduce_all_fusion_test, a_reduce_all_fusion_test_2) {
    ge::Graph graph("a_reduce_all_fusion_test_2");

    auto reduce_all_data = op::Data("reduce_all_data");
    std::vector<int64_t> dims_x{1, 1, 1};
    ge::Tensor begin_tensor;
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT);
    reduce_all_data.update_input_desc_x(tensorDescX);
    reduce_all_data.update_output_desc_y(tensorDescX);

    int32_t begin_size = tensorDescX.GetShape().GetShapeSize();
    tensorDescX.SetSize(begin_size * sizeof(int32_t));
    begin_tensor.SetTensorDesc(tensorDescX);
    int32_t* begin_data = nullptr;
    begin_data = new int32_t[begin_size];
    *(begin_data + 0) = 0;
    begin_tensor.SetData((uint8_t*)begin_data, begin_size * sizeof(int32_t));
    auto begin = op::Constant().set_attr_value(begin_tensor);

    delete []begin_data;

    auto reduce_all_ptr = op::ReduceAll("reduce_all");
    reduce_all_ptr.set_input_x(reduce_all_data).set_input_axes(begin).set_attr_keep_dims(false);

 
    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(reduce_all_ptr);

    std::vector<Operator> inputs{reduce_all_data, begin};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AReduceAllFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool find_reduce_all = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Relu") {
            find_reduce_all = true;
        }
    }
    EXPECT_EQ(find_reduce_all, true);
}
