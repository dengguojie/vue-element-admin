#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "nn_batch_norm_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class batchnormpreprocess_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {
    }
};

TEST_F(batchnormpreprocess_fusion_test, batchnormpreprocess_fusion_test_1) {
    ge::Graph graph("batchnormpreprocess_fusion_test_1");

    std::vector<int64_t> dims{3, 32, 16, 16};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape);

    auto data_op_1 = op::Data("input_x");
    data_op_1.update_input_desc_x(tensorDesc);
    data_op_1.update_output_desc_y(tensorDesc);

    auto data_op_2 = op::Data("input_scale");
    data_op_2.update_input_desc_x(tensorDesc);
    data_op_2.update_output_desc_y(tensorDesc);

    auto data_op_3 = op::Data("input_offset");
    data_op_3.update_input_desc_x(tensorDesc);
    data_op_3.update_output_desc_y(tensorDesc);

    auto bn_op = op::BatchNorm("batchnorm");
    bn_op.SetAttr("data_format", "NHWC");
    bn_op.SetAttr("is_training", true);
    bn_op.set_input_x(data_op_1);
    bn_op.set_input_scale(data_op_2);
    bn_op.set_input_offset(data_op_3);

    auto cast_op = op::Cast("cast_op");
    cast_op.set_input_x(bn_op, "reserve_space_3")
           .set_attr_dst_type(1); // dst_type 1: float16

    std::vector<Operator> inputs{data_op_1, data_op_2, data_op_3};
    std::vector<Operator> outputs{cast_op};
    //std::vector<Operator> outputs{bn_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::Status status = fe::FusionPassTestUtils::RunGraphFusionPass("BatchNormPreprocessFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    EXPECT_EQ(status, fe::SUCCESS);
}


TEST_F(batchnormpreprocess_fusion_test, batchnormpreprocess_fusion_test_2) {
    ge::Graph graph("batchnormpreprocess_fusion_test_2");

    std::vector<int64_t> dims{3, 32, 16, 16};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape);

    auto data_op_1 = op::Data("input_y");
    data_op_1.update_input_desc_x(tensorDesc);
    data_op_1.update_output_desc_y(tensorDesc);

    auto data_op_2 = op::Data("input_x");
    data_op_2.update_input_desc_x(tensorDesc);
    data_op_2.update_output_desc_y(tensorDesc);

    auto data_op_3 = op::Data("input_scale");
    data_op_3.update_input_desc_x(tensorDesc);
    data_op_3.update_output_desc_y(tensorDesc);

    auto data_op_4 = op::Data("input_reserve_space_1");
    data_op_4.update_input_desc_x(tensorDesc);
    data_op_4.update_output_desc_y(tensorDesc);

    auto data_op_5 = op::Data("input_reserve_space_2");
    data_op_5.update_input_desc_x(tensorDesc);
    data_op_5.update_output_desc_y(tensorDesc);

    auto data_op_6 = op::Data("input_reserve_space_3");
    data_op_6.update_input_desc_x(tensorDesc);
    data_op_6.update_output_desc_y(tensorDesc);

    auto bn_grad_op = op::BatchNormGrad("batchnormgrad");
    bn_grad_op.SetAttr("data_format", "NHWC");
    bn_grad_op.SetAttr("is_training", true);
    bn_grad_op.set_input_y_backprop(data_op_1);
    bn_grad_op.set_input_x(data_op_2);
    bn_grad_op.set_input_scale(data_op_3);
    bn_grad_op.set_input_reserve_space_1(data_op_4);
    bn_grad_op.set_input_reserve_space_2(data_op_5);
    bn_grad_op.set_input_reserve_space_3(data_op_6);


    std::vector<Operator> inputs{data_op_1, data_op_2, data_op_3, data_op_4, data_op_5, data_op_6};
    //std::vector<Operator> inputs{data_op_1, data_op_2, data_op_3, data_op_4, data_op_5};
    std::vector<Operator> outputs{bn_grad_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::Status status = fe::FusionPassTestUtils::RunGraphFusionPass("BatchNormGradPreprocessFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    //EXPECT_EQ(status, fe::NOT_CHANGED);
    EXPECT_EQ(status, fe::SUCCESS);
}
