#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "elewise_calculation_ops.h"
#include "state_ops.h"
#include "nn_batch_norm_ops.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class batch_norm_3d_grad_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "batch_norm_3d_grad_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "batch_norm_3d_grad_fusion_test TearDown" << std::endl;
    }
};

TEST_F(batch_norm_3d_grad_fusion_test, batch_norm_3d_grad_fusion_test_1) {
    ge::Graph graph("batch_norm_3d_grad_fusion_test_1");

    auto y_backpoop_data = op::Data("y_backpoop_data");
    auto x_data = op::Data("x_data");
    std::vector<int64_t> dim_y_backpoop{1, 1, 32, 32, 16};
    ge::Shape shape_y_backpoop(dim_y_backpoop);
    ge::TensorDesc tensorDescYback(shape_y_backpoop, FORMAT_NDHWC,  DT_FLOAT);
    y_backpoop_data.update_input_desc_x(tensorDescYback);
    y_backpoop_data.update_output_desc_y(tensorDescYback);

    x_data.update_input_desc_x(tensorDescYback);
    x_data.update_output_desc_y(tensorDescYback);

    auto bn_input_scale_data = op::Data("bn_input_scale_data");
    auto bn_input_reserve1_data = op::Data("bn_input_reserve1_data");
    auto bn_input_reserve2_data = op::Data("bn_input_reserve2_data");
    std::vector<int64_t> dims_scale{1,1,1,1,16};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NDHWC,  DT_FLOAT);
    bn_input_scale_data.update_input_desc_x(tensorDescScale);
    bn_input_scale_data.update_output_desc_y(tensorDescScale);

    bn_input_reserve1_data.update_input_desc_x(tensorDescScale);
    bn_input_reserve1_data.update_output_desc_y(tensorDescScale);

    bn_input_reserve2_data.update_input_desc_x(tensorDescScale);
    bn_input_reserve2_data.update_output_desc_y(tensorDescScale);

    auto batch_norm_3d_grad = op::BatchNorm3DGrad("batchnorm3dgrad_0");
    batch_norm_3d_grad.set_input_y_backprop(y_backpoop_data)
                      .set_input_x(x_data)
                      .set_input_scale(bn_input_scale_data)
                      .set_input_reserve_space_1(bn_input_reserve1_data)
                      .set_input_reserve_space_2(bn_input_reserve2_data)
                      .set_attr_epsilon(0.0001)
                      .set_attr_data_format("NDHWC")
                      .set_attr_is_training(true);



    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(batch_norm_3d_grad, "x_backprop");
    auto relu2_op = op::Relu("relu_op2");
    relu2_op.set_input_x(batch_norm_3d_grad, "scale_backprop");
    auto relu3_op = op::Relu("relu_op3");
    relu3_op.set_input_x(batch_norm_3d_grad, "offset_backprop");
    std::vector<Operator> inputs{y_backpoop_data,x_data, bn_input_scale_data, bn_input_reserve1_data, bn_input_reserve2_data};
    std::vector<Operator> outputs{relu_op, relu2_op, relu3_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("FusedBatchNorm3DGradFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findBnreduce = false;
    bool findBnupdate = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BN3DTrainingReduceGrad") {
            findBnreduce = true;
        }
        if (node->GetType() == "BN3DTrainingUpdateGrad") {
            findBnupdate = true;
        }
    }
    EXPECT_EQ(findBnreduce, true);
    EXPECT_EQ(findBnupdate, true);
}
