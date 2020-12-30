#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "state_ops.h"
#include "nn_batch_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class single_batch_norm_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "single_batch_norm_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "single_batch_norm_fusion_test TearDown" << std::endl;
    }
};

TEST_F(single_batch_norm_fusion_test, single_batch_norm_fusion_test_1) {
    ge::Graph graph("single_batch_norm_fusion_test_1");

    auto bn_input_x_data = op::Data("bn_input_x_data");
    std::vector<int64_t> dims_x{1, 2, 3, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT);
    bn_input_x_data.update_input_desc_x(tensorDescX);
    bn_input_x_data.update_output_desc_y(tensorDescX);

    auto bn_input_scale_data = op::Data("bn_input_scale_data");
    auto bn_input_offset_data = op::Data("bn_input_offset_data");
    std::vector<int64_t> dims_scale{4};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NHWC,  DT_FLOAT);
    bn_input_scale_data.update_input_desc_x(tensorDescScale);
    bn_input_scale_data.update_output_desc_y(tensorDescScale);

    bn_input_offset_data.update_input_desc_x(tensorDescScale);
    bn_input_offset_data.update_output_desc_y(tensorDescScale);
    
    auto bn_op = op::BatchNorm("batchnorm_0");
    bn_op.set_input_x(bn_input_x_data)
         .set_input_scale(bn_input_scale_data)
         .set_input_offset(bn_input_offset_data)
         .set_attr_is_training(true);

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(bn_op, "y");

    std::vector<Operator> inputs{bn_input_x_data, bn_input_scale_data, bn_input_offset_data};
    std::vector<Operator> outputs{relu_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("SingleBatchNormFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findBnreduce = false;
    bool findBnupdate = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BNTrainingReduce") {
            findBnreduce = true;
        }
        if (node->GetType() == "BNTrainingUpdateV3") {
            findBnupdate = true;
        }
    }
    EXPECT_EQ(findBnreduce, true);
    EXPECT_EQ(findBnupdate, true);
}
