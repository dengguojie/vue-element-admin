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

using namespace ge;
using namespace op;

class batch_norm3d_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "batch_norm3d_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "batch_norm3d_fusion_test TearDown" << std::endl;
    }
};

TEST_F(batch_norm3d_fusion_test, batch_norm3d_fusion_test_1) {
    ge::Graph graph("batch_norm3d_fusion_test_1");

    auto bn_input_x_data = op::Data("bn_input_x_data");
    std::vector<int64_t> dims_x{1, 1, 32, 32, 16};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NDHWC,  DT_FLOAT);
    bn_input_x_data.update_input_desc_x(tensorDescX);
    bn_input_x_data.update_output_desc_y(tensorDescX);

    auto bn_input_scale_data = op::Data("bn_input_scale_data");
    auto bn_input_offset_data = op::Data("bn_input_offset_data");
    std::vector<int64_t> dims_scale{1,1,1,1,16};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NDHWC,  DT_FLOAT);
    bn_input_scale_data.update_input_desc_x(tensorDescScale);
    bn_input_scale_data.update_output_desc_y(tensorDescScale);

    bn_input_offset_data.update_input_desc_x(tensorDescScale);
    bn_input_offset_data.update_output_desc_y(tensorDescScale);

    auto const_mul = op::Constant("const_mul");
    Tensor consttensor;
    float * dataValue = new float[1];
    * dataValue = 0.1;
    consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC));
    consttensor.SetData((uint8_t*)dataValue, 4);
    const_mul.set_attr_value(consttensor);
    delete []dataValue;

    auto bn_op = op::BatchNorm3D("batchnorm3d_0");
    bn_op.set_input_x(bn_input_x_data)
         .set_input_scale(bn_input_scale_data)
         .set_input_offset(bn_input_offset_data)
         .set_attr_epsilon(0.0001)
         .set_attr_data_format("NDHWC")
         .set_attr_is_training(true);

    auto var_mean = op::Variable("var_mean");
    var_mean.update_output_desc_y(tensorDescScale);

    auto sub1_op = op::Sub("sub1_op");
    sub1_op.set_input_x1(var_mean)
           .set_input_x2(bn_op, "batch_mean");

    auto mul1_op = op::Mul("mul1_op");
    mul1_op.set_input_x1(const_mul)
           .set_input_x2(sub1_op);

    auto assignsub1_op = op::AssignSub("assignsub1_op");
    assignsub1_op.set_input_var(var_mean)
                 .set_input_value(mul1_op);
    // ----------------
    auto var_var = op::Variable("var_var");
    var_var.update_output_desc_y(tensorDescScale);


    auto sub2_op = op::Sub("sub2_op");
    sub2_op.set_input_x1(var_var)
           .set_input_x2(bn_op, "batch_variance");

    auto mul2_op = op::Mul("mul2_op");
    mul2_op.set_input_x1(const_mul)
           .set_input_x2(sub2_op);


    auto assignsub2_op = op::AssignSub("assignsub2_op");
    assignsub2_op.set_input_var(var_var)
                 .set_input_value(mul2_op);

    auto relu_op = op::Relu("relu_op");
    relu_op.set_input_x(bn_op, "y");

    std::vector<Operator> inputs{bn_input_x_data, bn_input_scale_data, bn_input_offset_data, const_mul, var_mean, var_var};
    std::vector<Operator> outputs{relu_op, assignsub1_op, assignsub2_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("FusedBatchnorm3DFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findBnreduce = false;
    bool findBnupdate = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BN3DTrainingReduce") {
            findBnreduce = true;
        }
        if (node->GetType() == "BN3DTrainingUpdate") {
            findBnupdate = true;
        }
    }
    EXPECT_EQ(findBnreduce, true);
    EXPECT_EQ(findBnupdate, true);
}

TEST_F(batch_norm3d_fusion_test, batch_norm3d_fusion_test_2) {
    ge::Graph graph("batch_norm3d_fusion_test_2");

    auto bn_input_x_data = op::Data("bn_input_x_data");
    std::vector<int64_t> dims_x{2,3,4,5,5};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    bn_input_x_data.update_input_desc_x(tensorDescX);
    bn_input_x_data.update_output_desc_y(tensorDescX);

    auto bn_input_scale_data = op::Data("bn_input_scale_data");
    auto bn_input_offset_data = op::Data("bn_input_offset_data");
    auto bn_input_mean_data = op::Data("bn_input_mean_data");
    auto bn_input_variance_data = op::Data("bn_input_variance_data");

    std::vector<int64_t> dims_scale{3};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NCHW,  DT_FLOAT);
    bn_input_scale_data.update_input_desc_x(tensorDescScale);
    bn_input_scale_data.update_output_desc_y(tensorDescScale);

    bn_input_offset_data.update_input_desc_x(tensorDescScale);
    bn_input_offset_data.update_output_desc_y(tensorDescScale);

    bn_input_mean_data.update_input_desc_x(tensorDescScale);
    bn_input_mean_data.update_output_desc_y(tensorDescScale);

    bn_input_variance_data.update_input_desc_x(tensorDescScale);
    bn_input_variance_data.update_output_desc_y(tensorDescScale);

    auto bn_op = op::BatchNorm("batchnorm");
    bn_op.set_input_x(bn_input_x_data)
         .set_input_scale(bn_input_scale_data)
         .set_input_offset(bn_input_offset_data)
         .set_input_mean(bn_input_mean_data)
         .set_input_variance(bn_input_variance_data)
         .set_attr_epsilon(0.0001)
         .set_attr_data_format("NCHW")
         .set_attr_is_training(false);
    
    bn_op.SetAttr("onnx", "onnx");
    std::vector<Operator> inputs{bn_input_x_data, bn_input_scale_data, bn_input_offset_data, bn_input_mean_data, bn_input_variance_data};
    std::vector<Operator> outputs{bn_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchNorm3DFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool bfind = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BatchNorm3D") {
            bfind = true;
        }
    }
    EXPECT_EQ(bfind, true);
}
