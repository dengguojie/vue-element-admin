#include <algorithm>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "math_ops.h"
#include "elewise_calculation_ops.h"
#include "reduce_ops.h"
#include "state_ops.h"
#include "nn_batch_norm_ops.h"
#include "array_ops.h"
#include "nonlinear_fuc_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class fusedbatchnorminfgrad_fusion_pass_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "fusedbatchnorminfgrad_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fusedbatchnorminfgrad_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(fusedbatchnorminfgrad_fusion_pass_test, fusedbatchnorminfgrad_fusion_pass_test_1) {
    ge::Graph graph("fusedbatchnorminfgrad_fusion_pass_test");

    auto y_backprop = op::Data("y_backprop_data");
    auto bn_input_x_data = op::Data("bn_input_x_data");
    std::vector<int64_t> dims_x{2,3,4,2};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC,  DT_FLOAT);
    y_backprop.update_input_desc_x(tensorDescX);
    y_backprop.update_output_desc_y(tensorDescX);
    bn_input_x_data.update_input_desc_x(tensorDescX);
    bn_input_x_data.update_output_desc_y(tensorDescX);

    auto bn_input_scale_data = op::Data("bn_input_scale_data");
    auto bn_input_space_data_1 = op::Data("bn_input_space_data_1");
    auto bn_input_space_data_2 = op::Data("bn_input_space_data_2");
    std::vector<int64_t> dims_scale{2};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NHWC,  DT_FLOAT);
    bn_input_scale_data.update_input_desc_x(tensorDescScale);
    bn_input_scale_data.update_output_desc_y(tensorDescScale);

    bn_input_space_data_1.update_input_desc_x(tensorDescScale);
    bn_input_space_data_1.update_output_desc_y(tensorDescScale);

    bn_input_space_data_2.update_input_desc_x(tensorDescScale);
    bn_input_space_data_2.update_output_desc_y(tensorDescScale);

    auto const_mul = op::Constant("const_mul");
    Tensor consttensor;
    float * dataValue = new float[1];
    * dataValue = 0.1;
    consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC));
    consttensor.SetData((uint8_t*)dataValue, 4);
    const_mul.set_attr_value(consttensor);
    delete []dataValue;

    auto bn_op = op::BatchNormGrad("batchnormgrad_0");
    bn_op.set_input_y_backprop(y_backprop)
            .set_input_x(bn_input_x_data)
            .set_input_scale(bn_input_scale_data)
            .set_input_reserve_space_1(bn_input_space_data_1)
            .set_input_reserve_space_2(bn_input_space_data_2)
            .set_attr_epsilon(0.0001)
            .set_attr_data_format("NHWC")
            .set_attr_is_training(false);

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

    std::vector<Operator> inputs{y_backprop, bn_input_x_data, bn_input_scale_data, bn_input_space_data_1, 
                                 const_mul, bn_input_space_data_2, var_mean, var_var};
    std::vector<Operator> outputs{relu_op, assignsub1_op, assignsub2_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("BatchNormGradInfGradFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findBnreduce = false;
    bool findBnupdate = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BNInferGrad") {
            findBnreduce = true;
        }
        if (node->GetType() == "BNTrainingUpdateGrad") {
            findBnupdate = true;
        }
    }
    EXPECT_EQ(findBnreduce, true);
    EXPECT_EQ(findBnupdate, true);
}
