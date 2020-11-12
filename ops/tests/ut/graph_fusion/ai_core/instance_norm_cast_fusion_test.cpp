#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "elewise_calculation_ops.h"
#include "reduce_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class instance_norm_cast_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "instance_norm_cast_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "instance_norm_cast_fusion_test TearDown" << std::endl;
    }
};

TEST_F(instance_norm_cast_fusion_test, instance_norm_cast_fusion_test_1) {
    ge::Graph graph("instance_norm_cast_fusion_test_1");

    auto in_input_x_data = op::Data("in_input_x_data");
    std::vector<int64_t> dims_x{1, 2, 3, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDescX);
    in_input_x_data.update_output_desc_y(tensorDescX);

    std::vector<int64_t> dims_scale{1, 2, 1, 1};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NHWC,  DT_FLOAT);

    auto const_mean = op::Const("const_mean");
    Tensor consttensor1;
    float * dataValue = new float[2]{0.1, 0.1};

    consttensor1.SetTensorDesc(tensorDescScale);
    consttensor1.SetData((uint8_t*)dataValue, 8);
    const_mean.set_attr_value(consttensor1);

    auto const_var = op::Const("const_var");
    Tensor consttensor2;
    consttensor2.SetTensorDesc(tensorDescScale);
    consttensor2.SetData((uint8_t*)dataValue, 8);
    const_var.set_attr_value(consttensor2);


    auto const_gamma = op::Const("const_gamma");
    Tensor consttensor3;
    consttensor3.SetTensorDesc(tensorDescScale);
    consttensor3.SetData((uint8_t*)dataValue, 8);
    const_gamma.set_attr_value(consttensor3);


    auto const_beta = op::Const("const_beta");
    Tensor consttensor4;
    consttensor4.SetTensorDesc(tensorDescScale);
    consttensor4.SetData((uint8_t*)dataValue, 8);
    const_beta.set_attr_value(consttensor4);
    delete []dataValue;


    auto in_op = op::INInferV2("infernorm_0");
    in_op.set_input_x(in_input_x_data)
        .set_input_gamma(const_gamma)
        .set_input_beta(const_beta)
        .set_input_mean(const_mean)
        .set_input_variance(const_var)
        .set_attr_epsilon(0.00001);

    auto end_var = op::Square("sqrt_var");
    end_var.set_input_x(in_op, "batch_variance");

    auto end_mean = op::Square("sqrt_mean");
    end_mean.set_input_x(in_op, "batch_mean");

    auto relu_y = op::Relu("relu_y");
    relu_y.set_input_x(in_op, "y");

    std::vector<Operator> inputs{in_input_x_data, const_gamma, const_beta, const_mean, const_var};
    std::vector<Operator> outputs{relu_y, end_mean, end_var};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("HostINFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findInInfer = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "INInferV2D") {
            findInInfer = true;
        }
    }
    EXPECT_EQ(findInInfer, true);
}

TEST_F(instance_norm_cast_fusion_test, instance_norm_cast_fusion_test_2) {
    ge::Graph graph("instance_norm_cast_fusion_test_2");

    auto in_input_x_data = op::Data("in_input_x_data");
    std::vector<int64_t> dims_x{1, 2, 3, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDescX);
    in_input_x_data.update_output_desc_y(tensorDescX);

    std::vector<int64_t> dims_scale{1, 2, 1, 1};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NHWC,  DT_FLOAT);

    auto const_mean = op::Const("const_mean");
    Tensor consttensor1;
    float * dataValue = new float[2]{0.3, 0.3};

    consttensor1.SetTensorDesc(tensorDescScale);
    consttensor1.SetData((uint8_t*)dataValue, 8);
    const_mean.set_attr_value(consttensor1);

    auto const_var = op::Const("const_var");
    Tensor consttensor2;
    consttensor2.SetTensorDesc(tensorDescScale);
    consttensor2.SetData((uint8_t*)dataValue, 8);
    const_var.set_attr_value(consttensor2);


    auto const_gamma = op::Const("const_gamma");
    Tensor consttensor3;
    consttensor3.SetTensorDesc(tensorDescScale);
    consttensor3.SetData((uint8_t*)dataValue, 8);
    const_gamma.set_attr_value(consttensor3);


    auto const_beta = op::Const("const_beta");
    Tensor consttensor4;
    consttensor4.SetTensorDesc(tensorDescScale);
    consttensor4.SetData((uint8_t*)dataValue, 8);
    const_beta.set_attr_value(consttensor4);
    delete []dataValue;


    auto in_op = op::INInferV2("infernorm_0");
    in_op.set_input_x(in_input_x_data)
        .set_input_gamma(const_gamma)
        .set_input_beta(const_beta)
        .set_input_mean(const_mean)
        .set_input_variance(const_var)
        .set_attr_epsilon(0.00001);

    auto end_var = op::Square("sqrt_var");
    end_var.set_input_x(in_op, "batch_variance");

    auto end_mean = op::Square("sqrt_mean");
    end_mean.set_input_x(in_op, "batch_mean");

    auto relu_y = op::Relu("relu_y");
    relu_y.set_input_x(in_op, "y");

    std::vector<Operator> inputs{in_input_x_data, const_gamma, const_beta, const_mean, const_var};
    std::vector<Operator> outputs{relu_y, end_mean, end_var};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("HostINFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findInInfer = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "INInferV2D") {
            findInInfer = true;
        }
    }
    EXPECT_EQ(findInInfer, true);
}

TEST_F(instance_norm_cast_fusion_test, instance_norm_cast_fusion_test_3) {
    ge::Graph graph("instance_norm_cast_fusion_test_3");

    auto in_input_x_data = op::Data("in_input_x_data");
    std::vector<int64_t> dims_x{1, 2, 3, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDescX);
    in_input_x_data.update_output_desc_y(tensorDescX);

    std::vector<int64_t> dims_scale{1, 2, 1, 1};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NHWC,  DT_FLOAT);

    auto const_mean = op::Const("const_mean");
    Tensor consttensor1;
    float * dataValue = new float[2]{0.4, 0.4};

    consttensor1.SetTensorDesc(tensorDescScale);
    consttensor1.SetData((uint8_t*)dataValue, 8);
    const_mean.set_attr_value(consttensor1);

    auto const_var = op::Const("const_var");
    Tensor consttensor2;
    consttensor2.SetTensorDesc(tensorDescScale);
    consttensor2.SetData((uint8_t*)dataValue, 8);
    const_var.set_attr_value(consttensor2);


    auto const_gamma = op::Const("const_gamma");
    Tensor consttensor3;
    consttensor3.SetTensorDesc(tensorDescScale);
    consttensor3.SetData((uint8_t*)dataValue, 8);
    const_gamma.set_attr_value(consttensor3);


    auto const_beta = op::Const("const_beta");
    Tensor consttensor4;
    consttensor4.SetTensorDesc(tensorDescScale);
    consttensor4.SetData((uint8_t*)dataValue, 8);
    const_beta.set_attr_value(consttensor4);
    delete []dataValue;


    auto in_op = op::INInferV2("infernorm_0");
    in_op.set_input_x(in_input_x_data)
        .set_input_gamma(const_gamma)
        .set_input_beta(const_beta)
        .set_input_mean(const_mean)
        .set_input_variance(const_var)
        .set_attr_epsilon(0.00001);

    auto end_var = op::Square("sqrt_var");
    end_var.set_input_x(in_op, "batch_variance");

    auto end_mean = op::Square("sqrt_mean");
    end_mean.set_input_x(in_op, "batch_mean");

    auto relu_y = op::Relu("relu_y");
    relu_y.set_input_x(in_op, "y");

    std::vector<Operator> inputs{in_input_x_data, const_gamma, const_beta, const_mean, const_var};
    std::vector<Operator> outputs{relu_y, end_mean, end_var};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("HostINFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findInInfer = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "INInferV2D") {
            findInInfer = true;
        }
    }
    EXPECT_EQ(findInInfer, true);
}

TEST_F(instance_norm_cast_fusion_test, instance_norm_cast_fusion_test_4) {
    ge::Graph graph("instance_norm_cast_fusion_test_4");

    auto in_input_x_data = op::Data("in_input_x_data");
    std::vector<int64_t> dims_x{1, 2, 3, 4};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW,  DT_FLOAT);
    in_input_x_data.update_input_desc_x(tensorDescX);
    in_input_x_data.update_output_desc_y(tensorDescX);

    std::vector<int64_t> dims_scale{1, 2, 1, 1};
    ge::Shape shape_scale(dims_scale);
    ge::TensorDesc tensorDescScale(shape_scale, FORMAT_NHWC,  DT_FLOAT);

    auto const_mean = op::Const("const_mean");
    Tensor consttensor1;
    float * dataValue = new float[2]{0.2, 0.2};

    consttensor1.SetTensorDesc(tensorDescScale);
    consttensor1.SetData((uint8_t*)dataValue, 8);
    const_mean.set_attr_value(consttensor1);

    auto const_var = op::Const("const_var");
    Tensor consttensor2;
    consttensor2.SetTensorDesc(tensorDescScale);
    consttensor2.SetData((uint8_t*)dataValue, 8);
    const_var.set_attr_value(consttensor2);


    auto const_gamma = op::Const("const_gamma");
    Tensor consttensor3;
    consttensor3.SetTensorDesc(tensorDescScale);
    consttensor3.SetData((uint8_t*)dataValue, 8);
    const_gamma.set_attr_value(consttensor3);


    auto const_beta = op::Const("const_beta");
    Tensor consttensor4;
    consttensor4.SetTensorDesc(tensorDescScale);
    consttensor4.SetData((uint8_t*)dataValue, 8);
    const_beta.set_attr_value(consttensor4);
    delete []dataValue;


    auto in_op = op::INInferV2("infernorm_0");
    in_op.set_input_x(in_input_x_data)
        .set_input_gamma(const_gamma)
        .set_input_beta(const_beta)
        .set_input_mean(const_mean)
        .set_input_variance(const_var)
        .set_attr_epsilon(0.00001);

    auto end_var = op::Square("sqrt_var");
    end_var.set_input_x(in_op, "batch_variance");

    auto end_mean = op::Square("sqrt_mean");
    end_mean.set_input_x(in_op, "batch_mean");

    auto relu_y = op::Relu("relu_y");
    relu_y.set_input_x(in_op, "y");

    std::vector<Operator> inputs{in_input_x_data, const_gamma, const_beta, const_mean, const_var};
    std::vector<Operator> outputs{relu_y, end_mean, end_var};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("HostINFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findInInfer = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "INInferV2D") {
            findInInfer = true;
        }
    }
    EXPECT_EQ(findInInfer, true);
}