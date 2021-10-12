#include "nn_batch_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"
#include "elewise_calculation_ops.h"
#include "nn_calculation_ops.h"
#include "nonlinear_fuc_ops.h"

using namespace ge;
using namespace op;

class batchnorm_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() { std::cout << "batchnorm_fusion_pass_test SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "batchnorm_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(batchnorm_fusion_test, batchnorm_fusion_test_1) {
    ge::Graph graph("batchnorm_fusion_test_1");
    // const
    auto const_shape = ge::Shape({1});
    TensorDesc desc_add_1(const_shape, FORMAT_ND, DT_FLOAT16);
    Tensor const_tensor(desc_add_1);

    auto const1 = op::Const("const1")
        .set_attr_value(const_tensor);
    auto const2 = op::Const("const2")
        .set_attr_value(const_tensor);
    auto const3 = op::Const("const3")
        .set_attr_value(const_tensor);
    auto const4 = op::Const("const4")
        .set_attr_value(const_tensor);

    auto mul1 = op::Mul("mul1")
        .set_input_x1(const1)
        .set_input_x2(const2);

    auto mul2 = op::Mul("mul2")
        .set_input_x1(const3)
        .set_input_x2(mul1);

    auto sub1 = op::Sub("sub1")
        .set_input_x1(const4)
        .set_input_x2(mul2);

    auto shape_data = vector<int64_t>({1,3,9,9,1});
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NDHWC, DT_FLOAT16);

    // data op
    auto data = op::Data("data");
    data.update_input_desc_x(desc_data);
    data.update_output_desc_y(desc_data);

    TensorDesc weight_desc(ge::Shape({1,3,3,1,1}), FORMAT_DHWCN, DT_FLOAT16);
    Tensor weighttensor1(weight_desc);

    auto weight1 = op::Const().set_attr_value(weighttensor1);

    auto bias_shape = ge::Shape({1});
    TensorDesc desc_bias_1(bias_shape, FORMAT_ND, DT_FLOAT16);
    Tensor bias_tensor(desc_bias_1);
    auto conv_bias = op::Const("Conv3D/bias")
        .set_attr_value(bias_tensor);

    // conv3d op
    auto conv3d = op::Conv3D("Conv3d")
        .set_input_x(data)
        .set_input_filter(weight1)
        .set_input_bias(conv_bias)
        .set_attr_strides({1, 1, 1, 1, 1})
        .set_attr_pads({0, 0, 0, 0, 0, 0})
        .set_attr_data_format("NDHWC");

    TensorDesc conv3d_input_desc_x(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3d_input_desc_filter(ge::Shape(), FORMAT_DHWCN, DT_FLOAT16);
    TensorDesc conv3d_output_desc_y(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    conv3d.update_input_desc_x(conv3d_input_desc_x);
    conv3d.update_input_desc_filter(conv3d_input_desc_filter);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    auto mul3 = op::Mul("mul3")
         .set_input_x1(conv3d)
         .set_input_x2(mul1);

    auto add1 = op::Add("add1")
         .set_input_x1(mul3)
         .set_input_x2(sub1);

    auto relu_op = op::Relu("relu_op")
         .set_input_x(add1);

    std::vector<Operator> inputs{data};
    std::vector<Operator> outputs{relu_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ABatchnormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        std::cout << "node->GetType" << node->GetType() <<std::endl;
        if (node->GetType() == "BatchNorm") {
            findD = true;
        }
    }
    std::cout << "run batchnorm_fusion_test_1 end" <<std::endl;
    EXPECT_EQ(findD, true);
}

TEST_F(batchnorm_fusion_test, batchnorm_fusion_test_2) {
    ge::Graph graph("batchnorm_fusion_test_2");
    // const
    auto const_shape = ge::Shape({1});
    TensorDesc desc_add_1(const_shape, FORMAT_ND, DT_FLOAT16);
    Tensor const_tensor(desc_add_1);

    auto const1 = op::Const("const1")
        .set_attr_value(const_tensor);
    auto const2 = op::Const("const2")
        .set_attr_value(const_tensor);
    auto const3 = op::Const("const3")
        .set_attr_value(const_tensor);
    auto const4 = op::Const("const4")
        .set_attr_value(const_tensor);
    auto const5 = op::Const("const5")
        .set_attr_value(const_tensor);

    auto add1 = op::Add("add1")
        .set_input_x1(const1)
        .set_input_x2(const2);

    auto rsqrt1 = op::Rsqrt("rsqrt1")
        .set_input_x(add1);

    auto mul1 = op::Mul("mul1")
        .set_input_x1(rsqrt1)
        .set_input_x2(const3);

    auto mul2 = op::Mul("mul2")
        .set_input_x1(const4)
        .set_input_x2(mul1);

    auto sub1 = op::Sub("sub1")
        .set_input_x1(const5)
        .set_input_x2(mul2);

    auto shape_data = vector<int64_t>({1,3,9,9,1});
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NDHWC, DT_FLOAT16);

    // data op
    auto data = op::Data("data");
    data.update_input_desc_x(desc_data);
    data.update_output_desc_y(desc_data);

    TensorDesc weight_desc(ge::Shape({1,3,3,1,1}), FORMAT_DHWCN, DT_FLOAT16);
    Tensor weighttensor1(weight_desc);

    auto weight1 = op::Const().set_attr_value(weighttensor1);

    auto bias_shape = ge::Shape({1});
    TensorDesc desc_bias_1(bias_shape, FORMAT_ND, DT_FLOAT16);
    Tensor bias_tensor(desc_bias_1);
    auto conv_bias = op::Const("Conv3D/bias")
        .set_attr_value(bias_tensor);

    // conv3d op
    auto conv3d = op::Conv3D("Conv3d")
        .set_input_x(data)
        .set_input_filter(weight1)
        .set_input_bias(conv_bias)
        .set_attr_strides({1, 1, 1, 1, 1})
        .set_attr_pads({0, 0, 0, 0, 0, 0})
        .set_attr_data_format("NDHWC");

    TensorDesc conv3d_input_desc_x(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3d_input_desc_filter(ge::Shape(), FORMAT_DHWCN, DT_FLOAT16);
    TensorDesc conv3d_output_desc_y(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    conv3d.update_input_desc_x(conv3d_input_desc_x);
    conv3d.update_input_desc_filter(conv3d_input_desc_filter);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    auto mul3 = op::Mul("mul3")
         .set_input_x1(conv3d)
         .set_input_x2(mul1);

    auto add2 = op::Add("add2")
         .set_input_x1(mul3)
         .set_input_x2(sub1);

    auto relu_op = op::Relu("relu_op")
         .set_input_x(add2);

    std::vector<Operator> inputs{data};
    std::vector<Operator> outputs{relu_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ABatchnormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        std::cout << "node->GetType" << node->GetType() <<std::endl;
        if (node->GetType() == "BatchNorm") {
            findD = true;
        }
    }
    std::cout << "run batchnorm_fusion_test_1 end" <<std::endl;
    EXPECT_EQ(findD, true);
}
