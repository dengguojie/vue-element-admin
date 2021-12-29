#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "state_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class biasadd_conv_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "biasadd_conv_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "biasadd_conv_fusion_test TearDown" << std::endl;
    }
};

/* conv2d + biasadd */
TEST_F(biasadd_conv_fusion_test, biasadd_conv_fusion_test_1) {
    ge::Graph graph("biasadd_conv_fusion_test_1");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 56, 56, 64};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
    conv_input_x_data.update_input_desc_x(tensorDescX);
    conv_input_x_data.update_output_desc_y(tensorDescX);

    auto conv_input_filter_data = op::Const("conv_input_filter_data");
    auto conv_input_bias_data = op::Const("conv_input_bias_data");

    Tensor conv_input_filter_tensor;
    float *conv_input_filter_tensor_value = new float[64 * 64];
    for (int i = 0; i < 64 * 64; i++) {
        *(conv_input_filter_tensor_value + i) = 0.1;
    }
    conv_input_filter_tensor.SetData((uint8_t *) conv_input_filter_tensor_value, 64 * 64 * 4);

    std::vector<int64_t> dims_filter{64, 1, 1, 64};
    ge::Shape shape_filter(dims_filter);
    ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_NHWC, DT_FLOAT);
    conv_input_filter_tensor.SetTensorDesc(tensorDescFilter);
    conv_input_filter_data.set_attr_value(conv_input_filter_tensor);

    auto const_add = op::Const("const_add");

    Tensor const_add_tensor;
    float *const_add_tensor_value = new float[64];
    for (int i = 0; i < 64; i++) {
        *(const_add_tensor_value + i) = 0.1;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 64 * 4);

    std::vector<int64_t> dims_add{64};
    ge::Shape shape_add(dims_add);
    ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NHWC, DT_FLOAT);
    const_add_tensor.SetTensorDesc(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data);
    conv_op.set_attr_data_format("NHWC");
    conv_op.set_attr_strides({1, 2, 2, 1});
    conv_op.set_attr_pads({1, 1, 1, 1});

    auto biasadd_op = op::BiasAdd("biasadd_op");
    biasadd_op.set_input_x(conv_op)
               .set_input_bias(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(biasadd_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(biasadd_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AABiasaddConvFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    bool biasaddFlag = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BiasAdd") {
            biasaddFlag = true;
        }
    }
    EXPECT_EQ(biasaddFlag, false);

    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* conv2d + add, add's second input is variable, which is not a qualified case. */
TEST_F(biasadd_conv_fusion_test, biasadd_conv_fusion_test_2) {
  ge::Graph graph("biasadd_conv_fusion_test_2");

  auto conv_input_x_data = op::Data("conv_input_x_data");
  std::vector<int64_t> dims_x{1, 56, 56, 64};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
  conv_input_x_data.update_input_desc_x(tensorDescX);
  conv_input_x_data.update_output_desc_y(tensorDescX);

  auto conv_input_filter_data = op::Variable("conv_input_filter_data");
  auto conv_input_bias_data = op::Variable("conv_input_bias_data");

  Tensor conv_input_filter_tensor;
  std::vector<int64_t> dims_filter{64, 1, 1, 64};
  ge::Shape shape_filter(dims_filter);
  ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_NHWC, DT_FLOAT);
  conv_input_filter_tensor.SetTensorDesc(tensorDescFilter);
  conv_input_filter_data.set_attr_value(conv_input_filter_tensor);

  auto variable_add = op::Variable("const_add");

  Tensor add_tensor;
  std::vector<int64_t> dims_add{64};
  ge::Shape shape_add(dims_add);
  ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NHWC, DT_FLOAT);
  add_tensor.SetTensorDesc(tensorDescAdd);
  variable_add.set_attr_value(add_tensor);

  auto conv_op = op::Conv2D("conv_1");
  conv_op.set_input_x(conv_input_x_data)
      .set_input_filter(conv_input_filter_data);
  conv_op.set_attr_data_format("NHWC");
  conv_op.set_attr_strides({1, 2, 2, 1});
  conv_op.set_attr_pads({1, 1, 1, 1});

  auto add_op = op::Add("Add");
  add_op.set_input_x1(conv_op)
      .set_input_x2(variable_add);


  std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, variable_add};
  std::vector<Operator> outputs{add_op};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("AABiasaddConvFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  // compute_graph_ptr->Dump();
  bool biasaddFlag = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Add") {
      biasaddFlag = true;
    }
  }
  EXPECT_EQ(biasaddFlag, true);
}

/* conv2d(with bias) + biasadd not supported*/
TEST_F(biasadd_conv_fusion_test, biasadd_conv_fusion_test_3) {
    ge::Graph graph("biasadd_conv_fusion_test_3");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 56, 56, 64};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
    conv_input_x_data.update_input_desc_x(tensorDescX);
    conv_input_x_data.update_output_desc_y(tensorDescX);

    auto conv_input_filter_data = op::Const("conv_input_filter_data");
    auto conv_input_bias_data = op::Const("conv_input_bias_data");

    Tensor conv_input_filter_tensor;
    float *conv_input_filter_tensor_value = new float[64 * 64];
    for (int i = 0; i < 64 * 64; i++) {
        *(conv_input_filter_tensor_value + i) = 0.1;
    }
    conv_input_filter_tensor.SetData((uint8_t *) conv_input_filter_tensor_value, 64 * 64 * 4);

    std::vector<int64_t> dims_filter{64, 1, 1, 64};
    ge::Shape shape_filter(dims_filter);
    ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_NHWC, DT_FLOAT);
    conv_input_filter_tensor.SetTensorDesc(tensorDescFilter);
    conv_input_filter_data.set_attr_value(conv_input_filter_tensor);

    Tensor dims_bias_tensor;
    float *dims_bias_tensor_value = new float[64];
    for (int i = 0; i < 64; i++) {
        *(dims_bias_tensor_value + i) = 0.1;
    }
    dims_bias_tensor.SetData((uint8_t *) dims_bias_tensor_value, 64 * 4);

    std::vector<int64_t> dims_bias{64};
    ge::Shape shape_bias(dims_bias);
    ge::TensorDesc tensorDescBias(shape_bias, FORMAT_NHWC, DT_FLOAT);
    dims_bias_tensor.SetTensorDesc(tensorDescBias);
    conv_input_bias_data.set_attr_value(dims_bias_tensor);

    auto const_add = op::Const("const_add");

    Tensor const_add_tensor;
    float *const_add_tensor_value = new float[64];
    for (int i = 0; i < 64; i++) {
        *(const_add_tensor_value + i) = 0.1;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 64 * 4);

    std::vector<int64_t> dims_add{64};
    ge::Shape shape_add(dims_add);
    ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NHWC, DT_FLOAT);
    const_add_tensor.SetTensorDesc(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data)
            .set_input_bias(conv_input_bias_data);
    conv_op.set_attr_data_format("NHWC");
    conv_op.set_attr_strides({1, 2, 2, 1});
    conv_op.set_attr_pads({1, 1, 1, 1});

    auto biasadd_op = op::BiasAdd("biasadd_op");
    biasadd_op.set_input_x(conv_op)
               .set_input_bias(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(biasadd_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(biasadd_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AABiasaddConvFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    bool biasaddFlag = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "BiasAdd") {
            biasaddFlag = true;
        }
    }
    EXPECT_EQ(biasaddFlag, true);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* conv3d + reshape */
TEST_F(biasadd_conv_fusion_test, biasadd_conv_fusion_test_4) {
    ge::Graph graph("biasadd_conv_fusion_test_4");

    auto leakyrelu_input_x_data = op::Data("leakyrelu_input_x_data");
    std::vector<int64_t> dims_x{4, 16, 64, 64, 64};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_FLOAT);
    leakyrelu_input_x_data.update_input_desc_x(tensorDescX);
    leakyrelu_input_x_data.update_output_desc_y(tensorDescX);

    auto resize_input_x_data = op::Data("resize_input_x_data");
    std::vector<int64_t> dims_y{4, 4, 64, 4096};
    ge::Shape shape_y(dims_y);
    ge::TensorDesc tensorDescY(shape_y, FORMAT_NCHW, DT_FLOAT);
    resize_input_x_data.update_input_desc_x(tensorDescY);
    resize_input_x_data.update_output_desc_y(tensorDescY);

    auto conv3d_filter_data = op::Const("conv3d_filter_data");

    Tensor conv3d_filter_Tensor;
    float *conv3d_filter_Tensor_value = new float[4 * 16];
    for (int i = 0; i < 4 * 16; i++) {
        *(conv3d_filter_Tensor_value + i) = 0.1;
    }
    conv3d_filter_Tensor.SetData((uint8_t *) conv3d_filter_Tensor_value, 4 * 16 * 4);

    std::vector<int64_t> dims_filter{4, 16, 1, 1, 1};
    ge::Shape shape_filter(dims_filter);
    ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_NCDHW, DT_FLOAT);
    conv3d_filter_Tensor.SetTensorDesc(tensorDescFilter);
    conv3d_filter_data.set_attr_value(conv3d_filter_Tensor);

    auto reshape_resize_data = op::Const("reshape_resize_data");

    Tensor reshape_resize_tensor;
    int64_t *reshape_resize_tensor_value = new int64_t[5];
    *(reshape_resize_tensor_value + 0) = 4;
    *(reshape_resize_tensor_value + 1) = 4;
    *(reshape_resize_tensor_value + 2) = 64;
    *(reshape_resize_tensor_value + 3) = 64;
    *(reshape_resize_tensor_value + 4) = 64;
    reshape_resize_tensor.SetData((uint8_t *) reshape_resize_tensor_value, 5 * 8);

    std::vector<int64_t> dims_reshape{4, 4, 64, 64, 64};
    ge::Shape shape_reshape(dims_reshape);
    ge::TensorDesc tensorDescReshape(shape_reshape, FORMAT_ND, DT_INT64);
    reshape_resize_tensor.SetTensorDesc(tensorDescReshape);
    reshape_resize_data.set_attr_value(reshape_resize_tensor);

    auto conv_op = op::Conv3D("conv_1");
    conv_op.set_input_x(leakyrelu_input_x_data)
            .set_input_filter(conv3d_filter_data);
    conv_op.set_attr_data_format("NCDHW");
    conv_op.set_attr_strides({1, 1, 1, 1, 1});
    conv_op.set_attr_pads({0, 0, 0, 0, 0, 0});

    auto reshape_op = op::Reshape("reshape_op");
    reshape_op.set_input_x(resize_input_x_data)
               .set_input_shape(reshape_resize_data);

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
               .set_input_x2(reshape_op);

    std::vector<Operator> inputs{leakyrelu_input_x_data, resize_input_x_data, conv3d_filter_data, reshape_resize_data};
    std::vector<Operator> outputs{add_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AABiasaddConvFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    bool biasaddFlag = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Add") {
            biasaddFlag = true;
        }
    }
    EXPECT_EQ(biasaddFlag, true);

    delete[] conv3d_filter_Tensor_value;
    delete[] reshape_resize_tensor_value;
}
