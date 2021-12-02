#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class conv3d_mul_add_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "conv3d_mul_add_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv3d_mul_add_fusion_test TearDown" << std::endl;
    }
};

TEST_F(conv3d_mul_add_fusion_test, conv3d_mul_add_fusion_test1) {

    ge::Graph graph("conv3d_mul_add_fusion_test1");
    auto shape_data = vector<int64_t>({ 1,3,8,8,16});
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NDHWC, DT_FLOAT16);

    // data op
    auto data = op::Data("data");
    data.update_input_desc_x(desc_data);
    data.update_output_desc_y(desc_data);

    // custom op ,using method is the same with frame internal op
    // [Notice]: if you want to use custom self-define op, please prepare custom op according to custum op define user guides
    TensorDesc weight_desc(ge::Shape({3,3,3,16,16}), FORMAT_DHWCN, DT_FLOAT16);
    Tensor weighttensor1(weight_desc);

    auto weight1 = op::Const().set_attr_value(weighttensor1);

    auto bias_shape = ge::Shape({16});
    TensorDesc desc_bias_1(bias_shape, FORMAT_ND, DT_FLOAT16);
    Tensor bias_tensor(desc_bias_1);
    auto conv_bias = op::Const("Conv3D/bias")
        .set_attr_value(bias_tensor);

    // conv3d op
    auto conv3d = op::Conv3D("Conv3d")
        .set_input_x(data)
        .set_input_filter(weight1)
        .set_input_bias(conv_bias)
        .set_attr_strides({ 1, 1, 1, 1, 1 })
        .set_attr_pads({ 0, 0, 0, 0, 0, 0})
        .set_attr_dilations({ 0, 0, 0, 0, 0 })
        .set_attr_data_format("NDHWC");

    TensorDesc conv3d_input_desc_x(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3d_input_desc_filter(ge::Shape(), FORMAT_DHWCN, DT_FLOAT16);
    TensorDesc conv3d_output_desc_y(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    conv3d.update_input_desc_x(conv3d_input_desc_x);
    conv3d.update_input_desc_filter(conv3d_input_desc_filter);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    // add
    auto add1_shape = ge::Shape({1,1,1,1,16});
    TensorDesc desc_add_1(add1_shape, FORMAT_ND, DT_FLOAT16);
    Tensor add1_tensor(desc_add_1);

    auto conv_bias_mul1_add1 = op::Const("Conv3D/bias/mul/add")
        .set_attr_value(add1_tensor);
    auto add = op::Add("add1")
        .set_input_x1(conv3d)
        .set_input_x2(conv_bias_mul1_add1);

    std::vector<Operator> inputs{ data };
    std::vector<Operator> outputs{add};
    std::vector<std::pair<ge::Operator, std::string>> outputs_with_name = {{conv3d, "conv3d"}};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    // fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        std::cout << "node->GetType" << node->GetType() <<std::endl;
        if (node->GetType() == "Conv3D") {
            findD = true;
        }
    }
    std::cout << "run conv3d_mul_add_fusion_test1 end" <<std::endl;
    EXPECT_EQ(findD, true);
}

TEST_F(conv3d_mul_add_fusion_test, conv3d_mul_add_fusion_test2) {

    ge::Graph graph("conv3d_mul_add_fusion_test2");
    auto shape_data = vector<int64_t>({ 1,3,8,8,16});
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NDHWC, DT_FLOAT16);

    // data op
    auto data = op::Data("data");
    data.update_input_desc_x(desc_data);
    data.update_output_desc_y(desc_data);

    TensorDesc weight_desc(ge::Shape({3,3,3,16,16}), FORMAT_DHWCN, DT_FLOAT16);
    Tensor weighttensor1(weight_desc);

    auto weight1 = op::Const().set_attr_value(weighttensor1);

    // conv3d op
    auto conv3d = op::Conv3D("Conv3d")
        .set_input_x(data)
        .set_input_filter(weight1)
        .set_attr_strides({ 1, 1, 1, 1, 1 })
        .set_attr_pads({ 0, 0, 0, 0, 0, 0})
        .set_attr_dilations({ 0, 0, 0, 0, 0 })
        .set_attr_data_format("NDHWC");

    TensorDesc conv3d_input_desc_x(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3d_input_desc_filter(ge::Shape(), FORMAT_DHWCN, DT_FLOAT16);
    TensorDesc conv3d_output_desc_y(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    conv3d.update_input_desc_x(conv3d_input_desc_x);
    conv3d.update_input_desc_filter(conv3d_input_desc_filter);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    // add
    auto add1_shape = ge::Shape({1,1,1,1,16});
    TensorDesc desc_add_1(add1_shape, FORMAT_ND, DT_FLOAT16);
    Tensor add1_tensor(desc_add_1);

    auto conv_bias_mul1_add1 = op::Const("Conv3D/bias/mul/add")
        .set_attr_value(add1_tensor);
    auto add = op::Add("add1")
        .set_input_x1(conv3d)
        .set_input_x2(conv_bias_mul1_add1);

    std::vector<Operator> inputs{ data };
    std::vector<Operator> outputs{add};
    std::vector<std::pair<ge::Operator, std::string>> outputs_with_name = {{conv3d, "conv3d"}};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    // fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        std::cout << "node->GetType" << node->GetType() <<std::endl;
        if (node->GetType() == "Conv3D") {
            findD = true;
        }
    }
    std::cout << "run conv3d_mul_add_fusion_test2 end" <<std::endl;
    EXPECT_EQ(findD, true);
}

TEST_F(conv3d_mul_add_fusion_test, conv3d_mul_add_fusion_test3) {

    ge::Graph graph("conv3d_mul_add_fusion_test3");
    auto shape_data = vector<int64_t>({ 1,3,8,8,16});
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_NDHWC, DT_FLOAT16);

    // data op
    auto data = op::Data("data");
    data.update_input_desc_x(desc_data);
    data.update_output_desc_y(desc_data);

    TensorDesc weight_desc(ge::Shape({3,3,3,16,16}), FORMAT_DHWCN, DT_FLOAT16);
    Tensor weighttensor1(weight_desc);

    auto weight1 = op::Const().set_attr_value(weighttensor1);

    // conv3d op
    auto conv3d = op::Conv3D("Conv3d")
        .set_input_x(data)
        .set_input_filter(weight1)
        .set_attr_strides({ 1, 1, 1, 1, 1 })
        .set_attr_pads({ 0, 0, 0, 0, 0, 0})
        .set_attr_dilations({ 0, 0, 0, 0, 0 })
        .set_attr_data_format("NDHWC");

    TensorDesc conv3d_input_desc_x(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    TensorDesc conv3d_input_desc_filter(ge::Shape(), FORMAT_DHWCN, DT_FLOAT16);
    TensorDesc conv3d_output_desc_y(ge::Shape(), FORMAT_NDHWC, DT_FLOAT16);
    conv3d.update_input_desc_x(conv3d_input_desc_x);
    conv3d.update_input_desc_filter(conv3d_input_desc_filter);
    conv3d.update_output_desc_y(conv3d_output_desc_y);

    // mul
    auto mul_shape = ge::Shape({1,1,1,1,16});
    TensorDesc desc_mul(mul_shape, FORMAT_ND, DT_FLOAT16);
    Tensor mul_tensor(desc_mul);

    auto conv_bias_mul1_add1 = op::Const("Conv3D/bias/mul/add")
        .set_attr_value(mul_tensor);
    auto mul = op::Mul("mul")
        .set_input_x1(conv3d)
        .set_input_x2(conv_bias_mul1_add1);

    // add
    auto add1_shape = ge::Shape({1,1,1,1,16});
    TensorDesc desc_add_1(add1_shape, FORMAT_ND, DT_FLOAT16);
    Tensor add1_tensor(desc_add_1);

    auto conv_add1 = op::Const("Conv3D/mul/add")
        .set_attr_value(add1_tensor);
    auto add1 = op::Add("add1")
        .set_input_x1(mul)
        .set_input_x2(conv_add1);
    // add
    auto add2_shape = ge::Shape({1,1,1,1,16});
    TensorDesc desc_add_2(add2_shape, FORMAT_ND, DT_FLOAT16);
    Tensor add2_tensor(desc_add_2);

    auto conv_mul1_add2 = op::Const("Conv3D/mul/add2")
        .set_attr_value(add2_tensor);
    auto add2 = op::Add("add2")
        .set_input_x1(conv_mul1_add2)
        .set_input_x2(mul);

    std::vector<Operator> inputs{ data };
    std::vector<Operator> outputs{ add1 };
    std::vector<std::pair<ge::Operator, std::string>> outputs_with_name = {{conv3d, "conv3d"}};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    // fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AConv2dMulFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findD = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        std::cout << "node->GetType" << node->GetType() <<std::endl;
        if (node->GetType() == "Conv3D") {
            findD = true;
        }
    }
    std::cout << "run conv3d_mul_add_fusion_test3 end" <<std::endl;
    EXPECT_EQ(findD, true);
}

/* NHWC conv2d with bias(vector) + add(vector) */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_1) {
    ge::Graph graph("conv_add_fusion_test_1");

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

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 1);
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NHWC conv2d with bias(vector) + add(vector)
 * chanel is not equal
 */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_2) {
    ge::Graph graph("conv_add_fusion_test_2");

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
    float *const_add_tensor_value = new float[2];
    for (int i = 0; i < 2; i++) {
        *(const_add_tensor_value + i) = 0.1;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 2 * 4);

    std::vector<int64_t> dims_add{2};
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

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 1);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NHWC conv2d with bias(vector) + add(tensor) */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_3) {
    ge::Graph graph("conv_add_fusion_test_3");

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

    std::vector<int64_t> dims_add{1, 1, 1, 64};
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

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 1);
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NHWC conv2d with bias(vector) + add(tensor)
 * chanel is not equal
 */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_4) {
    ge::Graph graph("conv_add_fusion_test_4");

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
    float *const_add_tensor_value = new float[2];
    for (int i = 0; i < 2; i++) {
        *(const_add_tensor_value + i) = 0.1;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 2 * 4);

    std::vector<int64_t> dims_add{1, 1, 1, 2};
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

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 1);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NHWC conv2d with bias(vector) + add(tensor)
 * not channel wise
 */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_5) {
    ge::Graph graph("conv_add_fusion_test_5");

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
    float *const_add_tensor_value = new float[64 * 2];
    for (int i = 0; i < 64 * 2; i++) {
        *(const_add_tensor_value + i) = 0.1;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 64 * 2 * 4);

    std::vector<int64_t> dims_add{1, 2, 1, 64};
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

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 1);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NCHW conv2d with bias(vector) + add(vector) */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_6) {
    ge::Graph graph("conv_add_fusion_test_6");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 64, 56, 56};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_FLOAT);
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

    std::vector<int64_t> dims_filter{64, 64, 1, 1};
    ge::Shape shape_filter(dims_filter);
    ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_NCHW, DT_FLOAT);
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
    ge::TensorDesc tensorDescBias(shape_bias, FORMAT_NCHW, DT_FLOAT);
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
    ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NCHW, DT_FLOAT);
    const_add_tensor.SetTensorDesc(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data)
            .set_input_bias(conv_input_bias_data);
    conv_op.set_attr_data_format("NCHW");
    conv_op.set_attr_strides({1, 2, 2, 1});
    conv_op.set_attr_pads({1, 1, 1, 1});

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 1);
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NCHW conv2d with bias(vector) + add(vector)
 * chanel is not equal
 */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_7) {
    ge::Graph graph("conv_add_fusion_test_7");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 64, 56, 56};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_FLOAT);
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

    std::vector<int64_t> dims_filter{64, 64, 1, 1};
    ge::Shape shape_filter(dims_filter);
    ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_NCHW, DT_FLOAT);
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
    ge::TensorDesc tensorDescBias(shape_bias, FORMAT_NCHW, DT_FLOAT);
    dims_bias_tensor.SetTensorDesc(tensorDescBias);
    conv_input_bias_data.set_attr_value(dims_bias_tensor);

    auto const_add = op::Const("const_add");

    Tensor const_add_tensor;
    float *const_add_tensor_value = new float[2];
    for (int i = 0; i < 2; i++) {
        *(const_add_tensor_value + i) = 0.1;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 2 * 4);

    std::vector<int64_t> dims_add{2};
    ge::Shape shape_add(dims_add);
    ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NCHW, DT_FLOAT);
    const_add_tensor.SetTensorDesc(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data)
            .set_input_bias(conv_input_bias_data);
    conv_op.set_attr_data_format("NCHW");
    conv_op.set_attr_strides({1, 2, 2, 1});
    conv_op.set_attr_pads({1, 1, 1, 1});

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 1);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NCHW conv2d with bias(vector) + add(tensor) */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_8) {
    ge::Graph graph("conv_add_fusion_test_8");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 64, 56, 56};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_FLOAT);
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

    std::vector<int64_t> dims_filter{64, 64, 1, 1};
    ge::Shape shape_filter(dims_filter);
    ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_NCHW, DT_FLOAT);
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
    ge::TensorDesc tensorDescBias(shape_bias, FORMAT_NCHW, DT_FLOAT);
    dims_bias_tensor.SetTensorDesc(tensorDescBias);
    conv_input_bias_data.set_attr_value(dims_bias_tensor);

    auto const_add = op::Const("const_add");

    Tensor const_add_tensor;
    float *const_add_tensor_value = new float[64];
    for (int i = 0; i < 64; i++) {
        *(const_add_tensor_value + i) = 0.1;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 64 * 4);

    std::vector<int64_t> dims_add{1, 64, 1, 1};
    ge::Shape shape_add(dims_add);
    ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NCHW, DT_FLOAT);
    const_add_tensor.SetTensorDesc(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data)
            .set_input_bias(conv_input_bias_data);
    conv_op.set_attr_data_format("NCHW");
    conv_op.set_attr_strides({1, 2, 2, 1});
    conv_op.set_attr_pads({1, 1, 1, 1});

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 1);
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NCHW conv2d with bias(vector) + add(tensor)
 * chanel is not equal
 */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_9) {
    ge::Graph graph("conv_add_fusion_test_9");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 64, 56, 56};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_FLOAT);
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

    std::vector<int64_t> dims_filter{64, 64, 1, 1};
    ge::Shape shape_filter(dims_filter);
    ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_NCHW, DT_FLOAT);
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
    ge::TensorDesc tensorDescBias(shape_bias, FORMAT_NCHW, DT_FLOAT);
    dims_bias_tensor.SetTensorDesc(tensorDescBias);
    conv_input_bias_data.set_attr_value(dims_bias_tensor);

    auto const_add = op::Const("const_add");

    Tensor const_add_tensor;
    float *const_add_tensor_value = new float[2];
    for (int i = 0; i < 2; i++) {
        *(const_add_tensor_value + i) = 0.1;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 2 * 4);

    std::vector<int64_t> dims_add{1, 2, 1, 1};
    ge::Shape shape_add(dims_add);
    ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NCHW, DT_FLOAT);
    const_add_tensor.SetTensorDesc(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data)
            .set_input_bias(conv_input_bias_data);
    conv_op.set_attr_data_format("NCHW");
    conv_op.set_attr_strides({1, 2, 2, 1});
    conv_op.set_attr_pads({1, 1, 1, 1});

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 1);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NCHW conv2d with bias(vector) + add(tensor)
 * not channel wise
 */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_10) {
    ge::Graph graph("conv_add_fusion_test_10");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 64, 56, 56};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NCHW, DT_FLOAT);
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

    std::vector<int64_t> dims_filter{64, 64, 1, 1};
    ge::Shape shape_filter(dims_filter);
    ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_NCHW, DT_FLOAT);
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
    ge::TensorDesc tensorDescBias(shape_bias, FORMAT_NCHW, DT_FLOAT);
    dims_bias_tensor.SetTensorDesc(tensorDescBias);
    conv_input_bias_data.set_attr_value(dims_bias_tensor);

    auto const_add = op::Const("const_add");

    Tensor const_add_tensor;
    float *const_add_tensor_value = new float[64 * 2];
    for (int i = 0; i < 64 * 2; i++) {
        *(const_add_tensor_value + i) = 0.1;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 64 * 4 * 2);

    std::vector<int64_t> dims_add{1, 64, 1, 2};
    ge::Shape shape_add(dims_add);
    ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NCHW, DT_FLOAT);
    const_add_tensor.SetTensorDesc(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data)
            .set_input_bias(conv_input_bias_data);
    conv_op.set_attr_data_format("NCHW");
    conv_op.set_attr_strides({1, 2, 2, 1});
    conv_op.set_attr_pads({1, 1, 1, 1});

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 1);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] dims_bias_tensor_value;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NHWC conv2d without bias + add(scalar DT_FLOAT) */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_11) {
    ge::Graph graph("conv_add_fusion_test_11");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 56, 56, 64};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
    conv_input_x_data.update_input_desc_x(tensorDescX);
    conv_input_x_data.update_output_desc_y(tensorDescX);

    auto conv_input_filter_data = op::Const("conv_input_filter_data");

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
    float *const_add_tensor_value = new float[1];
    for (int i = 0; i < 1; i++) {
        *(const_add_tensor_value + i) = 0.1;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 1 * 4);

    std::vector<int64_t> dims_add{1};
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

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 0);

    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NHWC conv2d without bias + add(scalar DT_FLOAT16) */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_12) {
    ge::Graph graph("conv_add_fusion_test_12");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 56, 56, 64};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
    conv_input_x_data.update_input_desc_x(tensorDescX);
    conv_input_x_data.update_output_desc_y(tensorDescX);

    auto conv_input_filter_data = op::Const("conv_input_filter_data");

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
    int16_t *const_add_tensor_value = new int16_t[1];
    for (int i = 0; i < 1; i++) {
        *(const_add_tensor_value + i) = 2;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 1 * 2);

    std::vector<int64_t> dims_add{1};
    ge::Shape shape_add(dims_add);
    ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NHWC, DT_FLOAT16);
    const_add_tensor.SetTensorDesc(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data);
    conv_op.set_attr_data_format("NHWC");
    conv_op.set_attr_strides({1, 2, 2, 1});
    conv_op.set_attr_pads({1, 1, 1, 1});

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
            EXPECT_EQ(node->GetOutNodes().size(), 2);
        }
        if (node->GetType() == "Add") {
            AddCnt++;
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 0);

    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NHWC conv2d without bias + add(scalar DT_INT8) 
 * not support dtype
 */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_13) {
    ge::Graph graph("conv_add_fusion_test_13");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 56, 56, 64};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
    conv_input_x_data.update_input_desc_x(tensorDescX);
    conv_input_x_data.update_output_desc_y(tensorDescX);

    auto conv_input_filter_data = op::Const("conv_input_filter_data");

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
    char *const_add_tensor_value = new char[1];
    for (int i = 0; i < 1; i++) {
        *(const_add_tensor_value + i) = 5;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 1 * 1);

    std::vector<int64_t> dims_add{1};
    ge::Shape shape_add(dims_add);
    ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NHWC, DT_INT8);
    const_add_tensor.SetTensorDesc(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data);
    conv_op.set_attr_data_format("NHWC");
    conv_op.set_attr_strides({1, 2, 2, 1});
    conv_op.set_attr_pads({1, 1, 1, 1});

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(const_add);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, const_add};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
        }
        if (node->GetType() == "Add") {
            AddCnt++;
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}

/* NHWC conv2d without bias + add(reshape) */
TEST_F(conv3d_mul_add_fusion_test, conv_add_fusion_test_14) {
    ge::Graph graph("conv_add_fusion_test_12");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 56, 56, 64};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT);
    conv_input_x_data.update_input_desc_x(tensorDescX);
    conv_input_x_data.update_output_desc_y(tensorDescX);

    auto conv_input_filter_data = op::Const("conv_input_filter_data");

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
    int16_t *const_add_tensor_value = new int16_t[1];
    for (int i = 0; i < 1; i++) {
        *(const_add_tensor_value + i) = 2;
    }
    const_add_tensor.SetData((uint8_t *) const_add_tensor_value, 1 * 2);

    std::vector<int64_t> dims_add{1};
    ge::Shape shape_add(dims_add);
    ge::TensorDesc tensorDescAdd(shape_add, FORMAT_NHWC, DT_FLOAT16);
    const_add_tensor.SetTensorDesc(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data);
    conv_op.set_attr_data_format("NHWC");
    conv_op.set_attr_strides({1, 2, 2, 1});
    conv_op.set_attr_pads({1, 1, 1, 1});
    
    auto reshape_op = op::Reshape("reshape");
    std::vector<int64_t> reshape_input_dims{1, 128};
    std::vector<int64_t> reshape_output_dims{1, 1, 1, 128};
    ge::Shape reshape_input_shape(reshape_input_dims);
    ge::Shape reshape_output_shape(reshape_output_dims);
    ge::TensorDesc ReshapeInputTensorDesc(reshape_input_shape, FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc ReshapeOutputTensorDesc(reshape_output_shape, FORMAT_NCHW, DT_FLOAT16);
    reshape_op.update_input_desc_x(ReshapeInputTensorDesc);
    reshape_op.update_output_desc_y(ReshapeOutputTensorDesc);
    
    auto reshape_x_data = op::Data("reshape_x_data");
    std::vector<int64_t> dims_reshape{1, 128};
    ge::Shape reshape_x(dims_reshape);
    ge::TensorDesc tensorDesc(reshape_x, FORMAT_NHWC, DT_FLOAT);
    reshape_x_data.update_input_desc_x(tensorDesc);
    reshape_x_data.update_output_desc_y(tensorDesc);
    auto reshapeConst = op::Constant();
    reshape_op.set_input_x(reshape_x_data);
    reshape_op.set_input_shape(reshapeConst);

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(conv_op)
            .set_input_x2(reshape_op);

    auto end_op1 = op::Square("end_op1");
    end_op1.set_input_x(add_op);

    auto end_op2 = op::Asin("end_op2");
    end_op2.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, const_add, reshape_x_data, reshapeConst};
    std::vector<Operator> outputs{end_op1, end_op2};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("TBEConvAddFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    int Conv2Cnt = 0;
    int AddCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            Conv2Cnt++;
        }
        if (node->GetType() == "Add") {
            AddCnt++;
        }
    }
    EXPECT_EQ(Conv2Cnt, 1);
    EXPECT_EQ(AddCnt, 1);

    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}