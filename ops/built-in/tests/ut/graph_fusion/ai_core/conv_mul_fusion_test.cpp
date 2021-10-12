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

class conv_mul_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(conv_mul_fusion_test, conv_mul_fusion_test_1) {
    ge::Graph graph("conv_mul_fusion_test_1");

    auto conv_input_x_data = op::Data("conv_input_x_data");
    std::vector<int64_t> dims_x{1, 28, 28, 64};
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


    std::vector<int64_t> dims_filter{1, 1, 64, 64};
    ge::Shape shape_filter(dims_filter);
    ge::TensorDesc tensorDescFilter(shape_filter, FORMAT_HWCN, DT_FLOAT);
    tensorDescFilter.SetOriginFormat(FORMAT_HWCN);
    conv_input_filter_tensor.SetTensorDesc(tensorDescFilter);
    conv_input_filter_data.set_attr_value(conv_input_filter_tensor);
    conv_input_filter_data.update_output_desc_y(tensorDescFilter);

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

    auto const_mul = op::Const("const_mul");
    Tensor constmultensor;
    float *dataValue = new float[1];
    *dataValue = 0.1;
    constmultensor.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC));
    constmultensor.SetData((uint8_t *) dataValue, 4);
    const_mul.set_attr_value(constmultensor);


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
    const_add.update_output_desc_y(tensorDescAdd);
    const_add.set_attr_value(const_add_tensor);

    auto conv_op = op::Conv2D("conv_1");
    conv_op.set_input_x(conv_input_x_data)
            .set_input_filter(conv_input_filter_data)
            .set_input_bias(conv_input_bias_data);
    conv_op.set_attr_data_format("NHWC");
    conv_op.set_attr_strides({1, 1, 1, 1});
    conv_op.set_attr_pads({0, 0, 0, 0});

    auto mul_op = op::Mul("mul_op");
    mul_op.set_input_x1(conv_op)
            .set_input_x2(const_mul);

    auto add_op = op::Add("add_op");
    add_op.set_input_x1(mul_op)
            .set_input_x2(const_add);

    auto end_op = op::Square("end_op");
    end_op.set_input_x(add_op);

    std::vector<Operator> inputs{conv_input_x_data, conv_input_filter_data, conv_input_bias_data, const_mul, const_add};
    std::vector<Operator> outputs{end_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AConv2dMulFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    // compute_graph_ptr->Dump();
    bool findConv2Dreduce = false;
    int findMulCnt = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Conv2D") {
            findConv2Dreduce = true;
        }
        if (node->GetType() == "Mul") {
            findMulCnt += 1;
        }
    }
    EXPECT_EQ(findConv2Dreduce, true);
    EXPECT_EQ(findMulCnt, 2);
    delete[] dims_bias_tensor_value;
    delete[] dataValue;
    delete[] const_add_tensor_value;
    delete[] conv_input_filter_tensor_value;
}
