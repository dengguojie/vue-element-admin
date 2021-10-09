#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nonlinear_fuc_ops.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class argmax_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(argmax_fusion_test, argmax_equal_fusion_test_1) {
    ge::Graph graph("argmax_equal_fusion_test_1");
    auto shape_data = vector<int64_t>({2, 32});
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT16);
    auto data = op::Data("data");
    data.update_input_desc_x(desc_data);
    data.update_output_desc_y(desc_data);

    auto data1 = op::Data("data1");
    data1.update_input_desc_x(desc_data);
    data1.update_output_desc_y(desc_data);

    auto multiples_shape = ge::Shape({1});
    TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
    Tensor multiples_tensor(desc_input_size_1);
    uint32_t *multiples_tensor_value = new uint32_t[2]{1};
    multiples_tensor.SetData((uint8_t *) multiples_tensor_value, sizeof(uint32_t));
    auto argmax_multiples = op::Constant("multiples").set_attr_value(multiples_tensor);
    //argmax op
    auto argmaxv2 = op::ArgMaxV2("ArgMaxV2");
    argmaxv2.set_input_x(data);
    argmaxv2.set_input_dimension(argmax_multiples);

    auto argmaxv2b = op::ArgMaxV2("ArgMaxV2b");
    argmaxv2b.set_input_x(data1);
    argmaxv2b.set_input_dimension(argmax_multiples);

    std::vector<int64_t> equal_dims{2};
    ge::Shape equal_shape(equal_dims);
    ge::TensorDesc tensorDesc2(equal_shape, FORMAT_ND, DT_INT32);
    ge::TensorDesc tensorDesc3(equal_shape, FORMAT_ND, DT_INT32);
    auto equal_data = op::Data("equal_data");
    equal_data.update_input_desc_x(tensorDesc2);
    equal_data.update_output_desc_y(tensorDesc2);


    auto equal_op = op::Equal("equal");
    equal_op.set_input_x1(argmaxv2)
               .set_input_x2(argmaxv2b);

    std::vector<Operator> inputs{data,argmax_multiples, data1};
    std::vector<Operator> outputs{equal_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AArgMaxV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    delete[] multiples_tensor_value;

}

TEST_F(argmax_fusion_test, argmax_equal_fusion_test_2) {
    ge::Graph graph("argmax_equal_fusion_test_2");
    auto shape_data = vector<int64_t>({2, 32});
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT16);
    auto data = op::Data("data");
    data.update_input_desc_x(desc_data);
    data.update_output_desc_y(desc_data);

    auto data1 = op::Data("data1");
    data1.update_input_desc_x(desc_data);
    data1.update_output_desc_y(desc_data);

    auto multiples_shape = ge::Shape({1});
    TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT64);
    Tensor multiples_tensor(desc_input_size_1);
    uint32_t *multiples_tensor_value = new uint32_t[2]{1};
    multiples_tensor.SetData((uint8_t *) multiples_tensor_value, sizeof(uint32_t));
    auto argmax_multiples = op::Constant("multiples").set_attr_value(multiples_tensor);
    //argmax op
    auto argmaxv2 = op::ArgMaxV2("ArgMaxV2");
    argmaxv2.set_input_x(data);
    argmaxv2.set_input_dimension(argmax_multiples);

    auto argmaxv2b = op::ArgMaxV2("ArgMaxV2b");
    argmaxv2b.set_input_x(data1);
    argmaxv2b.set_input_dimension(argmax_multiples);

    std::vector<int64_t> equal_dims{2};
    ge::Shape equal_shape(equal_dims);
    ge::TensorDesc tensorDesc2(equal_shape, FORMAT_ND, DT_INT32);
    ge::TensorDesc tensorDesc3(equal_shape, FORMAT_ND, DT_INT32);
    auto equal_data = op::Data("equal_data");
    equal_data.update_input_desc_x(tensorDesc2);
    equal_data.update_output_desc_y(tensorDesc2);


    auto equal_op = op::Equal("equal");
    equal_op.set_input_x1(argmaxv2)
               .set_input_x2(argmaxv2b);

    std::vector<Operator> inputs{data,argmax_multiples, data1};
    std::vector<Operator> outputs{equal_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AArgMaxV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

}


TEST_F(argmax_fusion_test, argmax_equal_fusion_test_3) {
    ge::Graph graph("argmax_equal_fusion_test_3");
    auto shape_data = vector<int64_t>({-1, 5});
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT16);
    std::vector<std::pair<int64_t, int64_t>> range_x1 = {{1, 10}, {5,5}};
    desc_data.SetShapeRange(range_x1);
    auto data = op::Data("data");
    data.update_input_desc_x(desc_data);
    data.update_output_desc_y(desc_data);

    auto data1 = op::Data("data1");
    data1.update_input_desc_x(desc_data);
    data1.update_output_desc_y(desc_data);

    auto multiples_shape = ge::Shape({1});
    TensorDesc desc_input_size_1(multiples_shape, FORMAT_ND, DT_INT32);
    Tensor multiples_tensor(desc_input_size_1);
    uint32_t *multiples_tensor_value = new uint32_t[2]{0};
    multiples_tensor.SetData((uint8_t *) multiples_tensor_value, sizeof(uint32_t));
    auto argmax_multiples = op::Constant("multiples").set_attr_value(multiples_tensor);
    //argmax op
    auto argmaxv2 = op::ArgMaxV2("ArgMaxV2");
    argmaxv2.set_input_x(data);
    argmaxv2.set_input_dimension(argmax_multiples);

    auto argmaxv2b = op::ArgMaxV2("ArgMaxV2b");
    argmaxv2b.set_input_x(data1);
    argmaxv2b.set_input_dimension(argmax_multiples);

    std::vector<int64_t> equal_dims{5};
    ge::Shape equal_shape(equal_dims);
    ge::TensorDesc tensorDesc2(equal_shape, FORMAT_ND, DT_INT32);
    ge::TensorDesc tensorDesc3(equal_shape, FORMAT_ND, DT_INT32);
    auto equal_data = op::Data("equal_data");
    equal_data.update_input_desc_x(tensorDesc2);
    equal_data.update_output_desc_y(tensorDesc2);


    auto equal_op = op::Equal("equal");
    equal_op.set_input_x1(argmaxv2)
               .set_input_x2(argmaxv2b);

    std::vector<Operator> inputs{data,argmax_multiples, data1};
    std::vector<Operator> outputs{equal_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("AArgMaxV2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

}

