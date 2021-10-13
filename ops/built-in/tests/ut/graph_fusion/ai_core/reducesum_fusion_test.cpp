#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "nn_norm_ops.h"
#include "fp16_t.hpp"
#include "reduce_ops.h"

using namespace ge;
using namespace op;

class reducesum_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "inplace_add SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "inplace_add TearDown" << std::endl;
    }
};

TEST_F(reducesum_fusion_test, reducesum_fusion_test_1) {
    ge::Graph graph("reducesum_fusion_test_1");
    auto reducesum_input_data = op::Data("reducesum_input_data");
    std::vector<int64_t> dims{64,96,3,3};
    ge::Shape shape(dims);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    reducesum_input_data.update_input_desc_x(tensorDesc);
    reducesum_input_data.update_output_desc_y(tensorDesc);
    
    auto axis_shape = ge::Shape({1});
    TensorDesc desc_input_size_1(axis_shape, FORMAT_ND, DT_INT32);
    Tensor axis_tensor(desc_input_size_1);
    uint32_t *axis_tensor_value = new uint32_t[1]{0};
    axis_tensor.SetData((uint8_t *) axis_tensor_value, 1 * sizeof(uint32_t));
    auto begin = op::Constant().set_attr_value(axis_tensor);

    delete []axis_tensor_value;

    auto reducesum_op = op::ReduceSum("ReduceSum");
    reducesum_op.set_input_x(reducesum_input_data);
    reducesum_op.set_input_axes(begin);
    reducesum_op.set_attr_keep_dims(false);
   
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(reducesum_op);
    std::vector<Operator> inputs{reducesum_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrReduceSumFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool reducesum = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ReduceSumD") {
            reducesum = true;
        }
    }
    EXPECT_EQ(reducesum, true);
}

TEST_F(reducesum_fusion_test, reducesum_fusion_test_2) {
    ge::Graph graph("reducesum_fusion_test_2");
    auto reducesum_input_data = op::Data("reducesum_input_data");
    std::vector<int64_t> dims{64,96,3,3};
    ge::Shape shape(dims);
    std::vector<int64_t> dims1{64,96,3,3};
    ge::Shape shape1(dims1);
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    reducesum_input_data.update_input_desc_x(tensorDesc);
    reducesum_input_data.update_output_desc_y(tensorDesc1);
    
    ge::Tensor begin_tensor;
    std::vector<int64_t> begin_vec{0};
    ge::Shape begin_shape(begin_vec);
    ge::TensorDesc begin_desc(begin_shape, FORMAT_ND, DT_INT32);
    int32_t begin_size = begin_desc.GetShape().GetShapeSize();
    begin_desc.SetSize(begin_size * sizeof(int32_t));
    begin_tensor.SetTensorDesc(begin_desc);
    int32_t* begin_data = nullptr;
    begin_data = new int32_t[begin_size];
    *(begin_data + 0) = 0;
 
    begin_tensor.SetData((uint8_t*)begin_data, begin_size * sizeof(int32_t));
    delete [] begin_data;
    auto begin = op::Constant().set_attr_value(begin_tensor);
    
    auto reducesum_op = op::ReduceSum("ReduceSum");
    reducesum_op.set_input_x(reducesum_input_data);
    reducesum_op.set_input_axes(begin);
    reducesum_op.set_attr_keep_dims(true);
   
    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(reducesum_op);
    std::vector<Operator> inputs{reducesum_input_data};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrReduceSumFusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    bool reducesum = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ReduceSum") {
            reducesum = true;
        }
    }
    EXPECT_EQ(reducesum, true);
}
