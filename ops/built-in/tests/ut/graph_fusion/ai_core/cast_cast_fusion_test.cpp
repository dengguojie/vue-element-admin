#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class cast_cast_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "cast_cast_fusion SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "cast_cast_fusion TearDown" << std::endl;
    }
};

TEST_F(cast_cast_fusion_test, cast_cast_fusion_test_1) {
    ge::Graph graph("cast_cast_fusion_test_1");

    auto input_x = op::Data("input_x");

    std::vector<int64_t> dims_x{1, 32, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescData(shape_x, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    input_x.update_input_desc_x(tensorDescData);
    input_x.update_output_desc_y(tensorDescData);

    auto cast1_op = op::Cast("cast_1");
    ge::TensorDesc tensorDescCast1Out(shape_x, ge::FORMAT_NCHW, ge::DT_FLOAT);
    cast1_op.update_input_desc_x(tensorDescData);
    cast1_op.update_output_desc_y(tensorDescCast1Out);
    
    cast1_op.set_input_x(input_x)
            .set_attr_dst_type(0);

    auto cast2_op = op::Cast("cast_2");
     ge::TensorDesc tensorDescCast2Out(shape_x, ge::FORMAT_NCHW, ge::DT_INT32);
    cast2_op.update_input_desc_x(tensorDescCast1Out);
    cast2_op.update_output_desc_y(tensorDescCast2Out);

    cast2_op.set_input_x(cast1_op)
            .set_attr_dst_type(3);

    std::vector<Operator> inputs{input_x, cast1_op, cast2_op};
    std::vector<Operator> outputs{cast2_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "mul_maximum_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("CastCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "mul_maximum_fusion_test_1_after");
    bool findFusionCast = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Cast") {
            ge::GeTensorDesc castInputTensor = node->GetOpDesc()->GetInputDesc(0);
            ge::GeTensorDesc castOutputTensor = node->GetOpDesc()->GetOutputDesc(0);
            DataType castInputDataType = castInputTensor.GetDataType();
            DataType castOutputDataType = castOutputTensor.GetDataType();
            if (castInputDataType == ge::DT_FLOAT16 && castOutputDataType == ge::DT_INT32) {
                 findFusionCast = true;
            }
        }
    }
    EXPECT_EQ(findFusionCast, true);
}

TEST_F(cast_cast_fusion_test, cast_cast_fusion_test_2) {
    ge::Graph graph("cast_cast_fusion_test_2");

    auto input_x = op::Data("input_x");

    std::vector<int64_t> dims_x{1, 32, 32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescData(shape_x, ge::FORMAT_NCHW, ge::DT_INT32);
    input_x.update_input_desc_x(tensorDescData);
    input_x.update_output_desc_y(tensorDescData);

    auto cast1_op = op::Cast("cast_1");
    ge::TensorDesc tensorDescCast1Out(shape_x, ge::FORMAT_NCHW, ge::DT_BOOL);
    cast1_op.update_input_desc_x(tensorDescData);
    cast1_op.update_output_desc_y(tensorDescCast1Out);

    cast1_op.set_input_x(input_x)
            .set_attr_dst_type(12);

    auto cast2_op = op::Cast("cast_2");
     ge::TensorDesc tensorDescCast2Out(shape_x, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    cast2_op.update_input_desc_x(tensorDescCast1Out);
    cast2_op.update_output_desc_y(tensorDescCast2Out);

    cast2_op.set_input_x(cast1_op)
            .set_attr_dst_type(1);

    std::vector<Operator> inputs{input_x, cast1_op, cast2_op};
    std::vector<Operator> outputs{cast2_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "mul_maximum_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("CastCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "mul_maximum_fusion_test_1_after");
    bool findFusionCast = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Cast") {
            ge::GeTensorDesc castInputTensor = node->GetOpDesc()->GetInputDesc(0);
            ge::GeTensorDesc castOutputTensor = node->GetOpDesc()->GetOutputDesc(0);
            DataType castInputDataType = castInputTensor.GetDataType();
            DataType castOutputDataType = castOutputTensor.GetDataType();
            if (castInputDataType == ge::DT_FLOAT16 && castOutputDataType == ge::DT_INT32) {
                 findFusionCast = true;
            }
        }
    }
    EXPECT_EQ(findFusionCast, false);
}
