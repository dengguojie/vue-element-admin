#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/operator_reg.h"


using namespace ge;
using namespace op;

class confusion_transpose_nz_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "confusion_transpose_nz_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "confusion_transpose_nz_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(confusion_transpose_nz_fusion_pass_test, confusion_transpose_nz_fusion_pass_test_1) {
    ge::Graph graph("confusion_transpose_nz_fusion_pass_test_1");
    auto input_data = op::Data("input_data");
    std::vector<int64_t> data_dims{8, 8, 3, 6, 16, 16};
    ge::Shape data_shape(data_dims);
    std::vector<int64_t> confusion_transpose_ori_input_dims{8, 8, 96, 48};
    ge::Shape confusion_transpose_input_shape(data_dims);
    ge::Shape confusion_transpose_ori_input_shape(confusion_transpose_ori_input_dims);
    ge::TensorDesc TransposeInputTensorDesc(confusion_transpose_input_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TransposeInputTensorDesc.SetOriginShape(confusion_transpose_ori_input_shape);
    TransposeInputTensorDesc.SetOriginFormat(FORMAT_ND);
    input_data.update_input_desc_x(TransposeInputTensorDesc);
    input_data.update_output_desc_y(TransposeInputTensorDesc);

    auto confusion_transpose_op = op::ConfusionTransposeD("confusion_transpose_d");
    std::vector<int64_t> confusion_transpose_output_dims{8, 24, 6, 16, 16};
    ge::Shape confusion_transpose_output_shape(confusion_transpose_output_dims);
    std::vector<int64_t> confusion_transpose_ori_output_dims{8, 96, 384};
    ge::Shape confusion_transpose_ori_output_shape(confusion_transpose_ori_output_dims);
    ge::TensorDesc TransposeOutputTensorDesc(confusion_transpose_output_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TransposeOutputTensorDesc.SetOriginShape(confusion_transpose_ori_output_shape);
    TransposeOutputTensorDesc.SetOriginFormat(FORMAT_ND);
    confusion_transpose_op.update_input_desc_x(TransposeInputTensorDesc);
    confusion_transpose_op.update_output_desc_y(TransposeOutputTensorDesc);
    confusion_transpose_op.set_attr_perm({0,2,1,3});
    confusion_transpose_op.set_attr_shape({8, 96, 384});
    confusion_transpose_op.set_attr_transpose_first(true);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{24, 24, 16, 16};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_FRACTAL_NZ,  DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);
    confusion_transpose_op.set_input_x(input_data);
    auto bmOP = op::BatchMatMulV2("BatchMatMulV2");
    bmOP.set_input_x1(confusion_transpose_op);
    bmOP.set_input_x2(X2Data);

    std::vector<Operator> inputs{input_data, X2Data};
    std::vector<Operator> outputs{bmOP};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ConfusionTransposeNzFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findConfusionTransposeD = false;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ConfusionTransposeD") {
            findConfusionTransposeD = true;
            break;
        }
    }
    EXPECT_EQ(findConfusionTransposeD, false);
}

TEST_F(confusion_transpose_nz_fusion_pass_test, confusion_transpose_nz_fusion_pass_test_2) {
    ge::Graph graph("confusion_transpose_nz_fusion_pass_test_1");
    auto input_data = op::Data("input_data");
    std::vector<int64_t> data_dims{8, 8, 3, 6, 16, 16};
    ge::Shape data_shape(data_dims);
    std::vector<int64_t> confusion_transpose_ori_input_dims{8, 8, 96, 48};
    ge::Shape confusion_transpose_input_shape(data_dims);
    ge::Shape confusion_transpose_ori_input_shape(confusion_transpose_ori_input_dims);
    ge::TensorDesc TransposeInputTensorDesc(confusion_transpose_input_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TransposeInputTensorDesc.SetOriginShape(confusion_transpose_ori_input_shape);
    TransposeInputTensorDesc.SetOriginFormat(FORMAT_ND);
    input_data.update_input_desc_x(TransposeInputTensorDesc);
    input_data.update_output_desc_y(TransposeInputTensorDesc);

    auto confusion_transpose_op = op::ConfusionTransposeD("confusion_transpose_d");
    std::vector<int64_t> confusion_transpose_output_dims{8, 24, 6, 16, 16};
    ge::Shape confusion_transpose_output_shape(confusion_transpose_output_dims);
    std::vector<int64_t> confusion_transpose_ori_output_dims{8, 96, 384};
    ge::Shape confusion_transpose_ori_output_shape(confusion_transpose_ori_output_dims);
    ge::TensorDesc TransposeOutputTensorDesc(confusion_transpose_output_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TransposeOutputTensorDesc.SetOriginShape(confusion_transpose_ori_output_shape);
    TransposeOutputTensorDesc.SetOriginFormat(FORMAT_ND);
    confusion_transpose_op.update_input_desc_x(TransposeInputTensorDesc);
    confusion_transpose_op.update_output_desc_y(TransposeOutputTensorDesc);
    confusion_transpose_op.set_attr_perm({0,1,2,3});
    confusion_transpose_op.set_attr_shape({8, 96, 384});
    confusion_transpose_op.set_attr_transpose_first(true);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{24, 24, 16, 16};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_FRACTAL_NZ,  DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);
    confusion_transpose_op.set_input_x(input_data);
    auto bmOP = op::BatchMatMulV2("BatchMatMulV2");
    bmOP.set_input_x1(confusion_transpose_op);
    bmOP.set_input_x2(X2Data);

    std::vector<Operator> inputs{input_data, X2Data};
    std::vector<Operator> outputs{bmOP};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ConfusionTransposeNzFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findConfusionTransposeD = false;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ConfusionTransposeD") {
            findConfusionTransposeD = true;
            break;
        }
    }
    EXPECT_EQ(findConfusionTransposeD, true);
}

TEST_F(confusion_transpose_nz_fusion_pass_test, confusion_transpose_nz_fusion_pass_test_3) {
    ge::Graph graph("confusion_transpose_nz_fusion_pass_test_1");
    auto input_data = op::Data("input_data");
    std::vector<int64_t> data_dims{8, 3, 6, 16, 16};
    ge::Shape data_shape(data_dims);
    std::vector<int64_t> confusion_transpose_ori_input_dims{8, 96, 48};
    ge::Shape confusion_transpose_input_shape(data_dims);
    ge::Shape confusion_transpose_ori_input_shape(confusion_transpose_ori_input_dims);
    ge::TensorDesc TransposeInputTensorDesc(confusion_transpose_input_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TransposeInputTensorDesc.SetOriginShape(confusion_transpose_ori_input_shape);
    TransposeInputTensorDesc.SetOriginFormat(FORMAT_ND);
    input_data.update_input_desc_x(TransposeInputTensorDesc);
    input_data.update_output_desc_y(TransposeInputTensorDesc);

    auto confusion_transpose_op = op::ConfusionTransposeD("confusion_transpose_d");
    std::vector<int64_t> confusion_transpose_output_dims{24, 6, 16, 16};
    ge::Shape confusion_transpose_output_shape(confusion_transpose_output_dims);
    std::vector<int64_t> confusion_transpose_ori_output_dims{96, 384};
    ge::Shape confusion_transpose_ori_output_shape(confusion_transpose_ori_output_dims);
    ge::TensorDesc TransposeOutputTensorDesc(confusion_transpose_output_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    TransposeOutputTensorDesc.SetOriginShape(confusion_transpose_ori_output_shape);
    TransposeOutputTensorDesc.SetOriginFormat(FORMAT_ND);
    confusion_transpose_op.update_input_desc_x(TransposeInputTensorDesc);
    confusion_transpose_op.update_output_desc_y(TransposeOutputTensorDesc);
    confusion_transpose_op.set_attr_perm({0,2,1,3});
    confusion_transpose_op.set_attr_shape({96, 384});
    confusion_transpose_op.set_attr_transpose_first(true);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{24, 24, 16, 16};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_FRACTAL_NZ,  DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);
    confusion_transpose_op.set_input_x(input_data);
    auto bmOP = op::BatchMatMulV2("BatchMatMulV2");
    bmOP.set_input_x1(confusion_transpose_op);
    bmOP.set_input_x2(X2Data);

    std::vector<Operator> inputs{input_data, X2Data};
    std::vector<Operator> outputs{bmOP};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("ConfusionTransposeNzFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findConfusionTransposeD = false;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ConfusionTransposeD") {
            findConfusionTransposeD = true;
            break;
        }
    }
    EXPECT_EQ(findConfusionTransposeD, true);
}