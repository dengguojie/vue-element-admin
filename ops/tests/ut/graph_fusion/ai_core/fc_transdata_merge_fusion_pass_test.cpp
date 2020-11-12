#include "fc_transdata_merge_fusion_pass_test.h"
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/operator_reg.h"


using namespace ge;
using namespace op;

class fc_transdata_merge_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "fc_transdata_merge_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "fc_transdata_merge_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(fc_transdata_merge_fusion_pass_test, fc_transdata_merge_fusion_pass_test_1) {
    ge::Graph graph("fc_transdata_merge_fusion_pass_test_1");
    auto input_data = op::Data("Transdata_input_data");
    std::vector<int64_t> data_dims{21, 19, 16, 16};
    ge::Shape data_shape(data_dims);
    ge::TensorDesc dataInputTensorDesc(data_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_data.update_input_desc_x(dataInputTensorDesc);
    input_data.update_output_desc_y(dataInputTensorDesc);

    auto transdata1_op = op::TransData("transdata_1");
    std::vector<int64_t> transdata1_input_dims{21, 19, 16, 16};
    std::vector<int64_t> transdata1_output_dims{304, 324};
    ge::Shape transdata1_input_shape(transdata1_input_dims);
    ge::Shape transdata1_input_origin_shape(transdata1_output_dims);
    ge::Shape transdata1_output_shape(transdata1_output_dims);
    ge::TensorDesc Trasdata1InputTensorDesc(transdata1_input_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Trasdata1InputTensorDesc.SetOriginShape(transdata1_input_origin_shape);
    Trasdata1InputTensorDesc.SetOriginFormat(FORMAT_NCHW);
    ge::TensorDesc Trasdata1OutputTensorDesc(transdata1_output_shape, FORMAT_ND, DT_FLOAT16);
    transdata1_op.update_input_desc_src(Trasdata1InputTensorDesc);
    transdata1_op.update_output_desc_dst(Trasdata1OutputTensorDesc);

    auto reformat_op = op::ReFormat("reformat");
    std::vector<int64_t> reformat_dims{304, 324};
    ge::Shape reformat_shape(reformat_dims);
    ge::TensorDesc ReformatInputTensorDesc(reformat_shape, FORMAT_ND, DT_FLOAT16);
    ge::TensorDesc ReformatOutputTensorDesc(reformat_shape, FORMAT_NCHW, DT_FLOAT16);
    reformat_op.update_input_desc_x(ReformatInputTensorDesc);
    reformat_op.update_output_desc_y(ReformatOutputTensorDesc);

    auto reshape_op = op::Reshape("reshape");
    std::vector<int64_t> reshape_input_dims{304, 324};
    std::vector<int64_t> reshape_output_dims{304, 324, 1, 1};
    ge::Shape reshape_input_shape(reshape_input_dims);
    ge::Shape reshape_output_shape(reshape_output_dims);
    ge::TensorDesc ReshapeInputTensorDesc(reshape_input_shape, FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc ReshapeOutputTensorDesc(reshape_output_shape, FORMAT_NCHW, DT_FLOAT16);
    reshape_op.update_input_desc_x(ReshapeInputTensorDesc);
    reshape_op.update_output_desc_y(ReshapeOutputTensorDesc);

    auto transdata2_op = op::TransData("transdata_2");
    std::vector<int64_t> transdata2_input_dims{304, 324, 1, 1};
    std::vector<int64_t> transdata2_output_dims{304, 21, 1, 1, 16};
    ge::Shape transdata2_input_shape(transdata2_input_dims);
    ge::Shape transdata2_output_shape(transdata2_output_dims);
    ge::TensorDesc Trasdata2InputTensorDesc(transdata2_input_shape, FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc Trasdata2OutputTensorDesc(transdata2_output_shape, FORMAT_NC1HWC0, DT_FLOAT16);
    transdata2_op.update_input_desc_src(Trasdata2InputTensorDesc);
    transdata2_op.update_output_desc_dst(Trasdata2OutputTensorDesc);

    auto exp_op = op::Exp("exp");
    std::vector<int64_t> exp_dims{304, 21, 1, 1, 16};
    ge::Shape exp_shape(exp_dims);
    ge::TensorDesc ExpTensorDesc(exp_shape, FORMAT_NC1HWC0, DT_FLOAT16);
    exp_op.update_input_desc_x(ExpTensorDesc);
    exp_op.update_output_desc_y(ExpTensorDesc);

    auto reshapeConst = op::Constant();
    transdata1_op.set_input_src(input_data);
    reformat_op.set_input_x(transdata1_op);
    reshape_op.set_input_x(reformat_op);
    reshape_op.set_input_shape(reshapeConst);
    transdata2_op.set_input_src(reshape_op);
    exp_op.set_input_x(transdata2_op);
    std::vector<Operator> inputs{input_data};
    std::vector<Operator> outputs{exp_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("FCTransdataMergePass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool shapeMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransData") {
            auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
            auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
            std::vector<int64_t> indims = inputDesc.GetShape().GetDims();
            std::vector<int64_t> outdims = outputDesc.GetShape().GetDims();
            if ((indims == transdata1_input_dims) && (outdims == transdata2_output_dims)) {
                shapeMatch = true;
            }
        }
    }
    EXPECT_EQ(shapeMatch, true);
}

TEST_F(fc_transdata_merge_fusion_pass_test, fc_transdata_merge_fusion_pass_test_2) {
    ge::Graph graph("fc_transdata_merge_fusion_pass_test_2");
    auto input_data = op::Data("Transdata_input_data");
    std::vector<int64_t> data_dims{21, 19, 16, 16};
    ge::Shape data_shape(data_dims);
    ge::TensorDesc dataInputTensorDesc(data_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_data.update_input_desc_x(dataInputTensorDesc);
    input_data.update_output_desc_y(dataInputTensorDesc);
 
    auto transdata1_op = op::TransData("transdata_1");
    std::vector<int64_t> transdata1_input_dims{21, 19, 16, 16};
    std::vector<int64_t> transdata1_output_dims{304, 324};
    ge::Shape transdata1_input_shape(transdata1_input_dims);
    ge::Shape transdata1_input_origin_shape(transdata1_output_dims);
    ge::Shape transdata1_output_shape(transdata1_output_dims);
    ge::TensorDesc Trasdata1InputTensorDesc(transdata1_input_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Trasdata1InputTensorDesc.SetOriginShape(transdata1_input_origin_shape);
    ge::TensorDesc Trasdata1OutputTensorDesc(transdata1_output_shape, FORMAT_ND, DT_FLOAT16);
    transdata1_op.update_input_desc_src(Trasdata1InputTensorDesc);
    transdata1_op.update_output_desc_dst(Trasdata1OutputTensorDesc);

    auto reformat_op = op::ReFormat("reformat");
    std::vector<int64_t> reformat_dims{304, 324};
    ge::Shape reformat_shape(reformat_dims);
    ge::TensorDesc ReformatInputTensorDesc(reformat_shape, FORMAT_ND, DT_FLOAT16);
    ge::TensorDesc ReformatOutputTensorDesc(reformat_shape, FORMAT_NCHW, DT_FLOAT16);
    reformat_op.update_input_desc_x(ReformatInputTensorDesc);
    reformat_op.update_output_desc_y(ReformatOutputTensorDesc);

    auto reshape_op = op::Reshape("reshape");
    std::vector<int64_t> reshape_input_dims{304, 324};
    std::vector<int64_t> reshape_output_dims{304, 324, 1, 1};
    ge::Shape reshape_input_shape(reshape_input_dims);
    ge::Shape reshape_output_shape(reshape_output_dims);
    ge::TensorDesc ReshapeInputTensorDesc(reshape_input_shape, FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc ReshapeOutputTensorDesc(reshape_output_shape, FORMAT_NCHW, DT_FLOAT16);
    reshape_op.update_input_desc_x(ReshapeInputTensorDesc);
    reshape_op.update_output_desc_y(ReshapeOutputTensorDesc);

    auto transdata2_op = op::TransData("transdata_2");
    std::vector<int64_t> transdata2_input_dims{304, 324, 1, 1};
    std::vector<int64_t> transdata2_output_dims{304, 21, 1, 1, 16};
    ge::Shape transdata2_input_shape(transdata2_input_dims);
    ge::Shape transdata2_output_shape(transdata2_output_dims);
    ge::TensorDesc Trasdata2InputTensorDesc(transdata2_input_shape, FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc Trasdata2OutputTensorDesc(transdata2_output_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    transdata2_op.update_input_desc_src(Trasdata2InputTensorDesc);
    transdata2_op.update_output_desc_dst(Trasdata2OutputTensorDesc);

    auto reshapeConst = op::Constant();
    transdata1_op.set_input_src(input_data);
    reformat_op.set_input_x(transdata1_op);
    reshape_op.set_input_x(reformat_op);
    reshape_op.set_input_shape(reshapeConst);
    transdata2_op.set_input_src(reshape_op);
    std::vector<Operator> inputs{input_data};
    std::vector<Operator> outputs{transdata2_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("FCTransdataMergePass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findNode = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ReFormat") {
            findNode = true;
        }
    }
    EXPECT_EQ(findNode, true);
}
