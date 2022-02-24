#include "fc_transdata_merge_fusion_pass_test.h"
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

class transdata_cast_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "transdata_cast_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "transdata_cast_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(transdata_cast_fusion_pass_test, transdata_cast_fusion_pass_test_1) {
    ge::Graph graph("transdata_cast_fusion_pass_test_1");
    auto input_data = op::Data("Transdata_input_data");
    std::vector<int64_t> data_dims{1, 26, 1, 1, 32};
    ge::Shape data_shape(data_dims);
    ge::TensorDesc dataInputTensorDesc(data_shape, FORMAT_NC1HWC0, DT_BOOL);
    input_data.update_input_desc_x(dataInputTensorDesc);
    input_data.update_output_desc_y(dataInputTensorDesc);

    auto transdata1_op = op::TransData("transdata_1");
    std::vector<int64_t> transdata1_input_dims{1, 26, 1, 1, 32};
    std::vector<int64_t> transdata1_output_dims{1, 832, 1, 1};
    ge::Shape transdata1_input_shape(transdata1_input_dims);
    ge::Shape transdata1_input_origin_shape(transdata1_output_dims);
    ge::Shape transdata1_output_shape(transdata1_output_dims);
    ge::TensorDesc Trasdata1InputTensorDesc(transdata1_input_shape, FORMAT_NC1HWC0, DT_BOOL);
    Trasdata1InputTensorDesc.SetOriginShape(transdata1_input_origin_shape);
    Trasdata1InputTensorDesc.SetOriginFormat(FORMAT_NCHW);
    ge::TensorDesc Trasdata1OutputTensorDesc(transdata1_output_shape, FORMAT_NCHW, DT_BOOL);
    transdata1_op.update_input_desc_src(Trasdata1InputTensorDesc);
    transdata1_op.update_output_desc_dst(Trasdata1OutputTensorDesc);
    transdata1_op.SetAttr("src_format", FORMAT_NC1HWC0);
    transdata1_op.SetAttr("dst_format", FORMAT_NCHW);
    auto transdata1_desc = ge::OpDescUtils::GetOpDescFromOperator(transdata1_op);
    transdata1_desc->DelAttr("src_subformat");
    transdata1_desc->DelAttr("dst_subformat");

    auto reformat_op = op::ReFormat("reformat");
    std::vector<int64_t> reformat_dims{1, 832};
    ge::Shape reformat_shape(reformat_dims);
    ge::TensorDesc ReformatInputTensorDesc(reformat_shape, FORMAT_NCHW, DT_BOOL);
    ge::TensorDesc ReformatOutputTensorDesc(reformat_shape, FORMAT_ND, DT_BOOL);
    reformat_op.update_input_desc_x(ReformatInputTensorDesc);
    reformat_op.update_output_desc_y(ReformatOutputTensorDesc);

    auto reshape_op = op::Reshape("reshape");
    std::vector<int64_t> reshape_input_dims{1, 832, 1, 1};
    std::vector<int64_t> reshape_output_dims{1, 832};
    ge::Shape reshape_input_shape(reshape_input_dims);
    ge::Shape reshape_output_shape(reshape_output_dims);
    ge::TensorDesc ReshapeInputTensorDesc(reshape_input_shape, FORMAT_NCHW, DT_BOOL);
    ge::TensorDesc ReshapeOutputTensorDesc(reshape_output_shape, FORMAT_NCHW, DT_BOOL);
    reshape_op.update_input_desc_x(ReshapeInputTensorDesc);
    reshape_op.update_output_desc_y(ReshapeOutputTensorDesc);

    auto transdata2_op = op::TransData("transdata_2");
    std::vector<int64_t> transdata2_input_dims{1, 832};
    std::vector<int64_t> transdata2_output_dims{26, 1, 16, 32};
    ge::Shape transdata2_input_shape(transdata2_input_dims);
    ge::Shape transdata2_output_shape(transdata2_output_dims);
    ge::TensorDesc Trasdata2InputTensorDesc(transdata2_input_shape, FORMAT_ND, DT_BOOL);
    ge::TensorDesc Trasdata2OutputTensorDesc(transdata2_output_shape, FORMAT_FRACTAL_NZ, DT_BOOL);
    Trasdata2OutputTensorDesc.SetOriginShape(transdata2_output_shape);
    transdata2_op.update_input_desc_src(Trasdata2InputTensorDesc);
    transdata2_op.update_output_desc_dst(Trasdata2OutputTensorDesc);
    transdata2_op.SetAttr("src_format", FORMAT_ND);
    transdata2_op.SetAttr("dst_format", FORMAT_FRACTAL_NZ);

    auto fullyconnection_op = op::FullyConnection("fc");
    std::vector<int64_t> fc_dims{26, 1, 16, 32};
    ge::Shape fc_shape(fc_dims);
    ge::TensorDesc FCTensorDesc(fc_shape, FORMAT_FRACTAL_NZ, DT_BOOL);
    fullyconnection_op.update_input_desc_x(FCTensorDesc);
    fullyconnection_op.update_output_desc_y(FCTensorDesc);
    fullyconnection_op.set_attr_axis(1);

    auto reshapeConst = op::Constant();
    auto fcweight = op::Constant();
    transdata1_op.set_input_src(input_data);
    reshape_op.set_input_x(transdata1_op);
    reshape_op.set_input_shape(reshapeConst);
    reformat_op.set_input_x(reshape_op);
    transdata2_op.set_input_src(reformat_op);
    fullyconnection_op.set_input_x(transdata2_op);
    fullyconnection_op.set_input_w(fcweight);
    std::vector<Operator> inputs{input_data};
    std::vector<Operator> outputs{fullyconnection_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("TransdataCastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool dataTypeMatch = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransData") {
            auto inputDataType = node->GetOpDesc()->GetInputDesc(0).GetDataType();
            auto outputDataType = node->GetOpDesc()->GetOutputDesc(0).GetDataType();
            if ((inputDataType == DT_FLOAT16) && (outputDataType == DT_FLOAT16)) {
                dataTypeMatch = true;
            }
        }
    }
    EXPECT_EQ(dataTypeMatch, true);
}
