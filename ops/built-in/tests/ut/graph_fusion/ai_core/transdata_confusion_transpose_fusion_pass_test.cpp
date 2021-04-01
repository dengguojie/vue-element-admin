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

class transdata_confusion_transpose_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "transdata_confusion_transpose_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "transdata_confusion_transpose_fusion_pass_test TearDown" << std::endl;
    }
};

TEST_F(transdata_confusion_transpose_fusion_pass_test, transdata_confusion_transpose_fusion_pass_test_1) {
    ge::Graph graph("transdata_confusion_transpose_fusion_pass_test_1");
    auto input_data = op::Data("Transdata_input_data");
    std::vector<int64_t> data_dims{8, 8, 3, 6, 16, 16};
    ge::Shape data_shape(data_dims);
    ge::TensorDesc dataInputTensorDesc(data_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    input_data.update_input_desc_x(dataInputTensorDesc);
    input_data.update_output_desc_y(dataInputTensorDesc);

    auto transdata1_op = op::TransData("transdata_1");
    std::vector<int64_t> transdata1_input_dims{8, 8, 3, 6, 16, 16};
    std::vector<int64_t> transdata1_output_dims{8, 8, 96, 48};
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
    std::vector<int64_t> reformat_dims{8, 8, 96, 48};
    ge::Shape reformat_shape(reformat_dims);
    ge::TensorDesc ReformatInputTensorDesc(reformat_shape, FORMAT_ND, DT_FLOAT16);
    ge::TensorDesc ReformatOutputTensorDesc(reformat_shape, FORMAT_NCHW, DT_FLOAT16);
    reformat_op.update_input_desc_x(ReformatInputTensorDesc);
    reformat_op.update_output_desc_y(ReformatOutputTensorDesc);

    auto confusion_transpose_op = op::ConfusionTransposeD("confusion_transpose_d");
    std::vector<int64_t> confusion_transpose_input_dims{8, 8, 96, 48};
    std::vector<int64_t> confusion_transpose_output_dims{8, 96, 384};
    ge::Shape confusion_transpose_input_shape(confusion_transpose_input_dims);
    ge::Shape confusion_transpose_output_shape(confusion_transpose_output_dims);
    ge::TensorDesc TransposeInputTensorDesc(confusion_transpose_input_shape, FORMAT_ND, DT_FLOAT16);
    ge::TensorDesc TransposeOutputTensorDesc(confusion_transpose_output_shape, FORMAT_ND, DT_FLOAT16);
    confusion_transpose_op.update_input_desc_x(TransposeInputTensorDesc);
    confusion_transpose_op.update_output_desc_y(TransposeOutputTensorDesc);
    confusion_transpose_op.set_attr_perm({0,2,1,3});
    confusion_transpose_op.set_attr_shape({8, 96, 384});
    confusion_transpose_op.set_attr_transpose_first(true);


    auto transdata2_op = op::TransData("transdata_2");
    std::vector<int64_t> transdata2_input_dims{8, 96, 384};
    std::vector<int64_t> transdata2_output_dims{8, 24, 6, 16, 16};
    ge::Shape transdata2_input_shape(transdata2_input_dims);
    ge::Shape transdata2_output_shape(transdata2_output_dims);
    ge::TensorDesc Trasdata2InputTensorDesc(transdata2_input_shape, FORMAT_ND, DT_FLOAT16);
    ge::TensorDesc Trasdata2OutputTensorDesc(transdata2_output_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
    Trasdata2OutputTensorDesc.SetOriginShape(transdata2_input_shape);
    transdata2_op.update_input_desc_src(Trasdata2InputTensorDesc);
    transdata2_op.update_output_desc_dst(Trasdata2OutputTensorDesc);

    auto X2Data = op::Data("x2");
    std::vector<int64_t> dims_x2{24, 24, 16, 16};
    ge::Shape shape_x2(dims_x2);
    ge::TensorDesc tensorDescX2(shape_x2, FORMAT_FRACTAL_NZ,  DT_FLOAT16);
    X2Data.update_input_desc_x(tensorDescX2);
    X2Data.update_output_desc_y(tensorDescX2);

    transdata1_op.set_input_src(input_data);
    reformat_op.set_input_x(transdata1_op);
    confusion_transpose_op.set_input_x(reformat_op);
    transdata2_op.set_input_src(confusion_transpose_op);

    auto bmOP = op::BatchMatMulV2("BatchMatMulV2");
    bmOP.set_input_x1(transdata2_op);
    bmOP.set_input_x2(X2Data);

    std::vector<Operator> inputs{input_data, X2Data};
    std::vector<Operator> outputs{bmOP};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::RunGraphFusionPass("TransDataConfusionTransposeFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTransData = false;

    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "TransData") {
            findTransData = true;
        }
    }
    EXPECT_EQ(findTransData, true);
}
