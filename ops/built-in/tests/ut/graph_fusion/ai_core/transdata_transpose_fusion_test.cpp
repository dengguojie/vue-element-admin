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
#include "fc_transdata_merge_fusion_pass_test.h"

using namespace ge;
using namespace op;

class transdata_transpose_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "transdata_transpose_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "transdata_transpose_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(transdata_transpose_fusion_pass_test, transdata_transpose_fusion_pass_test1) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> data_dims{8, 3, 3, 16, 16};
  std::vector<int64_t> data_dims_out{8, 48, 48};
  ge::Shape data_shape(data_dims);
  ge::Shape data_shape_out(data_dims_out);
  ge::TensorDesc dataInputTensorDesc(data_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  dataInputTensorDesc.SetOriginShape(data_shape_out);
  input_data.update_input_desc_x(dataInputTensorDesc);
  input_data.update_output_desc_y(dataInputTensorDesc);

  auto transdata1_op = op::TransData("transdata_1");
  std::vector<int64_t> transdata1_input_dims{8, 3, 3, 16, 16};
  std::vector<int64_t> transdata1_output_dims{8, 48, 48};
  ge::Shape transdata1_input_shape(transdata1_input_dims);
  ge::Shape transdata1_output_shape(transdata1_output_dims);
  ge::TensorDesc Trasdata1InputTensorDesc(transdata1_input_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Trasdata1InputTensorDesc.SetOriginShape(transdata1_output_shape);
  ge::TensorDesc Trasdata1OutputTensorDesc(transdata1_output_shape, FORMAT_ND, DT_FLOAT16);
  transdata1_op.update_input_desc_src(Trasdata1InputTensorDesc);
  transdata1_op.update_output_desc_dst(Trasdata1OutputTensorDesc);

  auto reformat_op_trans1 = op::ReFormat("reformat1");
  std::vector<int64_t> reformat_dims{8, 48, 48};
  ge::Shape reformat_shape(reformat_dims);
  ge::TensorDesc ReformatInputTensorDesc(reformat_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc ReformatOutputTensorDesc(reformat_shape, FORMAT_NHWC, DT_FLOAT16);
  reformat_op_trans1.update_input_desc_x(ReformatInputTensorDesc);
  reformat_op_trans1.update_output_desc_y(ReformatOutputTensorDesc);

  auto reshape_op = op::Reshape("reshape");
  std::vector<int64_t> reshape_input_dims{8, 48, 48};
  std::vector<int64_t> reshape_dims{48 * 8, 48};
  ge::Shape reshape_input_shape(reshape_input_dims);
  ge::Shape reshape_shape(reshape_dims);
  ge::TensorDesc ReshapeIutputTensorDesc(reshape_input_shape, FORMAT_NHWC, DT_FLOAT16);
  ge::TensorDesc ReshapeOutputTensorDesc(reshape_shape, FORMAT_NHWC, DT_FLOAT16);
  reshape_op.update_input_desc_x(ReshapeIutputTensorDesc);
  reshape_op.update_output_desc_y(ReshapeOutputTensorDesc);

  auto reformat_op_trans2 = op::ReFormat("reformat2");
  std::vector<int64_t> reformat_dims2{48 * 8, 48};
  ge::Shape reformat_shape2(reformat_dims2);
  ge::TensorDesc ReformatInputTensorDesc2(reformat_shape2, FORMAT_NHWC, DT_FLOAT16);
  ge::TensorDesc ReformatOutputTensorDesc2(reformat_shape2, FORMAT_ND, DT_FLOAT16);
  reformat_op_trans2.update_input_desc_x(ReformatInputTensorDesc2);
  reformat_op_trans2.update_output_desc_y(ReformatOutputTensorDesc2);

  auto transdata2_op = op::TransData("transdata_2");
  std::vector<int64_t> transdata2_input_dims{48 * 8, 48};
  std::vector<int64_t> transdata2_output_dims{3, 24, 16, 16};
  ge::Shape transdata2_input_shape(transdata2_input_dims);
  ge::Shape transdata2_output_shape(transdata2_output_dims);
  ge::TensorDesc Trasdata2inputTensorDesc(transdata2_input_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc Trasdata2OutputTensorDesc(transdata2_output_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  transdata2_op.update_input_desc_src(Trasdata2inputTensorDesc);
  transdata2_op.update_output_desc_dst(Trasdata2OutputTensorDesc);

  transdata1_op.set_input_src(input_data);
  reformat_op_trans1.set_input_x(transdata1_op);
  reshape_op.set_input_x(reformat_op_trans1);
  reformat_op_trans2.set_input_x(reshape_op);
  transdata2_op.set_input_src(reformat_op_trans2);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataTransposeFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    std::cout << "lllllllllllll" << node->GetType() << endl;
    auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
    auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
    std::vector<int64_t> indims = inputDesc.GetShape().GetDims();
    std::vector<int64_t> origin = inputDesc.GetOriginShape().GetDims();
    std::vector<int64_t> outdims = outputDesc.GetShape().GetDims();
    std::cout << "77777777777777777777" << origin.size() << endl;
    for (int i = 0; i < indims.size(); ++i) {
      std::cout << node->GetType() << "input dim " << indims[i] << endl;
    }

    for (int i = 0; i < outdims.size(); ++i) {
      std::cout << node->GetType() << "output dim " << outdims[i] << endl;
    }
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, true);
}

TEST_F(transdata_transpose_fusion_pass_test, transdata_transpose_fusion_pass_test2) {
  ge::Graph graph("transdata_transpose_fusion_pass_test2");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> data_dims{32, 1, 1, 16, 16};
  std::vector<int64_t> data_dims_out{32, 16, 16};
  ge::Shape data_shape(data_dims);
  ge::Shape data_shape_out(data_dims_out);
  ge::TensorDesc dataInputTensorDesc(data_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  dataInputTensorDesc.SetOriginShape(data_shape_out);
  input_data.update_input_desc_x(dataInputTensorDesc);
  input_data.update_output_desc_y(dataInputTensorDesc);

  auto transdata1_op = op::TransData("transdata_1");
  std::vector<int64_t> transdata1_input_dims{32, 1, 1, 16, 16};
  std::vector<int64_t> transdata1_output_dims{16, 2, 16, 16};
  ge::Shape transdata1_input_shape(transdata1_input_dims);
  ge::Shape transdata1_output_shape(transdata1_output_dims);
  ge::TensorDesc Trasdata1InputTensorDesc(transdata1_input_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Trasdata1InputTensorDesc.SetOriginShape(transdata1_output_shape);
  ge::TensorDesc Trasdata1OutputTensorDesc(transdata1_output_shape, FORMAT_ND, DT_FLOAT16);
  transdata1_op.update_input_desc_src(Trasdata1InputTensorDesc);
  transdata1_op.update_output_desc_dst(Trasdata1OutputTensorDesc);

  auto reformat_op_trans1 = op::ReFormat("reformat1");
  std::vector<int64_t> reformat_dims{16, 2, 16, 16};
  ge::Shape reformat_shape(reformat_dims);
  ge::TensorDesc ReformatInputTensorDesc(reformat_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc ReformatOutputTensorDesc(reformat_shape, FORMAT_NHWC, DT_FLOAT16);
  reformat_op_trans1.update_input_desc_x(ReformatInputTensorDesc);
  reformat_op_trans1.update_output_desc_y(ReformatOutputTensorDesc);

  auto reshape_op = op::Reshape("reshape");
  std::vector<int64_t> reshape_input_dims{16, 2, 16, 16};
  std::vector<int64_t> reshape_dims{16, 32, 16};
  ge::Shape reshape_input_shape(reshape_input_dims);
  ge::Shape reshape_shape(reshape_dims);
  ge::TensorDesc ReshapeIutputTensorDesc(reshape_input_shape, FORMAT_NHWC, DT_FLOAT16);
  ge::TensorDesc ReshapeOutputTensorDesc(reshape_shape, FORMAT_NHWC, DT_FLOAT16);
  reshape_op.update_input_desc_x(ReshapeIutputTensorDesc);
  reshape_op.update_output_desc_y(ReshapeOutputTensorDesc);

  auto reformat_op_trans2 = op::ReFormat("reformat2");
  std::vector<int64_t> reformat_dims2{16, 32, 16};
  ge::Shape reformat_shape2(reformat_dims2);
  ge::TensorDesc ReformatInputTensorDesc2(reformat_shape2, FORMAT_NHWC, DT_FLOAT16);
  ge::TensorDesc ReformatOutputTensorDesc2(reformat_shape2, FORMAT_ND, DT_FLOAT16);
  reformat_op_trans2.update_input_desc_x(ReformatInputTensorDesc2);
  reformat_op_trans2.update_output_desc_y(ReformatOutputTensorDesc2);

  auto transdata2_op = op::TransData("transdata_2");
  std::vector<int64_t> transdata2_input_dims{16, 32, 16};
  std::vector<int64_t> transdata2_output_dims{16, 1, 2, 16};
  ge::Shape transdata2_input_shape(transdata2_input_dims);
  ge::Shape transdata2_output_shape(transdata2_output_dims);
  ge::TensorDesc Trasdata2inputTensorDesc(transdata2_input_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc Trasdata2OutputTensorDesc(transdata2_output_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  transdata2_op.update_input_desc_src(Trasdata2inputTensorDesc);
  transdata2_op.update_output_desc_dst(Trasdata2OutputTensorDesc);

  transdata1_op.set_input_src(input_data);
  reformat_op_trans1.set_input_x(transdata1_op);
  reshape_op.set_input_x(reformat_op_trans1);
  reformat_op_trans2.set_input_x(reshape_op);
  transdata2_op.set_input_src(reformat_op_trans2);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataTransposeFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    std::cout << "lllllllllllll" << node->GetType() << endl;
    auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
    auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
    std::vector<int64_t> indims = inputDesc.GetShape().GetDims();
    std::vector<int64_t> origin = inputDesc.GetOriginShape().GetDims();
    std::vector<int64_t> outdims = outputDesc.GetShape().GetDims();
    std::cout << "77777777777777777777" << origin.size() << endl;
    for (int i = 0; i < indims.size(); ++i) {
      std::cout << node->GetType() << "input dim " << indims[i] << endl;
    }

    for (int i = 0; i < outdims.size(); ++i) {
      std::cout << node->GetType() << "output dim " << outdims[i] << endl;
    }
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, true);
}

TEST_F(transdata_transpose_fusion_pass_test, transdata_transpose_fusion_pass_test3) {
  ge::Graph graph("transdata_transpose_fusion_pass_test3");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> data_dims{16, 1, 1, 16, 16};
  std::vector<int64_t> data_dims_out{16, 16, 16};
  ge::Shape data_shape(data_dims);
  ge::Shape data_shape_out(data_dims_out);
  ge::TensorDesc dataInputTensorDesc(data_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  dataInputTensorDesc.SetOriginShape(data_shape_out);
  input_data.update_input_desc_x(dataInputTensorDesc);
  input_data.update_output_desc_y(dataInputTensorDesc);

  auto transdata1_op = op::TransData("transdata_1");
  std::vector<int64_t> transdata1_input_dims{16, 1, 1, 16, 16};
  std::vector<int64_t> transdata1_output_dims{16, 16, 16};
  ge::Shape transdata1_input_shape(transdata1_input_dims);
  ge::Shape transdata1_output_shape(transdata1_output_dims);
  ge::TensorDesc Trasdata1InputTensorDesc(transdata1_input_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  Trasdata1InputTensorDesc.SetOriginShape(transdata1_output_shape);
  ge::TensorDesc Trasdata1OutputTensorDesc(transdata1_output_shape, FORMAT_ND, DT_FLOAT16);
  transdata1_op.update_input_desc_src(Trasdata1InputTensorDesc);
  transdata1_op.update_output_desc_dst(Trasdata1OutputTensorDesc);

  auto reformat_op_trans1 = op::ReFormat("reformat1");
  std::vector<int64_t> reformat_dims{16, 16, 16};
  ge::Shape reformat_shape(reformat_dims);
  ge::TensorDesc ReformatInputTensorDesc(reformat_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc ReformatOutputTensorDesc(reformat_shape, FORMAT_NHWC, DT_FLOAT16);
  reformat_op_trans1.update_input_desc_x(ReformatInputTensorDesc);
  reformat_op_trans1.update_output_desc_y(ReformatOutputTensorDesc);

  auto reshape_op = op::Reshape("reshape");
  std::vector<int64_t> reshape_input_dims{16,16, 16};
  std::vector<int64_t> reshape_dims{16, 256};
  ge::Shape reshape_input_shape(reshape_input_dims);
  ge::Shape reshape_shape(reshape_dims);
  ge::TensorDesc ReshapeIutputTensorDesc(reshape_input_shape, FORMAT_NHWC, DT_FLOAT16);
  ge::TensorDesc ReshapeOutputTensorDesc(reshape_shape, FORMAT_NHWC, DT_FLOAT16);
  reshape_op.update_input_desc_x(ReshapeIutputTensorDesc);
  reshape_op.update_output_desc_y(ReshapeOutputTensorDesc);

  auto reformat_op_trans2 = op::ReFormat("reformat2");
  std::vector<int64_t> reformat_dims2{16, 256};
  ge::Shape reformat_shape2(reformat_dims2);
  ge::TensorDesc ReformatInputTensorDesc2(reformat_shape2, FORMAT_NHWC, DT_FLOAT16);
  ge::TensorDesc ReformatOutputTensorDesc2(reformat_shape2, FORMAT_ND, DT_FLOAT16);
  reformat_op_trans2.update_input_desc_x(ReformatInputTensorDesc2);
  reformat_op_trans2.update_output_desc_y(ReformatOutputTensorDesc2);

  auto transdata2_op = op::TransData("transdata_2");
  std::vector<int64_t> transdata2_input_dims{16, 256};
  std::vector<int64_t> transdata2_output_dims{16, 1, 16, 16};
  ge::Shape transdata2_input_shape(transdata2_input_dims);
  ge::Shape transdata2_output_shape(transdata2_output_dims);
  ge::TensorDesc Trasdata2inputTensorDesc(transdata2_input_shape, FORMAT_ND, DT_FLOAT16);
  ge::TensorDesc Trasdata2OutputTensorDesc(transdata2_output_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  transdata2_op.update_input_desc_src(Trasdata2inputTensorDesc);
  transdata2_op.update_output_desc_dst(Trasdata2OutputTensorDesc);

  transdata1_op.set_input_src(input_data);
  reformat_op_trans1.set_input_x(transdata1_op);
  reshape_op.set_input_x(reformat_op_trans1);
  reformat_op_trans2.set_input_x(reshape_op);
  transdata2_op.set_input_src(reformat_op_trans2);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataTransposeFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    std::cout << "lllllllllllll" << node->GetType() << endl;
    auto inputDesc = node->GetOpDesc()->GetInputDesc(0);
    auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
    std::vector<int64_t> indims = inputDesc.GetShape().GetDims();
    std::vector<int64_t> origin = inputDesc.GetOriginShape().GetDims();
    std::vector<int64_t> outdims = outputDesc.GetShape().GetDims();
    std::cout << "77777777777777777777" << origin.size() << endl;
    for (int i = 0; i < indims.size(); ++i) {
      std::cout << node->GetType() << "input dim " << indims[i] << endl;
    }

    for (int i = 0; i < outdims.size(); ++i) {
      std::cout << node->GetType() << "output dim " << outdims[i] << endl;
    }
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, true);
}
