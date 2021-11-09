#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "matrix_calculation_ops.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "nonlinear_fuc_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/operator_reg.h"
#include "fc_transdata_merge_fusion_pass_test.h"

using namespace ge;
using namespace op;

class transdata_reshape_transpose_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "transdata_reshape_transpose_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "transdata_reshape_transpose_fusion_pass_test TearDown" << std::endl;
  }
};

TEST_F(transdata_reshape_transpose_fusion_pass_test, transdata_reshape_transpose_test1) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> in_dims{2048, 1, 20, 16, 16};
  std::vector<int64_t> in_origin_dims{2048, 320, 16};
  std::vector<int64_t> out_dims{2048, 20, 1, 16, 16};
  std::vector<int64_t> out_origin_dims{2048, 16, 320};
  ge::Shape in_shape(in_dims);
  ge::Shape in_origin_shape(in_origin_dims);
  ge::Shape out_shape(out_dims);
  ge::Shape out_origin_shape(out_origin_dims);

  ge::TensorDesc in_tensor_desc(in_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_tensor_desc.SetOriginShape(in_origin_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  ge::TensorDesc out_tensor_desc(out_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  out_tensor_desc.SetOriginShape(out_origin_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);

  input_data.update_input_desc_x(in_tensor_desc);
  input_data.update_output_desc_y(in_tensor_desc);

  ge::TensorDesc reshape_in_tensor(in_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(in_origin_shape);
  ge::TensorDesc reshape_out_tensor(out_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(out_origin_shape);

  auto transdata1_op = op::TransData("transdata_1");
  transdata1_op.update_input_desc_src(in_tensor_desc);
  transdata1_op.update_output_desc_dst(reshape_in_tensor);

  Tensor tensor_value;
  tensor_value.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));
  int32_t *data_value = new int32_t[1];
  *data_value = 3;
  tensor_value.SetData((uint8_t*)data_value, 4);
  auto const_op = op::Const("const").set_attr_value(tensor_value);
  auto reshape_op = op::Reshape("reshape").set_input_shape(const_op);

  reshape_op.update_input_desc_x(reshape_in_tensor);
  reshape_op.update_output_desc_y(reshape_out_tensor);

  auto transdata2_op = op::TransData("transdata_2");
  transdata2_op.update_input_desc_src(reshape_out_tensor);
  transdata2_op.update_output_desc_dst(out_tensor_desc);

  auto relu_op = op::Relu("relu_op");
  relu_op.update_input_desc_x(out_tensor_desc);
  relu_op.update_output_desc_y(out_tensor_desc);

  transdata1_op.set_input_src(input_data);
  reshape_op.set_input_x(transdata1_op);
  transdata2_op.set_input_src(reshape_op);
  relu_op.set_input_x(transdata2_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataZReshapeTransposeFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  delete[] data_value;
  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 3);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, true);
}

TEST_F(transdata_reshape_transpose_fusion_pass_test, transdata_reshape_transpose_test2) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> in_dims{1, 4, 16, 16};
  std::vector<int64_t> in_origin_dims{64, 16};
  std::vector<int64_t> out_dims{4, 1, 16, 16};
  std::vector<int64_t> out_origin_dims{16, 64};
  ge::Shape in_shape(in_dims);
  ge::Shape in_origin_shape(in_origin_dims);
  ge::Shape out_shape(out_dims);
  ge::Shape out_origin_shape(out_origin_dims);

  ge::TensorDesc in_tensor_desc(in_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_tensor_desc.SetOriginShape(in_origin_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  ge::TensorDesc out_tensor_desc(out_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  out_tensor_desc.SetOriginShape(out_origin_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);

  input_data.update_input_desc_x(in_tensor_desc);
  input_data.update_output_desc_y(in_tensor_desc);

  ge::TensorDesc reshape_in_tensor(in_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(in_origin_shape);
  ge::TensorDesc reshape_out_tensor(out_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(out_origin_shape);

  auto transdata1_op = op::TransData("transdata_1");
  transdata1_op.update_input_desc_src(in_tensor_desc);
  transdata1_op.update_output_desc_dst(reshape_in_tensor);

  auto reformat1_op = op::ReFormat("reformat1");
  reformat1_op.update_input_desc_x(reshape_in_tensor);
  reformat1_op.update_output_desc_y(reshape_in_tensor);

  auto reshape_op = op::Reshape("reshape");
  reshape_op.update_input_desc_x(reshape_in_tensor);
  reshape_op.update_output_desc_y(reshape_out_tensor);

  auto reformat2_op = op::ReFormat("reformat2");
  reformat2_op.update_input_desc_x(reshape_out_tensor);
  reformat2_op.update_output_desc_y(reshape_out_tensor);

  auto transdata2_op = op::TransData("transdata_2");
  transdata2_op.update_input_desc_src(reshape_out_tensor);
  transdata2_op.update_output_desc_dst(out_tensor_desc);

  transdata1_op.set_input_src(input_data);
  reformat1_op.set_input_x(transdata1_op);
  reshape_op.set_input_x(reformat1_op);
  reformat2_op.set_input_x(reshape_op);
  transdata2_op.set_input_src(reformat2_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataZReshapeTransposeFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 2);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, true);
}

TEST_F(transdata_reshape_transpose_fusion_pass_test, transdata_reshape_transpose_test3) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> in_dims{1, 4, 16, 16};
  std::vector<int64_t> in_origin_dims{64, 16};
  std::vector<int64_t> out_dims{4, 1, 16, 16};
  std::vector<int64_t> out_origin_dims{16, 64};
  ge::Shape in_shape(in_dims);
  ge::Shape in_origin_shape(in_origin_dims);
  ge::Shape out_shape(out_dims);
  ge::Shape out_origin_shape(out_origin_dims);

  ge::TensorDesc in_tensor_desc(in_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_tensor_desc.SetOriginShape(in_origin_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  ge::TensorDesc out_tensor_desc(out_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  out_tensor_desc.SetOriginShape(out_origin_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);

  input_data.update_input_desc_x(in_tensor_desc);
  input_data.update_output_desc_y(in_tensor_desc);

  ge::TensorDesc reshape_in_tensor(in_origin_shape, FORMAT_NCHW, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(in_origin_shape);
  ge::TensorDesc reshape_out_tensor(out_origin_shape, FORMAT_NCHW, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(out_origin_shape);

  auto transdata1_op = op::TransData("transdata_1");
  transdata1_op.update_input_desc_src(in_tensor_desc);
  transdata1_op.update_output_desc_dst(reshape_in_tensor);

  auto reshape_op = op::Reshape("reshape");
  reshape_op.update_input_desc_x(reshape_in_tensor);
  reshape_op.update_output_desc_y(reshape_out_tensor);

  auto transdata2_op = op::TransData("transdata_2");
  transdata2_op.update_input_desc_src(reshape_out_tensor);
  transdata2_op.update_output_desc_dst(out_tensor_desc);

  transdata1_op.set_input_src(input_data);
  reshape_op.set_input_x(transdata1_op);
  transdata2_op.set_input_src(reshape_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataZReshapeTransposeFusionPass",
                                              fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 4);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, false);
}

TEST_F(transdata_reshape_transpose_fusion_pass_test, transdata_reshape_transpose_test4) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> in_dims{1, 4, 16, 16};
  std::vector<int64_t> in_origin_dims{64, 16};
  std::vector<int64_t> out_dims{4, 1, 16, 16};
  std::vector<int64_t> out_origin_dims{16, 64};
  ge::Shape in_shape(in_dims);
  ge::Shape in_origin_shape(in_origin_dims);
  ge::Shape out_shape(out_dims);
  ge::Shape out_origin_shape(out_origin_dims);

  ge::TensorDesc in_tensor_desc(in_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_tensor_desc.SetOriginShape(in_origin_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  ge::TensorDesc out_tensor_desc(out_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  out_tensor_desc.SetOriginShape(out_origin_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);

  input_data.update_input_desc_x(in_tensor_desc);
  input_data.update_output_desc_y(in_tensor_desc);

  ge::TensorDesc reshape_in_tensor(in_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(in_origin_shape);
  ge::TensorDesc reshape_out_tensor(out_origin_shape, FORMAT_NCHW, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(out_origin_shape);

  auto transdata1_op = op::TransData("transdata_1");
  transdata1_op.update_input_desc_src(in_tensor_desc);
  transdata1_op.update_output_desc_dst(reshape_in_tensor);

  auto reshape_op = op::Reshape("reshape");
  reshape_op.update_input_desc_x(reshape_in_tensor);
  reshape_op.update_output_desc_y(reshape_out_tensor);

  auto transdata2_op = op::TransData("transdata_2");
  transdata2_op.update_input_desc_src(reshape_out_tensor);
  transdata2_op.update_output_desc_dst(out_tensor_desc);

  transdata1_op.set_input_src(input_data);
  reshape_op.set_input_x(transdata1_op);
  transdata2_op.set_input_src(reshape_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataZReshapeTransposeFusionPass",
  fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
  *compute_graph_ptr);

  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 4);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, false);
}

TEST_F(transdata_reshape_transpose_fusion_pass_test, transdata_reshape_transpose_test5) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> in_dims{4, 16, 16};
  std::vector<int64_t> in_origin_dims{64, 16};
  std::vector<int64_t> out_dims{4, 1, 16, 16};
  std::vector<int64_t> out_origin_dims{16, 64};
  ge::Shape in_shape(in_dims);
  ge::Shape in_origin_shape(in_origin_dims);
  ge::Shape out_shape(out_dims);
  ge::Shape out_origin_shape(out_origin_dims);

  ge::TensorDesc in_tensor_desc(in_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_tensor_desc.SetOriginShape(in_origin_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  ge::TensorDesc out_tensor_desc(out_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  out_tensor_desc.SetOriginShape(out_origin_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);

  input_data.update_input_desc_x(in_tensor_desc);
  input_data.update_output_desc_y(in_tensor_desc);

  ge::TensorDesc reshape_in_tensor(in_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(in_origin_shape);
  ge::TensorDesc reshape_out_tensor(out_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(out_origin_shape);

  auto transdata1_op = op::TransData("transdata_1");
  transdata1_op.update_input_desc_src(in_tensor_desc);
  transdata1_op.update_output_desc_dst(reshape_in_tensor);

  auto reshape_op = op::Reshape("reshape");
  reshape_op.update_input_desc_x(reshape_in_tensor);
  reshape_op.update_output_desc_y(reshape_out_tensor);

  auto transdata2_op = op::TransData("transdata_2");
  transdata2_op.update_input_desc_src(reshape_out_tensor);
  transdata2_op.update_output_desc_dst(out_tensor_desc);

  transdata1_op.set_input_src(input_data);
  reshape_op.set_input_x(transdata1_op);
  transdata2_op.set_input_src(reshape_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataZReshapeTransposeFusionPass",
  fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
  *compute_graph_ptr);

  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 4);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, false);
}

TEST_F(transdata_reshape_transpose_fusion_pass_test, transdata_reshape_transpose_test6) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> in_dims{1, 4, 16, 16};
  std::vector<int64_t> in_origin_dims{64, 16};
  std::vector<int64_t> out_dims{1, 16, 16};
  std::vector<int64_t> out_origin_dims{16, 64};
  ge::Shape in_shape(in_dims);
  ge::Shape in_origin_shape(in_origin_dims);
  ge::Shape out_shape(out_dims);
  ge::Shape out_origin_shape(out_origin_dims);

  ge::TensorDesc in_tensor_desc(in_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_tensor_desc.SetOriginShape(in_origin_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  ge::TensorDesc out_tensor_desc(out_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  out_tensor_desc.SetOriginShape(out_origin_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);

  input_data.update_input_desc_x(in_tensor_desc);
  input_data.update_output_desc_y(in_tensor_desc);

  ge::TensorDesc reshape_in_tensor(in_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(in_origin_shape);
  ge::TensorDesc reshape_out_tensor(out_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(out_origin_shape);

  auto transdata1_op = op::TransData("transdata_1");
  transdata1_op.update_input_desc_src(in_tensor_desc);
  transdata1_op.update_output_desc_dst(reshape_in_tensor);

  auto reshape_op = op::Reshape("reshape");
  reshape_op.update_input_desc_x(reshape_in_tensor);
  reshape_op.update_output_desc_y(reshape_out_tensor);

  auto transdata2_op = op::TransData("transdata_2");
  transdata2_op.update_input_desc_src(reshape_out_tensor);
  transdata2_op.update_output_desc_dst(out_tensor_desc);

  transdata1_op.set_input_src(input_data);
  reshape_op.set_input_x(transdata1_op);
  transdata2_op.set_input_src(reshape_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataZReshapeTransposeFusionPass",
  fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
  *compute_graph_ptr);

  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 4);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, false);
}

TEST_F(transdata_reshape_transpose_fusion_pass_test, transdata_reshape_transpose_test7) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> in_dims{1, 4, 16, 16};
  std::vector<int64_t> in_origin_dims{16};
  std::vector<int64_t> out_dims{4, 1, 16, 16};
  std::vector<int64_t> out_origin_dims{16, 64};
  ge::Shape in_shape(in_dims);
  ge::Shape in_origin_shape(in_origin_dims);
  ge::Shape out_shape(out_dims);
  ge::Shape out_origin_shape(out_origin_dims);

  ge::TensorDesc in_tensor_desc(in_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_tensor_desc.SetOriginShape(in_origin_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  ge::TensorDesc out_tensor_desc(out_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  out_tensor_desc.SetOriginShape(out_origin_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);

  input_data.update_input_desc_x(in_tensor_desc);
  input_data.update_output_desc_y(in_tensor_desc);

  ge::TensorDesc reshape_in_tensor(in_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(in_origin_shape);
  ge::TensorDesc reshape_out_tensor(out_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(out_origin_shape);

  auto transdata1_op = op::TransData("transdata_1");
  transdata1_op.update_input_desc_src(in_tensor_desc);
  transdata1_op.update_output_desc_dst(reshape_in_tensor);

  auto reshape_op = op::Reshape("reshape");
  reshape_op.update_input_desc_x(reshape_in_tensor);
  reshape_op.update_output_desc_y(reshape_out_tensor);

  auto transdata2_op = op::TransData("transdata_2");
  transdata2_op.update_input_desc_src(reshape_out_tensor);
  transdata2_op.update_output_desc_dst(out_tensor_desc);

  transdata1_op.set_input_src(input_data);
  reshape_op.set_input_x(transdata1_op);
  transdata2_op.set_input_src(reshape_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataZReshapeTransposeFusionPass",
  fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
  *compute_graph_ptr);

  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 4);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, false);
}

TEST_F(transdata_reshape_transpose_fusion_pass_test, transdata_reshape_transpose_test8) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> in_dims{1, 4, 16, 16};
  std::vector<int64_t> in_origin_dims{64, 32};
  std::vector<int64_t> out_dims{4, 1, 16, 16};
  std::vector<int64_t> out_origin_dims{32, 16};
  ge::Shape in_shape(in_dims);
  ge::Shape in_origin_shape(in_origin_dims);
  ge::Shape out_shape(out_dims);
  ge::Shape out_origin_shape(out_origin_dims);

  ge::TensorDesc in_tensor_desc(in_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_tensor_desc.SetOriginShape(in_origin_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  ge::TensorDesc out_tensor_desc(out_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  out_tensor_desc.SetOriginShape(out_origin_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);

  input_data.update_input_desc_x(in_tensor_desc);
  input_data.update_output_desc_y(in_tensor_desc);

  ge::TensorDesc reshape_in_tensor(in_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(in_origin_shape);
  ge::TensorDesc reshape_out_tensor(out_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(out_origin_shape);

  auto transdata1_op = op::TransData("transdata_1");
  transdata1_op.update_input_desc_src(in_tensor_desc);
  transdata1_op.update_output_desc_dst(reshape_in_tensor);

  auto reshape_op = op::Reshape("reshape");
  reshape_op.update_input_desc_x(reshape_in_tensor);
  reshape_op.update_output_desc_y(reshape_out_tensor);

  auto transdata2_op = op::TransData("transdata_2");
  transdata2_op.update_input_desc_src(reshape_out_tensor);
  transdata2_op.update_output_desc_dst(out_tensor_desc);

  transdata1_op.set_input_src(input_data);
  reshape_op.set_input_x(transdata1_op);
  transdata2_op.set_input_src(reshape_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataZReshapeTransposeFusionPass",
  fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
  *compute_graph_ptr);

  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 4);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, false);
}

TEST_F(transdata_reshape_transpose_fusion_pass_test, transdata_reshape_transpose_test9) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> in_dims{1, 4, 16, 16};
  std::vector<int64_t> in_origin_dims{64, 18};
  std::vector<int64_t> out_dims{4, 1, 16, 16};
  std::vector<int64_t> out_origin_dims{18, 64};
  ge::Shape in_shape(in_dims);
  ge::Shape in_origin_shape(in_origin_dims);
  ge::Shape out_shape(out_dims);
  ge::Shape out_origin_shape(out_origin_dims);

  ge::TensorDesc in_tensor_desc(in_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_tensor_desc.SetOriginShape(in_origin_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  ge::TensorDesc out_tensor_desc(out_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  out_tensor_desc.SetOriginShape(out_origin_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);

  input_data.update_input_desc_x(in_tensor_desc);
  input_data.update_output_desc_y(in_tensor_desc);

  ge::TensorDesc reshape_in_tensor(in_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(in_origin_shape);
  ge::TensorDesc reshape_out_tensor(out_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(out_origin_shape);

  auto transdata1_op = op::TransData("transdata_1");
  transdata1_op.update_input_desc_src(in_tensor_desc);
  transdata1_op.update_output_desc_dst(reshape_in_tensor);

  auto reshape_op = op::Reshape("reshape");
  reshape_op.update_input_desc_x(reshape_in_tensor);
  reshape_op.update_output_desc_y(reshape_out_tensor);

  auto transdata2_op = op::TransData("transdata_2");
  transdata2_op.update_input_desc_src(reshape_out_tensor);
  transdata2_op.update_output_desc_dst(out_tensor_desc);

  transdata1_op.set_input_src(input_data);
  reshape_op.set_input_x(transdata1_op);
  transdata2_op.set_input_src(reshape_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataZReshapeTransposeFusionPass",
  fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
  *compute_graph_ptr);

  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 4);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, false);
}

TEST_F(transdata_reshape_transpose_fusion_pass_test, transdata_reshape_transpose_test10) {
  ge::Graph graph("transdata_transpose_fusion_pass_test1");
  auto input_data = op::Data("Transdata_input_data");
  std::vector<int64_t> in_dims{1, 4, 16, 16};
  std::vector<int64_t> in_origin_dims{64};
  std::vector<int64_t> out_dims{4, 1, 16, 16};
  std::vector<int64_t> out_origin_dims{64};
  ge::Shape in_shape(in_dims);
  ge::Shape in_origin_shape(in_origin_dims);
  ge::Shape out_shape(out_dims);
  ge::Shape out_origin_shape(out_origin_dims);

  ge::TensorDesc in_tensor_desc(in_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  in_tensor_desc.SetOriginShape(in_origin_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  ge::TensorDesc out_tensor_desc(out_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  out_tensor_desc.SetOriginShape(out_origin_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);

  input_data.update_input_desc_x(in_tensor_desc);
  input_data.update_output_desc_y(in_tensor_desc);

  ge::TensorDesc reshape_in_tensor(in_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(in_origin_shape);
  ge::TensorDesc reshape_out_tensor(out_origin_shape, FORMAT_ND, DT_FLOAT16);
  reshape_in_tensor.SetOriginFormat(FORMAT_ND);
  reshape_in_tensor.SetOriginShape(out_origin_shape);

  auto transdata1_op = op::TransData("transdata_1");
  transdata1_op.update_input_desc_src(in_tensor_desc);
  transdata1_op.update_output_desc_dst(reshape_in_tensor);

  auto reshape_op = op::Reshape("reshape");
  reshape_op.update_input_desc_x(reshape_in_tensor);
  reshape_op.update_output_desc_y(reshape_out_tensor);

  auto transdata2_op = op::TransData("transdata_2");
  transdata2_op.update_input_desc_src(reshape_out_tensor);
  transdata2_op.update_output_desc_dst(out_tensor_desc);

  transdata1_op.set_input_src(input_data);
  reshape_op.set_input_x(transdata1_op);
  transdata2_op.set_input_src(reshape_op);

  std::vector<Operator> inputs{input_data};
  std::vector<Operator> outputs{transdata2_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("TransdataZReshapeTransposeFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS,
  *compute_graph_ptr);

  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 4);
  bool findTransPose = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "TransposeD") {
      findTransPose = true;
    }
  }
  EXPECT_EQ(findTransPose, false);
}