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
#include "nonlinear_fuc_ops.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
using namespace op;

class force_fp16_cast_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "force_fp16_cast_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "force_fp16_cast_fusion_test TearDown" << std::endl;
  }
};

TEST_F(force_fp16_cast_fusion_test, force_fp16_cast_fusion_test_1) {
  ge::Graph graph("force_fp16_cast_fusion_test");
  std::cout << "force_fp16_cast_fusion_test_1 SetUp" << std::endl;
  auto cast_input_data = op::Data("cast_input_data");
  std::vector<int64_t> dims{32, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  cast_input_data.update_input_desc_x(tensorDesc);
  cast_input_data.update_output_desc_y(tensorDesc);
  auto square_op = op::Square("square_0");
  square_op.set_input_x(cast_input_data);
  auto cast_op = op::Cast("cast_0");
  cast_op.set_input_x(square_op).set_attr_dst_type(1);
  auto relu_op_1 = op::Relu("relu_1");
  relu_op_1.set_input_x(cast_op);

  std::vector<Operator> inputs{cast_input_data};
  std::vector<Operator> outputs{relu_op_1};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ForceFp16CastFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool castMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Cast") {
      castMatch = true;
    }
  }
  EXPECT_EQ(castMatch, true);
}
TEST_F(force_fp16_cast_fusion_test, force_fp16_cast_fusion_test_2) {
  ge::Graph graph("force_fp16_cast_fusion_test");
  std::cout << "force_fp16_cast_fusion_test_2 SetUp" << std::endl;
  auto cast_input_data = op::Data("cast_input_data");
  std::vector<int64_t> dims{32, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  cast_input_data.update_input_desc_x(tensorDesc);
  cast_input_data.update_output_desc_y(tensorDesc);
  auto square_op = op::Square("square_0");
  square_op.set_input_x(cast_input_data);
  auto cast_op = op::Cast("cast_0");
  cast_op.set_input_x(square_op).set_attr_dst_type(4);
  auto relu_op_1 = op::Relu("relu_1");
  relu_op_1.set_input_x(cast_op);

  std::vector<Operator> inputs{cast_input_data};
  std::vector<Operator> outputs{relu_op_1};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ForceFp16CastFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool castMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Cast") {
      castMatch = true;
    }
  }
  EXPECT_EQ(castMatch, true);
}
TEST_F(force_fp16_cast_fusion_test, force_fp16_cast_fusion_test_3) {
  ge::Graph graph("force_fp16_cast_fusion_test");
  std::cout << "force_fp16_cast_fusion_test_3 SetUp" << std::endl;
  auto cast_input_data = op::Data("cast_input_data");
  std::vector<int64_t> dims{32, 32};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  cast_input_data.update_input_desc_x(tensorDesc);
  cast_input_data.update_output_desc_y(tensorDesc);
  auto square_op = op::Square("square_0");
  square_op.set_input_x(cast_input_data);
  auto cast_op = op::Cast("cast_0");
  cast_op.set_input_x(square_op).set_attr_dst_type(3);
  auto relu_op_1 = op::Relu("relu_1");
  relu_op_1.set_input_x(cast_op);

  std::vector<Operator> inputs{cast_input_data};
  std::vector<Operator> outputs{relu_op_1};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ForceFp16CastFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool castMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Cast") {
      castMatch = true;
    }
  }
  EXPECT_EQ(castMatch, true);
}
TEST_F(force_fp16_cast_fusion_test, force_fp16_cast_fusion_test_4) {
  ge::Graph graph("force_fp16_cast_fusion_test");
  std::cout << "force_fp16_cast_fusion_test_4 SetUp" << std::endl;
  auto cast_input_data = op::Data("cast_input_data");
  std::vector<int64_t> dims{
      1,
      4,
      4,
      32,
  };
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  cast_input_data.update_input_desc_x(tensorDesc);
  cast_input_data.update_output_desc_y(tensorDesc);

  auto square_op = op::AvgPool("square_0");
  square_op.set_input_x(cast_input_data)
      .set_attr_ksize({1, 4, 4, 1})
      .set_attr_strides({1, 1, 1, 1})
      .set_attr_padding("VALID")
      .set_attr_data_format("NHWC");
  auto cast_op = op::Cast("cast_0");
  cast_op.set_input_x(square_op).set_attr_dst_type(3);
  auto relu_op_1 = op::Relu("relu_1");
  relu_op_1.set_input_x(cast_op);

  std::vector<Operator> inputs{cast_input_data};
  std::vector<Operator> outputs{relu_op_1};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ForceFp16CastFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr,
                                              false);
  bool castMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Cast") {
      castMatch = true;
    }
  }
  EXPECT_EQ(castMatch, true);
}
TEST_F(force_fp16_cast_fusion_test, force_fp16_cast_fusion_test_6) {
  ge::Graph graph("force_fp16_cast_fusion_test");
  std::cout << "force_fp16_cast_fusion_test_6 SetUp" << std::endl;
  auto cast_input_data = op::Data("cast_input_data");
  std::vector<int64_t> dims{2, 2};
  ge::Shape shape(dims);
  ge::TensorDesc tensorDesc(shape, ge::FORMAT_NHWC, ge::DT_INT32);
  cast_input_data.update_input_desc_x(tensorDesc);
  cast_input_data.update_output_desc_y(tensorDesc);
  auto square_op = op::Square("square_0");
  square_op.set_input_x(cast_input_data);
  auto cast_op = op::Cast("cast_0");
  cast_op.set_input_x(square_op).set_attr_dst_type(1);
  auto relu_op_1 = op::Relu("relu_1");
  relu_op_1.set_input_x(cast_op);

  std::vector<Operator> inputs{cast_input_data};
  std::vector<Operator> outputs{relu_op_1};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ForceFp16CastFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool castMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Cast") {
      castMatch = true;
    }
  }
  EXPECT_EQ(castMatch, true);
}
