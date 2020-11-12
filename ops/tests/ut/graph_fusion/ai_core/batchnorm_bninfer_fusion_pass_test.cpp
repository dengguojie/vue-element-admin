#include "nn_batch_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"

using namespace ge;
using namespace op;

class batchnorm_bninfer_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() { std::cout << "inplace_add SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "inplace_add TearDown" << std::endl;
  }
};

TEST_F(batchnorm_bninfer_fusion_test, batchnorm_bninfer_fusion_test_1) {
  ge::Graph graph("batchnorm_bninfer_fusion_test_1");

  auto input0 = op::Data("input0");
  std::vector<int64_t> dims_ms{3, 32};
  ge::Shape shape_ms(dims_ms);
  ge::TensorDesc tensorDescMs(shape_ms);
  input0.update_input_desc_x(tensorDescMs);
  input0.update_output_desc_y(tensorDescMs);

  auto input1 = op::Const("intput1");
  Tensor axis;
  float *dataValue = new float[1];
  *dataValue = 1.1;
  axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));
  axis.SetData((uint8_t *)dataValue, 4);
  input1.set_attr_value(axis);
  input1.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));
  delete [] dataValue;

  auto input2 = op::Const("input2");
  input2.set_attr_value(axis);
  input2.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto input3 = op::Const("input3");
  input3.set_attr_value(axis);
  input3.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto input4 = op::Const("input4");
  input4.set_attr_value(axis);
  input4.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto batchnorm = op::BatchNorm("batchnorm");
  batchnorm.set_input_x(input0);
  batchnorm.set_input_scale(input1);
  batchnorm.set_input_offset(input2);
  batchnorm.set_input_mean(input3);
  batchnorm.set_input_variance(input4);
  batchnorm.set_attr_is_training(false);

  std::vector<Operator> inputs{input0, input1, input2, input3, input4};
  std::vector<Operator> outputs{batchnorm};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "batchnorm_bninfer_fusion_test_1_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchNormBnInferFusionPass",
                                              fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "batchnorm_bninfer_fusion_test_1_after");
}

TEST_F(batchnorm_bninfer_fusion_test, batchnorm_bninfer_fusion_test_2) {
  ge::Graph graph("batchnorm_bninfer_fusion_test_2");

  auto input0 = op::Data("input0");
  std::vector<int64_t> dims_ms{3, 32};
  ge::Shape shape_ms(dims_ms);
  ge::TensorDesc tensorDescMs(shape_ms);
  input0.update_input_desc_x(tensorDescMs);
  input0.update_output_desc_y(tensorDescMs);

  auto input1 = op::Const("input1");
  Tensor axis;
  float *dataValue = new float[1];
  *dataValue = 1.1;
  axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));
  axis.SetData((uint8_t *)dataValue, 4);
  input1.set_attr_value(axis);
  input1.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));
  delete[] dataValue;

  auto input2 = op::Const("input2");
  input2.set_attr_value(axis);
  input2.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto input3 = op::Const("input3");
  input3.set_attr_value(axis);
  input3.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto input4 = op::Data("input4");
  input4.update_input_desc_x(tensorDescMs);
  input4.update_output_desc_y(tensorDescMs);

  auto batchnorm = op::BatchNorm("batchnorm");
  batchnorm.set_input_x(input0);
  batchnorm.set_input_scale(input1);
  batchnorm.set_input_offset(input2);
  batchnorm.set_input_mean(input3);
  batchnorm.set_input_variance(input4);
  batchnorm.set_attr_is_training(false);

  std::vector<Operator> inputs{input0, input1, input2, input3, input4};
  std::vector<Operator> outputs{batchnorm};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "batchnorm_bninfer_fusion_test_2_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("BatchNormBnInferFusionPass",
                                              fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "batchnorm_bninfer_fusion_test_2_after");
}
