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

class hostbn_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() { std::cout << "hostbn_fusion_test SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "hostbn_fusion_test TearDown" << std::endl;
  }
};

TEST_F(hostbn_fusion_test, hostbn_fusion_test_1) {
  ge::Graph graph("hostbn_fusion_test_1");

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

  auto input5 = op::Const("input5");
  input4.set_attr_value(axis);
  input4.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_NCHW, DT_FLOAT));

  auto bn = op::BNInference("BNInference");
  bn.set_input_x(input0);
  bn.set_input_scale(input1);
  bn.set_input_offset(input2);
  bn.set_input_mean(input3);
  bn.set_input_variance(input4);
  bn.set_input_momentum(input5);
  bn.set_attr_epsilon(1e-5f);
  bn.set_attr_use_global_stats(true);
  bn.set_attr_mode(1);

  std::vector<Operator> inputs{input0, input1, input2, input3, input4, input5};
  std::vector<Operator> outputs{bn};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr =
      ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "HostBNFusionPass_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("HostBNFusionPass",
                                              fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  GE_DUMP(compute_graph_ptr, "HostBNFusionPass_after");
}
