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
#include "quantize_ops.h"
#include "matrix_calculation_ops.h"

using namespace ge;
using namespace op;

class ascend_deq_quant_antiq_maxpool_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "threshold_relu_fusion SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "threshold_relu_fusion TearDown" << std::endl;
  }
};

TEST_F(ascend_deq_quant_antiq_maxpool_fusion_test, ascend_deq_quant_antiq_maxpool_fusion_test_1) {
  //第一部分：使用IR进行构图，注意要对input和output赋属性描述
  ge::Graph graph("ascend_deq_quant_antiq_maxpool_fusion_test_1");

  auto fc_input_data1 = op::Data("fc_input_data0");
  auto fc_input_data2 = op::Data("fc_input_data1");
  std::vector<int64_t> dims1{2, 1280};
  ge::Shape shape1(dims1);
  ge::TensorDesc tensorDesc1(shape1, ge::FORMAT_NCHW, ge::DT_INT8);
  std::vector<int64_t> dims2{1000, 1280};
  ge::Shape shape2(dims2);
  ge::TensorDesc tensorDesc2(shape2, ge::FORMAT_NCHW, ge::DT_INT8);
  std::vector<int64_t> dims3{2, 1000};
  ge::Shape shape3(dims3);
  ge::TensorDesc tensorDesc3(shape3, ge::FORMAT_NCHW, ge::DT_INT32);
  auto fc_op =
      op::FullyConnection("fc_0").set_input_x(fc_input_data1).set_input_w(fc_input_data2).set_attr_num_output(1);
  fc_op.update_input_desc_x(tensorDesc1);
  fc_op.update_input_desc_w(tensorDesc2);
  fc_op.update_output_desc_y(tensorDesc3);

  float deq_scale = 1.0;
  ge::Shape shape({1});
  TensorDesc tensorDesc4(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  ge::Tensor scale_tensor(tensorDesc4, reinterpret_cast<uint8_t*>(&deq_scale), sizeof(float));
  auto const_op = op::Const("deq_scale").set_attr_value(scale_tensor);
  auto deq_op = op::AscendDequant("deq_op_0");
  deq_op.set_input_x(fc_op).set_input_deq_scale(const_op).set_attr_dtype(DT_FLOAT16);

  float scale = 1.0;
  float offset = 0.0;
  auto quant_op = op::AscendQuant("quant_op_0");
  quant_op.set_input_x(deq_op).set_attr_scale(scale).set_attr_offset(offset);

  auto antiq_op = op::AscendAntiQuant("antiq_op_0")
                      .set_input_x(quant_op)
                      .set_attr_scale(scale)
                      .set_attr_offset(offset)
                      .set_attr_dtype(DT_FLOAT16);

  ge::Operator::OpListInt maxpool_ksize = {1, 1, 1, 1};
  ge::Operator::OpListInt maxpool_strides = {1, 1, 1, 1};
  auto maxpool_op = op::MaxPool("maxpool_op_0")
                        .set_input_x(antiq_op)
                        .set_attr_ksize(maxpool_ksize)
                        .set_attr_strides(maxpool_strides)
                        .set_attr_padding("VALID")
                        .set_attr_data_format("NCHW");

  std::vector<Operator> inputs{fc_input_data1, fc_input_data2};
  std::vector<Operator> outputs{maxpool_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  //调用融合规则测试的Utils对图进行infershape
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  //调用融合Pass，需要指定融合规则名字
  fe::FusionPassTestUtils::RunGraphFusionPass("AscendDequantQuantAntiquantMaxpoolFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  bool fcMatch = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "FullyConnection") {
      fcMatch = true;
    }
  }
  EXPECT_EQ(fcMatch, true);
}