#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"
#include "fp16_t.hpp"

using namespace ge;

namespace fe {

class conv2d_to_fullyconnection_fusion_pass_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "conv2d_to_fullyconnection_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "conv2d_to_fullyconnection_fusion_pass_test TearDown" << std::endl;
  }

  /******************************************
   *
   *  inputs     filter   bias(if exist)
   *        \      |     /
   *              \ /
   *             conv2d
   *               |
   *
   ******************************************/
  void BuildGraph(ComputeGraphPtr &compute_graph) {
    ge::Graph graph("test");
    auto data_0 = op::Data("data_0");
    ge::Shape shape_x({1,7,7,16});
    ge::TensorDesc tensor_desc_x(shape_x, FORMAT_NHWC, DT_FLOAT16);
    data_0.update_input_desc_x(tensor_desc_x);
    data_0.update_output_desc_y(tensor_desc_x);

    TensorDesc filter_desc(ge::Shape({7,7,16,4}), FORMAT_HWCN, DT_FLOAT16);
    filter_desc.SetOriginFormat(FORMAT_HWCN);
    auto filter_0 = op::Const("filter_0");
    Tensor filter;
    fp16_t * filter_value = new fp16_t[7*7*16*4];
    filter.SetTensorDesc(filter_desc);
    filter.SetData((uint8_t*)filter_value, 7*7*16*4*sizeof(fp16_t));
    filter_0.set_attr_value(filter);
    filter_0.update_output_desc_y(filter_desc);

    TensorDesc bias_desc(ge::Shape({4}), FORMAT_ND, DT_FLOAT16);
    bias_desc.SetOriginFormat(FORMAT_ND);
    auto bias_0 = op::Const("bias_0");
    Tensor bias;
    fp16_t * biasValue = new fp16_t[4];
    bias.SetTensorDesc(bias_desc);
    bias.SetData((uint8_t*)biasValue, 4*sizeof(fp16_t));
    bias_0.set_attr_value(bias);
    bias_0.update_output_desc_y(bias_desc);

    TensorDesc out_desc(ge::Shape({1,1,1,4}), FORMAT_NHWC, DT_FLOAT16);
    out_desc.SetOriginFormat(FORMAT_NHWC);

    auto conv2d_layer = op::Conv2D("conv2d");
    conv2d_layer.set_input_x(data_0)
                .set_input_filter(filter_0)
                .set_input_bias(bias_0)
                .set_attr_strides({1,1,1,1})
                .set_attr_pads({0,0,0,0});

    conv2d_layer.update_input_desc_x(tensor_desc_x);
    conv2d_layer.update_input_desc_filter(filter_desc);
    conv2d_layer.update_input_desc_bias(bias_desc);
    conv2d_layer.update_output_desc_y(out_desc);

    auto relu_0 = op::Relu("relu_0");
    relu_0.set_input_x(conv2d_layer);
    relu_0.update_input_desc_x(out_desc);
    relu_0.update_output_desc_y(out_desc);

    delete[] filter_value;
    delete[] biasValue;

    std::vector<Operator> inputs{data_0, filter_0, bias_0};
    std::vector<Operator> outputs{relu_0};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};

TEST_F(conv2d_to_fullyconnection_fusion_pass_test, conv2d_to_fullyconnection_fusion_pass_test_1) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ConvToFullyConnectionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool find_fullyconnection_flag = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "FullyConnection") {
      find_fullyconnection_flag = true;
      break;
    }
  }

  EXPECT_EQ(find_fullyconnection_flag, true);

}

} // namespace fe
