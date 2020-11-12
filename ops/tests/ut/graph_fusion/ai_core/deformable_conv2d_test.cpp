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

class deformable_conv2d_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "deformable_conv2d_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "deformable_conv2d_test TearDown" << std::endl;
  }

  /******************************************
   *
   *  inputs filter    offsets bias(if exist)
   *        \    \       /    /
   *                \/
   *         deformable_conv2d
   *                |
   *
   ******************************************/
  void BuildGraph(ComputeGraphPtr &compute_graph) {
    ge::Graph graph("test");
    auto data_0 = op::Data("data_0");
    ge::Shape shape_x({1, 7, 7, 16});
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT16);
    data_0.update_input_desc_x(tensorDescX);
    data_0.update_output_desc_y(tensorDescX);

    TensorDesc filterDesc(ge::Shape({3,3,16,4}), FORMAT_HWCN, DT_FLOAT16);
    filterDesc.SetOriginFormat(FORMAT_HWCN);
    auto filter_0 = op::Const("filter_0");
    Tensor filter;
    fp16_t * filterValue = new fp16_t[3*3*16*4];
    filter.SetTensorDesc(filterDesc);
    filter.SetData((uint8_t*)filterValue, 3*3*16*4*sizeof(fp16_t));
    filter_0.set_attr_value(filter);
    filter_0.update_output_desc_y(filterDesc);

    TensorDesc offsetDesc(ge::Shape({1,7,7,4}), FORMAT_NHWC, DT_FLOAT16);
    offsetDesc.SetOriginFormat(FORMAT_NHWC);
    auto offset_0 = op::Const("offset_0");
    Tensor offset;
    fp16_t * offsetValue = new fp16_t[7*7*4*1];
    offset.SetTensorDesc(offsetDesc);
    offset.SetData((uint8_t*)offsetValue, 7*7*4*1*sizeof(fp16_t));
    offset_0.set_attr_value(offset);
    offset_0.update_output_desc_y(offsetDesc);

    TensorDesc biasDesc(ge::Shape({4}), FORMAT_ND, DT_FLOAT16);
    biasDesc.SetOriginFormat(FORMAT_ND);
    auto bias_0 = op::Const("bias_0");
    Tensor bias;
    fp16_t * biasValue = new fp16_t[4];
    bias.SetTensorDesc(biasDesc);
    bias.SetData((uint8_t*)biasValue, 4*sizeof(fp16_t));
    bias_0.set_attr_value(bias);
    bias_0.update_output_desc_y(biasDesc);

    auto dfm_conv2d_layer = op::DeformableConv2D("dfm_conv2d");
    dfm_conv2d_layer.set_input_x(data_0)
                    .set_input_filter(filter_0)
                    .set_input_offsets(offset_0)
                    .set_input_bias(bias_0)
                    .set_attr_strides({0,1,1,0})
                    .set_attr_pads({1,1,1,1});

    dfm_conv2d_layer.update_input_desc_x(tensorDescX);
    dfm_conv2d_layer.update_output_desc_y(offsetDesc);

    auto relu_0 = op::Relu("relu_0");
    relu_0.set_input_x(dfm_conv2d_layer);
    relu_0.update_input_desc_x(offsetDesc);
    relu_0.update_output_desc_y(offsetDesc);

    delete[] filterValue;
    delete[] offsetValue;
    delete[] biasValue;

    std::vector<Operator> inputs{data_0, filter_0, offset_0, bias_0};
    std::vector<Operator> outputs{relu_0};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};




TEST_F(deformable_conv2d_test, deformable_conv2d_test_1) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ADeformableConv2dPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool findSplit = false;
  bool findConcat = false;
  bool ksizeEqual = false;
  bool padsEqual = false;
  bool stridesEqual = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "DeformableOffsets") {
      findSplit = true;
      OpDescPtr offsetDesc = node->GetOpDesc();
      std::vector<int64_t> expksize(2,3);
      std::vector<int64_t> ksize;
      AttrUtils::GetListInt(offsetDesc, "ksize", ksize),
      ksizeEqual = ksize == expksize;
    }
    if (node->GetType() == "Conv2D") {
      findConcat = true;
      OpDescPtr convDesc = node->GetOpDesc();
      std::vector<int64_t> expstrides = {0, 3, 3, 0};
      std::vector<int64_t> strides;
      AttrUtils::GetListInt(convDesc, "strides", strides),
      stridesEqual = strides == expstrides;
      std::vector<int64_t> exppads(4);
      std::vector<int64_t> pads;
      AttrUtils::GetListInt(convDesc, "pads", pads),
      padsEqual = pads == exppads;
    }
  }
  EXPECT_EQ(findSplit, true);
  EXPECT_EQ(ksizeEqual, true);
  EXPECT_EQ(padsEqual, true);
  EXPECT_EQ(stridesEqual, true);
  EXPECT_EQ(findConcat, true);
}
} // namespace fe