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
    ge::TensorDesc tensor_desc_x(shape_x, FORMAT_NHWC, DT_FLOAT16);
    data_0.update_input_desc_x(tensor_desc_x);
    data_0.update_output_desc_y(tensor_desc_x);

    TensorDesc filter_desc(ge::Shape({3,3,16,4}), FORMAT_HWCN, DT_FLOAT16);
    filter_desc.SetOriginFormat(FORMAT_HWCN);
    auto filter_0 = op::Const("filter_0");
    Tensor filter;
    fp16_t * filter_value = new fp16_t[3*3*16*4];
    filter.SetTensorDesc(filter_desc);
    filter.SetData((uint8_t*)filter_value, 3*3*16*4*sizeof(fp16_t));
    filter_0.set_attr_value(filter);
    filter_0.update_output_desc_y(filter_desc);

    TensorDesc offset_desc(ge::Shape({1,7,7,27}), FORMAT_NHWC, DT_FLOAT16);
    offset_desc.SetOriginFormat(FORMAT_NHWC);
    auto offset_0 = op::Const("offset_0");
    Tensor offset;
    fp16_t * offset_value = new fp16_t[7*7*27*1];
    offset.SetTensorDesc(offset_desc);
    offset.SetData((uint8_t*)offset_value, 7*7*27*1*sizeof(fp16_t));
    offset_0.set_attr_value(offset);
    offset_0.update_output_desc_y(offset_desc);

    TensorDesc bias_desc(ge::Shape({4}), FORMAT_ND, DT_FLOAT16);
    bias_desc.SetOriginFormat(FORMAT_ND);
    auto bias_0 = op::Const("bias_0");
    Tensor bias;
    fp16_t * biasValue = new fp16_t[4];
    bias.SetTensorDesc(bias_desc);
    bias.SetData((uint8_t*)biasValue, 4*sizeof(fp16_t));
    bias_0.set_attr_value(bias);
    bias_0.update_output_desc_y(bias_desc);

    TensorDesc out_desc(ge::Shape({1,4,4,4}), FORMAT_NHWC, DT_FLOAT16);
    out_desc.SetOriginFormat(FORMAT_NHWC);

    auto dfm_conv2d_layer = op::DeformableConv2D("dfm_conv2d");
    dfm_conv2d_layer.set_input_x(data_0)
                    .set_input_filter(filter_0)
                    .set_input_offsets(offset_0)
                    .set_input_bias(bias_0)
                    .set_attr_strides({1,2,2,1})
                    .set_attr_pads({1,1,1,1});

    dfm_conv2d_layer.update_input_desc_x(tensor_desc_x);
    dfm_conv2d_layer.update_output_desc_y(out_desc);

    auto relu_0 = op::Relu("relu_0");
    relu_0.set_input_x(dfm_conv2d_layer);
    relu_0.update_input_desc_x(out_desc);
    relu_0.update_output_desc_y(out_desc);

    delete[] filter_value;
    delete[] offset_value;
    delete[] biasValue;

    std::vector<Operator> inputs{data_0, filter_0, offset_0, bias_0};
    std::vector<Operator> outputs{relu_0};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }

  void BuildGraph1(ComputeGraphPtr &compute_graph) {
    ge::Graph graph("test");
    auto data_0 = op::Data("data_0");
    ge::Shape shape_x({1, 16, 7, 7});
    ge::TensorDesc tensor_desc_x(shape_x, FORMAT_NCHW, DT_FLOAT16);
    data_0.update_input_desc_x(tensor_desc_x);
    data_0.update_output_desc_y(tensor_desc_x);

    TensorDesc filter_desc(ge::Shape({3,3,16,4}), FORMAT_HWCN, DT_FLOAT16);
    filter_desc.SetOriginFormat(FORMAT_HWCN);
    auto filter_0 = op::Const("filter_0");
    Tensor filter;
    fp16_t * filter_value = new fp16_t[3*3*16*4];
    filter.SetTensorDesc(filter_desc);
    filter.SetData((uint8_t*)filter_value, 3*3*16*4*sizeof(fp16_t));
    filter_0.set_attr_value(filter);
    filter_0.update_output_desc_y(filter_desc);

    TensorDesc offset_desc(ge::Shape({1,27,7,7}), FORMAT_NCHW, DT_FLOAT16);
    offset_desc.SetOriginFormat(FORMAT_NCHW);
    auto offset_0 = op::Const("offset_0");
    Tensor offset;
    fp16_t * offset_value = new fp16_t[7*7*27*1];
    offset.SetTensorDesc(offset_desc);
    offset.SetData((uint8_t*)offset_value, 7*7*27*1*sizeof(fp16_t));
    offset_0.set_attr_value(offset);
    offset_0.update_output_desc_y(offset_desc);

    TensorDesc bias_desc(ge::Shape({4}), FORMAT_ND, DT_FLOAT16);
    bias_desc.SetOriginFormat(FORMAT_ND);
    auto bias_0 = op::Const("bias_0");
    Tensor bias;
    fp16_t * biasValue = new fp16_t[4];
    bias.SetTensorDesc(bias_desc);
    bias.SetData((uint8_t*)biasValue, 4*sizeof(fp16_t));
    bias_0.set_attr_value(bias);
    bias_0.update_output_desc_y(bias_desc);

    TensorDesc out_desc(ge::Shape({1,4,4,4}), FORMAT_NHWC, DT_FLOAT16);
    out_desc.SetOriginFormat(FORMAT_NHWC);

    auto dfm_conv2d_layer = op::DeformableConv2D("dfm_conv2d");
    dfm_conv2d_layer.set_input_x(data_0)
                    .set_input_filter(filter_0)
                    .set_input_offsets(offset_0)
                    .set_input_bias(bias_0)
                    .set_attr_strides({1,1,2,2})
                    .set_attr_pads({1,1,1,1});

    dfm_conv2d_layer.update_input_desc_x(tensor_desc_x);
    dfm_conv2d_layer.update_output_desc_y(out_desc);

    auto relu_0 = op::Relu("relu_0");
    relu_0.set_input_x(dfm_conv2d_layer);
    relu_0.update_input_desc_x(out_desc);
    relu_0.update_output_desc_y(out_desc);

    delete[] filter_value;
    delete[] offset_value;
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
  bool find_offsets = false;
  bool find_conv2d = false;
  bool offset_out_equal = false;
  bool conv_in_equal = false;
  bool ksize_equal = false;
  bool pads_equal = false;
  bool strides_equal = false;
  bool bias_equal = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "DeformableOffsets") {
      find_offsets = true;
      OpDescPtr offset_desc = node->GetOpDesc();
      auto y_tensor = offset_desc->GetOutputDesc(0);
      vector<int64_t> out_shape = y_tensor.GetOriginShape().GetDims();
      vector<int64_t> exp_out = {1, 12, 12, 16};
      offset_out_equal = out_shape == exp_out;
      std::vector<int64_t> exp_ksize(2,3);
      std::vector<int64_t> ksize;
      AttrUtils::GetListInt(offset_desc, "ksize", ksize),
      ksize_equal = ksize == exp_ksize;
    }
    if (node->GetType() == "Conv2D") {
      find_conv2d = true;
      OpDescPtr convDesc = node->GetOpDesc();
      std::vector<int64_t> exp_strides = {1, 3, 3, 1};
      std::vector<int64_t> strides;
      AttrUtils::GetListInt(convDesc, "strides", strides),
      strides_equal = strides == exp_strides;
      std::vector<int64_t> exp_pads(4);
      std::vector<int64_t> pads;
      AttrUtils::GetListInt(convDesc, "pads", pads),
      pads_equal = pads == exp_pads;
      auto x_tensor = convDesc->GetInputDesc(0);
      vector<int64_t> in_shape = x_tensor.GetOriginShape().GetDims();
      vector<int64_t> exp_in = {1, 12, 12, 16};
      conv_in_equal = in_shape == exp_in;
      auto bias_tensor = convDesc->GetInputDesc(2);
      vector<int64_t> bias_shape = bias_tensor.GetOriginShape().GetDims();
      vector<int64_t> exp_bias = {4};
      bias_equal = bias_shape == exp_bias;
    }
  }
  EXPECT_EQ(find_offsets, true);
  EXPECT_EQ(ksize_equal, true);
  EXPECT_EQ(pads_equal, true);
  EXPECT_EQ(strides_equal, true);
  EXPECT_EQ(find_conv2d, true);
  EXPECT_EQ(offset_out_equal, true);
  EXPECT_EQ(conv_in_equal, true);
  EXPECT_EQ(bias_equal, true);
}

TEST_F(deformable_conv2d_test, deformable_conv2d_test_2) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph1(compute_graph);
  FusionPassTestUtils::InferShapeAndType(compute_graph);
  FusionPassTestUtils::RunGraphFusionPass("ADeformableConv2dPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
  bool find_offsets = false;
  bool find_conv2d = false;
  bool offset_out_equal = false;
  bool conv_in_equal = false;
  bool conv_out_equal = false;
  bool ksize_equal = false;
  bool pads_equal = false;
  bool strides_equal = false;
  for (auto node: compute_graph->GetAllNodes()) {
    if (node->GetType() == "DeformableOffsets") {
      find_offsets = true;
      OpDescPtr offset_desc = node->GetOpDesc();
      auto y_tensor = offset_desc->GetOutputDesc(0);
      vector<int64_t> out_shape = y_tensor.GetOriginShape().GetDims();
      vector<int64_t> exp_out = {1, 16, 12, 12};
      offset_out_equal = out_shape == exp_out;
      std::vector<int64_t> exp_ksize(2,3);
      std::vector<int64_t> ksize;
      AttrUtils::GetListInt(offset_desc, "ksize", ksize),
      ksize_equal = ksize == exp_ksize;
    }
    if (node->GetType() == "Conv2D") {
      find_conv2d = true;
      OpDescPtr convDesc = node->GetOpDesc();
      std::vector<int64_t> exp_strides = {1, 1, 3, 3};
      std::vector<int64_t> strides;
      AttrUtils::GetListInt(convDesc, "strides", strides),
      strides_equal = strides == exp_strides;
      std::vector<int64_t> exp_pads(4);
      std::vector<int64_t> pads;
      AttrUtils::GetListInt(convDesc, "pads", pads),
      pads_equal = pads == exp_pads;
      auto x_tensor = convDesc->GetInputDesc(0);
      vector<int64_t> in_shape = x_tensor.GetOriginShape().GetDims();
      vector<int64_t> exp_in = {1, 16, 12, 12};
      conv_in_equal = in_shape == exp_in;
      auto y_tensor = convDesc->GetOutputDesc(0);
      vector<int64_t> out_shape = y_tensor.GetOriginShape().GetDims();
      vector<int64_t> exp_out = {1, 4, 4, 4};
      conv_out_equal = out_shape == exp_out;
    }
  }
  EXPECT_EQ(find_offsets, true);
  EXPECT_EQ(ksize_equal, true);
  EXPECT_EQ(pads_equal, true);
  EXPECT_EQ(strides_equal, true);
  EXPECT_EQ(find_conv2d, true);
  EXPECT_EQ(offset_out_equal, true);
  EXPECT_EQ(conv_in_equal, true);
  EXPECT_EQ(conv_out_equal, true);
}
} // namespace fe