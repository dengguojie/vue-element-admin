#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "fusion_pass_test_utils.h"


class depthwise_conv2d_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "depthwise_conv2d_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "depthwise_conv2d_fusion_pass_test TearDown" << std::endl;
    }
};

/************************************
 *                  x1
 *                   |
 *                  transdata
 *                 /   \
 *       x    filter  netoutput
 *          \  /         |
 *         conv2d
 *            |
 *
 *
 *************************************/
TEST_F(depthwise_conv2d_fusion_pass_test, fuse_n_c_dim_test) {
  ge::OpDescPtr x = std::make_shared<ge::OpDesc>("x", "Data");
  ge::OpDescPtr x1 = std::make_shared<ge::OpDesc>("x1", "Data");
  ge::OpDescPtr filter = std::make_shared<ge::OpDesc>("filter", "Data");
  ge::OpDescPtr d_conv = std::make_shared<ge::OpDesc>("depthwise_conv2d", "DepthwiseConv2D");
  ge::OpDescPtr relu = std::make_shared<ge::OpDesc>("relu", "Relu");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");

  ge::GeShape input_shape({1, 16, 256, 256});
  ge::GeTensorDesc x_desc(input_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  x_desc.SetOriginFormat(ge::FORMAT_NCHW);
  x_desc.SetOriginDataType(ge::DT_FLOAT);
  x_desc.SetOriginShape(input_shape);

  ge::GeShape output_shape({1, 4, 256, 256});
  ge::GeTensorDesc y_desc(output_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  y_desc.SetOriginFormat(ge::FORMAT_NCHW);
  y_desc.SetOriginDataType(ge::DT_FLOAT);
  y_desc.SetOriginShape(output_shape);

  ge::GeShape filter_shape({1, 1, 16, 4});
  ge::GeTensorDesc filter_desc(filter_shape, ge::FORMAT_HWCN, ge::DT_FLOAT);
  filter_desc.SetOriginFormat(ge::FORMAT_HWCN);
  filter_desc.SetOriginDataType(ge::DT_FLOAT);
  filter_desc.SetOriginShape(filter_shape);

  x->AddOutputDesc(x_desc);
  filter->AddInputDesc(filter_desc);
  filter->AddOutputDesc(filter_desc);
  d_conv->AddInputDesc("x", x_desc);
  d_conv->AddInputDesc("filter", filter_desc);
  d_conv->AddOutputDesc(y_desc);
  x1->AddOutputDesc(filter_desc);
  relu->AddOutputDesc(filter_desc);
  relu->AddInputDesc(filter_desc);
  netoutput->AddInputDesc(filter_desc);

  ge::AttrUtils::SetListInt(d_conv, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(d_conv, "pads", {0, 0, 0, 0});
  ge::AttrUtils::SetListInt(d_conv, "strides", {1, 1, 1, 1});

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr x_node = compute_graph_ptr->AddNode(x);
  ge::NodePtr filter_node = compute_graph_ptr->AddNode(filter);
  ge::NodePtr d_conv_node = compute_graph_ptr->AddNode(d_conv);
  ge::NodePtr x1_node = compute_graph_ptr->AddNode(x1);
  ge::NodePtr relu_node = compute_graph_ptr->AddNode(relu);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);


  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), d_conv_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), d_conv_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), filter_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  vector<int64_t> out_shape;
  vector<int64_t> in_shape;
  vector<int64_t> exp_in = {1, 1, 16, 4};
  vector<int64_t> exp_out = {1, 1, 1, 64};
  bool has_attr = false;
  fe::FusionPassTestUtils::RunGraphFusionPass("ADepthwiseFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  for (ge::NodePtr &node : compute_graph_ptr->GetDirectNode()) {
    std::cout << "type:" << node->GetType() << ", name:" << node->GetName() << std::endl;
    if (node->GetName() == "filter/Reshape") {
      ge::OpDescPtr desc = node->GetOpDesc();
      auto out_tensor = desc->GetOutputDesc(0);
      out_shape = out_tensor.GetOriginShape().GetDims();
      auto in_tensor = desc->GetInputDesc(0);
      in_shape = in_tensor.GetOriginShape().GetDims();
    }
  }
  EXPECT_EQ(in_shape, exp_in);
  EXPECT_EQ(out_shape, exp_out);
}

/************************************
 *                  x1
 *                   |
 *                  relu
 *                 /   \
 *       x    filter  netoutput
 *          \  /         |
 *         conv2d
 *            |
 *
 *
 *************************************/
TEST_F(depthwise_conv2d_fusion_pass_test, format_nhwc_test) {
  ge::OpDescPtr x = std::make_shared<ge::OpDesc>("x", "Data");
  ge::OpDescPtr x1 = std::make_shared<ge::OpDesc>("x1", "Data");
  ge::OpDescPtr filter = std::make_shared<ge::OpDesc>("filter", "Data");
  ge::OpDescPtr d_conv = std::make_shared<ge::OpDesc>("depthwise_conv2d", "DepthwiseConv2D");
  ge::OpDescPtr relu = std::make_shared<ge::OpDesc>("relu", "Relu");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");

  ge::GeShape input_shape({1, 16, 256, 256});
  ge::GeTensorDesc x_desc(input_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  x_desc.SetOriginFormat(ge::FORMAT_NCHW);
  x_desc.SetOriginDataType(ge::DT_FLOAT);
  x_desc.SetOriginShape(input_shape);

  ge::GeShape output_shape({1, 4, 256, 256});
  ge::GeTensorDesc y_desc(output_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  y_desc.SetOriginFormat(ge::FORMAT_NCHW);
  y_desc.SetOriginDataType(ge::DT_FLOAT);
  y_desc.SetOriginShape(output_shape);

  ge::GeShape filter_shape({4, 1, 1, 16});
  ge::GeTensorDesc filter_desc(filter_shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  filter_desc.SetOriginFormat(ge::FORMAT_NHWC);
  filter_desc.SetOriginDataType(ge::DT_FLOAT);
  filter_desc.SetOriginShape(filter_shape);

  x->AddOutputDesc(x_desc);
  filter->AddInputDesc(filter_desc);
  filter->AddOutputDesc(filter_desc);
  d_conv->AddInputDesc("x", x_desc);
  d_conv->AddInputDesc("filter", filter_desc);
  d_conv->AddOutputDesc(y_desc);
  x1->AddOutputDesc(filter_desc);
  relu->AddOutputDesc(filter_desc);
  relu->AddInputDesc(filter_desc);
  netoutput->AddInputDesc(filter_desc);

  ge::AttrUtils::SetListInt(d_conv, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(d_conv, "pads", {0, 0, 0, 0});
  ge::AttrUtils::SetListInt(d_conv, "strides", {1, 1, 1, 1});

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr x_node = compute_graph_ptr->AddNode(x);
  ge::NodePtr filter_node = compute_graph_ptr->AddNode(filter);
  ge::NodePtr d_conv_node = compute_graph_ptr->AddNode(d_conv);
  ge::NodePtr x1_node = compute_graph_ptr->AddNode(x1);
  ge::NodePtr relu_node = compute_graph_ptr->AddNode(relu);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);


  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), d_conv_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), d_conv_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), filter_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  fe::Status res = fe::FusionPassTestUtils::RunGraphFusionPass("ADepthwiseFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  EXPECT_EQ(res, fe::NOT_CHANGED);
}

/************************************
 *                  x1
 *                   |
 *                  relu
 *                 /   \
 *       x    filter  netoutput
 *          \  /         |
 *         conv2d
 *            |
 *
 *
 *************************************/
TEST_F(depthwise_conv2d_fusion_pass_test, has_been_set_test) {
  ge::OpDescPtr x = std::make_shared<ge::OpDesc>("x", "Data");
  ge::OpDescPtr x1 = std::make_shared<ge::OpDesc>("x1", "Data");
  ge::OpDescPtr filter = std::make_shared<ge::OpDesc>("filter", "Data");
  ge::OpDescPtr d_conv = std::make_shared<ge::OpDesc>("depthwise_conv2d", "DepthwiseConv2D");
  ge::OpDescPtr relu = std::make_shared<ge::OpDesc>("relu", "Relu");
  ge::OpDescPtr netoutput = std::make_shared<ge::OpDesc>("netoutput", "NetOutput");

  ge::GeShape input_shape({1, 16, 256, 256});
  ge::GeTensorDesc x_desc(input_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  x_desc.SetOriginFormat(ge::FORMAT_NCHW);
  x_desc.SetOriginDataType(ge::DT_FLOAT);
  x_desc.SetOriginShape(input_shape);

  ge::GeShape output_shape({1, 4, 256, 256});
  ge::GeTensorDesc y_desc(output_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  y_desc.SetOriginFormat(ge::FORMAT_NCHW);
  y_desc.SetOriginDataType(ge::DT_FLOAT);
  y_desc.SetOriginShape(output_shape);

  ge::GeShape filter_shape({1, 1, 16, 4});
  ge::GeTensorDesc filter_desc(filter_shape, ge::FORMAT_HWCN, ge::DT_FLOAT);
  filter_desc.SetOriginFormat(ge::FORMAT_HWCN);
  filter_desc.SetOriginDataType(ge::DT_FLOAT);
  filter_desc.SetOriginShape(filter_shape);

  x->AddOutputDesc(x_desc);
  filter->AddInputDesc(filter_desc);
  filter->AddOutputDesc(filter_desc);
  d_conv->AddInputDesc("x", x_desc);
  d_conv->AddInputDesc("filter", filter_desc);
  d_conv->AddOutputDesc(y_desc);
  x1->AddOutputDesc(filter_desc);
  relu->AddOutputDesc(filter_desc);
  relu->AddInputDesc(filter_desc);
  netoutput->AddInputDesc(filter_desc);

  ge::AttrUtils::SetBool(filter, "_has_been_changed", true);
  ge::AttrUtils::SetListInt(d_conv, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(d_conv, "pads", {0, 0, 0, 0});
  ge::AttrUtils::SetListInt(d_conv, "strides", {1, 1, 1, 1});

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr x_node = compute_graph_ptr->AddNode(x);
  ge::NodePtr filter_node = compute_graph_ptr->AddNode(filter);
  ge::NodePtr d_conv_node = compute_graph_ptr->AddNode(d_conv);
  ge::NodePtr x1_node = compute_graph_ptr->AddNode(x1);
  ge::NodePtr relu_node = compute_graph_ptr->AddNode(relu);
  ge::NodePtr netoutput_node = compute_graph_ptr->AddNode(netoutput);


  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), d_conv_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), d_conv_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), filter_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  fe::FusionPassTestUtils::RunGraphFusionPass("ADepthwiseFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
}

/************************************
 *       x    filter (not 4d)
 *          \  /        
 *         conv2d
 *            |
 *
 *************************************/
TEST_F(depthwise_conv2d_fusion_pass_test, filter_input_not_4d_test) {
  ge::OpDescPtr x = std::make_shared<ge::OpDesc>("x", "Data");
  ge::OpDescPtr filter = std::make_shared<ge::OpDesc>("filter", "Data");
  ge::OpDescPtr d_conv = std::make_shared<ge::OpDesc>("depthwise_conv2d", "DepthwiseConv2D");

  ge::GeShape input_shape({1, 16, 256, 256});
  ge::GeTensorDesc x_desc(input_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  x_desc.SetOriginFormat(ge::FORMAT_NCHW);
  x_desc.SetOriginDataType(ge::DT_FLOAT);
  x_desc.SetOriginShape(input_shape);

  ge::GeShape output_shape({1, 4, 256, 256});
  ge::GeTensorDesc y_desc(output_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  y_desc.SetOriginFormat(ge::FORMAT_NCHW);
  y_desc.SetOriginDataType(ge::DT_FLOAT);
  y_desc.SetOriginShape(output_shape);

  ge::GeShape filter_5d_shape({4, 1, 1, 1, 16});
  ge::GeTensorDesc filter_5d_desc(filter_5d_shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  filter_5d_desc.SetOriginFormat(ge::FORMAT_NC1HWC0);
  filter_5d_desc.SetOriginDataType(ge::DT_FLOAT);
  filter_5d_desc.SetOriginShape(filter_5d_shape);

  x->AddOutputDesc(x_desc);
  filter->AddInputDesc(filter_5d_desc);
  filter->AddOutputDesc(filter_5d_desc);
  d_conv->AddInputDesc("x", x_desc);
  d_conv->AddInputDesc("filter", filter_5d_desc);
  d_conv->AddOutputDesc(y_desc);

  ge::AttrUtils::SetListInt(d_conv, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(d_conv, "pads", {0, 0, 0, 0});
  ge::AttrUtils::SetListInt(d_conv, "strides", {1, 1, 1, 1});

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr x_node = compute_graph_ptr->AddNode(x);
  ge::NodePtr filter_node = compute_graph_ptr->AddNode(filter);
  ge::NodePtr d_conv_node = compute_graph_ptr->AddNode(d_conv);


  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), d_conv_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), d_conv_node->GetInDataAnchor(1));

  fe::Status res = fe::FusionPassTestUtils::RunGraphFusionPass("ADepthwiseFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  EXPECT_EQ(res, fe::FAILED);
}

/************************************
 *
 *       x    filter
 *          \  /
 *         conv2d
 *            |
 *
 *************************************/
TEST_F(depthwise_conv2d_fusion_pass_test, all_filter_input_empty_test) {
  ge::OpDescPtr x = std::make_shared<ge::OpDesc>("x", "Data");
  ge::OpDescPtr x1 = std::make_shared<ge::OpDesc>("x1", "Data");
  ge::OpDescPtr filter = std::make_shared<ge::OpDesc>("filter", "Data");
  ge::OpDescPtr d_conv = std::make_shared<ge::OpDesc>("depthwise_conv2d", "DepthwiseConv2D");

  ge::GeShape input_shape({1, 16, 256, 256});
  ge::GeTensorDesc x_desc(input_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  x_desc.SetOriginFormat(ge::FORMAT_NCHW);
  x_desc.SetOriginDataType(ge::DT_FLOAT);
  x_desc.SetOriginShape(input_shape);

  ge::GeShape output_shape({1, 4, 256, 256});
  ge::GeTensorDesc y_desc(output_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  y_desc.SetOriginFormat(ge::FORMAT_NCHW);
  y_desc.SetOriginDataType(ge::DT_FLOAT);
  y_desc.SetOriginShape(output_shape);

  ge::GeShape filter_shape({1, 1, 16, 4});
  ge::GeTensorDesc filter_desc(filter_shape, ge::FORMAT_HWCN, ge::DT_FLOAT);
  filter_desc.SetOriginFormat(ge::FORMAT_HWCN);
  filter_desc.SetOriginDataType(ge::DT_FLOAT);
  filter_desc.SetOriginShape(filter_shape);

  x->AddOutputDesc(x_desc);
  filter->AddInputDesc(filter_desc);
  filter->AddOutputDesc(filter_desc);
  d_conv->AddInputDesc("x", x_desc);
  d_conv->AddInputDesc("filter", filter_desc);
  d_conv->AddOutputDesc(y_desc);

  ge::AttrUtils::SetListInt(d_conv, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(d_conv, "pads", {0, 0, 0, 0});
  ge::AttrUtils::SetListInt(d_conv, "strides", {1, 1, 1, 1});

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr x_node = compute_graph_ptr->AddNode(x);
  ge::NodePtr filter_node = compute_graph_ptr->AddNode(filter);
  ge::NodePtr d_conv_node = compute_graph_ptr->AddNode(d_conv);

  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), d_conv_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), d_conv_node->GetInDataAnchor(1));

  vector<int64_t> out_shape;
  vector<int64_t> in_shape;
  vector<int64_t> exp_in = {1, 1, 16, 4};
  vector<int64_t> exp_out = {1, 1, 1, 64};
  bool has_attr = false;
  fe::FusionPassTestUtils::RunGraphFusionPass("ADepthwiseFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  for (ge::NodePtr &node : compute_graph_ptr->GetDirectNode()) {
    std::cout << "type:" << node->GetType() << ", name:" << node->GetName() << std::endl;
    if (node->GetName() == "filter/Reshape") {
      ge::OpDescPtr desc = node->GetOpDesc();
      auto out_tensor = desc->GetOutputDesc(0);
      out_shape = out_tensor.GetOriginShape().GetDims();
      auto in_tensor = desc->GetInputDesc(0);
      in_shape = in_tensor.GetOriginShape().GetDims();
    }
  }
  EXPECT_EQ(in_shape, exp_in);
  EXPECT_EQ(out_shape, exp_out);
}