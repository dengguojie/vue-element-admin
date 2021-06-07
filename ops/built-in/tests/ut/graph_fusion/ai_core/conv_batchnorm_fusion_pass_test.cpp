#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "fusion_pass_test_utils.h"

using namespace ge;

class conv_batchnorm_fusion_pass_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "conv_batchnorm_fusion_pass_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "conv_batchnorm_fusion_pass_test TearDown" << std::endl;
    }
};

/* conv2d + biasadd */
TEST_F(conv_batchnorm_fusion_pass_test, biasadd_conv_fusion_test_1) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
  OpDescPtr filter_const = std::make_shared<OpDesc>("filter_const", "Const");
  OpDescPtr scale_const = std::make_shared<OpDesc>("scale_const", "Const");
  OpDescPtr offset_const = std::make_shared<OpDesc>("offset_const", "Const");
  OpDescPtr mean_const = std::make_shared<OpDesc>("mean_const", "Const");
  OpDescPtr variance_const = std::make_shared<OpDesc>("variance_const", "Const");
  OpDescPtr conv = std::make_shared<OpDesc>("conv", "Conv2D");
  OpDescPtr batch_norm = std::make_shared<OpDesc>("batch_norm", "BatchNorm");

  vector<int64_t> input_dim = {10, 3, 320, 320};
  GeShape input_shape(input_dim);
  GeTensorDesc input_tenosr_desc(input_shape, FORMAT_NCHW, DT_FLOAT);
  input_tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  input_tenosr_desc.SetOriginDataType(DT_FLOAT);
  input_tenosr_desc.SetOriginShape(input_shape);

  vector<int64_t> output_dim = {10, 3, 320, 320};
  GeShape output_shape(output_dim);
  GeTensorDesc output_tenosr_desc(output_shape, FORMAT_NCHW, DT_FLOAT);
  output_tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  output_tenosr_desc.SetOriginDataType(DT_FLOAT);
  output_tenosr_desc.SetOriginShape(output_shape);

  vector<int64_t> filter_dim = {40, 3, 3, 3};
  GeShape filter_shape(filter_dim);
  GeTensorDesc filter_tenosr_desc(filter_shape, FORMAT_NCHW, DT_FLOAT);
  filter_tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  filter_tenosr_desc.SetOriginDataType(DT_FLOAT);
  filter_tenosr_desc.SetOriginShape(filter_shape);

  vector<int64_t> const_dim = {40, 3, 3, 3};
  GeShape const_shape(const_dim);
  GeTensorDesc const_tenosr_desc(const_shape, FORMAT_NCHW, DT_FLOAT);
  const_tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  const_tenosr_desc.SetOriginDataType(DT_FLOAT);
  const_tenosr_desc.SetOriginShape(const_shape);

  data->AddOutputDesc(input_tenosr_desc);
  filter_const->AddOutputDesc(filter_tenosr_desc);
  scale_const->AddOutputDesc(const_tenosr_desc);
  offset_const->AddOutputDesc(const_tenosr_desc);
  mean_const->AddOutputDesc(const_tenosr_desc);
  variance_const->AddOutputDesc(const_tenosr_desc);

  conv->AddInputDesc("x", input_tenosr_desc);
  conv->AddInputDesc("filter", filter_tenosr_desc);
  conv->AddOutputDesc(output_tenosr_desc);
  batch_norm->AddInputDesc("x", output_tenosr_desc);
  batch_norm->AddInputDesc("scale", const_tenosr_desc);
  batch_norm->AddInputDesc("offset", const_tenosr_desc);
  batch_norm->AddInputDesc("mean", const_tenosr_desc);
  batch_norm->AddInputDesc("variance", const_tenosr_desc);
  batch_norm->AddOutputDesc(output_tenosr_desc);

  ge::AttrUtils::SetListInt(conv, "dilations", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(conv, "pads", {1, 1, 1, 1});
  ge::AttrUtils::SetListInt(conv, "strides", {1, 1, 2, 2});

  ge::ComputeGraphPtr compute_graph_ptr = std::make_shared<ComputeGraph>("test");
  NodePtr data_node = compute_graph_ptr->AddNode(data);
  NodePtr filter_node = compute_graph_ptr->AddNode(filter_const);
  NodePtr scale_node = compute_graph_ptr->AddNode(scale_const);
  NodePtr offset_node = compute_graph_ptr->AddNode(offset_const);
  NodePtr mean_node = compute_graph_ptr->AddNode(mean_const);
  NodePtr variance_node = compute_graph_ptr->AddNode(variance_const);
  NodePtr conv_node = compute_graph_ptr->AddNode(conv);
  NodePtr batch_norm_node = compute_graph_ptr->AddNode(batch_norm);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), batch_norm_node->GetInDataAnchor(0));

  GraphUtils::AddEdge(scale_node->GetOutDataAnchor(0), batch_norm_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(offset_node->GetOutDataAnchor(0), batch_norm_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(mean_node->GetOutDataAnchor(0), batch_norm_node->GetInDataAnchor(3));
  GraphUtils::AddEdge(variance_node->GetOutDataAnchor(0), batch_norm_node->GetInDataAnchor(4));

  GraphUtils::AddEdge(data_node->GetOutControlAnchor(), filter_node->GetInControlAnchor());
  GraphUtils::AddEdge(conv_node->GetOutControlAnchor(), scale_node->GetInControlAnchor());
  GraphUtils::AddEdge(conv_node->GetOutControlAnchor(), offset_node->GetInControlAnchor());
  GraphUtils::AddEdge(conv_node->GetOutControlAnchor(), mean_node->GetInControlAnchor());
  GraphUtils::AddEdge(conv_node->GetOutControlAnchor(), variance_node->GetInControlAnchor());

  fe::FusionPassTestUtils::RunGraphFusionPass("ConvBatchnormFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  for (ge::NodePtr &node : compute_graph_ptr->GetDirectNode()) {
    std::cout << "type:" << node->GetType() << ", name:" << node->GetName() << std::endl;
  }
}
