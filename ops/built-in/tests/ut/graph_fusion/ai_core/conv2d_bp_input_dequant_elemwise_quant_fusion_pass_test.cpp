#include "gtest/gtest.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_calculation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "fusion_pass_test_utils.h"
#include "quantize_ops.h"

using namespace ge;
using namespace op;

class tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test TearDown" << std::endl;
  }
};

namespace fe {
Status RunConv2dDxQuantBufferFusionPass(string fusion_pass_name, BufferFusionPassType pass_type,
                                        ge::ComputeGraph &compute_graph_ptr, size_t idx_pattern = 0) {
  std::map<string, BufferFusionPassRegistry::CreateFn> create_fns =
      BufferFusionPassRegistry::GetInstance().GetCreateFnByType(pass_type);
  const auto &iter = create_fns.find(fusion_pass_name);
  bool flag = (iter != create_fns.end());
  if (iter != create_fns.end()) {
    if (pass_type == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
      auto buffer_fusion_pass_base_ptr =
          std::unique_ptr<BufferFusionPassBase>(dynamic_cast<BufferFusionPassBase *>(iter->second()));
      if (buffer_fusion_pass_base_ptr == nullptr) {
        return FAILED;
      }
      buffer_fusion_pass_base_ptr->SetName(fusion_pass_name);
      vector<BufferFusionPattern *> patterns = buffer_fusion_pass_base_ptr->DefinePatterns();
      std::vector<BufferFusionOpDesc *> desc = patterns[idx_pattern]->GetOpDescs();
      ge::ComputeGraph::Vistor<ge::NodePtr> node_ptrs = compute_graph_ptr.GetAllNodes();

      vector<ge::NodePtr> dx_nodes;
      vector<ge::NodePtr> dequant_nodes;
      vector<ge::NodePtr> elemwise_nodes;
      vector<ge::NodePtr> quant_nodes;
      for (auto i : node_ptrs) {
        auto op_desc = i->GetOpDesc();
        if (op_desc->GetType() == "Conv2DBackpropInputD") {
          dx_nodes.push_back(i);
        }
        if (op_desc->GetType() == "AscendDequant") {
          dequant_nodes.push_back(i);
        }
        if (op_desc->GetType() == "PRelu") {
          elemwise_nodes.push_back(i);
        }
        if (op_desc->GetType() == "Add") {
          elemwise_nodes.push_back(i);
        }
        if (op_desc->GetType() == "AscendQuant") {
          quant_nodes.push_back(i);
        }
      }

      BufferFusionMapping mapping;
      for (auto i : desc) {
        if (i->desc_name == "conv2dbackpropinput") {
          mapping[i] = dx_nodes;
        }
        if (i->desc_name == "dequant") {
          mapping[i] = dequant_nodes;
        }
        if (i->desc_name == "elemwise") {
          mapping[i] = elemwise_nodes;
        }
        if (i->desc_name == "quant") {
          mapping[i] = quant_nodes;
        }
      }
      vector<ge::NodePtr> fusion_nodes;
      buffer_fusion_pass_base_ptr->GetFusionNodes(mapping, fusion_nodes);
      return SUCCESS;
    }
  }
  return FAILED;
}
}  // namespace fe

TEST_F(tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test,
       tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test_1) {
  ge::Graph graph("tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test_1");

  // create conv2d dx
  auto shape_filter = vector<int64_t>({4, 8, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(shape_filter), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_filter = op::Data("node_filter");
  node_filter.update_input_desc_x(desc_filter);
  node_filter.update_output_desc_y(desc_filter);

  auto shape_out_backprop = vector<int64_t>({1, 4, 3, 3});
  ge::TensorDesc desc_out_backprop(ge::Shape(shape_out_backprop), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_out_backprop = op::Data("node_out_backprop");
  node_out_backprop.update_input_desc_x(desc_out_backprop);
  node_out_backprop.update_output_desc_y(desc_out_backprop);

  auto node_dx = op::Conv2DBackpropInputD("Conv2DBackpropInputD")
                     .set_input_filter(node_filter)
                     .set_input_out_backprop(node_out_backprop)
                     .set_attr_input_size({1, 8, 5, 5})
                     .set_attr_strides({1, 1, 1, 1})
                     .set_attr_pads({0, 0, 0, 0})
                     .set_attr_dilations({1, 1, 1, 1})
                     .set_attr_data_format("NCHW");
  TensorDesc dx_input_desc_filter(ge::Shape(shape_filter), FORMAT_NCHW, DT_FLOAT16);
  TensorDesc dx_input_desc_out_backprop(ge::Shape(shape_out_backprop), FORMAT_NCHW, DT_FLOAT16);
  node_dx.update_input_desc_filter(desc_filter);
  node_dx.update_input_desc_out_backprop(desc_out_backprop);
  auto shape_y = vector<int64_t>({1, 8, 5, 5});
  TensorDesc dx_output_desc_y(ge::Shape(shape_y), FORMAT_NCHW, DT_FLOAT16);
  node_dx.update_output_desc_y(dx_output_desc_y);

  // create ascend_dequant
  auto node_dequant = op::AscendDequant("dequant");
  node_dequant.set_input_x(node_dx);

  // create PRelu
  auto shape_weight = vector<int64_t>({1, 8, 5, 5});
  ge::TensorDesc desc_weight(ge::Shape(shape_weight), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_weight = op::Data("node_weight");
  node_weight.update_input_desc_x(desc_weight);
  node_weight.update_output_desc_y(desc_weight);

  auto prelu = op::PRelu("prelu")
                  .set_input_x(node_dequant)
                  .set_input_weight(node_weight);
  prelu.update_input_desc_x(dx_output_desc_y);
  prelu.update_input_desc_weight(desc_weight);
  auto shape_prelu = vector<int64_t>({1, 8, 5, 5});
  TensorDesc prelu_output_desc_y(ge::Shape(shape_prelu), FORMAT_ND, DT_FLOAT16);
  prelu.update_output_desc_y(prelu_output_desc_y);

  // create ascend_quant
  auto node_quant = op::AscendQuant("quant");
  node_quant.set_input_x(prelu);

  // create graph
  std::vector<Operator> inputs{node_filter, node_out_backprop, node_weight};
  std::vector<Operator> outputs{node_quant};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  Status ret = fe::RunConv2dDxQuantBufferFusionPass("TbeDxDeqElemQuantPass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                                    *compute_graph_ptr);
  bool find_node_dx = false;
  bool find_node_prelu = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Conv2DBackpropInputD") {
      find_node_dx = true;
    }
    if (node->GetType() == "PRelu") {
      find_node_prelu = true;
    }
  }
  EXPECT_EQ(find_node_dx, true);
  EXPECT_EQ(find_node_prelu, true);
  EXPECT_EQ(ret, SUCCESS);
}

// check dynamic shape
TEST_F(tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test,
       tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test_2) {
  ge::Graph graph("tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test_2");

  // create conv2d dx
  auto shape_filter = vector<int64_t>({4, -1, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(shape_filter), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_filter = op::Data("node_filter");
  node_filter.update_input_desc_x(desc_filter);
  node_filter.update_output_desc_y(desc_filter);

  auto shape_out_backprop = vector<int64_t>({1, -1, 3, 3});
  ge::TensorDesc desc_out_backprop(ge::Shape(shape_out_backprop), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_out_backprop = op::Data("node_out_backprop");
  node_out_backprop.update_input_desc_x(desc_out_backprop);
  node_out_backprop.update_output_desc_y(desc_out_backprop);

  auto node_dx = op::Conv2DBackpropInputD("Conv2DBackpropInputD")
                     .set_input_filter(node_filter)
                     .set_input_out_backprop(node_out_backprop)
                     .set_attr_input_size({1, 8, 5, 5})
                     .set_attr_strides({1, 1, 1, 1})
                     .set_attr_pads({0, 0, 0, 0})
                     .set_attr_dilations({1, 1, 1, 1})
                     .set_attr_data_format("NCHW");
  TensorDesc dx_input_desc_filter(ge::Shape(shape_filter), FORMAT_NCHW, DT_FLOAT16);
  TensorDesc dx_input_desc_out_backprop(ge::Shape(shape_out_backprop), FORMAT_NCHW, DT_FLOAT16);
  node_dx.update_input_desc_filter(desc_filter);
  node_dx.update_input_desc_out_backprop(desc_out_backprop);
  auto shape_y = vector<int64_t>({1, 8, 5, 5});
  TensorDesc dx_output_desc_y(ge::Shape(shape_y), FORMAT_NCHW, DT_FLOAT16);
  node_dx.update_output_desc_y(dx_output_desc_y);

  // create ascend_dequant
  auto node_dequant = op::AscendDequant("dequant");
  node_dequant.set_input_x(node_dx);

  // create PRelu
  auto shape_weight = vector<int64_t>({1, 8, 5, 5});
  ge::TensorDesc desc_weight(ge::Shape(shape_weight), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_weight = op::Data("node_weight");
  node_weight.update_input_desc_x(desc_weight);
  node_weight.update_output_desc_y(desc_weight);

  auto prelu = op::PRelu("prelu")
                  .set_input_x(node_dequant)
                  .set_input_weight(node_weight);
  prelu.update_input_desc_x(dx_output_desc_y);
  prelu.update_input_desc_weight(desc_weight);
  auto shape_prelu = vector<int64_t>({1, 8, 5, 5});
  TensorDesc prelu_output_desc_y(ge::Shape(shape_prelu), FORMAT_ND, DT_FLOAT16);
  prelu.update_output_desc_y(prelu_output_desc_y);

  // create ascend_quant
  auto node_quant = op::AscendQuant("quant");
  node_quant.set_input_x(prelu);

  // create graph
  std::vector<Operator> inputs{node_filter, node_out_backprop, node_weight};
  std::vector<Operator> outputs{node_quant};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  Status ret = fe::RunConv2dDxQuantBufferFusionPass("TbeDxDeqElemQuantPass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                                    *compute_graph_ptr);
  bool find_node_dx = false;
  bool find_node_prelu = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Conv2DBackpropInputD") {
      find_node_dx = true;
    }
    if (node->GetType() == "PRelu") {
      find_node_prelu = true;
    }
  }
  EXPECT_EQ(find_node_dx, true);
  EXPECT_EQ(find_node_prelu, true);
  EXPECT_EQ(ret, SUCCESS);
}

// check elemwise node, only support LeakyRelu or Prelu
TEST_F(tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test,
       tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test_3) {
  ge::Graph graph("tbe_conv2d_backprop_input_dequant_elemwise_quant_fusion_pass_test_3");

  // create conv2d dx
  auto shape_filter = vector<int64_t>({4, 8, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(shape_filter), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_filter = op::Data("node_filter");
  node_filter.update_input_desc_x(desc_filter);
  node_filter.update_output_desc_y(desc_filter);

  auto shape_out_backprop = vector<int64_t>({1, 4, 3, 3});
  ge::TensorDesc desc_out_backprop(ge::Shape(shape_out_backprop), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_out_backprop = op::Data("node_out_backprop");
  node_out_backprop.update_input_desc_x(desc_out_backprop);
  node_out_backprop.update_output_desc_y(desc_out_backprop);

  auto node_dx = op::Conv2DBackpropInputD("Conv2DBackpropInputD")
                     .set_input_filter(node_filter)
                     .set_input_out_backprop(node_out_backprop)
                     .set_attr_input_size({1, 8, 5, 5})
                     .set_attr_strides({1, 1, 1, 1})
                     .set_attr_pads({0, 0, 0, 0})
                     .set_attr_dilations({1, 1, 1, 1})
                     .set_attr_data_format("NCHW");
  TensorDesc dx_input_desc_filter(ge::Shape(shape_filter), FORMAT_NCHW, DT_FLOAT16);
  TensorDesc dx_input_desc_out_backprop(ge::Shape(shape_out_backprop), FORMAT_NCHW, DT_FLOAT16);
  node_dx.update_input_desc_filter(desc_filter);
  node_dx.update_input_desc_out_backprop(desc_out_backprop);
  auto shape_y = vector<int64_t>({1, 8, 5, 5});
  TensorDesc dx_output_desc_y(ge::Shape(shape_y), FORMAT_NCHW, DT_FLOAT16);
  node_dx.update_output_desc_y(dx_output_desc_y);

  // create ascend_dequant
  auto node_dequant = op::AscendDequant("dequant");
  node_dequant.set_input_x(node_dx);

  // create add
  auto shape_weight = vector<int64_t>({1, 8, 5, 5});
  ge::TensorDesc desc_weight(ge::Shape(shape_weight), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_weight = op::Data("node_weight");
  node_weight.update_input_desc_x(desc_weight);
  node_weight.update_output_desc_y(desc_weight);

  auto node_add = op::Add("add")
                      .set_input_x1(node_dequant)
                      .set_input_x2(node_weight);
  auto shape_add = vector<int64_t>({1, 8, 5, 5});
  TensorDesc add_output_desc_y(ge::Shape(shape_add), FORMAT_NCHW, DT_FLOAT16);
  node_add.update_output_desc_y(add_output_desc_y);

  // create ascend_quant
  auto node_quant = op::AscendQuant("quant");
  node_quant.set_input_x(node_add);

  // create graph
  std::vector<Operator> inputs{node_filter, node_out_backprop, node_weight};
  std::vector<Operator> outputs{node_quant};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  Status ret = fe::RunConv2dDxQuantBufferFusionPass("TbeDxDeqElemQuantPass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                                    *compute_graph_ptr);
  bool find_node_dx = false;
  bool find_node_prelu = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Conv2DBackpropInputD") {
      find_node_dx = true;
    }
    if (node->GetType() == "PRelu") {
      find_node_prelu = true;
    }
  }
  EXPECT_EQ(find_node_dx, true);
  EXPECT_EQ(find_node_prelu, false);
  EXPECT_EQ(ret, SUCCESS);
}