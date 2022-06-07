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

using namespace ge;
using namespace op;

class tbe_conv2d_backprop_elemwise_fusion_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tbe_conv2d_backprop_elemwise_fusion_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tbe_conv2d_backprop_elemwise_fusion_pass_test TearDown" << std::endl;
  }
};

namespace fe {
Status RunConv2dDxBufferFusionPass(string fusion_pass_name, BufferFusionPassType pass_type,
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
      vector<ge::NodePtr> elemwise_nodes;
      vector<ge::NodePtr> elemwise_nodes1;
      for (auto i : node_ptrs) {
        auto op_desc = i->GetOpDesc();
        if (op_desc->GetType() == "Conv2DBackpropInputD") {
          dx_nodes.push_back(i);
        }
        if (op_desc->GetType() == "Add") {
          elemwise_nodes.push_back(i);
        }
        if (op_desc->GetType() == "Mul") {
          elemwise_nodes1.push_back(i);
        }
      }

      BufferFusionMapping mapping;
      for (auto i : desc) {
        if (i->desc_name == "conv2dBackpropInput") {
          mapping[i] = dx_nodes;
        }
        if (i->desc_name == "eltwise") {
          mapping[i] = elemwise_nodes;
        }
        if (i->desc_name == "eltwise1") {
          mapping[i] = elemwise_nodes1;
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

TEST_F(tbe_conv2d_backprop_elemwise_fusion_pass_test, tbe_conv2d_backprop_elemwise_fusion_pass_test_1) {
  ge::Graph graph("tbe_conv2d_backprop_elemwise_fusion_pass_test_1");

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

  // create add
  auto shape_data = vector<int64_t>({1, 8, 5, 5});
  ge::TensorDesc desc_data1(ge::Shape(shape_data), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_data1 = op::Data("node_data1");
  node_data1.update_input_desc_x(desc_data1);
  node_data1.update_output_desc_y(desc_data1);

  auto node_add = op::Add("Add").set_input_x1(node_dx).set_input_x2(node_data1);
  node_add.update_output_desc_y(desc_data1);

  // create mul
  ge::TensorDesc desc_data2(ge::Shape(shape_data), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto node_data2 = op::Data("node_data2");
  node_data2.update_input_desc_x(desc_data2);
  node_data2.update_output_desc_y(desc_data2);

  auto node_mul = op::Mul("Mul").set_input_x1(node_add).set_input_x2(node_data2);

  // create graph
  std::vector<Operator> inputs{node_filter, node_out_backprop, node_data1, node_data2};
  std::vector<Operator> outputs{node_mul};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  Status ret = fe::RunConv2dDxBufferFusionPass("TbeConv2DBackpropElemwiseFusionPass",
                                               fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, *compute_graph_ptr);
  bool find_node_dx = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Conv2DBackpropInputD") {
      find_node_dx = true;
    }
  }
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(find_node_dx, true);
}