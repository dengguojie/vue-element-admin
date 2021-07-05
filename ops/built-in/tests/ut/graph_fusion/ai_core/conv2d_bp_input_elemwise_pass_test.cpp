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

class conv2d_bp_input_elemwise_pass_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "conv2d_bp_input_elemwise_pass_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "conv2d_bp_input_elemwise_pass_test TearDown" << std::endl;
  }
};

namespace fe {
Status RunBufferFusionPass(string fusionPassName, BufferFusionPassType passType, ge::ComputeGraph& computeGraph, size_t idx_pattern=0) {
  std::map<string, BufferFusionPassRegistry::CreateFn> createFns =
      BufferFusionPassRegistry::GetInstance().GetCreateFnByType(passType);
  const auto& iter = createFns.find(fusionPassName);
  if (iter != createFns.end()) {
    if (passType == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
      auto BufferFusionPassBasePtr =
          std::unique_ptr<BufferFusionPassBase>(dynamic_cast<BufferFusionPassBase*>(iter->second()));
      if (BufferFusionPassBasePtr == nullptr) {
        return FAILED;
      }
      BufferFusionPassBasePtr->SetName(fusionPassName);
      vector<BufferFusionPattern*> patterns = BufferFusionPassBasePtr->DefinePatterns();

      std::vector<BufferFusionOpDesc*> desc = patterns[idx_pattern]->GetOpDescs();

      ge::ComputeGraph::Vistor<ge::NodePtr> NodePtrs = computeGraph.GetAllNodes();

      vector<ge::NodePtr> elemwiseNodes;
      vector<ge::NodePtr> dxNodes;
      for (auto i : NodePtrs) {
        auto opDesc = i->GetOpDesc();
        if (opDesc->GetType() == "Conv2DBackpropInputD") {
          dxNodes.push_back(i);
        }
        if (opDesc->GetType() == "PRelu") {
          elemwiseNodes.push_back(i);
        }
        if (opDesc->GetType() == "Mul") {
          elemwiseNodes.push_back(i);
        }
      }

      BufferFusionMapping mapping;
      for (auto i : desc) {
        if (i->desc_name == "conv2dbackpropinput") {
          mapping[i] = dxNodes;
        }
        if (i->desc_name == "elemwise") {
          mapping[i] = elemwiseNodes;
        }
      }
      vector<ge::NodePtr> fusion_nodes;
      BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
      return SUCCESS;
    }
  }

  return FAILED;
}
}  // namespace fe
TEST_F(conv2d_bp_input_elemwise_pass_test, conv2d_bp_input_elemwise_pass_test_1) {
  ge::Graph graph("conv2d_bp_input_elemwise_pass_test_1");

  // create conv2d_b
  auto nchw_shape_filter = vector<int64_t>({1, 4, 5, 5});
  ge::TensorDesc desc_filter(ge::Shape(nchw_shape_filter), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto data_filter = op::Data("data_filter");
  data_filter.update_input_desc_x(desc_filter);
  data_filter.update_output_desc_y(desc_filter);

  auto nchw_shape_out_backprop = vector<int64_t>({4, 8, 3, 3});
  ge::TensorDesc desc_out_backprop(ge::Shape(nchw_shape_out_backprop), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto data_out_backprop = op::Data("data_out_backprop");
  data_out_backprop.update_input_desc_x(desc_out_backprop);
  data_out_backprop.update_output_desc_y(desc_out_backprop);

  auto dx = op::Conv2DBackpropInputD("Conv2DBackpropInputD")
                .set_input_filter(data_filter)
                .set_input_out_backprop(data_out_backprop)
                .set_attr_input_size({1, 8, 5, 5})
                .set_attr_strides({1, 1, 1, 1})
                .set_attr_pads({0, 0, 0, 0})
                .set_attr_dilations({1, 1, 1, 1})
                .set_attr_data_format("NCHW");
  TensorDesc dx_input_desc_filter(ge::Shape(nchw_shape_filter), FORMAT_NCHW, DT_FLOAT16);
  TensorDesc dx_input_desc_out_backprop(ge::Shape(nchw_shape_out_backprop), FORMAT_NCHW, DT_FLOAT16);
  dx.update_input_desc_filter(desc_filter);
  dx.update_input_desc_out_backprop(desc_out_backprop);
  auto nc1hwc0_shape_y_dx = vector<int64_t>({1, 8, 5, 5});
  TensorDesc dx_output_desc_y(ge::Shape(nc1hwc0_shape_y_dx), FORMAT_NCHW, DT_FLOAT16);
  dx.update_output_desc_y(dx_output_desc_y);

  // create PRelu
  auto nchw_shape_weight = vector<int64_t>({1, 8, 5, 5});
  ge::TensorDesc desc_weight(ge::Shape(nchw_shape_weight), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto data_weight = op::Data("data_weight");
  data_weight.update_input_desc_x(desc_weight);
  data_weight.update_output_desc_y(desc_weight);

  auto prelu = op::PRelu("prelu")
                  .set_input_x(dx)
                  .set_input_weight(data_weight);
  prelu.update_input_desc_x(dx_output_desc_y);
  prelu.update_input_desc_weight(desc_weight);
  auto nc1hwc0_shape_y_prelu = vector<int64_t>({1, 8, 5, 5});
  TensorDesc prelu_output_desc_y(ge::Shape(nc1hwc0_shape_y_prelu), FORMAT_ND, DT_FLOAT16);
  prelu.update_output_desc_y(prelu_output_desc_y);

  std::vector<Operator> inputs{data_filter, data_out_backprop, data_weight};
  std::vector<Operator> outputs{prelu};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::RunBufferFusionPass("TbeDxElemwisePass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                          *compute_graph_ptr, 1);

  bool find_mul = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Conv2DBackpropInputD") {
      find_mul = true;
    }
  }
  EXPECT_EQ(find_mul, true);
}

TEST_F(conv2d_bp_input_elemwise_pass_test, conv2d_bp_input_elemwise_pass_test_2) {
    ge::Graph graph("conv2d_bp_input_elemwise_pass_test_2");

    auto dedy_shape = vector<int64_t>({1, 128, 214, 214});
    ge::TensorDesc desc_dedy(ge::Shape(dedy_shape), FORMAT_NCHW, DT_FLOAT16);
    auto data1 = op::Data("data1");
    data1.update_input_desc_x(desc_dedy);
    data1.update_output_desc_y(desc_dedy);

    auto dedx_shape = vector<int64_t>({1, 64, 214, 214});
    ge::TensorDesc desc_dedx(ge::Shape(dedx_shape), FORMAT_NCHW, DT_FLOAT16);
    auto data2 = op::Data("data2").set_attr_index(1);
    data2.update_input_desc_x(desc_dedx);
    data2.update_output_desc_y(desc_dedx);

    auto conv2dbackpropfilterd = op::Conv2DBackpropFilterD("conv2dbackpropfilterd")
        .set_input_x(data2)
        .set_input_out_backprop(data1)
        .set_attr_filter_size({128,16,1,1})
        .set_attr_strides({1,1,1,1})
        .set_attr_pads({0,0,0,0})
        .set_attr_dilations({1,1,1,1})
        .set_attr_groups({4})
        .set_attr_data_format("NCHW");
    ge::TensorDesc input_desc_outbackprop(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc input_desc_x(ge::Shape(), FORMAT_NCHW, DT_FLOAT16);
    ge::TensorDesc output_desc_y(ge::Shape(), FORMAT_NCHW, DT_FLOAT);
    conv2dbackpropfilterd.update_input_desc_out_backprop(input_desc_outbackprop);
    conv2dbackpropfilterd.update_input_desc_x(input_desc_x);
    conv2dbackpropfilterd.update_output_desc_y(output_desc_y);

    auto x2_shape = vector<int64_t>({1, 64, 214, 214});
    ge::TensorDesc x2_desc(ge::Shape(x2_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    auto data_x2 = op::Data("data_x2");
    data_x2.update_input_desc_x(x2_desc);
    data_x2.update_output_desc_y(x2_desc);

    auto mul = op::Mul("mul")
        .set_input_x1(conv2dbackpropfilterd)
        .set_input_x2(data_x2);

    std::vector<Operator> inputs{data1, data2};
    std::vector<Operator> outputs{mul};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::RunBufferFusionPass("TbeDxElemwisePass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, *compute_graph_ptr);

    bool find_mul = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            find_mul = true;
        }
    }
    EXPECT_EQ(find_mul, true);
}
