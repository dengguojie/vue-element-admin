#include "array_ops.h"
#include "framework/common/types.h"
#include "fusion_pass_test_utils.h"
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_calculation_ops.h"
#include "transformation_ops.h"
#include "buffer_fusion/ub_fusion/ai_core/conv2d_backprop_input/tbe_dx_transdata_fusion_pass.h"

#define private public
#define protected public
#include "common/util/platform_info.h"

using namespace ge;
using namespace op;

class tbe_dx_transdata_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tbe_dx_transdata_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tbe_dx_transdata_fusion_test TearDown" << std::endl;
  }
};

namespace fe {
static Status RunBufferFusionPass(string fusion_pass_name, BufferFusionPassType pass_type,
                                  ge::ComputeGraph &compute_graph, vector<ge::NodePtr> &fusion_nodes) {
  std::map<string, BufferFusionPassRegistry::CreateFn> create_fns =
    BufferFusionPassRegistry::GetInstance().GetCreateFnByType(pass_type);
  const auto &iter = create_fns.find(fusion_pass_name);
  if (iter != create_fns.end()) {
    if (pass_type == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
      auto BufferFusionPassBasePtr =
        std::unique_ptr<BufferFusionPassBase>(dynamic_cast<BufferFusionPassBase*>(iter->second()));
      if (BufferFusionPassBasePtr == nullptr) {
        return FAILED;
      }
      BufferFusionPassBasePtr->SetName(fusion_pass_name);
      vector<BufferFusionPattern*> patterns = BufferFusionPassBasePtr->DefinePatterns();

      for (auto pattern : patterns) {
        std::vector<BufferFusionOpDesc*> node_descs = pattern->GetOpDescs();
        ge::ComputeGraph::Vistor<ge::NodePtr> nodes = compute_graph.GetAllNodes();

        vector<ge::NodePtr> cube_nodes;
        vector<ge::NodePtr> transdata1_nodes;
        vector<ge::NodePtr> transdata2_nodes;
        for (auto node : nodes) {
          auto op_desc = node->GetOpDesc();
          if (op_desc->GetName() == "conv2d_bp_input") {
            cube_nodes.push_back(node);
          }
          if (op_desc->GetName() == "transdata1") {
            transdata1_nodes.push_back(node);
          }
          if (op_desc->GetName() == "transdata2") {
            transdata2_nodes.push_back(node);
          }
        }

        BufferFusionMapping mapping;
        for (auto node_desc : node_descs) {
          if (node_desc->desc_name == "cube") {
            mapping[node_desc] = cube_nodes;
          }
          if (node_desc->desc_name == "transdata1") {
            mapping[node_desc] = transdata1_nodes;
          }
          if (node_desc->desc_name == "transdata2") {
            mapping[node_desc] = transdata2_nodes;
          }
        }
        BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
      }
      return SUCCESS;
    }
  }
  return FAILED;
}
}

TEST_F(tbe_dx_transdata_fusion_test, tbe_dx_transdata_fusion_test_1) {
  ge::Graph graph("tbe_dx_transdata_fusion_test_1");

  vector<int64_t> ori_shape_filter = {3, 3, 32, 64};
  vector<int64_t> shape_filter = {18, 4, 16, 16};
  ge::TensorDesc desc_filter(ge::Shape(shape_filter), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
  desc_filter.SetOriginShape(ge::Shape(ori_shape_filter));
  desc_filter.SetOriginFormat(ge::FORMAT_HWCN);
  auto data_filter = op::Data("data_filter");
  data_filter.update_input_desc_x(desc_filter);
  data_filter.update_output_desc_y(desc_filter);

  // nchw: (16, 32, 64, 64)
  vector<int64_t> shape_out_backprop = {-1, -1, -1, -1};
  ge::TensorDesc desc_out_backprop(ge::Shape(shape_out_backprop), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_out_backprop = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};
  desc_out_backprop.SetShapeRange(range_out_backprop);
  auto data_out_backprop = op::Data("data_out_backprop");
  data_out_backprop.update_input_desc_x(desc_out_backprop);
  data_out_backprop.update_output_desc_y(desc_out_backprop);

  auto transdata1 = op::TransData("transdata1").set_input_src(data_out_backprop);
  vector<int64_t> shape_transdata1 = {-1, -1, -1, -1, 16};
  ge::TensorDesc desc_transdata1(ge::Shape(shape_transdata1), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  transdata1.update_input_desc_src(desc_out_backprop);
  transdata1.update_output_desc_dst(desc_transdata1);

  auto input_size = op::Const("input_size");
  vector<int64_t> input_size_shape = {4};
  ge::TensorDesc input_size_desc(ge::Shape(input_size_shape), ge::FORMAT_NCHW, ge::DT_INT32);
  std::vector<std::pair<int64_t,int64_t>> range_transdata1 = {{1, -1}, {1, -1}, {1, -1}, {1, -1}, {16, 16}};
  desc_transdata1.SetShapeRange(range_transdata1);
  Tensor input_size_tensor;
  int32_t input_size_value[] = {-1, -1, -1, -1};
  input_size_tensor.SetTensorDesc(input_size_desc);
  input_size_tensor.SetData((uint8_t *)input_size_value, sizeof(input_size_value));
  input_size.set_attr_value(input_size_tensor);
  input_size.update_output_desc_y(input_size_desc);

  auto dx = op::Conv2DBackpropInput("conv2d_bp_input")
                .set_input_input_size(input_size)
                .set_input_filter(data_filter)
                .set_input_out_backprop(transdata1)
                .set_attr_strides({1, 1, 1, 1})
                .set_attr_pads({-1, -1, -1, -1})
                .set_attr_dilations({1, 1, 1, 1})
                .set_attr_groups({1})
                .set_attr_data_format("NCHW");
  dx.update_input_desc_filter(desc_filter);
  dx.update_input_desc_out_backprop(desc_transdata1);
  vector<int64_t> shape_y = {-1, -1, -1, -1, 16};
  TensorDesc desc_y(ge::Shape(shape_y), FORMAT_NC1HWC0, DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> range_y = {{1, -1}, {1, -1}, {1, -1}, {1, -1}, {16, 16}};
  desc_y.SetShapeRange(range_y);
  dx.update_output_desc_y(desc_y);

  auto transdata2 = op::TransData("transdata2").set_input_src(dx);
  vector<int64_t> shape_transdata2 = {-1, -1, -1, -1};
  ge::TensorDesc desc_transdata2(ge::Shape(shape_transdata2), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  transdata2.update_input_desc_src(desc_y);
  transdata2.update_output_desc_dst(desc_transdata2);

  std::vector<Operator> inputs{data_filter, data_out_backprop};
  std::vector<Operator> outputs{transdata2};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  platform_info.soc_info.ai_core_cnt = 32;
  opti_compilation_info.soc_version = "Ascend910A";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

  vector<ge::NodePtr> fusion_nodes;
  Status res = fe::RunBufferFusionPass("TbeDxTransDataFusionPass", fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                       *compute_graph_ptr, fusion_nodes);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  bool find_dx = false;
  int count_transdata = 0;
  for (auto node : fusion_nodes) {
    if (node->GetType() == "Conv2DBackpropInput") {
      find_dx = true;
    }
    if (node->GetType() == "TransData") {
      count_transdata++;
    }
  }
  EXPECT_EQ(res, SUCCESS);
  EXPECT_EQ(find_dx, true);
  EXPECT_EQ(count_transdata, 2);
}
