#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

#define private public
#include "common/util/platform_info.h"

using namespace fe;

class bnupdate_reluv2_bnreduce_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "bnupdate_reluv2_bnreduce_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "bnupdate_reluv2_bnreduce_fusion_test TearDown" << std::endl;
  }

  /******************************************************
   *
  *  x sum square_sum scale offset mean variance
  *   \   \          \  |  /      /    /
  *                               -----
  *                  bnupdate      /->    share the same
  *               /->                     memory address
  *            ---------
  *     /     /        /   |      \
  *    y  mean variance batch_mean batch_variance
  *     \
  *      reluv2
  *        \    \
  *         |     mask
  *           |        weight bias(if exist)
  *            \      /      /
  *               conv2d
  *                /  \
  *        bureduce    others
  *               |    |
   *
   *******************************************************/
  void BuildGraph(ge::ComputeGraphPtr &compute_graph) {
    ge::Graph graph("test");
    auto x = ge::op::Data("x");
    ge::TensorDesc tensor_desc_x(ge::Shape({256,56,56,128}), ge::FORMAT_NHWC, ge::DT_FLOAT);
    x.update_input_desc_x(tensor_desc_x);
    x.update_output_desc_y(tensor_desc_x);

    auto sum = ge::op::Data("sum");
    ge::TensorDesc tensor_desc_s(ge::Shape({128}), ge::FORMAT_NHWC, ge::DT_FLOAT);
    sum.update_input_desc_x(tensor_desc_s);
    sum.update_output_desc_y(tensor_desc_s);

    auto square_sum = ge::op::Data("square_sum");
    square_sum.update_input_desc_x(tensor_desc_s);
    square_sum.update_output_desc_y(tensor_desc_s);

    auto scale = ge::op::Data("scale");
    scale.update_input_desc_x(tensor_desc_s);
    scale.update_output_desc_y(tensor_desc_s);

    auto offset = ge::op::Data("offset");
    offset.update_input_desc_x(tensor_desc_s);
    offset.update_output_desc_y(tensor_desc_s);

    auto mean = ge::op::Data("mean");
    mean.update_input_desc_x(tensor_desc_s);
    mean.update_output_desc_y(tensor_desc_s);

    auto variance = ge::op::Data("variance");
    variance.update_input_desc_x(tensor_desc_s);
    variance.update_output_desc_y(tensor_desc_s);

    auto bn1 = ge::op::BNTrainingUpdate("bn1");
    bn1.set_input_x(x)
        .set_input_sum(sum)
        .set_input_square_sum(square_sum)
        .set_input_scale(scale)
        .set_input_offset(offset)
        .set_input_mean(mean)
        .set_input_variance(variance)
        .set_attr_factor(0.1)
        .set_attr_epsilon(0.0001);

    bn1.update_output_desc_y(tensor_desc_x);
    bn1.update_output_desc_mean(tensor_desc_s);
    bn1.update_output_desc_variance(tensor_desc_s);
    bn1.update_output_desc_batch_mean(tensor_desc_s);
    bn1.update_output_desc_batch_variance(tensor_desc_s);


    ge::TensorDesc tensor_desc_mask(ge::Shape({256,56,56,128}), ge::FORMAT_NHWC, ge::DT_UINT8);
    auto reluv2 = ge::op::ReluV2("reluv2");
    reluv2.set_input_x(bn1, 0);
    reluv2.update_output_desc_y(tensor_desc_x);
    reluv2.update_output_desc_mask(tensor_desc_mask);

    auto filter = ge::op::Data("filter");
    ge::TensorDesc tensor_desc_filter(ge::Shape({3,3,128,128}), ge::FORMAT_HWCN, ge::DT_FLOAT);
    filter.update_input_desc_x(tensor_desc_filter);
    filter.update_output_desc_y(tensor_desc_filter);

    auto conv2d = ge::op::Conv2D("conv2d");
    conv2d.set_input_x(reluv2, 0)
        .set_input_filter(filter)
        .set_attr_strides({1,2,2,1})
        .set_attr_pads({0,1,0,1})
        .set_attr_dilations({1,1,1,1})
        .set_attr_groups(1);
    conv2d.update_output_desc_y(tensor_desc_x);

    auto bn2 = ge::op::BNTrainingReduce("bn2");
    bn2.set_input_x(conv2d);
    bn2.update_output_desc_sum(tensor_desc_s);
    bn2.update_output_desc_square_sum(tensor_desc_s);

    auto relu1 = ge::op::Conv2D("relu1");
    relu1.set_input_x(conv2d);
    relu1.update_output_desc_y(tensor_desc_x);

    auto relu2 = ge::op::Conv2D("relu2");
    relu2.set_input_x(bn2, 0);
    relu2.update_output_desc_y(tensor_desc_x);

    std::vector<ge::Operator> inputs{x, sum, square_sum, scale, offset, mean, variance, filter};
    std::vector<ge::Operator> outputs{relu1, relu2};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};

TEST_F(bnupdate_reluv2_bnreduce_fusion_test, bnupdate_reluv2_bnreduce_fusion_test_1) {
  ge::ComputeGraphPtr compute_graph;
  BuildGraph(compute_graph);
  FusionPassTestUtils::InferShapeAndType(compute_graph);

    // set soc_version
    fe::PlatformInfo platform_info;
    fe::OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 32;
    opti_compilation_info.soc_version = "Ascend910A";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910A"] = platform_info;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);



  FusionPassTestUtils::RunGraphFusionPass("ZBNupdateReluV2Conv2DBNreducePass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
      fe::PlatformInfoManager::Instance().platform_info_map_.clear();

  bool find_fused_node = false;
  size_t input_name_size = 0;
  size_t output_name_size = 0;
  for (auto node: compute_graph->GetAllNodes()) {
      std::cout << "node->GetType(): " <<node->GetType()<< std::endl;
    if (node->GetType() == "FusedBN2ReluConvBN1") {
      find_fused_node = true;
      auto fused_desc = node->GetOpDesc();
      input_name_size = fused_desc->GetAllInputName().size();
      output_name_size = fused_desc->GetAllOutputName().size();
    }
  }
  EXPECT_EQ(find_fused_node, true);
  std::cout << "input_name_size: " <<input_name_size <<" output_name_size: "<<output_name_size<< std::endl;
//   EXPECT_EQ(input_name_size, 9);
//   EXPECT_EQ(output_name_size, 8);
}
