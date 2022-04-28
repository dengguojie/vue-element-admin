#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "array_ops.h"
#include "nn_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;
using namespace op;

class DepthwiseConv2DTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DepthwiseConv2DTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DepthwiseConv2DTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    tmp = *(int32_t *)(data.c_str() + i);
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

TEST_F(DepthwiseConv2DTiling, DepthwiseConv2d_tiling_dynamic_nhw) {
  using namespace optiling;
  std::string op_name = "DepthwiseConv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"_pattern": "DepthwiseConvolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )";
  compileInfo += R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )";
  compileInfo += R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )";
  compileInfo += R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )";
  compileInfo += R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )";
  compileInfo += R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )";
  compileInfo += R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})";

  ge::Graph graph("depthwiseconv2d_op_tiling_test_0");

  auto x_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter").set_attr_index(1);
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto depthwiseconv2d = op::DepthwiseConv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  depthwiseconv2d.update_input_desc_x(desc_x);
  depthwiseconv2d.update_input_desc_filter(desc_filter);
  depthwiseconv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{depthwiseconv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("DepthwiseConv2d_tiling_dynamic_nhw", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(depthwiseconv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}

TEST_F(DepthwiseConv2DTiling, DepthwiseConv2d_tiling_dynamic_None) {
  using namespace optiling;
  std::string op_name = "DepthwiseConv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"_pattern": "DepthwiseConvolution", "push_status": 0, "tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]}, "block_dim": {"10000": 1}, "_vars": {"10000": ["batch_n"]}, "_custom_vars": {"10000": ["batch_n"]}, )";
  compileInfo += R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )";
  compileInfo += R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )";
  compileInfo += R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )";
  compileInfo += R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )";
  compileInfo += R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )";
  compileInfo += R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})";

  ge::Graph graph("depthwiseconv2d_op_tiling_test_1");

  auto x_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter").set_attr_index(1);
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto depthwiseconv2d = op::DepthwiseConv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  depthwiseconv2d.update_input_desc_x(desc_x);
  depthwiseconv2d.update_input_desc_filter(desc_filter);
  depthwiseconv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{depthwiseconv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("DepthwiseConv2d_tiling_dynamic_None", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(depthwiseconv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 ");
}