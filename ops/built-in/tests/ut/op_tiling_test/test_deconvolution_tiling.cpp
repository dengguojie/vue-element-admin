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
#include "register/op_tiling_registry.h"

using namespace std;
using namespace ge;
using namespace op;

class DeConvlutionTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DeConvlutionTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DeConvlutionTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

TEST_F(DeConvlutionTiling, DeConvlution_tiling_dynamic_hw) {
  using namespace optiling;
  std::string op_name = "Deconvolution";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}})";
  
  ge::Graph graph("deconvolution_op_tiling_test_0");

  auto x_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x").set_attr_index(1);
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto deconvolution = op::Deconvolution(op_name)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  deconvolution.update_input_desc_x(desc_x);
  deconvolution.update_input_desc_filter(desc_filter);
  deconvolution.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{deconvolution};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("DeConvlution_tiling_dynamic_hw", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(deconvolution, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}


TEST_F(DeConvlutionTiling, DeConvlution_tiling_dynamic_n) {
  using namespace optiling;
  std::string op_name = "Deconvolution";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  
  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "tiling_range": {"10000":[1,4]}, "block_dim": {"10000": 2}, "correct_range_flag": false, "_vars": {"10000": ["batch_n"]}})";

  ge::Graph graph("deconvolution_op_tiling_test_1");

  auto x_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x").set_attr_index(1);
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto deconvolution = op::Deconvolution(op_name)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  deconvolution.update_input_desc_x(desc_x);
  deconvolution.update_input_desc_filter(desc_filter);
  deconvolution.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{deconvolution};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("DeConvlution_tiling_dynamic_n", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(deconvolution, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 ");
}

// fuzz build compile list input
TEST_F(DeConvlutionTiling, DeConvlution_tiling_fuzz_build_list_input) {
  using namespace optiling;
  std::string op_name = "Deconvolution";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"([{"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"0": [1, 10, 10, 25, 10, 25]}, "block_dim": {"0": 2}, "_vars": {"0": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}}, {"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"1": [10, 20, 10, 25, 10, 25]}, "block_dim": {"1": 2}, "_vars": {"1": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}}])";
  
  ge::Graph graph("deconvolution_op_tiling_test_2");

  auto x_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x").set_attr_index(1);
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto deconvolution = op::Deconvolution(op_name)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  deconvolution.update_input_desc_x(desc_x);
  deconvolution.update_input_desc_filter(desc_filter);
  deconvolution.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{deconvolution};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("DeConvlution_tiling_fuzz_build_list_input", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(deconvolution, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 0);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}