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

class Conv2DTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DTiling TearDown" << std::endl;
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

TEST_F(Conv2DTiling, Conv2d_tiling_dynamic_nhw) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("conv2d_op_tiling_test_0");

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

  auto conv2d = op::Conv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_dynamic_nhw", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(conv2d, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}

TEST_F(Conv2DTiling, Conv2d_tiling_dynamic_None) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]}, "block_dim": {"10000": 1}, "_vars": {"10000": ["batch_n"]}})";

  ge::Graph graph("conv2d_op_tiling_test_1");

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

  auto conv2d = op::Conv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_dynamic_None", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(conv2d, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 ");
}

TEST_F(Conv2DTiling, Conv2d_tiling_dynamic_channel) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "fmap_c1": 5, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("conv2d_op_tiling_test_2");

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

  auto conv2d = op::Conv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_dynamic_channel", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(conv2d, op_compile_info, runInfo));
}

// fuzz build compile list input
TEST_F(Conv2DTiling, Conv2d_tiling_fuzz_build_list_input) {
  // new ut ops
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"([{"_pattern": "Convolution", "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"0": [16, 32, 16, 32, 16, 32]}, "block_dim": {"0": 16}, "_vars": {"0": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}},{"_pattern": "Convolution", "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"1": [16, 32, 64, 128, 64, 128]}, "block_dim": {"1": 16}, "_vars": {"1": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}}])";

  ge::Graph graph("conv2d_op_tiling_test_3");

  auto x_shape = vector<int64_t>({16, 3, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({33, 3, 3, 5});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter").set_attr_index(1);
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto conv2d = op::Conv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({16, 33, 14, 12});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_fuzz_build_list_input", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(conv2d, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 16);
  EXPECT_EQ(runInfo.GetTilingKey(), 0);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 16 14 16 12 ");
}