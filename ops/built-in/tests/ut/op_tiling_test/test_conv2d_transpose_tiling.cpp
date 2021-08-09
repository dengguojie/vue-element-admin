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

class Conv2DTransposeTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DTransposeTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DTransposeTiling TearDown" << std::endl;
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

TEST_F(Conv2DTransposeTiling, Conv2d_transpose_tiling_dynamic_nhw) {
  using namespace optiling;
  std::string op_name = "Conv2DTranspose";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}})";

  ge::Graph graph("conv2dtranspose_op_tiling_test_0");

  auto input_size_shape = vector<int64_t>({4});
  ge::TensorDesc desc_input_size(ge::Shape(input_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(desc_input_size);
  input_size.update_output_desc_y(desc_input_size);

  auto x_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto conv2dtranspose = op::Conv2DTranspose(op_name)
      .set_input_input_size(input_size)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dtranspose.update_input_desc_input_size(desc_input_size);
  conv2dtranspose.update_input_desc_x(desc_x);
  conv2dtranspose.update_input_desc_filter(desc_x);
  conv2dtranspose.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, filter, input_size};
  std::vector<Operator> outputs{conv2dtranspose};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_transpose_tiling_dynamic_nhw", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(conv2dtranspose, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}

TEST_F(Conv2DTransposeTiling, Conv2d_transpose_compile_info_empty) {
  using namespace optiling;
  std::string op_name = "Conv2DTranspose";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({})";

  ge::Graph graph("conv2d_transpose_compile_info_empty");

  auto input_size_shape = vector<int64_t>({4});
  ge::TensorDesc desc_input_size(ge::Shape(input_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(desc_input_size);
  input_size.update_output_desc_y(desc_input_size);

  auto x_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto conv2dtranspose = op::Conv2DTranspose(op_name)
      .set_input_input_size(input_size)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dtranspose.update_input_desc_input_size(desc_input_size);
  conv2dtranspose.update_input_desc_x(desc_x);
  conv2dtranspose.update_input_desc_filter(desc_x);
  conv2dtranspose.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, filter, input_size};
  std::vector<Operator> outputs{conv2dtranspose};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_transpose_compile_info_empty", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(conv2dtranspose, op_compile_info, runInfo));
}

TEST_F(Conv2DTransposeTiling, Conv2d_transpose_compile_info_not_have_vars) {
  using namespace optiling;
  std::string op_name = "Conv2DTranspose";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}})";

  ge::Graph graph("conv2d_transpose_compile_info_not_have_vars");

  auto input_size_shape = vector<int64_t>({4});
  ge::TensorDesc desc_input_size(ge::Shape(input_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(desc_input_size);
  input_size.update_output_desc_y(desc_input_size);

  auto x_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto conv2dtranspose = op::Conv2DTranspose(op_name)
      .set_input_input_size(input_size)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dtranspose.update_input_desc_input_size(desc_input_size);
  conv2dtranspose.update_input_desc_x(desc_x);
  conv2dtranspose.update_input_desc_filter(desc_x);
  conv2dtranspose.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, filter, input_size};
  std::vector<Operator> outputs{conv2dtranspose};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_transpose_compile_info_not_have_vars", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(conv2dtranspose, op_compile_info, runInfo));
}

// fuzz build compile list input
TEST_F(Conv2DTransposeTiling, Conv2d_transpose_tiling_fuzz_build_list_input) {
  using namespace optiling;
  std::string op_name = "Conv2DTranspose";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"([{"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"0": [1, 10, 10, 25, 10, 25]}, "block_dim": {"0": 2}, "_vars": {"0": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}}, {"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"1": [10, 20, 15, 35, 14, 35]}, "block_dim": {"1": 2}, "_vars": {"1": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}}])";
  ge::Graph graph("conv2dtranspose_op_tiling_test_0");

  auto input_size_shape = vector<int64_t>({4});
  ge::TensorDesc desc_input_size(ge::Shape(input_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(desc_input_size);
  input_size.update_output_desc_y(desc_input_size);

  auto x_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto conv2dtranspose = op::Conv2DTranspose(op_name)
      .set_input_input_size(input_size)
      .set_input_x(x)
      .set_input_filter(filter);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dtranspose.update_input_desc_input_size(desc_input_size);
  conv2dtranspose.update_input_desc_x(desc_x);
  conv2dtranspose.update_input_desc_filter(desc_x);
  conv2dtranspose.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, filter, input_size};
  std::vector<Operator> outputs{conv2dtranspose};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_transpose_tiling_fuzz_build_list_input", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(conv2dtranspose, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 0);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}