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
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

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
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}

TEST_F(Conv2DTiling, Conv2d_tiling_dynamic_batch_n) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {"10000": 1}, "tiling_range":{"10000":[1,3]},"block_dim": {"10000": 8}, "_vars": {"10000": ["batch_n"]}, "_custom_vars": {"10000": ["batch_n"]}})";

  ge::Graph graph("conv2d_op_tiling_test_8");

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

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_dynamic_batch_n", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 8);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 ");
}

TEST_F(Conv2DTiling, Conv2d_tiling_dynamic_None) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]}, "block_dim": {"10000": 1}, "_vars": {"10000": ["batch_n"]}, "_custom_vars": {"10000": ["batch_n"]}})";

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
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 ");
}

TEST_F(Conv2DTiling, Conv2d_tiling_dynamic_channel) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "fmap_c1": 5, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

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
  RUN_TILING_V4_FALSE(conv2d, iter->second, compileInfo, runInfo);
}

TEST_F(Conv2DTiling, Conv2d_tiling_dynamic_nhw_repo) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {"10000":[1, 10, 10]}, "repo_range": {"10000": [1, 10, 10, 25, 10, 25]}, "cost_range": {}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("conv2d_op_tiling_test_9");

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

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_dynamic_nhw_repo", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}
// fuzz build compile list input
TEST_F(Conv2DTiling, Conv2d_tiling_fuzz_build_list_input) {
  // new ut ops
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"([{"_pattern": "Convolution", "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [16, 32, 16, 32, 16, 32]}, "block_dim": {"10000": 16}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}},{"_pattern": "Convolution", "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10001": [16, 32, 64, 128, 64, 128]}, "block_dim": {"10001": 16}, "_vars": {"10001": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10001": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}}])";

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
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 16);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 16 14 16 12 ");
}

TEST_F(Conv2DTiling, Conv2d_tiling_dynamic_nhwc) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("conv2d_op_tiling_test_4");

  auto x_shape = vector<int64_t>({1, 16, 16, 32});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NHWC, DT_FLOAT16);
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

  auto y_shape = vector<int64_t>({1, 16, 16, 64});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_dynamic_nhwc", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}

TEST_F(Conv2DTiling, Conv2d_vadd_fusion_tiling_dynamic_nhwc) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["_dim_0_0", "_dim_2_0", "batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("conv2d_op_tiling_test_5");

  auto x_shape = vector<int64_t>({1, 16, 16, 32});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NHWC, DT_FLOAT16);
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

  auto y_shape = vector<int64_t>({1, 16, 16, 64});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_vadd_fusion_tiling_dynamic_nhwc", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 256 1 16 16 16 16 ");
}

TEST_F(Conv2DTiling, Conv2d_vadd_fusion_tiling_dynamic_nhwc_invalid) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["dim", "batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("conv2d_op_tiling_test_5");

  auto x_shape = vector<int64_t>({1, 16, 16, 32});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NHWC, DT_FLOAT16);
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

  auto y_shape = vector<int64_t>({1, 16, 16, 64});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_vadd_fusion_tiling_dynamic_nhwc_invalid", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "");
}

TEST_F(Conv2DTiling, Conv2d_vadd_fusion_tiling_dynamic_nhwc_invalid1) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["xyz", "batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("conv2d_op_tiling_test_5");

  auto x_shape = vector<int64_t>({1, 16, 16, 32});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NHWC, DT_FLOAT16);
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

  auto y_shape = vector<int64_t>({1, 16, 16, 64});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_vadd_fusion_tiling_dynamic_nhwc_invalid1", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
  }


TEST_F(Conv2DTiling, Conv2d_tiling_binary_case0) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("conv2d_op_tiling_test_0");

  auto x_shape = vector<int64_t>({1, 2, 16, 16, 16});
  auto x_ori_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NC1HWC0, DT_FLOAT16);
  desc_x.SetOriginShape(ge::Shape(x_ori_shape));
  desc_x.SetOriginFormat(FORMAT_NCHW);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({18, 4, 16, 16});
  auto filter_ori_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_FRACTAL_Z, DT_FLOAT16);
  desc_filter.SetOriginShape(ge::Shape(filter_ori_shape));
  desc_filter.SetOriginFormat(FORMAT_NCHW);
  auto filter = op::Data("filter").set_attr_index(1);
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto conv2d = op::Conv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter)
      .set_attr_strides({1, 1, 1, 1})
      .set_attr_pads({1, 1, 1, 1})
      .set_attr_dilations({1, 1, 1, 1})
      .set_attr_groups(1)
      .set_attr_data_format("NCHW");

  auto y_shape = vector<int64_t>({1, 4, 16, 16, 16});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_binary_case0", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 8);
  EXPECT_EQ(runInfo.GetTilingKey(), 32857);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 1 1 16 16 16 16 2 4 3 3 1 1 1 1 0 0 1 4 2 1 1 1 2 18 20 1 18 1 1 18 ");
}

TEST_F(Conv2DTiling, Conv2d_tiling_binary_case1) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("Conv2d_tiling_binary_case1");

  auto x_shape = vector<int64_t>({1, 2, 56, 56, 16});
  auto x_ori_shape = vector<int64_t>({1, 32, 56, 56});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NC1HWC0, DT_FLOAT16);
  desc_x.SetOriginShape(ge::Shape(x_ori_shape));
  desc_x.SetOriginFormat(FORMAT_NCHW);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({2, 4, 16, 16});
  auto filter_ori_shape = vector<int64_t>({64, 32, 1, 1});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_FRACTAL_Z, DT_FLOAT16);
  desc_filter.SetOriginShape(ge::Shape(filter_ori_shape));
  desc_filter.SetOriginFormat(FORMAT_NCHW);
  auto filter = op::Data("filter").set_attr_index(1);
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto conv2d = op::Conv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter)
      .set_attr_strides({1, 1, 1, 1})
      .set_attr_pads({0, 0, 0, 0})
      .set_attr_dilations({1, 1, 1, 1})
      .set_attr_groups(1)
      .set_attr_data_format("NCHW");

  auto y_shape = vector<int64_t>({1, 4, 56, 56, 16});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_binary_case1", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 16);
  EXPECT_EQ(runInfo.GetTilingKey(), 65);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 1 1 56 56 56 56 2 4 1 1 0 0 0 0 0 0 1 1 16 1 4 1 8 2 1 1 2 1 1 2 ");
}

TEST_F(Conv2DTiling, Conv2d_tiling_binary_case_cin_lessthan_16) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("Conv2d_tiling_binary_case_cin_lessthan_16");

  auto x_shape = vector<int64_t>({1, 1, 56, 56, 16});
  auto x_ori_shape = vector<int64_t>({1, 15, 56, 56});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NC1HWC0, DT_FLOAT16);
  desc_x.SetOriginShape(ge::Shape(x_ori_shape));
  desc_x.SetOriginFormat(FORMAT_NCHW);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({1, 4, 16, 16});
  auto filter_ori_shape = vector<int64_t>({64, 15, 1, 1});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_FRACTAL_Z, DT_FLOAT16);
  desc_filter.SetOriginShape(ge::Shape(filter_ori_shape));
  desc_filter.SetOriginFormat(FORMAT_NCHW);
  auto filter = op::Data("filter").set_attr_index(1);
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto conv2d = op::Conv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter)
      .set_attr_strides({1, 1, 1, 1})
      .set_attr_pads({0, 0, 0, 0})
      .set_attr_dilations({1, 1, 1, 1})
      .set_attr_groups(1)
      .set_attr_data_format("NCHW");

  auto y_shape = vector<int64_t>({1, 4, 56, 56, 16});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_binary_case_cin_lessthan_16", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 25);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 1 1 56 56 56 56 1 4 1 1 0 0 0 0 0 0 1 1 32 1 4 1 4 1 1 1 1 1 1 1 ");
}


TEST_F(Conv2DTiling, Conv2d_tiling_binary_case_stride_2) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("Conv2d_tiling_binary_case_stride_2");

  auto x_shape = vector<int64_t>({1, 2, 56, 56, 16});
  auto x_ori_shape = vector<int64_t>({1, 32, 56, 56});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NC1HWC0, DT_FLOAT16);
  desc_x.SetOriginShape(ge::Shape(x_ori_shape));
  desc_x.SetOriginFormat(FORMAT_NCHW);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({2, 4, 16, 16});
  auto filter_ori_shape = vector<int64_t>({64, 32, 1, 1});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_FRACTAL_Z, DT_FLOAT16);
  desc_filter.SetOriginShape(ge::Shape(filter_ori_shape));
  desc_filter.SetOriginFormat(FORMAT_NCHW);
  auto filter = op::Data("filter").set_attr_index(1);
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto conv2d = op::Conv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter)
      .set_attr_strides({1, 1, 2, 2})
      .set_attr_pads({0, 0, 0, 0})
      .set_attr_dilations({1, 1, 1, 1})
      .set_attr_groups(1)
      .set_attr_data_format("NCHW");

  auto y_shape = vector<int64_t>({1, 4, 28, 28, 16});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_binary_case_stride_2", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 16);
  EXPECT_EQ(runInfo.GetTilingKey(), 32833);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 2 2 1 56 28 56 28 2 4 1 1 0 0 0 0 0 0 1 1 16 1 4 1 4 2 1 1 2 1 1 2 ");
}

TEST_F(Conv2DTiling, Conv2d_tiling_binary_case_dilation_2) {
  using namespace optiling;
  std::string op_name = "Conv2D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("Conv2d_tiling_binary_case_dilation_2");

  auto x_shape = vector<int64_t>({1, 2, 56, 56, 16});
  auto x_ori_shape = vector<int64_t>({1, 32, 56, 56});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NC1HWC0, DT_FLOAT16);
  desc_x.SetOriginShape(ge::Shape(x_ori_shape));
  desc_x.SetOriginFormat(FORMAT_NCHW);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_shape = vector<int64_t>({18, 4, 16, 16});
  auto filter_ori_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_FRACTAL_Z, DT_FLOAT16);
  desc_filter.SetOriginShape(ge::Shape(filter_ori_shape));
  desc_filter.SetOriginFormat(FORMAT_NCHW);
  auto filter = op::Data("filter").set_attr_index(1);
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto conv2d = op::Conv2D(op_name)
      .set_input_x(x)
      .set_input_filter(filter)
      .set_attr_strides({1, 1, 1, 1})
      .set_attr_pads({0, 0, 0, 0})
      .set_attr_dilations({1, 1, 2, 2})
      .set_attr_groups(1)
      .set_attr_data_format("NCHW");

  auto y_shape = vector<int64_t>({1, 4, 52, 52, 16});
  ge::TensorDesc desc_output(ge::Shape(y_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  conv2d.update_input_desc_x(desc_x);
  conv2d.update_input_desc_filter(desc_filter);
  conv2d.update_output_desc_y(desc_output);

  std::vector<Operator> inputs{x, filter};
  std::vector<Operator> outputs{conv2d};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_tiling_binary_case_dilation_2", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V4(conv2d, iter->second, compileInfo, runInfo);
  EXPECT_EQ(runInfo.GetBlockDim(), 16);
  EXPECT_EQ(runInfo.GetTilingKey(), 32857);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 2 1 1 1 56 52 56 52 2 4 3 3 0 0 0 0 0 0 1 2 8 1 2 1 2 18 19 1 50 1 1 18 ");
}