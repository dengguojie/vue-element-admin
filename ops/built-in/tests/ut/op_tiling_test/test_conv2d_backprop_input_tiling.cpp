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

using namespace std;
using namespace ge;
using namespace op;

class Conv2DBackpropInputTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DBackpropInputTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DBackpropInputTiling TearDown" << std::endl;
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

TEST_F(Conv2DBackpropInputTiling, Conv2d_bp_input_tiling_dynamic_nhw) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropInput";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}})";

  ge::Graph graph("conv2dbackprop_input_op_tiling_test_0");

  auto input_size_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_input_size(ge::Shape(input_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(desc_input_size);
  input_size.update_output_desc_y(desc_input_size);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto out_backprop_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropinput = op::Conv2DBackpropInput(op_name)
      .set_input_input_size(input_size)
      .set_input_filter(filter)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropinput.update_input_desc_input_size(desc_input_size);
  conv2dbackpropinput.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropinput.update_input_desc_filter(desc_filter);
  conv2dbackpropinput.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{filter, out_backprop, input_size};
  std::vector<Operator> outputs{conv2dbackpropinput};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_tiling_dynamic_nhw", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(conv2dbackpropinput, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}

TEST_F(Conv2DBackpropInputTiling, Conv2d_bp_input_tiling_compile_info_empty) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropInput";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({})";

  ge::Graph graph("conv2d_bp_input_tiling_compile_info_empty");

  auto input_size_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_input_size(ge::Shape(input_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(desc_input_size);
  input_size.update_output_desc_y(desc_input_size);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto out_backprop_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropinput = op::Conv2DBackpropInput(op_name)
      .set_input_input_size(input_size)
      .set_input_filter(filter)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropinput.update_input_desc_input_size(desc_input_size);
  conv2dbackpropinput.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropinput.update_input_desc_filter(desc_filter);
  conv2dbackpropinput.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{filter, out_backprop, input_size};
  std::vector<Operator> outputs{conv2dbackpropinput};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_input_tiling_compile_info_empty", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(conv2dbackpropinput, op_compile_info, runInfo));
}

TEST_F(Conv2DBackpropInputTiling, Conv2d_bp_input_tiling_compile_info_not_have_vars) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropInput";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}})";

  ge::Graph graph("conv2d_bp_input_tiling_compile_info_not_have_vars");

  auto input_size_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_input_size(ge::Shape(input_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(desc_input_size);
  input_size.update_output_desc_y(desc_input_size);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto out_backprop_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropinput = op::Conv2DBackpropInput(op_name)
      .set_input_input_size(input_size)
      .set_input_filter(filter)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropinput.update_input_desc_input_size(desc_input_size);
  conv2dbackpropinput.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropinput.update_input_desc_filter(desc_filter);
  conv2dbackpropinput.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{filter, out_backprop, input_size};
  std::vector<Operator> outputs{conv2dbackpropinput};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_input_tiling_compile_info_not_have_vars", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(conv2dbackpropinput, op_compile_info, runInfo));
}

TEST_F(Conv2DBackpropInputTiling, Conv2d_bp_input_tiling_no_repo_seeds) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropInput";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}})";

  ge::Graph graph("conv2dbackprop_input_op_tiling_test_1");

  auto input_size_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_input_size(ge::Shape(input_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(desc_input_size);
  input_size.update_output_desc_y(desc_input_size);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto out_backprop_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropinput = op::Conv2DBackpropInput(op_name)
      .set_input_input_size(input_size)
      .set_input_filter(filter)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropinput.update_input_desc_input_size(desc_input_size);
  conv2dbackpropinput.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropinput.update_input_desc_filter(desc_filter);
  conv2dbackpropinput.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{filter, out_backprop, input_size};
  std::vector<Operator> outputs{conv2dbackpropinput};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_input_tiling_no_repo_seeds", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(conv2dbackpropinput, op_compile_info, runInfo));
}

TEST_F(Conv2DBackpropInputTiling, Conv2d_bp_input_dynamic_None) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropInput";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]}, "block_dim": {"10000": 1}, "_vars": {"10000": ["batch_n"]}})";

  ge::Graph graph("conv2dbackprop_input_op_tiling_test_2");

  auto input_size_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_input_size(ge::Shape(input_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(desc_input_size);
  input_size.update_output_desc_y(desc_input_size);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto out_backprop_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropinput = op::Conv2DBackpropInput(op_name)
      .set_input_input_size(input_size)
      .set_input_filter(filter)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropinput.update_input_desc_input_size(desc_input_size);
  conv2dbackpropinput.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropinput.update_input_desc_filter(desc_filter);
  conv2dbackpropinput.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{filter, out_backprop, input_size};
  std::vector<Operator> outputs{conv2dbackpropinput};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_input_dynamic_None", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(conv2dbackpropinput, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 ");
}

// fuzz build compile list input
TEST_F(Conv2DBackpropInputTiling, Conv2d_bp_input_fuzz_build_list_input) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropInput";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"([{"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"0": [1, 10, 10, 25, 10, 25]}, "block_dim": {"0": 2}, "_vars": {"0": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}}, {"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"1": [10, 100, 15, 30, 15, 30]}, "block_dim": {"1": 2}, "_vars": {"1": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}}])";

  ge::Graph graph("conv2dbackprop_input_op_tiling_test_3");

  auto input_size_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_input_size(ge::Shape(input_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(desc_input_size);
  input_size.update_output_desc_y(desc_input_size);

  auto filter_shape = vector<int64_t>({64, 32, 3, 3});
  ge::TensorDesc desc_filter(ge::Shape(filter_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(desc_filter);
  filter.update_output_desc_y(desc_filter);

  auto out_backprop_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropinput = op::Conv2DBackpropInput(op_name)
      .set_input_input_size(input_size)
      .set_input_filter(filter)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropinput.update_input_desc_input_size(desc_input_size);
  conv2dbackpropinput.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropinput.update_input_desc_filter(desc_filter);
  conv2dbackpropinput.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{filter, out_backprop, input_size};
  std::vector<Operator> outputs{conv2dbackpropinput};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_input_fuzz_build_list_input", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(conv2dbackpropinput, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 0);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}