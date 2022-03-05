#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#define private public
#include "register/op_tiling_registry.h"

using namespace std;
using namespace ge;
using namespace op;

class AvgPoolTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPoolTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPoolTiling TearDown" << std::endl;
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

TEST_F(AvgPoolTiling, AvgPool_tiling_dynamic_nhw) {
  using namespace optiling;
  std::string op_name = "AvgPool";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Convolution", "push_status": 0,
  "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {},
  "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2},
  "strides_h" : 60, "strides_w" : 60, "k_size_h": 2, "k_size_w": 2,
  "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("avg_pool_op_tiling_test0");

  auto x_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto avg_pool_op = op::AvgPool(op_name);
  avg_pool_op.set_input_x(x);

  auto output_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(output_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  avg_pool_op.update_input_desc_x(desc_x);
  avg_pool_op.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{avg_pool_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("AvgPool_tiling_dynamic_nhw", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(avg_pool_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}

TEST_F(AvgPoolTiling, AvgPool_tiling_dynamic_None) {
  using namespace optiling;
  std::string op_name = "AvgPool";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  const ge::AscendString compileInfo = R"({"_pattern": "Convolution", "push_status": 0,
  "tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]},
  "block_dim": {"10000": 1}, "strides_h": 60, "strides_w" : 60, "k_size_h": 2, "k_size_w": 2,
  "_vars": {"10000": ["batch_n"]}})";

  ge::Graph graph("avg_pool_op_tiling_test1");

  auto x_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto avg_pool_op = op::AvgPool(op_name);
  avg_pool_op.set_input_x(x);

  auto output_shape = vector<int64_t>({1, 64, 16, 16});
  ge::TensorDesc output_desc_y(ge::Shape(output_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  avg_pool_op.update_input_desc_x(desc_x);
  avg_pool_op.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{avg_pool_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("AvgPool_tiling_dynamic_None", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(avg_pool_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 ");
}

TEST_F(AvgPoolTiling, AvgPool_tiling_dynamic_Vector) {
  using namespace optiling;
  std::string op_name = "AvgPool";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"strides_h" : 64, "strides_w" : 64, "k_size_h": 1, "k_size_w": 1,
  "vars": {"ub_ele": 126976, "core_num": 30, "ksize_h": 1, "ksize_w": 1,
  "strides_h": 64, "strides_w": 64, "padding": 0}})";

  ge::Graph graph("avg_pool_op_tiling_test1");

  auto x_shape = vector<int64_t>({16, 13, 79, 69, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto avg_pool_op = op::AvgPool(op_name);
  avg_pool_op.set_input_x(x);

  auto output_shape = vector<int64_t>({16, 13, 79, 69, 16});
  ge::TensorDesc output_desc_y(ge::Shape(output_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  avg_pool_op.update_input_desc_x(desc_x);
  avg_pool_op.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{avg_pool_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("AvgPool_tiling_dynamic_Vector", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(avg_pool_op, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 30 7 5 79 69 2 2 65 65 0 0 0 0 1 1 1 2 0 2 0 ");
}

// fuzz build compile list input
TEST_F(AvgPoolTiling, AvgPool_tiling_fuzz_build_list_input) {
  using namespace optiling;
  std::string op_name = "AvgPool";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"([{"_pattern": "Convolution",
  "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {},
  "cost_range": {"0": [16, 32, 16, 32, 16, 32]}, "strides_h" : 60,
  "strides_w" : 60, "block_dim": {"0": 16}, "k_size_h": 2, "k_size_w": 2,
  "_vars": {"0": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}},
  {"_pattern": "Convolution", "tiling_type": "dynamic_tiling",
  "repo_seeds": {}, "repo_range": {},
  "cost_range": {"1": [16, 32, 64, 128, 64, 128]}, "block_dim": {"1": 16},
  "_vars": {"1": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}}])";

  ge::Graph graph("avg_pool_op_tiling_test2");

  auto x_shape = vector<int64_t>({16, 3, 16, 16, 16});
  ge::TensorDesc desc_x(ge::Shape(x_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto avg_pool_op = op::AvgPool(op_name);
  avg_pool_op.set_input_x(x);

  auto output_shape = vector<int64_t>({16, 3, 14, 12, 16});
  ge::TensorDesc output_desc_y(ge::Shape(output_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);

  avg_pool_op.update_input_desc_x(desc_x);
  avg_pool_op.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{avg_pool_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  optiling::utils::OpCompileInfo op_compile_info("AvgPool_tiling_fuzz_build_list_input", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(avg_pool_op, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 16 14 16 12 ");  
}