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
#include "register/op_tiling_registry.h"

using namespace std;
using namespace ge;
using namespace op;

class AvgPoolV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPoolV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPoolV2Tiling TearDown" << std::endl;
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

TEST_F(AvgPoolV2Tiling, AvgPoolV2_tiling_dynamic_nhw) {
  using namespace optiling;
  std::string op_name = "AvgPoolV2";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "strides_h" : 60, "strides_w" : 60, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("avg_pool_v2_op_tiling_test0");

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

  optiling::utils::OpCompileInfo op_compile_info("AvgPoolV2_tiling_dynamic_nhw", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(avg_pool_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}

TEST_F(AvgPoolV2Tiling, AvgPoolV2_tiling_dynamic_None) {
  using namespace optiling;
  std::string op_name = "AvgPoolV2";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  
  const ge::AscendString compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]}, "block_dim": {"10000": 1}, "strides_h" : 60, "strides_w" : 60, "_vars": {"10000": ["batch_n"]}})";

  ge::Graph graph("avg_pool_v2_op_tiling_test1");

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

  optiling::utils::OpCompileInfo op_compile_info("AvgPoolV2_tiling_dynamic_None", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(avg_pool_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 ");
}

TEST_F(AvgPoolV2Tiling, AvgPoolV2_tiling_dynamic_nhwc) {
  using namespace optiling;
  std::string op_name = "AvgPoolV2";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "strides_h" : 60, "strides_w" : 60, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  ge::Graph graph("avg_pool_v2_op_tiling_test2");

  auto x_shape = vector<int64_t>({1, 16, 16, 32});
  ge::TensorDesc desc_x(ge::Shape(x_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto avg_pool_op = op::AvgPool(op_name);
  avg_pool_op.set_input_x(x);

  auto output_shape = vector<int64_t>({1, 16, 16, 64});
  ge::TensorDesc output_desc_y(ge::Shape(output_shape), ge::FORMAT_NHWC, ge::DT_FLOAT16);

  avg_pool_op.update_input_desc_x(desc_x);
  avg_pool_op.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x};
  std::vector<Operator> outputs{avg_pool_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("AvgPoolV2_tiling_dynamic_nhwc", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(avg_pool_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}