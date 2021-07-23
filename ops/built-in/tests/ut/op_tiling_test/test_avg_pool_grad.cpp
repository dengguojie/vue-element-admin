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


class AvgPoolGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPoolGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPoolGradTiling TearDown" << std::endl;
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
//mode dynamic_nhw
TEST_F(AvgPoolGradTiling, Avg_pool_grad_tiling_dynamic_nhw) {
  using namespace optiling;
  std::string op_name = "AvgPoolGrad";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}})";

  ge::Graph graph("avg_pool_grad_op_tiling_test0");

  auto orig_input_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_orig_input(ge::Shape(orig_input_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto orig_input = op::Data("orig_input");
  orig_input.update_input_desc_x(desc_orig_input);
  orig_input.update_output_desc_y(desc_orig_input);

  auto input_grad_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_input_grad(ge::Shape(input_grad_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto input_grad = op::Data("input_grad");
  input_grad.update_input_desc_x(desc_input_grad);
  input_grad.update_output_desc_y(desc_input_grad);

  auto avg_pool_grad_op = op::AvgPoolGrad(op_name);
  avg_pool_grad_op.set_input_orig_input_shape(orig_input);
  avg_pool_grad_op.set_input_input_grad(input_grad);

  auto out_grad_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_out_grad(ge::Shape(out_grad_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  avg_pool_grad_op.update_input_desc_orig_input_shape(desc_orig_input);
  avg_pool_grad_op.update_input_desc_input_grad(desc_input_grad);
  avg_pool_grad_op.update_output_desc_out_grad(output_desc_out_grad);

  std::vector<Operator> inputs{orig_input, input_grad};
  std::vector<Operator> outputs{avg_pool_grad_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Avg_pool_grad_tiling_dynamic_nhw", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(avg_pool_grad_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 16 16 16 16 ");
}
//mode dynamic_n
TEST_F(AvgPoolGradTiling, Avg_pool_grad_dynamic_n) {
  using namespace optiling;
  std::string op_name = "AvgPoolGrad";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  
  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]}, "block_dim": {"10000": 1}, "_vars": {"10000": ["batch_n"]}})";

  ge::Graph graph("avg_pool_grad_op_tiling_test1");

  auto orig_input_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_orig_input(ge::Shape(orig_input_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto orig_input = op::Data("orig_input");
  orig_input.update_input_desc_x(desc_orig_input);
  orig_input.update_output_desc_y(desc_orig_input);

  auto input_grad_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc desc_input_grad(ge::Shape(input_grad_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto input_grad = op::Data("input_grad");
  input_grad.update_input_desc_x(desc_input_grad);
  input_grad.update_output_desc_y(desc_input_grad);

  auto avg_pool_grad_op = op::AvgPoolGrad(op_name);
  avg_pool_grad_op.set_input_orig_input_shape(orig_input);
  avg_pool_grad_op.set_input_input_grad(input_grad);

  auto out_grad_shape = vector<int64_t>({1, 32, 16, 16});
  ge::TensorDesc output_desc_out_grad(ge::Shape(out_grad_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  avg_pool_grad_op.update_input_desc_orig_input_shape(desc_orig_input);
  avg_pool_grad_op.update_input_desc_input_grad(desc_input_grad);
  avg_pool_grad_op.update_output_desc_out_grad(output_desc_out_grad);

  std::vector<Operator> inputs{orig_input, input_grad};
  std::vector<Operator> outputs{avg_pool_grad_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Avg_pool_grad_dynamic_None", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(avg_pool_grad_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 ");
}

// fuzzy compile single input 
TEST_F(AvgPoolGradTiling, Avg_pool_grad_fuzzy_compile_single_input) {
  using namespace optiling;
  std::string op_name = "AvgPoolGrad";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_input", "_vars": {"0": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}, "block_dim": {"0": 1}, "correct_range_flag": false, "cost_range": {"0": [1, 1, 1, 9, 61, 124]}, "kernelId": 0, "repo_range": {}, "repo_seeds": {}, "tiling_type": "dynamic_tiling"})";

  ge::Graph graph("Avg_pool_grad_fuzzy_compile_single_input");

  auto orig_input_shape = vector<int64_t>({1, 5, 4, 66});
  ge::TensorDesc desc_orig_input(ge::Shape(orig_input_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto orig_input = op::Data("orig_input");
  orig_input.update_input_desc_x(desc_orig_input);
  orig_input.update_output_desc_y(desc_orig_input);

  auto input_grad_shape = vector<int64_t>({1, 5, 2, 17});
  ge::TensorDesc desc_input_grad(ge::Shape(input_grad_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto input_grad = op::Data("input_grad");
  input_grad.update_input_desc_x(desc_input_grad);
  input_grad.update_output_desc_y(desc_input_grad);

  auto avg_pool_grad_op = op::AvgPoolGrad(op_name);
  avg_pool_grad_op.set_input_orig_input_shape(orig_input);
  avg_pool_grad_op.set_input_input_grad(input_grad);

  auto out_grad_shape = vector<int64_t>({1, 5, 4, 66});
  ge::TensorDesc output_desc_out_grad(ge::Shape(out_grad_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  avg_pool_grad_op.update_input_desc_orig_input_shape(desc_orig_input);
  avg_pool_grad_op.update_input_desc_input_grad(desc_input_grad);
  avg_pool_grad_op.update_output_desc_out_grad(output_desc_out_grad);

  std::vector<Operator> inputs{orig_input, input_grad};
  std::vector<Operator> outputs{avg_pool_grad_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Avg_pool_grad_fuzzy_compile_single_input", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(avg_pool_grad_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 0);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 2 4 17 66 ");
}

// fuzzy compile list input 
TEST_F(AvgPoolGradTiling, Avg_pool_grad_fuzzy_compile_list_input) {
  using namespace optiling;
  std::string op_name = "AvgPoolGrad";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"([{"_pattern": "Conv2d_backprop_input", "_vars": {"0": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}, "block_dim": {"0": 1}, "correct_range_flag": false, "cost_range": {"0": [1, 1, 1, 9, 61, 124]}, "kernelId": 0, "repo_range": {}, "repo_seeds": {}, "tiling_type": "dynamic_tiling"}, {"_pattern": "Conv2d_backprop_input", "_vars": {"0": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}, "block_dim": {"0": 1}, "correct_range_flag": false, "cost_range": {"0": [1, 1, 1, 9, 61, 124]}, "kernelId": 0, "repo_range": {}, "repo_seeds": {}, "tiling_type": "dynamic_tiling"}])";

  ge::Graph graph("Avg_pool_grad_fuzzy_compile_list_input");

  auto orig_input_shape = vector<int64_t>({1, 5, 4, 66});
  ge::TensorDesc desc_orig_input(ge::Shape(orig_input_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto orig_input = op::Data("orig_input");
  orig_input.update_input_desc_x(desc_orig_input);
  orig_input.update_output_desc_y(desc_orig_input);

  auto input_grad_shape = vector<int64_t>({1, 5, 2, 17});
  ge::TensorDesc desc_input_grad(ge::Shape(input_grad_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  auto input_grad = op::Data("input_grad");
  input_grad.update_input_desc_x(desc_input_grad);
  input_grad.update_output_desc_y(desc_input_grad);

  auto avg_pool_grad_op = op::AvgPoolGrad(op_name);
  avg_pool_grad_op.set_input_orig_input_shape(orig_input);
  avg_pool_grad_op.set_input_input_grad(input_grad);

  auto out_grad_shape = vector<int64_t>({1, 5, 4, 66});
  ge::TensorDesc output_desc_out_grad(ge::Shape(out_grad_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  avg_pool_grad_op.update_input_desc_orig_input_shape(desc_orig_input);
  avg_pool_grad_op.update_input_desc_input_grad(desc_input_grad);
  avg_pool_grad_op.update_output_desc_out_grad(output_desc_out_grad);

  std::vector<Operator> inputs{orig_input, input_grad};
  std::vector<Operator> outputs{avg_pool_grad_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Avg_pool_grad_fuzzy_compile_list_input", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(avg_pool_grad_op, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(runInfo.GetTilingKey(), 0);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 2 4 17 66 ");
}
