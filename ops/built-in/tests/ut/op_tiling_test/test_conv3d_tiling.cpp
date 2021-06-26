#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include "array_ops.h"
#include "nn_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

using namespace std;
using namespace ge;
using namespace op;

class Conv3DTiling : public testing::Test
{
 protected:
  static void SetUpTestCase()
  {
    std::cout << "Conv3DTiling SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "Conv3DTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data)
{
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

TEST_F(Conv3DTiling, Conv3d_tiling_dynamic_w)
{
  using namespace optiling;
  std::string op_name = "Conv3D";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = "{\"_pattern\": \"conv3d\", \"push_status\": 0, \"tiling_type\": \"dynamic_tiling\", \"repo_seeds\": {\"10000\": [32, 16, 56, 56]}, \"repo_range\": {\"10000\": [32, 32, 16, 16, 56, 56, 24, 456]}, \"cost_range\": {}, \"block_dim\": {\"10000\": 32}, \"_vars\": {\"10000\": [\"fmap_w\", \"w_out\"]}}";

  ge::Graph graph("Conv3d_tiling_dynamic_w");

  auto x_shape = std::vector<int64_t>({32, 16, 1, 56, 56, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_shape = std::vector<int64_t>({27, 2, 16, 16, 1, 49});
  ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_FRACTAL_Z_3D, ge::DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(filter_desc);
  filter.update_output_desc_y(filter_desc);

  auto conv3d = op::Conv3D(op_name.c_str())
    .set_input_x(x_size)
    .set_input_filter(filter);

  auto y_shape = std::vector<int64_t>({32, 15, 1, 56, 56, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  conv3d.update_input_desc_x(x_desc);
  conv3d.update_input_desc_filter(filter_desc);
  conv3d.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{x_size, filter};
  std::vector<ge::Operator> outputs{conv3d};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_tiling_dynamic_w", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(conv3d, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "56 56 ");
}

TEST_F(Conv3DTiling, Conv3d_tiling_dynamic_batch)
{
  using namespace optiling;
  std::string op_name = "Conv3D";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = "{\"_pattern\": \"conv3d\", \"push_status\": 0, \"tiling_type\": \"dynamic_tiling\", \"repo_seeds\": {\"10000\": 32}, \"tiling_range\": {\"10000\": [1, 35]}, \"block_dim\": {\"10000\": 32}, \"_vars\": {\"10000\": [\"batch_n\"]}}";

  ge::Graph graph("Conv3d_tiling_dynamic_batch");

  auto x_shape = std::vector<int64_t>({32, 16, 1, 56, 56, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_shape = std::vector<int64_t>({27, 2, 16, 16, 1, 49});
  ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_FRACTAL_Z_3D, ge::DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(filter_desc);
  filter.update_output_desc_y(filter_desc);

  auto conv3d = op::Conv3D(op_name.c_str())
    .set_input_x(x_size)
    .set_input_filter(filter);

  auto y_shape = std::vector<int64_t>({32, 15, 1, 56, 56, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  conv3d.update_input_desc_x(x_desc);
  conv3d.update_input_desc_filter(filter_desc);
  conv3d.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{x_size, filter};
  std::vector<ge::Operator> outputs{conv3d};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_tiling_dynamic_batch", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(conv3d, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 ");
}

TEST_F(Conv3DTiling, Conv3d_tiling_dynamic_batch_invalid_C)
{
  using namespace optiling;
  std::string op_name = "Conv3D";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = "{\"_pattern\": \"conv3d\", \"push_status\": 0, \"fmap_c1\": 233, \"tiling_type\": \"dynamic_tiling\", \"repo_seeds\": {\"10000\": 32}, \"tiling_range\": {\"10000\": [1, 35]}, \"block_dim\": {\"10000\": 32}, \"_vars\": {\"10000\": [\"batch_n\"]}}";

  ge::Graph graph("Conv3d_tiling_dynamic_batch_invalid_C");

  auto x_shape = std::vector<int64_t>({32, 16, 1, 56, 56, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_shape = std::vector<int64_t>({27, 2, 16, 16, 1, 49});
  ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_FRACTAL_Z_3D, ge::DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(filter_desc);
  filter.update_output_desc_y(filter_desc);

  auto conv3d = op::Conv3D(op_name.c_str())
    .set_input_x(x_size)
    .set_input_filter(filter);

  auto y_shape = std::vector<int64_t>({32, 15, 1, 56, 56, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  conv3d.update_input_desc_x(x_desc);
  conv3d.update_input_desc_filter(filter_desc);
  conv3d.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{x_size, filter};
  std::vector<ge::Operator> outputs{conv3d};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_tiling_dynamic_batch_invalid_C", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_FALSE(iter->second(conv3d, op_compile_info, runInfo));
}

// fuzz build compile list input
TEST_F(Conv3DTiling, Conv3d_tiling_fuzzy_build)
{
  using namespace optiling;
  std::string op_name = "Conv3D";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "conv3d","_vars": {"0": ["batch_n","fmap_d","d_out","fmap_h","h_out","fmap_w","w_out"]},"block_dim": {"0":32},"correct_range_flag":false,"cost_range":{"0": [2,3,8,15,8,15,8,15]},"fmap_c1":20,"kernelId":0,"repo_range": {},"repo_seeds": {},"tiling_type": "dynamic_tiling"})";

  ge::Graph graph("Conv3d_tiling_fuzzy_build");

  auto x_shape = std::vector<int64_t>({2, 8, 20, 8, 8, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_shape = std::vector<int64_t>({320, 2, 20, 2, 2, 16});
  ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_FRACTAL_Z_3D, ge::DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(filter_desc);
  filter.update_output_desc_y(filter_desc);

  auto conv3d = op::Conv3D(op_name.c_str())
    .set_input_x(x_size)
    .set_input_filter(filter);

  auto y_shape = std::vector<int64_t>({2, 4, 20, 4, 4, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  conv3d.update_input_desc_x(x_desc);
  conv3d.update_input_desc_filter(filter_desc);
  conv3d.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{x_size, filter};
  std::vector<ge::Operator> outputs{conv3d};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_tiling_fuzzy_build", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second(conv3d, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 0);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 8 4 8 4 8 4 ");
}
