#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

using namespace std;
using namespace ge;
using namespace op;

class AvgPool3DTiling : public testing::Test
{
 protected:
  static void SetUpTestCase()
  {
    std::cout << "AvgPool3DTiling SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "AvgPool3DTiling TearDown" << std::endl;
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

TEST_F(AvgPool3DTiling, avg_pool3d_tiling_dynamic_w)
{
  using namespace optiling;
  std::string op_name = "AvgPool3D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = "{\"_pattern\": \"conv3d\", \"push_status\": 0, \"tiling_type\": \"dynamic_tiling\", \"repo_seeds\": {\"10000\": [32, 16, 56, 56]}, \"repo_range\": {\"10000\": [32, 32, 16, 16, 56, 56, 24, 456]}, \"cost_range\": {}, \"block_dim\": {\"10000\": 32}, \"_vars\": {\"10000\": [\"fmap_w\", \"w_out\"]}}";

  ge::Graph graph("avg_pool3d_tiling_dynamic_w");

  auto x_shape = std::vector<int64_t>({32, 16, 1, 56, 56, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto avgPool3D = op::AvgPool3D(op_name.c_str())
    .set_input_x(x_size);

  auto y_shape = std::vector<int64_t>({32, 15, 1, 56, 56, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  avgPool3D.update_input_desc_x(x_desc);
  avgPool3D.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{x_size};
  std::vector<ge::Operator> outputs{avgPool3D};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("avg_pool3d_tiling_dynamic_w", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_v2_(avgPool3D, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "56 56 ");
}

TEST_F(AvgPool3DTiling, avg_pool3d_tiling_dynamic_batch_invalid_dim)
{
  using namespace optiling;
  std::string op_name = "AvgPool3D";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = "{\"_pattern\": \"conv3d\", \"push_status\": 0, \"fmap_c1\": 233, \"tiling_type\": \"dynamic_tiling\", \"repo_seeds\": {\"10000\": 32}, \"tiling_range\": {\"10000\": [1, 35]}, \"block_dim\": {\"10000\": 32}, \"_vars\": {\"10000\": [\"batch_n\"]}}";

  ge::Graph graph("avg_pool3d_tiling_dynamic_batch_invalid_dim");

  auto x_shape = std::vector<int64_t>({32, 16, 1, 56, 56});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto avgPool3D = op::AvgPool3D(op_name.c_str())
    .set_input_x(x_size);

  auto y_shape = std::vector<int64_t>({32, 15, 1, 56, 56, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  avgPool3D.update_input_desc_x(x_desc);
  avgPool3D.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{x_size};
  std::vector<ge::Operator> outputs{avgPool3D};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("avg_pool3d_tiling_dynamic_batch_invalid_dim", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_FALSE(iter->second.tiling_func_v2_(avgPool3D, op_compile_info, runInfo));
}
