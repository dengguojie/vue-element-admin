#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
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

class Conv3DTransposeTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DTransposeTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DTransposeTiling TearDown" << std::endl;
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

TEST_F(Conv3DTransposeTiling, Conv3d_transpose_tiling_dynamic_dhw_not_cover)
{
  using namespace optiling;
  std::string op_name = "Conv3DTranspose";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv3d_backprop_input","tiling_type": "dynamic_tiling", "repo_seeds": {"10000": [52,112,32]},"cost_range": {}, "repo_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Conv3d_transpose_tiling_dynamic_dhw_not_cover");

  auto input_size_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc input_size_desc(ge::Shape(input_size_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(input_size_desc);
  input_size.update_output_desc_y(input_size_desc);

  auto x_shape = std::vector<int64_t>({1, 24, 4, 92, 128, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(filter_desc);
  filter.update_output_desc_y(filter_desc);

  auto conv3d_transpose = op::Conv3DTranspose(op_name.c_str())
    .set_input_input_size(input_size)
    .set_input_x(x_size)
    .set_input_filter(filter);

  auto y_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  conv3d_transpose.update_input_desc_input_size(input_size_desc);
  conv3d_transpose.update_input_desc_x(x_desc);
  conv3d_transpose.update_input_desc_filter(filter_desc);
  conv3d_transpose.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{input_size, x_size, filter};
  std::vector<ge::Operator> outputs{conv3d_transpose};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_transpose_tiling_dynamic_dhw_not_cover", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_FALSE(iter->second.tiling_func_v2_(conv3d_transpose, op_compile_info, runInfo));
}

TEST_F(Conv3DTransposeTiling, Conv3d_transpose_tiling_dynamic_dhw_repo_range)
{
  using namespace optiling;
  std::string op_name = "Conv3DTranspose";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"push_status": 0, "_pattern": "Conv3d_backprop_input","tiling_type": "dynamic_tiling","repo_seeds": {"10000": [1,52,112,32]},"cost_range": {}, "repo_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Conv3d_transpose_tiling_dynamic_dhw_repo_range");

  auto input_size_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc input_size_desc(ge::Shape(input_size_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(input_size_desc);
  input_size.update_output_desc_y(input_size_desc);

  auto x_shape = std::vector<int64_t>({1, 24, 4, 92, 128, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(filter_desc);
  filter.update_output_desc_y(filter_desc);

  auto conv3d_transpose = op::Conv3DTranspose(op_name.c_str())
    .set_input_input_size(input_size)
    .set_input_x(x_size)
    .set_input_filter(filter);

  auto y_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  conv3d_transpose.update_input_desc_input_size(input_size_desc);
  conv3d_transpose.update_input_desc_x(x_desc);
  conv3d_transpose.update_input_desc_filter(filter_desc);
  conv3d_transpose.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{input_size, x_size, filter};
  std::vector<ge::Operator> outputs{conv3d_transpose};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_transpose_tiling_dynamic_dhw_repo_range", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_v2_(conv3d_transpose, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "24 24 92 92 128 128 ");
}

TEST_F(Conv3DTransposeTiling, Conv3d_transpose_tiling_dynamic_dhw_cost_range)
{
  using namespace optiling;
  std::string op_name = "Conv3DTranspose";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"push_status": 0,"_pattern": "Conv3d_backprop_input","tiling_type": "dynamic_tiling","repo_seeds": {"10000": [1,52,112,32]},"cost_range": {"10001": [1,1,24,54,92,122,128,158]},"repo_range": {"10000": [1,1,12,12,92,122,128,158]},"block_dim": {"10000": 2,"10001": 4},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Conv3d_transpose_tiling_dynamic_dhw_cost_range");

  auto input_size_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc input_size_desc(ge::Shape(input_size_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(input_size_desc);
  input_size.update_output_desc_y(input_size_desc);

  auto x_shape = std::vector<int64_t>({1, 24, 4, 92, 128, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(filter_desc);
  filter.update_output_desc_y(filter_desc);

  auto conv3d_transpose = op::Conv3DTranspose(op_name.c_str())
    .set_input_input_size(input_size)
    .set_input_x(x_size)
    .set_input_filter(filter);

  auto y_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  conv3d_transpose.update_input_desc_input_size(input_size_desc);
  conv3d_transpose.update_input_desc_x(x_desc);
  conv3d_transpose.update_input_desc_filter(filter_desc);
  conv3d_transpose.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{input_size, x_size, filter};
  std::vector<ge::Operator> outputs{conv3d_transpose};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_transpose_tiling_dynamic_dhw_cost_range", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_v2_(conv3d_transpose, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 4);
  EXPECT_EQ(runInfo.GetTilingKey(), 10001);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "24 24 92 92 128 128 ");
}

TEST_F(Conv3DTransposeTiling, Conv3d_transpose_tiling_dynamic_batch)
{
  using namespace optiling;
  std::string op_name = "Conv3DTranspose";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv3d_backprop_input","tiling_type": "dynamic_tiling","repo_seeds": {"10000": [1,52,112,32]},"repo_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Conv3d_transpose_tiling_dynamic_batch");

  auto input_size_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc input_size_desc(ge::Shape(input_size_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(input_size_desc);
  input_size.update_output_desc_y(input_size_desc);

  auto x_shape = std::vector<int64_t>({1, 24, 4, 92, 128, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(filter_desc);
  filter.update_output_desc_y(filter_desc);

  auto conv3d_transpose = op::Conv3DTranspose(op_name.c_str())
    .set_input_input_size(input_size)
    .set_input_x(x_size)
    .set_input_filter(filter);

  auto y_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  conv3d_transpose.update_input_desc_input_size(input_size_desc);
  conv3d_transpose.update_input_desc_x(x_desc);
  conv3d_transpose.update_input_desc_filter(filter_desc);
  conv3d_transpose.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{input_size, x_size, filter};
  std::vector<ge::Operator> outputs{conv3d_transpose};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_transpose_tiling_dynamic_batch", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_v2_(conv3d_transpose, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "24 24 92 92 128 128 ");
}

TEST_F(Conv3DTransposeTiling, Conv3d_transpose_tiling_default_tiling)
{
  using namespace optiling;
  std::string op_name = "Conv3DTranspose";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv3d_backprop_input","tiling_type": "default_tiling","default_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Conv3d_transpose_tiling_default_tiling");

  auto input_size_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc input_size_desc(ge::Shape(input_size_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(input_size_desc);
  input_size.update_output_desc_y(input_size_desc);

  auto x_shape = std::vector<int64_t>({1, 24, 4, 92, 128, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(filter_desc);
  filter.update_output_desc_y(filter_desc);

  auto conv3d_transpose = op::Conv3DTranspose(op_name.c_str())
    .set_input_input_size(input_size)
    .set_input_x(x_size)
    .set_input_filter(filter);

  auto y_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  conv3d_transpose.update_input_desc_input_size(input_size_desc);
  conv3d_transpose.update_input_desc_x(x_desc);
  conv3d_transpose.update_input_desc_filter(filter_desc);
  conv3d_transpose.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{input_size, x_size, filter};
  std::vector<ge::Operator> outputs{conv3d_transpose};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_transpose_tiling_default_tiling", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_v2_(conv3d_transpose, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(runInfo.GetTilingKey(), 10000);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "24 24 92 92 128 128 ");
}

TEST_F(Conv3DTransposeTiling, Conv3d_transpose_tiling_dynamic_batch_invalid_C)
{
  using namespace optiling;
  std::string op_name = "Conv3DTranspose";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv3d_backprop_input", "dedy_c1": 233, "tiling_type": "dynamic_tiling","repo_seeds": {"10000": [1,52,112,32]},"repo_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Conv3d_transpose_tiling_dynamic_batch_invalid_C");

  auto input_size_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc input_size_desc(ge::Shape(input_size_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto input_size = op::Data("input_size");
  input_size.update_input_desc_x(input_size_desc);
  input_size.update_output_desc_y(input_size_desc);

  auto x_shape = std::vector<int64_t>({1, 24, 4, 92, 128, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc filter_desc(ge::Shape(filter_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto filter = op::Data("filter");
  filter.update_input_desc_x(filter_desc);
  filter.update_output_desc_y(filter_desc);

  auto conv3d_transpose = op::Conv3DTranspose(op_name.c_str())
    .set_input_input_size(input_size)
    .set_input_x(x_size)
    .set_input_filter(filter);

  auto y_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);

  conv3d_transpose.update_input_desc_input_size(input_size_desc);
  conv3d_transpose.update_input_desc_x(x_desc);
  conv3d_transpose.update_input_desc_filter(filter_desc);
  conv3d_transpose.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{input_size, x_size, filter};
  std::vector<ge::Operator> outputs{conv3d_transpose};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_transpose_tiling_dynamic_batch_invalid_C", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_FALSE(iter->second.tiling_func_v2_(conv3d_transpose, op_compile_info, runInfo));
}
