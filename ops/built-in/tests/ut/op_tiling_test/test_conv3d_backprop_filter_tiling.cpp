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

class Conv3DBackpropFilterTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DBackpropFilterTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DBackpropFilterTiling TearDown" << std::endl;
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

TEST_F(Conv3DBackpropFilterTiling, Conv3d_bp_filter_tiling_default_tiling_invalid_fmap_C)
{
  using namespace optiling;
  std::string op_name = "Conv3DBackpropFilter";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv3d_backprop_filter","tiling_type": "default_tiling","fmap_c1": 233, "default_range": {"10000": [3,3,3,3,3,3,32,32]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Conv3d_bp_filter_tiling_default_tiling_invalid_fmap_C");

  auto x_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_size_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc filter_size_desc(ge::Shape(filter_size_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto filter_size = op::Data("filter_size");
  filter_size.update_input_desc_x(filter_size_desc);
  filter_size.update_output_desc_y(filter_size_desc);

  auto out_backprop_shape = std::vector<int64_t>({1, 24, 4, 92, 128, 16});
  ge::TensorDesc out_backprop_desc(ge::Shape(out_backprop_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop");
  out_backprop.update_input_desc_x(out_backprop_desc);
  out_backprop.update_output_desc_y(out_backprop_desc);

  auto conv3dw = op::Conv3DBackpropFilter(op_name.c_str())
    .set_input_x(x_size)
    .set_input_filter_size(filter_size)
    .set_input_out_backprop(out_backprop);

  auto y_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);

  conv3dw.update_input_desc_x(x_desc);
  conv3dw.update_input_desc_filter_size(filter_size_desc);
  conv3dw.update_input_desc_out_backprop(out_backprop_desc);
  conv3dw.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{x_size, filter_size, out_backprop};
  std::vector<ge::Operator> outputs{conv3dw};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_bp_filter_tiling_default_tiling_invalid_fmap_C", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_FALSE(iter->second(conv3dw, op_compile_info, runInfo));
}

TEST_F(Conv3DBackpropFilterTiling, Conv3d_bp_filter_tiling_default_tiling_invalid_dedy_C)
{
  using namespace optiling;
  std::string op_name = "Conv3DBackpropFilter";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv3d_backprop_filter","tiling_type": "default_tiling", "fmap_c1": 2, "dedy_c1": 233, "default_range": {"10000": [3,3,3,3,3,3,32,32]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Conv3d_bp_filter_tiling_default_tiling_invalid_dedy_C");

  auto x_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_size_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc filter_size_desc(ge::Shape(filter_size_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto filter_size = op::Data("filter_size");
  filter_size.update_input_desc_x(filter_size_desc);
  filter_size.update_output_desc_y(filter_size_desc);

  auto out_backprop_shape = std::vector<int64_t>({1, 24, 4, 92, 128, 16});
  ge::TensorDesc out_backprop_desc(ge::Shape(out_backprop_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop");
  out_backprop.update_input_desc_x(out_backprop_desc);
  out_backprop.update_output_desc_y(out_backprop_desc);

  auto conv3dw = op::Conv3DBackpropFilter(op_name.c_str())
    .set_input_x(x_size)
    .set_input_filter_size(filter_size)
    .set_input_out_backprop(out_backprop);

  auto y_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);

  conv3dw.update_input_desc_x(x_desc);
  conv3dw.update_input_desc_filter_size(filter_size_desc);
  conv3dw.update_input_desc_out_backprop(out_backprop_desc);
  conv3dw.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{x_size, filter_size, out_backprop};
  std::vector<ge::Operator> outputs{conv3dw};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_bp_filter_tiling_default_tiling_invalid_dedy_C", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_FALSE(iter->second(conv3dw, op_compile_info, runInfo));
}

TEST_F(Conv3DBackpropFilterTiling, Conv3d_bp_filter_tiling_default_tiling_invalid_test_C)
{
  using namespace optiling;
  std::string op_name = "Conv3DBackpropFilter";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv3d_backprop_filter","tiling_type": "default_tiling", "fmap_c1": 2, "dedy_c1": 233, "default_range": {"10000": [3,3,3,3,3,3,32,32]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Conv3d_bp_filter_tiling_default_tiling_invalid_test_C");

  auto x_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_size_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc filter_size_desc(ge::Shape(filter_size_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto filter_size = op::Data("filter_size");
  filter_size.update_input_desc_x(filter_size_desc);
  filter_size.update_output_desc_y(filter_size_desc);

  auto out_backprop_shape = std::vector<int64_t>({1, 24, 4, 92, 128, 16});
  ge::TensorDesc out_backprop_desc(ge::Shape(out_backprop_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop");
  out_backprop.update_input_desc_x(out_backprop_desc);
  out_backprop.update_output_desc_y(out_backprop_desc);

  auto conv3dw = op::Conv3DBackpropFilter(op_name.c_str())
    .set_input_x(x_size)
    .set_input_filter_size(filter_size)
    .set_input_out_backprop(out_backprop);

  auto y_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);

  conv3dw.update_input_desc_x(x_desc);
  conv3dw.update_input_desc_filter_size(filter_size_desc);
  conv3dw.update_input_desc_out_backprop(out_backprop_desc);
  conv3dw.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{x_size, filter_size, out_backprop};
  std::vector<ge::Operator> outputs{conv3dw};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_bp_filter_tiling_default_tiling_invalid_test_C", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_FALSE(iter->second(conv3dw, op_compile_info, runInfo));
}

TEST_F(Conv3DBackpropFilterTiling, Conv3d_bp_filter_tiling_default_tiling_invalid_test)
{
  using namespace optiling;
  std::string op_name = "Conv3DBackpropFilter";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv3d_backprop_filter","tiling_type": "default_tiling", "default_range": {"10000": [3,3,3,3,3,3,32,32]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Conv3d_bp_filter_tiling_default_tiling_invalid_test");

  auto x_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto x_size = op::Data("x");
  x_size.update_input_desc_x(x_desc);
  x_size.update_output_desc_y(x_desc);

  auto filter_size_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc filter_size_desc(ge::Shape(filter_size_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto filter_size = op::Data("filter_size");
  filter_size.update_input_desc_x(filter_size_desc);
  filter_size.update_output_desc_y(filter_size_desc);

  auto out_backprop_shape = std::vector<int64_t>({1, 24, 4, 92, 128, 16});
  ge::TensorDesc out_backprop_desc(ge::Shape(out_backprop_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop");
  out_backprop.update_input_desc_x(out_backprop_desc);
  out_backprop.update_output_desc_y(out_backprop_desc);

  auto conv3dw = op::Conv3DBackpropFilter(op_name.c_str())
    .set_input_x(x_size)
    .set_input_filter_size(filter_size)
    .set_input_out_backprop(out_backprop);

  auto y_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);

  conv3dw.update_input_desc_x(x_desc);
  conv3dw.update_input_desc_filter_size(filter_size_desc);
  conv3dw.update_input_desc_out_backprop(out_backprop_desc);
  conv3dw.update_output_desc_y(y_desc);

  std::vector<ge::Operator> inputs{x_size, filter_size, out_backprop};
  std::vector<ge::Operator> outputs{conv3dw};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Conv3d_bp_filter_tiling_default_tiling_invalid_test", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_FALSE(iter->second(conv3dw, op_compile_info, runInfo));
}
