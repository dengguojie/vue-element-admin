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

class AvgPool3DGradTiling : public testing::Test {
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

TEST_F(AvgPool3DGradTiling, Avg_Pool3D_Grad_tiling_invalid_test_A)
{
  using namespace optiling;
  std::string op_name = "AvgPool3DGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Avg_pool3d_grad","tiling_type": "default_tiling","default_range": {"10000": [3,3,3,3,3,3,32,32]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Avg_Pool3D_Grad_tiling_invalid_test_A");

  auto orig_input_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc orig_input_shape_desc(ge::Shape(orig_input_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto orig_input_shape_size = op::Data("orig_input_shape");
  orig_input_shape_size.update_input_desc_x(orig_input_shape_desc);
  orig_input_shape_size.update_output_desc_y(orig_input_shape_desc);

  auto grads_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc grads_desc(ge::Shape(grads_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto grads = op::Data("grads");
  grads.update_input_desc_x(grads_desc);
  grads.update_output_desc_y(grads_desc);

  auto avgPool3DGrad = op::AvgPool3DGrad(op_name.c_str())
    .set_input_orig_input_shape(orig_input_shape_size)
    .set_input_grads(grads);

  auto output_shape = std::vector<int64_t>({3, 3, 3, 32, 64});
  ge::TensorDesc output_desc(ge::Shape(output_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);

  avgPool3DGrad.update_input_desc_orig_input_shape(orig_input_shape_desc);
  avgPool3DGrad.update_input_desc_grads(grads_desc);
  avgPool3DGrad.update_output_desc_output(output_desc);

  std::vector<ge::Operator> inputs{orig_input_shape_size, grads};
  std::vector<ge::Operator> outputs{avgPool3DGrad};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Avg_Pool3D_Grad_tiling_invalid_test_A", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_FALSE(iter->second.tiling_func_v2_(avgPool3DGrad, op_compile_info, runInfo));
}

TEST_F(AvgPool3DGradTiling, Avg_Pool3D_Grad_tiling_invalid_test_B)
{
  using namespace optiling;
  std::string op_name = "AvgPool3DGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Avg_pool3d_grad","tiling_type": "default_tiling","default_range": {"10000": [3,3,3,3,3,3,32,32]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  ge::Graph graph("Avg_Pool3D_Grad_tiling_invalid_test_A");

  auto orig_input_shape = std::vector<int64_t>({1, 24, 2, 92, 128, 16});
  ge::TensorDesc orig_input_shape_desc(ge::Shape(orig_input_shape), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  auto orig_input_shape_size = op::Data("orig_input_shape");
  orig_input_shape_size.update_input_desc_x(orig_input_shape_desc);
  orig_input_shape_size.update_output_desc_y(orig_input_shape_desc);

  auto grads_shape = std::vector<int64_t>({3, 3, 3, 32, 64, 1});
  ge::TensorDesc grads_desc(ge::Shape(grads_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);
  auto grads = op::Data("grads");
  grads.update_input_desc_x(grads_desc);
  grads.update_output_desc_y(grads_desc);

  auto avgPool3DGrad = op::AvgPool3DGrad(op_name.c_str())
    .set_input_orig_input_shape(orig_input_shape_size)
    .set_input_grads(grads);

  auto output_shape = std::vector<int64_t>({3, 3, 3, 32, 64, 1});
  ge::TensorDesc output_desc(ge::Shape(output_shape), ge::FORMAT_DHWCN, ge::DT_FLOAT16);

  avgPool3DGrad.update_input_desc_orig_input_shape(orig_input_shape_desc);
  avgPool3DGrad.update_input_desc_grads(grads_desc);
  avgPool3DGrad.update_output_desc_output(output_desc);

  std::vector<ge::Operator> inputs{orig_input_shape_size, grads};
  std::vector<ge::Operator> outputs{avgPool3DGrad};

  graph.SetInputs(inputs).SetOutputs(outputs);

  optiling::utils::OpCompileInfo op_compile_info("Avg_Pool3D_Grad_tiling_invalid_test_A", compileInfo);
  optiling::utils::OpRunInfo runInfo;

  ASSERT_FALSE(iter->second.tiling_func_v2_(avgPool3DGrad, op_compile_info, runInfo));
}
