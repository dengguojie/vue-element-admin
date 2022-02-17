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

class Conv2DBackpropFilterTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DBackpropFilterTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DBackpropFilterTiling TearDown" << std::endl;
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

TEST_F(Conv2DBackpropFilterTiling, Conv2d_bp_filter_tiling_dynamic_nhw) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropFilter";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_filter", "push_status": 1, "tiling_type": "dynamic_tiling", "repo_seeds": {"10000": [8, 52, 635]}, "repo_range": {"10000": [8, 8, 52, 52, 635, 635]}, "cost_range": {}, "block_dim": {"10000": 16}, "correct_range_flag": false, "_vars": {"10000": ["batch", "fmap_h", "dedy_h", "fmap_w", "dedy_w"]}})";

  ge::Graph graph("conv2dbackprop_filter_op_tiling_test_0");

  auto x_shape = vector<int64_t>({8, 5, 52, 635});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_size_shape = vector<int64_t>({257, 5, 1, 1});
  ge::TensorDesc desc_filter_size(ge::Shape(filter_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter_size = op::Data("filter_size").set_attr_index(1);
  filter_size.update_input_desc_x(desc_filter_size);
  filter_size.update_output_desc_y(desc_filter_size);

  auto out_backprop_shape = vector<int64_t>({8, 257, 13, 159});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropfilter = op::Conv2DBackpropFilter(op_name)
      .set_input_x(x)
      .set_input_filter_size(filter_size)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({257, 5, 1, 1});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropfilter.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropfilter.update_input_desc_x(desc_x);
  conv2dbackpropfilter.update_input_desc_filter_size(desc_filter_size);
  conv2dbackpropfilter.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, out_backprop, filter_size};
  std::vector<Operator> outputs{conv2dbackpropfilter};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_filter_tiling_dynamic_nhw", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(conv2dbackpropfilter, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 16);
}

TEST_F(Conv2DBackpropFilterTiling, Conv2d_bp_filter_tiling_dynamic_n) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropFilter";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_filter", "push_status": 1, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "tiling_range": {"10000": [1, 7]}, "block_dim": {"10000": 16}, "correct_range_flag": true, "_vars": {"10000": ["batch"]}})";

  ge::Graph graph("conv2dbackprop_filter_op_tiling_test_1");

  auto x_shape = vector<int64_t>({8, 5, 52, 635});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_size_shape = vector<int64_t>({257, 5, 1, 1});
  ge::TensorDesc desc_filter_size(ge::Shape(filter_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter_size = op::Data("filter_size").set_attr_index(1);
  filter_size.update_input_desc_x(desc_filter_size);
  filter_size.update_output_desc_y(desc_filter_size);

  auto out_backprop_shape = vector<int64_t>({8, 257, 13, 159});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropfilter = op::Conv2DBackpropFilter(op_name)
      .set_input_x(x)
      .set_input_filter_size(filter_size)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({257, 5, 1, 1});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropfilter.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropfilter.update_input_desc_x(desc_x);
  conv2dbackpropfilter.update_input_desc_filter_size(desc_filter_size);
  conv2dbackpropfilter.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, out_backprop, filter_size};
  std::vector<Operator> outputs{conv2dbackpropfilter};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_filter_tiling_dynamic_n", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(conv2dbackpropfilter, op_compile_info, runInfo));
}

TEST_F(Conv2DBackpropFilterTiling, Conv2d_bp_filter_tiling_dynamic_compile_info_empty) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropFilter";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({})";

  ge::Graph graph("conv2d_bp_filter_tiling_dynamic_compile_info_empty");

  auto x_shape = vector<int64_t>({8, 5, 52, 635});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_size_shape = vector<int64_t>({257, 5, 1, 1});
  ge::TensorDesc desc_filter_size(ge::Shape(filter_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter_size = op::Data("filter_size").set_attr_index(1);
  filter_size.update_input_desc_x(desc_filter_size);
  filter_size.update_output_desc_y(desc_filter_size);

  auto out_backprop_shape = vector<int64_t>({8, 257, 13, 159});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropfilter = op::Conv2DBackpropFilter(op_name)
      .set_input_x(x)
      .set_input_filter_size(filter_size)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({257, 5, 1, 1});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropfilter.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropfilter.update_input_desc_x(desc_x);
  conv2dbackpropfilter.update_input_desc_filter_size(desc_filter_size);
  conv2dbackpropfilter.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, out_backprop, filter_size};
  std::vector<Operator> outputs{conv2dbackpropfilter};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_filter_tiling_dynamic_compile_info_empty", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(conv2dbackpropfilter, op_compile_info, runInfo));
}

TEST_F(Conv2DBackpropFilterTiling, Conv2d_bp_filter_tiling_dynamic_compile_info_not_have_vars) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropFilter";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_filter", "push_status": 1, "tiling_type": "dynamic_tiling", "repo_seeds": {"10000": [8, 52, 635]}, "repo_range": {"10000": [8, 8, 52, 52, 635, 635]}, "cost_range": {}, "block_dim": {"10000": 16}, "correct_range_flag": false})";

  ge::Graph graph("conv2d_bp_filter_tiling_dynamic_compile_info_not_have_vars");

  auto x_shape = vector<int64_t>({8, 5, 52, 635});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_size_shape = vector<int64_t>({257, 5, 1, 1});
  ge::TensorDesc desc_filter_size(ge::Shape(filter_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter_size = op::Data("filter_size").set_attr_index(1);
  filter_size.update_input_desc_x(desc_filter_size);
  filter_size.update_output_desc_y(desc_filter_size);

  auto out_backprop_shape = vector<int64_t>({8, 257, 13, 159});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropfilter = op::Conv2DBackpropFilter(op_name)
      .set_input_x(x)
      .set_input_filter_size(filter_size)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({257, 5, 1, 1});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropfilter.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropfilter.update_input_desc_x(desc_x);
  conv2dbackpropfilter.update_input_desc_filter_size(desc_filter_size);
  conv2dbackpropfilter.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, out_backprop, filter_size};
  std::vector<Operator> outputs{conv2dbackpropfilter};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_filter_tiling_dynamic_compile_info_not_have_vars", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_v2_(conv2dbackpropfilter, op_compile_info, runInfo));
}

// fuzz build compile list input
TEST_F(Conv2DBackpropFilterTiling, Conv2d_bp_filter_tiling_fuzz_build_list_input) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropFilter";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"([{"_pattern": "Conv2d_backprop_filter", "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"0": [1, 16, 50, 53, 630, 640]}, "block_dim": {"0": 16}, "_vars": {"0": ["batch", "fmap_h", "dedy_h", "fmap_w", "dedy_w"]}},{"_pattern": "Conv2d_backprop_filter", "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"1": [16, 32, 64, 128, 64, 128]}, "block_dim": {"1": 16}, "_vars": {"1": ["batch", "fmap_h", "dedy_h", "fmap_w", "dedy_w"]}}])";

  ge::Graph graph("conv2dbackprop_filter_op_tiling_test_2");

  auto x_shape = vector<int64_t>({8, 5, 52, 635});
  ge::TensorDesc desc_x(ge::Shape(x_shape), FORMAT_NCHW, DT_FLOAT16);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_size_shape = vector<int64_t>({257, 5, 1, 1});
  ge::TensorDesc desc_filter_size(ge::Shape(filter_size_shape), FORMAT_NCHW, DT_FLOAT16);
  auto filter_size = op::Data("filter_size").set_attr_index(1);
  filter_size.update_input_desc_x(desc_filter_size);
  filter_size.update_output_desc_y(desc_filter_size);

  auto out_backprop_shape = vector<int64_t>({8, 257, 13, 159});
  ge::TensorDesc desc_out_backprop(ge::Shape(out_backprop_shape), FORMAT_NCHW, DT_FLOAT16);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropfilter = op::Conv2DBackpropFilter(op_name)
      .set_input_x(x)
      .set_input_filter_size(filter_size)
      .set_input_out_backprop(out_backprop);

  auto y_shape = vector<int64_t>({257, 5, 1, 1});
  ge::TensorDesc output_desc_y(ge::Shape(y_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);

  conv2dbackpropfilter.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropfilter.update_input_desc_x(desc_x);
  conv2dbackpropfilter.update_input_desc_filter_size(desc_filter_size);
  conv2dbackpropfilter.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, out_backprop, filter_size};
  std::vector<Operator> outputs{conv2dbackpropfilter};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_filter_tiling_fuzz_build_list_input", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(conv2dbackpropfilter, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 16);
  EXPECT_EQ(runInfo.GetTilingKey(), 0);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8 52 13 635 159 ");
}

TEST_F(Conv2DBackpropFilterTiling, Conv2d_bp_filter_tiling_binary_mode_normal) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropFilter";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_filter", "block_dim": {"CORE_NUM": 32}, "tiling_type": "binary", "max_core_num": 16, "attrs": {"dilation_h":1,"dilation_w":1,"groups":1,"padd":0,"padl":0,"padr":0,"padu":0,"stride_h":1,"stride_w":1}})";

  ge::Graph graph("conv2dbackprop_filter_op_tiling_test_binary_mode_normal");


  auto x_shape_vec = vector<int64_t>({8, 5, 56, 56});
  ge::Shape x_shape(x_shape_vec);
  ge::TensorDesc desc_x(x_shape, FORMAT_NCHW, DT_FLOAT16);
  desc_x.SetOriginShape(x_shape);
  desc_x.SetOriginFormat(FORMAT_NCHW);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_size_shape_vec = vector<int64_t>({257, 5, 3, 3});
  ge::Shape filter_size_shape = ge::Shape(filter_size_shape_vec);
  ge::TensorDesc desc_filter_size(filter_size_shape, FORMAT_NCHW, DT_FLOAT16);
  desc_filter_size.SetOriginShape(filter_size_shape);
  desc_filter_size.SetOriginFormat(FORMAT_NCHW);
  auto filter_size = op::Data("filter_size").set_attr_index(1);
  filter_size.update_input_desc_x(desc_filter_size);
  filter_size.update_output_desc_y(desc_filter_size);

  auto out_backprop_shape_vec = vector<int64_t>({8, 257, 54, 54});
  ge::Shape out_backprop_shape(out_backprop_shape_vec);
  ge::TensorDesc desc_out_backprop(out_backprop_shape, FORMAT_NCHW, DT_FLOAT16);
  desc_out_backprop.SetOriginShape(out_backprop_shape);
  desc_out_backprop.SetOriginFormat(FORMAT_NCHW);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropfilter = op::Conv2DBackpropFilter(op_name)
                                  .set_input_x(x)
                                  .set_input_filter_size(filter_size)
                                  .set_input_out_backprop(out_backprop)
                                  .set_attr_strides({1, 1, 1, 1})
                                  .set_attr_pads({0, 0, 0, 0})
                                  .set_attr_dilations({1, 1, 1, 1})
                                  .set_attr_groups({1})
                                  .set_attr_data_format("NCHW");

  auto y_shape_vec = vector<int64_t>({257, 5, 3, 3});
  ge::Shape y_shape = ge::Shape(y_shape_vec);
  ge::TensorDesc output_desc_y(y_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  output_desc_y.SetOriginShape(y_shape);
  output_desc_y.SetOriginFormat(FORMAT_NCHW);

  conv2dbackpropfilter.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropfilter.update_input_desc_x(desc_x);
  conv2dbackpropfilter.update_input_desc_filter_size(desc_filter_size);
  conv2dbackpropfilter.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, out_backprop, filter_size};
  std::vector<Operator> outputs{conv2dbackpropfilter};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_filter_tiling_binary_mode_normal", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(conv2dbackpropfilter, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 12);
  EXPECT_EQ(runInfo.GetTilingKey(), 14100625);
}

TEST_F(Conv2DBackpropFilterTiling, Conv2d_bp_filter_tiling_binary_mode_l1_full_load) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropFilter";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_filter", "block_dim": {"CORE_NUM": 32}, "tiling_type": "binary", "max_core_num": 32, "attrs": {"dilation_h":1,"dilation_w":1,"groups":1,"padd":0,"padl":0,"padr":0,"padu":0,"stride_h":1,"stride_w":1}})";

  ge::Graph graph("conv2dbackprop_filter_op_tiling_test_binary_mode_l1_full_load");


  auto x_shape_vec = vector<int64_t>({1, 256, 13, 13});
  ge::Shape x_shape(x_shape_vec);
  ge::TensorDesc desc_x(x_shape, FORMAT_NCHW, DT_FLOAT16);
  desc_x.SetOriginShape(x_shape);
  desc_x.SetOriginFormat(FORMAT_NCHW);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_size_shape_vec = vector<int64_t>({256, 512, 1, 1});
  ge::Shape filter_size_shape = ge::Shape(filter_size_shape_vec);
  ge::TensorDesc desc_filter_size(filter_size_shape, FORMAT_NCHW, DT_FLOAT16);
  desc_filter_size.SetOriginShape(filter_size_shape);
  desc_filter_size.SetOriginFormat(FORMAT_NCHW);
  auto filter_size = op::Data("filter_size").set_attr_index(1);
  filter_size.update_input_desc_x(desc_filter_size);
  filter_size.update_output_desc_y(desc_filter_size);

  auto out_backprop_shape_vec = vector<int64_t>({1, 512, 13, 13});
  ge::Shape out_backprop_shape(out_backprop_shape_vec);
  ge::TensorDesc desc_out_backprop(out_backprop_shape, FORMAT_NCHW, DT_FLOAT16);
  desc_out_backprop.SetOriginShape(out_backprop_shape);
  desc_out_backprop.SetOriginFormat(FORMAT_NCHW);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropfilter = op::Conv2DBackpropFilter(op_name)
                                  .set_input_x(x)
                                  .set_input_filter_size(filter_size)
                                  .set_input_out_backprop(out_backprop)
                                  .set_attr_strides({1, 1, 1, 1})
                                  .set_attr_pads({0, 0, 0, 0})
                                  .set_attr_dilations({1, 1, 1, 1})
                                  .set_attr_groups({1})
                                  .set_attr_data_format("NCHW");;

  auto y_shape_vec = vector<int64_t>({256, 512, 1, 1});
  ge::Shape y_shape = ge::Shape(y_shape_vec);
  ge::TensorDesc output_desc_y(y_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  output_desc_y.SetOriginShape(y_shape);
  output_desc_y.SetOriginFormat(FORMAT_NCHW);

  conv2dbackpropfilter.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropfilter.update_input_desc_x(desc_x);
  conv2dbackpropfilter.update_input_desc_filter_size(desc_filter_size);
  conv2dbackpropfilter.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, out_backprop, filter_size};
  std::vector<Operator> outputs{conv2dbackpropfilter};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_filter_tiling_binary_mode_l1_full_load", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(conv2dbackpropfilter, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 64);
  EXPECT_EQ(runInfo.GetTilingKey(), 12500000);
}

TEST_F(Conv2DBackpropFilterTiling, Conv2d_bp_filter_tiling_binary_mode_pads_upadate) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropFilter";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  const ge::AscendString compileInfo = R"({"_pattern": "Conv2d_backprop_filter", "block_dim": {"CORE_NUM": 32}, "tiling_type": "binary", "max_core_num": 16, "attrs": {"dilation_h":1,"dilation_w":1,"groups":1,"padd":0,"padl":0,"padr":0,"padu":0,"stride_h":1,"stride_w":1}})";

  ge::Graph graph("conv2dbackprop_filter_op_tiling_test_binary_mode_normal");


  auto x_shape_vec = vector<int64_t>({8, 5, 56, 56});
  ge::Shape x_shape(x_shape_vec);
  ge::TensorDesc desc_x(x_shape, FORMAT_NCHW, DT_FLOAT16);
  desc_x.SetOriginShape(x_shape);
  desc_x.SetOriginFormat(FORMAT_NCHW);
  auto x = op::Data("x");
  x.update_input_desc_x(desc_x);
  x.update_output_desc_y(desc_x);

  auto filter_size_shape_vec = vector<int64_t>({257, 5, 3, 3});
  ge::Shape filter_size_shape = ge::Shape(filter_size_shape_vec);
  ge::TensorDesc desc_filter_size(filter_size_shape, FORMAT_NCHW, DT_FLOAT16);
  desc_filter_size.SetOriginShape(filter_size_shape);
  desc_filter_size.SetOriginFormat(FORMAT_NCHW);
  auto filter_size = op::Data("filter_size").set_attr_index(1);
  filter_size.update_input_desc_x(desc_filter_size);
  filter_size.update_output_desc_y(desc_filter_size);

  auto out_backprop_shape_vec = vector<int64_t>({8, 257, 56, 56});
  ge::Shape out_backprop_shape(out_backprop_shape_vec);
  ge::TensorDesc desc_out_backprop(out_backprop_shape, FORMAT_NCHW, DT_FLOAT16);
  desc_out_backprop.SetOriginShape(out_backprop_shape);
  desc_out_backprop.SetOriginFormat(FORMAT_NCHW);
  auto out_backprop = op::Data("out_backprop").set_attr_index(1);
  out_backprop.update_input_desc_x(desc_out_backprop);
  out_backprop.update_output_desc_y(desc_out_backprop);

  auto conv2dbackpropfilter = op::Conv2DBackpropFilter(op_name)
                                  .set_input_x(x)
                                  .set_input_filter_size(filter_size)
                                  .set_input_out_backprop(out_backprop)
                                  .set_attr_strides({1, 1, 1, 1})
                                  .set_attr_pads({-1, -1, -1, -1})
                                  .set_attr_dilations({1, 1, 1, 1})
                                  .set_attr_groups({1})
                                  .set_attr_data_format("NCHW");

  auto y_shape_vec = vector<int64_t>({257, 5, 3, 3});
  ge::Shape y_shape = ge::Shape(y_shape_vec);
  ge::TensorDesc output_desc_y(y_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  output_desc_y.SetOriginShape(y_shape);
  output_desc_y.SetOriginFormat(FORMAT_NCHW);

  conv2dbackpropfilter.update_input_desc_out_backprop(desc_out_backprop);
  conv2dbackpropfilter.update_input_desc_x(desc_x);
  conv2dbackpropfilter.update_input_desc_filter_size(desc_filter_size);
  conv2dbackpropfilter.update_output_desc_y(output_desc_y);

  std::vector<Operator> inputs{x, out_backprop, filter_size};
  std::vector<Operator> outputs{conv2dbackpropfilter};

  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  optiling::utils::OpCompileInfo op_compile_info("Conv2d_bp_filter_tiling_binary_mode_pads_upadate", compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(conv2dbackpropfilter, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 28);
  EXPECT_EQ(runInfo.GetTilingKey(), 23866250);
}
