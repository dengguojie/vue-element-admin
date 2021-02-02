#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

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

TEST_F(Conv2DBackpropFilterTiling, Conv2d_bp_filter_tiling_dynamic_hw) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropFilter";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv2d_backprop_filter", "push_status": 0, "dynamic_mode": "dynamic_hw", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [4, 4, 4, 4]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["fmap_h", "fmap_w", "dedy_h", "dedy_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {2, 128, 4, 4},
    {256, 128, 3, 3},
    {2, 256, 4, 4},
  };
  std::vector<int64_t> output {256, 128, 3, 3};
  std::vector<std::string> input_types{"float16", "int32", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NCHW", "NCHW", "NCHW"};
  std::string output_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = input_formats[i];
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;
  tensor_output.format = output_format;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "Conv2d_bp_filter_tiling_dynamic_hw";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 2);
}

TEST_F(Conv2DBackpropFilterTiling, Conv2d_bp_filter_tiling_dynamic_n) {
  using namespace optiling;
  std::string op_name = "Conv2DBackpropFilter";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv2d_backprop_filter", "push_status": 0, "dynamic_mode": "dynamic_batch", "repo_seeds": {}, "repo_range": {}, "tiling_range": {"10000": [2, 2]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["fmap_n"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {2, 128, 4, 4},
    {256, 128, 3, 3},
    {2, 256, 4, 4},
  };
  std::vector<int64_t> output {256, 128, 3, 3};
  std::vector<std::string> input_types{"float16", "int32", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NCHW", "NCHW", "NCHW"};
  std::string output_format = "NCHW";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = input_formats[i];
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;
  tensor_output.format = output_format;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "Conv2d_bp_filter_tiling_dynamic_n";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 2);
}