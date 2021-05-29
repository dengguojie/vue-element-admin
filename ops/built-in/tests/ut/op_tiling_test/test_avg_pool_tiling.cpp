#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

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
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "strides_h" : 60, "strides_w" : 60, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 32, 16, 16},
    {64, 32, 3, 3},
  };
  std::vector<int64_t> output {1, 64, 16, 16};
  std::vector<std::string> input_types{"float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NCHW", "NCHW"};
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
  op_compile_info.key = "AvgPool_tiling_dynamic_nhw";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 2);
  EXPECT_EQ(runInfo.tiling_key, 10000);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 16 16 16 16 ");
}

TEST_F(AvgPoolTiling, AvgPool_tiling_dynamic_None) {
  using namespace optiling;
  std::string op_name = "AvgPool";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]}, "block_dim": {"10000": 1}, "strides_h" : 60, "strides_w" : 60, "_vars": {"10000": ["batch_n"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 32, 16, 16},
    {64, 32, 3, 3},
  };
  std::vector<int64_t> output {1, 64, 16, 16};
  std::vector<std::string> input_types{"float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NCHW", "NCHW"};
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
  op_compile_info.key = "AvgPool_tiling_dynamic_None";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(runInfo.tiling_key, 10000);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 ");
}
TEST_F(AvgPoolTiling, AvgPool_tiling_dynamic_Vector) {
  using namespace optiling;
  std::string op_name = "AvgPool";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"strides_h" : 64, "strides_w" : 64, "vars": {"ub_ele": 126976, "core_num": 30, "ksize_h": 1, "ksize_w": 1, "strides_h": 64, "strides_w": 64, "padding": 0}})";

  std::vector<std::vector<int64_t>> inputs {
    {16, 13, 79, 69, 16}
  };
  std::vector<int64_t> output {16, 13, 79, 69, 16};
  std::vector<std::string> input_types{"float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NC1HWC0"};
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
  op_compile_info.key = "AvgPool_tiling_dynamic_Vector";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 30 7 5 79 69 2 2 65 65 0 0 0 0 1 1 1 2 0 2 0 ");
}