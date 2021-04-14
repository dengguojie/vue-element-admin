#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class GEMMTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GEMMTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GEMMTiling TearDown" << std::endl;
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

TEST_F(GEMMTiling, GEMM_op_tiling_obj) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Matmul", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 1, 3, 1, 3]}, "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}})";

  std::vector<std::vector<int64_t>> ori_inputs {
    {2, 3},
    {3, 4}
  };
  std::vector<std::vector<int64_t>> inputs {
    {1, 1, 16, 16},
    {1, 1, 16, 16}
  };
  std::vector<int64_t> ori_output {2, 4};
  std::vector<int64_t> output {1, 1, 16, 16};
  std::vector<std::string> input_types{"float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"FRACTAL_NZ", "FRACTAL_NZ"};
  std::string origin_format = "ND";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = input_formats[i];
    tensor_input.ori_shape = ori_inputs[i];
    tensor_input.ori_format = origin_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;
  tensor_output.ori_shape = ori_output;
  tensor_output.format = "FRACTAL_NZ";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "Matmul";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 2);
  EXPECT_EQ(runInfo.tiling_key, 10000);
}

TEST_F(GEMMTiling, GEMM_op_tiling_arr) {
  using namespace optiling;
  std::string op_name = "MatMul";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"([{"_pattern": "Matmul", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 1, 3, 1, 3]}, "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}},{"_pattern": "Matmul", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10001": [1, 3, 1, 3, 4, 7]}, "block_dim": {"10001": 2}, "attrs":{"transpose_a": false, "transpose_b": false}}])";

  std::vector<std::vector<int64_t>> ori_inputs {
    {2, 3},
    {3, 4}
  };
  std::vector<std::vector<int64_t>> inputs {
    {1, 1, 16, 16},
    {1, 1, 16, 16}
  };
  std::vector<int64_t> ori_output {2, 4};
  std::vector<int64_t> output {1, 1, 16, 16};
  std::vector<std::string> input_types{"float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"FRACTAL_NZ", "FRACTAL_NZ"};
  std::string origin_format = "ND";

  TeOpParas opParas;
  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    TeOpTensorArg tensor_arg;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = input_formats[i];
    tensor_input.ori_shape = ori_inputs[i];
    tensor_input.ori_format = origin_format;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensor_arg);
  }
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;
  tensor_output.ori_shape = ori_output;
  tensor_output.format = "FRACTAL_NZ";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "GEMM_op_tiling_arr";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 2);
  EXPECT_EQ(runInfo.tiling_key, 10000);
}