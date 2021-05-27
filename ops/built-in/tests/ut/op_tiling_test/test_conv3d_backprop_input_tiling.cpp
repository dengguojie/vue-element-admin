#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class Conv3DBackpropInputTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv3DBackpropInputTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv3DBackpropInputTiling TearDown" << std::endl;
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

TEST_F(Conv3DBackpropInputTiling, Conv3d_bp_input_tiling_dynamic_dhw_not_cover) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropInput";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv3d_backprop_input","tiling_type": "dynamic_tiling", "repo_seeds": {"10000": [52,112,32]},"cost_range": {}, "repo_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128, 16},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128, 16},
  };
  std::vector<int64_t> output {1, 24, 2, 92, 128, 16};
  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDHWC", "NDHWC", "NDHWC"};
  std::string output_format = "NDHWC";

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
  op_compile_info.key = "Conv3d_bp_input_tiling_dynamic_dhw_not_cover";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(Conv3DBackpropInputTiling, Conv3d_bp_input_tiling_dynamic_dhw_repo_range) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropInput";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"push_status": 0, "_pattern": "Conv3d_backprop_input","tiling_type": "dynamic_tiling","repo_seeds": {"10000": [1,52,112,32]},"cost_range": {}, "repo_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128, 16},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128, 16},
  };
  std::vector<int64_t> output {1, 24, 2, 92, 128, 16};
  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDHWC", "NDHWC", "NDHWC"};
  std::string output_format = "NDHWC";

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
  op_compile_info.key = "Conv3d_bp_tiling_dynamic_dhw";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 2);
  EXPECT_EQ(runInfo.tiling_key, 10000);
  EXPECT_EQ(to_string(runInfo.tiling_data), "24 24 92 92 128 128 ");
}


TEST_F(Conv3DBackpropInputTiling, Conv3d_bp_input_tiling_dynamic_dhw_cost_range) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropInput";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"push_status": 0,"_pattern": "Conv3d_backprop_input","tiling_type": "dynamic_tiling","repo_seeds": {"10000": [1,52,112,32]},"cost_range": {"10001": [1,1,24,54,92,122,128,158]},"repo_range": {"10000": [1,1,12,12,92,122,128,158]},"block_dim": {"10000": 2,"10001": 4},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128, 16},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128, 16},
  };
  std::vector<int64_t> output {1, 24, 2, 92, 128, 16};
  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDHWC", "NDHWC", "NDHWC"};
  std::string output_format = "NDHWC";

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
  op_compile_info.key = "Conv3d_bp_input_tiling_dynamic_dhw_cost_range";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 4);
  EXPECT_EQ(runInfo.tiling_key, 10001);
  EXPECT_EQ(to_string(runInfo.tiling_data), "24 24 92 92 128 128 ");
}

TEST_F(Conv3DBackpropInputTiling, Conv3d_bp_input_tiling_dynamic_batch) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropInput";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv3d_backprop_input","tiling_type": "dynamic_tiling","repo_seeds": {"10000": [1,52,112,32]},"repo_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128, 16},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128, 16},
  };
  std::vector<int64_t> output {1, 24, 2, 92, 128, 16};
  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDHWC", "NDHWC", "NDHWC"};
  std::string output_format = "NDHWC";

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
  op_compile_info.key = "Conv3d_bp_tiling_dynamic_dhw";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 2);
  EXPECT_EQ(runInfo.tiling_key, 10000);
  EXPECT_EQ(to_string(runInfo.tiling_data), "24 24 92 92 128 128 ");
}

TEST_F(Conv3DBackpropInputTiling, Conv3d_bp_input_tiling_default_tiling) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropInput";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv3d_backprop_input","tiling_type": "default_tiling","default_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128, 16},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128, 16},
  };
  std::vector<int64_t> output {1, 24, 2, 92, 128, 16};
  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDHWC", "NDHWC", "NDHWC"};
  std::string output_format = "NDHWC";

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
  op_compile_info.key = "Conv3d_bp_tiling_dynamic_default_tiling";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 2);
  EXPECT_EQ(runInfo.tiling_key, 10000);
  EXPECT_EQ(to_string(runInfo.tiling_data), "24 24 92 92 128 128 ");
}


TEST_F(Conv3DBackpropInputTiling, Conv3d_bp_input_tiling_dynamic_batch_invalid_C) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropInput";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv3d_backprop_input", "dedy_c1": 233, "tiling_type": "dynamic_tiling","repo_seeds": {"10000": [1,52,112,32]},"repo_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128, 16},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128, 16},
  };
  std::vector<int64_t> output {1, 24, 2, 92, 128, 16};
  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDHWC", "NDHWC", "NDHWC"};
  std::string output_format = "NDHWC";

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
  op_compile_info.key = "Conv3d_bp_tiling_dynamic_dhw_invalid_C";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}


// fuzz build compile list input
TEST_F(Conv3DBackpropInputTiling, Conv3d_bp_input_fuzz_build_list_input) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropInput";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv3d_backprop_input","_vars": {"4": ["batch_n","dedx_d","dedy_d","dedx_h","dedy_h","dedx_w","dedy_w"]},"block_dim": {"0":28,"1":32,"2":32,"3":32,"4":32},"correct_range_flag":null,"cost_range":{},"deby_c1":4,"kernelId":4,"repo_range": {"0":[1,1,8,8,56,56,56,56],"1":[1,1,8,8,56,56,56,56],"2":[1,1,8,8,56,56,56,56],"3":[1,1,8,8,56,56,56,56],"4":[1,1,8,8,56,56,56,56]},"repo_seeds": {"0":[1,8,56,56],"1":[1,52,192,288],"2":[1,52,160,160],"3":[1,52,160,160],"4":[1,52,160,160]},"tiling_type": "dynamic_tiling"})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 8, 16, 56, 56, 16},
    {1, 1, 1, 256, 64},
    {1, 8, 4, 56, 56, 16},
  };
  std::vector<int64_t> output {1, 8, 16, 56, 56, 16};
  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDHWC", "NDHWC", "NDHWC"};
  std::string output_format = "NDHWC";

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
  op_compile_info.key = "Conv3d_bp_input_fuzz_build_list_input";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 28);
  EXPECT_EQ(runInfo.tiling_key, 0);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 8 8 56 56 56 56 ");
}



TEST_F(Conv3DBackpropInputTiling, Conv3d_bp_input_fuzz_build_list_compile_info_array) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropInput";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"([{"_pattern": "Conv3d_backprop_input","_vars": {"4": ["batch_n","dedx_d","dedy_d","dedx_h","dedy_h","dedx_w","dedy_w"]},"block_dim": {"0":28,"1":32,"2":32,"3":32,"4":32},"correct_range_flag":null,"cost_range":{},"deby_c1":4,"kernelId":4,"repo_range": {"0":[1,1,8,8,56,56,56,56],"1":[1,1,8,8,56,56,56,56],"2":[1,1,8,8,56,56,56,56],"3":[1,1,8,8,56,56,56,56],"4":[1,1,8,8,56,56,56,56]},"repo_seeds": {"0":[1,8,56,56],"1":[1,52,192,288],"2":[1,52,160,160],"3":[1,52,160,160],"4":[1,52,160,160]},"tiling_type": "dynamic_tiling"}, {"_pattern": "Conv3d_backprop_input","_vars": {"4": ["batch_n","dedx_d","dedy_d","dedx_h","dedy_h","dedx_w","dedy_w"]},"block_dim": {"0":28,"1":32,"2":32,"3":32,"4":32},"correct_range_flag":null,"cost_range":{},"deby_c1":4,"kernelId":4,"repo_range": {"0":[1,1,8,8,56,56,56,56],"1":[1,1,8,8,56,56,56,56],"2":[1,1,8,8,56,56,56,56],"3":[1,1,8,8,56,56,56,56],"4":[1,1,8,8,56,56,56,56]},"repo_seeds": {"0":[1,8,56,56],"1":[1,52,192,288],"2":[1,52,160,160],"3":[1,52,160,160],"4":[1,52,160,160]},"tiling_type": "dynamic_tiling"}])";

  std::vector<std::vector<int64_t>> inputs {
    {1, 8, 16, 56, 56, 16},
    {1, 1, 1, 256, 64},
    {1, 8, 4, 56, 56, 16},
  };
  std::vector<int64_t> output {1, 8, 16, 56, 56, 16};
  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDHWC", "NDHWC", "NDHWC"};
  std::string output_format = "NDHWC";

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
  op_compile_info.key = "Conv3d_bp_input_fuzz_build_list_compile_info_array";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 28);
  EXPECT_EQ(runInfo.tiling_key, 0);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 8 8 56 56 56 56 ");
}


TEST_F(Conv3DBackpropInputTiling, Conv3d_bp_input_tiling_dynamic_batch_test_C) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropInput";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv3d_backprop_input", "dedy_c1": 233, "tiling_type": "dynamic_tiling","repo_seeds": {"10000": [1,52,112,32]},"repo_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128},
  };
  std::vector<int64_t> output {1, 24, 2, 92, 128, 16};
  std::vector<std::string> input_types{"float16", "float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDHWC", "NDHWC", "NDHWC"};
  std::string output_format = "NDHWC";

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
  op_compile_info.key = "Conv3d_bp_input_tiling_dynamic_batch_test_C";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}
