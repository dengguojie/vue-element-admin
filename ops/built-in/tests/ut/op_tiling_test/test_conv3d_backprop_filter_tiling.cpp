#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

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

TEST_F(Conv3DBackpropFilterTiling, Conv3d_bp_filter_tiling_default_tiling_invalid_fmap_C) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropFilter";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv3d_backprop_filter","tiling_type": "default_tiling","fmap_c1": 233, "default_range": {"10000": [3,3,3,3,3,3,32,32]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128, 16},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128, 16},
  };
  std::vector<int64_t> output {3, 3, 3, 32, 64};
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
  op_compile_info.key = "Conv3d_bpfilter_tiling_dynamic_default_tiling_invalid_fmap_C";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(Conv3DBackpropFilterTiling, Conv3d_bp_filter_tiling_default_tiling_invalid_dedy_C) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropFilter";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv3d_backprop_filter","tiling_type": "default_tiling", "fmap_c1": 2, "dedy_c1": 233, "default_range": {"10000": [3,3,3,3,3,3,32,32]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128, 16},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128, 16},
  };
  std::vector<int64_t> output {3, 3, 3, 32, 64};
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
  op_compile_info.key = "Conv3d_bpfilter_tiling_dynamic_default_tiling_invalid_dedy_C";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(Conv3DBackpropFilterTiling, Conv3d_bp_filter_tiling_default_tiling_invalid_test_C) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropFilter";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv3d_backprop_filter","tiling_type": "default_tiling", "fmap_c1": 2, "dedy_c1": 233, "default_range": {"10000": [3,3,3,3,3,3,32,32]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128},
  };
  std::vector<int64_t> output {3, 3, 3, 32, 64};
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
  op_compile_info.key = "Conv3d_bp_filter_tiling_default_tiling_invalid_test_C";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(Conv3DBackpropFilterTiling, Conv3d_bp_filter_tiling_default_tiling_invalid_test) {
  using namespace optiling;
  std::string op_name = "Conv3DBackpropFilter";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_pattern": "Conv3d_backprop_filter","tiling_type": "default_tiling", "default_range": {"10000": [3,3,3,3,3,3,32,32]},"block_dim": {"10000": 2},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128, 16},
    {3, 3, 3, 32, 64},
    {1, 24, 4, 92, 128, 16},
  };
  std::vector<int64_t> output {3, 3, 3, 32, 64};
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
  op_compile_info.key = "Conv3d_bpfilter_tiling_dynamic_default_tiling_invalid_test";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}