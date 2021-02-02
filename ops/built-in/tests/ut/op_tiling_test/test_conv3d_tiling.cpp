#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class Conv3DTiling : public testing::Test
{
 protected:
  static void SetUpTestCase()
  {
    std::cout << "Conv3DTiling SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "Conv3DTiling TearDown" << std::endl;
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

TEST_F(Conv3DTiling, Conv3d_tiling_dynamic_dhw)
{
  using namespace optiling;
  std::string op_name = "Conv3D";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"_pattern\": \"conv3d\", \"dynamic_mode\": \"dynamic_dhw\", \"repo_seeds\": {\"10000\": [128, 128, 128], \"10001\": [48, 184, 256], \"10002\": [16, 120, 176], \"10003\": [208, 208, 208], \"10004\": [32, 256, 384]}, \"repo_range\": {\"10000\": [48, 78, 184, 214, 256, 286], \"10001\": [48, 78, 184, 214, 256, 286], \"10002\": [48, 78, 184, 214, 256, 286], \"10003\": [48, 78, 184, 214, 256, 286], \"10004\": [48, 78, 184, 214, 256, 286]}, \"cost_range\": {}, \"block_dim\": {\"10000\": 32, \"10001\": 32, \"10002\": 32, \"10003\": 32, \"10004\": 32}}";

  std::vector<std::vector<int64_t>> inputs {
    {1, 48, 1, 184, 256, 16},
    {27, 2, 16, 16, 1, 49},
  };
  std::vector<int64_t> output {1, 48, 2, 184, 256, 16};
  std::vector<std::string> input_types{"float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDC1HWC0", "FRACTAL_Z_3D"};
  std::string output_format = "NDC1HWC0";

  TeOpParas opParas;

  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = input_formats[i];

    TeOpTensorArg tensor_arg;
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
  op_compile_info.key = "Conv3d_tiling_dynamic_dhw";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "10001 48 184 256 48 184 256 ");
}

TEST_F(Conv3DTiling, Conv3d_tiling_dynamic_batch)
{
  using namespace optiling;
  std::string op_name = "Conv3D";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"_pattern\": \"conv3d\", \"dynamic_mode\": \"dynamic_batch\", \"tiling_range\": {\"10000\": [1, 31]}, \"repo_seeds\": {\"10000\": 1}, \"block_dim\": {\"10000\": 32}}";

  std::vector<std::vector<int64_t>> inputs {
    {1, 24, 2, 92, 128, 16},
    {54, 4, 16, 16, 32, 49},
  };
  std::vector<int64_t> output {1, 24, 4, 92, 128, 16};
  std::vector<std::string> input_types{"float16", "float16"};
  std::string output_dtype = "float16";
  std::vector<std::string> input_formats{"NDC1HWC0", "FRACTAL_Z_3D"};
  std::string output_format = "NDC1HWC0";

  TeOpParas opParas;

  for (size_t i = 0; i < inputs.size(); i++) {
    TeOpTensor tensor_input;
    tensor_input.shape = inputs[i];
    tensor_input.dtype = input_types[i];
    tensor_input.format = input_formats[i];

    TeOpTensorArg tensor_arg;
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
  op_compile_info.key = "Conv3d_tiling_dynamic_batch";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "10000 1 ");
}
