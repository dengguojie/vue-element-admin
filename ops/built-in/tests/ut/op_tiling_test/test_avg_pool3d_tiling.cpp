#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class AvgPool3DTiling : public testing::Test
{
 protected:
  static void SetUpTestCase()
  {
    std::cout << "AvgPool3DTiling SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "AvgPool3DTiling TearDown" << std::endl;
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

TEST_F(AvgPool3DTiling, avg_pool3d_tiling_dynamic_w)
{
  using namespace optiling;
  std::string op_name = "AvgPool3D";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"_pattern\": \"conv3d\", \"push_status\": 0, \"tiling_type\": \"dynamic_tiling\", \"repo_seeds\": {\"10000\": [32, 16, 56, 56]}, \"repo_range\": {\"10000\": [32, 32, 16, 16, 56, 56, 24, 456]}, \"cost_range\": {}, \"block_dim\": {\"10000\": 32}, \"_vars\": {\"10000\": [\"fmap_w\", \"w_out\"]}}";

  std::vector<std::vector<int64_t>> inputs {
    {32, 16, 1, 56, 56, 16},
    {27, 2, 16, 16, 1, 49},
  };
  std::vector<int64_t> output {32, 15, 1, 56, 56, 16};
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
  op_compile_info.key = "AvgPool3D_tiling_dynamic_w";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(runInfo.tiling_key, 10000);
  EXPECT_EQ(to_string(runInfo.tiling_data), "56 56 ");
}

TEST_F(AvgPool3DTiling, avg_pool3d_tiling_dynamic_batch_invalid_dim)
{
  using namespace optiling;
  std::string op_name = "AvgPool3D";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"_pattern\": \"conv3d\", \"push_status\": 0, \"fmap_c1\": 233, \"tiling_type\": \"dynamic_tiling\", \"repo_seeds\": {\"10000\": 32}, \"tiling_range\": {\"10000\": [1, 35]}, \"block_dim\": {\"10000\": 32}, \"_vars\": {\"10000\": [\"batch_n\"]}}";

  std::vector<std::vector<int64_t>> inputs {
    {32, 16, 1, 56, 56},
    {27, 2, 16, 16, 1, 49},
  };
  std::vector<int64_t> output {32, 15, 1, 56, 56, 16};
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
  op_compile_info.key = "Conv3d_tiling_dynamic_batch_invalid_C";

  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}