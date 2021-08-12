#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class MirrorPadTiling : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "MirrorPadTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MirrorPadTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

TEST_F(MirrorPadTiling, mirror_pad_tiling_0) {
  using namespace optiling;
  std::string op_name = "MirrorPad";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"output_dim4_max_cnt\": 64, \"core_num\": 2, \"dtype_rate\": 2}}";

  std::vector<int64_t> input{64, 64, 64, 128};
  std::vector<int64_t> padding_shape{4, 2};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> output{64, 64, 64, 128};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_padding;
  tensor_padding.shape = padding_shape;
  tensor_padding.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_padding_arg;
  tensor_padding_arg.tensor.push_back(tensor_padding);
  tensor_padding_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["paddings"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)padding_value.data(), padding_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_padding_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 1 64 64 64 256 0 0 0 0 0 0 0 0 0 0 2 131072 131072 ");
}

TEST_F(MirrorPadTiling, mirror_pad_tiling_1) {
  using namespace optiling;
  std::string op_name = "MirrorPad";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"output_dim4_max_cnt\": 64, \"core_num\": 2, \"dtype_rate\": 2}}";

  std::vector<int64_t> input{64, 64, 64, 4};
  std::vector<int64_t> padding_shape{4, 2};
  std::vector<int32_t> padding_value{1, 1, 1, 1, 1, 1, 0, 0};
  std::vector<int64_t> output{66, 66, 66, 4};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_padding;
  tensor_padding.shape = padding_shape;
  tensor_padding.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_padding_arg;
  tensor_padding_arg.tensor.push_back(tensor_padding);
  tensor_padding_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["paddings"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)padding_value.data(), padding_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_padding_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 64 64 64 8 0 0 1 1 1 1 1 1 0 0 1 0 4356 ");
}
