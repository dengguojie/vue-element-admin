#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class SpaceToBatchNDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SpaceToBatchNDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SpaceToBatchNDTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
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

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_0) {
  using namespace optiling;
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 2}}";

  std::vector<int64_t> input{4, 2, 2, 2, 16};
  std::vector<int64_t> input_pads{2, 2};
  std::vector<int32_t> pads{1, 1, 1, 1};
  std::vector<int64_t> output{16, 2, 2, 2, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  TeOpTensor tensor_input_pads;
  tensor_input_pads.shape = input_pads;
  tensor_input_pads.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_pads_arg;
  tensor_input_pads_arg.tensor.push_back(tensor_input_pads);
  tensor_input_pads_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["paddings"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)pads.data(), pads.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_pads_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 8 1 1 4 0 2 2 0 0 1 1 1 1 0 2 2 2 16 16 0 2 2 ");
}
TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_1) {
  using namespace optiling;
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  std::vector<int64_t> input{2, 2, 2, 2, 2, 16};
  std::vector<int64_t> input_block{3};
  std::vector<int32_t> block{2, 2, 2};
  std::vector<int64_t> input_pads{2, 2};
  std::vector<int32_t> pads{1, 1, 1, 1, 1, 1};
  std::vector<int64_t> output{16, 2, 2, 2, 2, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NDC1HWC0";
  tensor_input.ori_format = "NDHWC";
  TeOpTensor tensor_input_block;
  tensor_input_block.shape = input_block;
  tensor_input_block.dtype = "int32";
  TeOpTensor tensor_input_pads;
  tensor_input_pads.shape = input_pads;
  tensor_input_pads.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NDC1HWC0";
  tensor_output.ori_format = "NDHWC";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_block_arg;
  tensor_input_block_arg.tensor.push_back(tensor_input_block);
  tensor_input_block_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_pads_arg;
  tensor_input_pads_arg.tensor.push_back(tensor_input_pads);
  tensor_input_pads_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["block_shape"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)block.data(), block.size() * 4, ge::Tensor());
  opParas.const_inputs["paddings"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)pads.data(), pads.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_block_arg);
  opParas.inputs.push_back(tensor_input_pads_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234561";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "6 2 1 1 2 2 2 2 1 1 1 1 1 1 2 2 2 2 16 16 2 2 2 ");
}
TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_2) {
  using namespace optiling;
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  std::vector<int64_t> input{4, 2, 2, 2, 16};
  std::vector<int64_t> input_block{3};
  std::vector<int32_t> block{1, 2, 2};
  std::vector<int64_t> input_pads{3, 2};
  std::vector<int32_t> pads{0, 0, 1, 1, 1, 1};
  std::vector<int64_t> output{16, 2, 2, 2, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NCHW";
  TeOpTensor tensor_input_block;
  tensor_input_block.shape = input_block;
  tensor_input_block.dtype = "int32";
  TeOpTensor tensor_input_pads;
  tensor_input_pads.shape = input_pads;
  tensor_input_pads.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NCHW";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_block_arg;
  tensor_input_block_arg.tensor.push_back(tensor_input_block);
  tensor_input_block_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_pads_arg;
  tensor_input_pads_arg.tensor.push_back(tensor_input_pads);
  tensor_input_pads_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["block_shape"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)block.data(), block.size() * 4, ge::Tensor());
  opParas.const_inputs["paddings"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)pads.data(), pads.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_block_arg);
  opParas.inputs.push_back(tensor_input_pads_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234562";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 8 1 1 4 0 2 2 0 0 1 1 1 1 0 2 2 2 16 16 0 2 2 ");
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_3) {
  using namespace optiling;
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  std::vector<int64_t> input{2, 2, 2, 2, 2, 16};
  std::vector<int64_t> input_block{4};
  std::vector<int32_t> block{1, 2, 2, 2};
  std::vector<int64_t> input_pads{4, 2};
  std::vector<int32_t> pads{0, 0, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> output{16, 2, 2, 2, 2, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NDC1HWC0";
  tensor_input.ori_format = "NCDHW";
  TeOpTensor tensor_input_block;
  tensor_input_block.shape = input_block;
  tensor_input_block.dtype = "int32";
  TeOpTensor tensor_input_pads;
  tensor_input_pads.shape = input_pads;
  tensor_input_pads.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NDC1HWC0";
  tensor_output.ori_format = "NCDHW";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_block_arg;
  tensor_input_block_arg.tensor.push_back(tensor_input_block);
  tensor_input_block_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_pads_arg;
  tensor_input_pads_arg.tensor.push_back(tensor_input_pads);
  tensor_input_pads_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["block_shape"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)block.data(), block.size() * 4, ge::Tensor());
  opParas.const_inputs["paddings"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)pads.data(), pads.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_block_arg);
  opParas.inputs.push_back(tensor_input_pads_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234563";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "6 2 1 1 2 2 2 2 1 1 1 1 1 1 2 2 2 2 16 16 2 2 2 ");
}
TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_4) {
  using namespace optiling;
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  std::vector<int64_t> input{2, 2, 7952, 1, 16};
  std::vector<int64_t> input_block{3};
  std::vector<int32_t> block{1, 1, 1};
  std::vector<int64_t> input_pads{3, 2};
  std::vector<int32_t> pads{0, 0, 0, 0, 0, 0};
  std::vector<int64_t> output{2, 2, 7952, 1, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NCHW";
  TeOpTensor tensor_input_block;
  tensor_input_block.shape = input_block;
  tensor_input_block.dtype = "int32";
  TeOpTensor tensor_input_pads;
  tensor_input_pads.shape = input_pads;
  tensor_input_pads.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NCHW";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_block_arg;
  tensor_input_block_arg.tensor.push_back(tensor_input_block);
  tensor_input_block_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_pads_arg;
  tensor_input_pads_arg.tensor.push_back(tensor_input_pads);
  tensor_input_pads_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["block_shape"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)block.data(), block.size() * 4, ge::Tensor());
  opParas.const_inputs["paddings"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)pads.data(), pads.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_block_arg);
  opParas.inputs.push_back(tensor_input_pads_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "12 32 249 233 2 0 1 1 0 0 0 0 0 0 0 2 7952 1 16 2 0 7952 1 ");
}

TEST_F(SpaceToBatchNDTiling, SpaceToBatchND_tiling_5) {
  using namespace optiling;
  std::string op_name = "SpaceToBatchND";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 0}}";

  std::vector<int64_t> input{2, 2, 1, 4487, 16};
  std::vector<int64_t> input_block{3};
  std::vector<int32_t> block{1, 1, 2};
  std::vector<int64_t> input_pads{3, 2};
  std::vector<int32_t> pads{0, 0, 0, 0, 0, 1};
  std::vector<int64_t> output{4, 2, 1, 244, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NCHW";
  TeOpTensor tensor_input_block;
  tensor_input_block.shape = input_block;
  tensor_input_block.dtype = "int32";
  TeOpTensor tensor_input_pads;
  tensor_input_pads.shape = input_pads;
  tensor_input_pads.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NCHW";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_block_arg;
  tensor_input_block_arg.tensor.push_back(tensor_input_block);
  tensor_input_block_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_pads_arg;
  tensor_input_pads_arg.tensor.push_back(tensor_input_pads);
  tensor_input_pads_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["block_shape"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)block.data(), block.size() * 4, ge::Tensor());
  opParas.const_inputs["paddings"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)pads.data(), pads.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_block_arg);
  opParas.inputs.push_back(tensor_input_pads_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456b";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 4 1 1 2 0 2 1 0 0 0 1 0 0 0 2 4487 1 16 4 0 2244 1 ");
}
