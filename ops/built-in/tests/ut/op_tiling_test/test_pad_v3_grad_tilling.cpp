#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class PadV3GradTiling : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "PadV3GradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PadV3GradTiling TearDown" << std::endl;
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

TEST_F(PadV3GradTiling, rpad_v3_grad_tiling_0) {
  using namespace optiling;
  std::string op_name = "PadV3Grad";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  std::vector<int64_t> input{64, 64, 64, 64};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> output{64, 64, 64, 64};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_padding;
  tensor_padding.shape = padding_shape;
  tensor_padding.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

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
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 64 64 64 64 64 64 64 64 32 0 0 0 0 128 128 ");
}

TEST_F(PadV3GradTiling, rpad_v3_grad_tiling_1) {
  using namespace optiling;
  std::string op_name = "PadV3Grad";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  std::vector<int64_t> input{1, 512, 42, 12};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<int64_t> output{1, 512, 40, 10};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_padding;
  tensor_padding.shape = padding_shape;
  tensor_padding.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

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
  op_compile_info.key = "1234561";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 1 512 42 12 1 512 40 10 32 1 1 1 1 16 16 ");
}

TEST_F(PadV3GradTiling, rpad_v3_grad_tiling_2) {
  using namespace optiling;
  std::string op_name = "PadV3Grad";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  std::vector<int64_t> input{1, 512, 42, 1002};
  std::vector<int64_t> padding_shape{4};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<int64_t> output{1, 512, 40, 1000};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_padding;
  tensor_padding.shape = padding_shape;
  tensor_padding.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

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
  op_compile_info.key = "1234562";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 1 512 42 1002 1 512 40 1000 32 1 1 1 1 16 16 ");
}

TEST_F(PadV3GradTiling, rpad_v3_grad_tiling_3) {
  using namespace optiling;
  std::string op_name = "PadV3Grad";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  std::vector<int64_t> input{1, 512, 400, 1002};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<int64_t> output{1, 512, 398, 1000};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_padding;
  tensor_padding.shape = padding_shape;
  tensor_padding.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

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
  op_compile_info.key = "1234563";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 512 400 1002 1 512 398 1000 32 1 1 1 1 16 16 ");
}

TEST_F(PadV3GradTiling, rpad_v3_grad_tiling_4) {
  using namespace optiling;
  std::string op_name = "PadV3Grad";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  std::vector<int64_t> input{64, 64, 66, 66};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 2, 1, 2, 1};
  std::vector<int64_t> output{64, 64, 63, 63};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_padding;
  tensor_padding.shape = padding_shape;
  tensor_padding.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

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
  op_compile_info.key = "1654326";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 64 64 66 66 64 64 63 63 32 2 1 2 1 128 128 ");
}

TEST_F(PadV3GradTiling, rpad_v3_grad_tiling_5) {
  using namespace optiling;
  std::string op_name = "PadV3Grad";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  std::vector<int64_t> input{2, 512, 1000, 1000};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 2, 3, 2, 3};
  std::vector<int64_t> output{2, 512, 995, 995};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_padding;
  tensor_padding.shape = padding_shape;
  tensor_padding.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

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
  op_compile_info.key = "1654327";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 2 512 1000 1000 2 512 995 995 32 2 3 2 3 32 32 ");
}

TEST_F(PadV3GradTiling, rpad_v3_grad_tiling_6) {
  using namespace optiling;
  std::string op_name = "PadV3Grad";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  std::vector<int64_t> input{2, 512, 52, 964};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<int64_t> output{2, 512, 50, 962};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_padding;
  tensor_padding.shape = padding_shape;
  tensor_padding.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

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
  op_compile_info.key = "1654328";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 2 512 52 964 2 512 50 962 32 1 1 1 1 32 32 ");
}

TEST_F(PadV3GradTiling, rpad_v3_grad_tiling_7) {
  using namespace optiling;
  std::string op_name = "PadV3Grad";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  std::vector<int64_t> input{2, 512, 964, 52};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<int64_t> output{2, 512, 962, 50};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_padding;
  tensor_padding.shape = padding_shape;
  tensor_padding.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

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
  op_compile_info.key = "1654329";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 2 512 964 52 2 512 962 50 32 1 1 1 1 32 32 ");
}
