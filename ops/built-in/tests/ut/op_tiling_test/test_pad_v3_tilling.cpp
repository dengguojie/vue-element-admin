#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class PadV3Tiling : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "PadV3Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PadV3Tiling TearDown" << std::endl;
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

TEST_F(PadV3Tiling, rpad_v3_tiling_0) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 1, \"mode\": \"constant\", \"padding_contiguous\": false}}";

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
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 1 1 1 1 1 16777216 0 0 0 0 0 0 0 0 0 0 0 0 -1 ");
}

TEST_F(PadV3Tiling, rpad_v3_tiling_1) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 2, \"mode\": \"constant\", \"padding_contiguous\": false}}";

  std::vector<int64_t> input{1, 512, 40, 10};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 1, 1, 0, 0, 1, 1};
  std::vector<int64_t> output{1, 512, 42, 12};

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
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 1 1 1 512 40 20 0 0 0 0 0 0 0 0 1 1 2 2 2 ");
}

TEST_F(PadV3Tiling, rpad_v3_tiling_2) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 2, \"mode\": \"constant\", \"padding_contiguous\": false}}";

  std::vector<int64_t> input{1, 512, 40, 1000};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 1, 1, 0, 0, 1, 1};
  std::vector<int64_t> output{1, 512, 42, 1002};

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
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 1 1 512 40 2000 0 0 0 0 0 0 0 0 1 1 2 2 1 ");
}

TEST_F(PadV3Tiling, rpad_v3_tiling_3) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 2, \"mode\": \"constant\", \"padding_contiguous\": false}}";

  std::vector<int64_t> input{1, 512, 40, 10000};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 1, 1, 0, 0, 1, 1};
  std::vector<int64_t> output{1, 512, 42, 10002};

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
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 1 1 512 40 20000 0 0 0 0 0 0 0 0 1 1 2 2 1 ");
}

TEST_F(PadV3Tiling, rpad_v3_tiling_4) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 2, \"mode\": \"constant\", \"padding_contiguous\": true}}";

  std::vector<int64_t> input{400, 1000};
  std::vector<int64_t> padding_shape{4};
  std::vector<int32_t> padding_value{1, 1, 1, 1};
  std::vector<int64_t> output{402, 1002};

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
  op_compile_info.key = "1234564";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 1 1 1 400 2000 0 0 0 0 0 0 0 0 1 1 2 2 1 ");
}

TEST_F(PadV3Tiling, rpad_v3_tiling_5) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 2, \"mode\": \"constant\", \"padding_contiguous\": true}}";

  std::vector<int64_t> input{400, 100000};
  std::vector<int64_t> padding_shape{4};
  std::vector<int32_t> padding_value{1, 1, 1, 1};
  std::vector<int64_t> output{402, 100002};

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
  op_compile_info.key = "1234565";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 1 1 1 1 400 200000 0 0 0 0 0 0 0 0 1 1 2 2 0 ");
}

TEST_F(PadV3Tiling, rpad_v3_tiling_6) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"mode\": \"reflect\", \"padding_contiguous\": false}}";

  std::vector<int64_t> input{64, 64, 64, 64};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 1, 1, 0, 0, 1, 1};
  std::vector<int64_t> output{64, 64, 66, 66};

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
  op_compile_info.key = "1654321";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 64 64 64 64 64 64 66 66 32 1 1 1 1 128 128 ");
}

TEST_F(PadV3Tiling, rpad_v3_tiling_7) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"mode\": \"reflect\", \"padding_contiguous\": true}}";

  std::vector<int64_t> input{2, 512, 10, 10};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<int64_t> output{2, 512, 12, 12};

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
  op_compile_info.key = "1654322";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 2 512 10 10 2 512 12 12 32 1 1 1 1 32 32 ");
}

TEST_F(PadV3Tiling, rpad_v3_tiling_8) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"mode\": \"reflect\", \"padding_contiguous\": false}}";

  std::vector<int64_t> input{2, 512, 10, 211};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 1, 1, 0, 0, 1, 1};
  std::vector<int64_t> output{2, 512, 12, 213};

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
  op_compile_info.key = "1654323";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 2 512 10 211 2 512 12 213 32 1 1 1 1 32 32 ");
}

TEST_F(PadV3Tiling, rpad_v3_tiling_9) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 1, \"dtype_rate\": 1, \"mode\": \"constant\", \"padding_contiguous\": false}}";

  std::vector<int64_t> input{1, 1280, 8, 8};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 1, 1, 0, 0, 1, 1};
  std::vector<int64_t> output{1, 1280, 10, 10};

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
  op_compile_info.key = "2234220";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "5 1 1 1 1280 8 8 0 0 0 0 0 0 0 0 1 1 1 1 -1 ");
}

TEST_F(PadV3Tiling, rpad_v3_tiling_10) {
  using namespace optiling;
  std::string op_name = "PadV3";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 65536, \"core_num\": 1, \"dtype_rate\": 1, \"mode\": \"constant\", \"padding_contiguous\": false}}";

  std::vector<int64_t> input{1, 1280, 8, 8};
  std::vector<int64_t> padding_shape{8};
  std::vector<int32_t> padding_value{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> output{1, 1280, 8, 8};

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
  op_compile_info.key = "2234231";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 1 1 1 1280 8 8 0 0 0 0 0 0 0 0 0 0 0 0 -1 ");
}
