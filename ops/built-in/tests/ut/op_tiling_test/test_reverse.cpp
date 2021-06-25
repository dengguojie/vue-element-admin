#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class ReverseV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ReverseV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ReverseV2Tiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
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

TEST_F(ReverseV2Tiling, rReverseV2_tiling_0) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "2}}";

  std::vector<int64_t> input{200, 200, 200, 4};
  std::vector<int64_t> reverse_axes_shape{1};
  std::vector<int32_t> reverse_axes_value{2};
  std::vector<int64_t> output{200, 200, 200, 4};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "11 1 1 1 1 1250 200 8 0 0 0 0 0 1 0 1 1 1 1 1 1 32 0 0 0 0 0 0 0 0 19 1250 3 1 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_1) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "2}}";

  std::vector<int64_t> input{200, 200, 200, 200};
  std::vector<int64_t> reverse_axes_shape{2};
  std::vector<int32_t> reverse_axes_value{0, 2};
  std::vector<int64_t> output{200, 200, 200, 200};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234561";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 1 1 1 1 1 200 400 0 0 0 0 0 1 0 1 1 1 1 25 8 200 0 0 0 0 1 1 0 1 153 200 2 3 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_2) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "2}}";

  std::vector<int64_t> input{200, 200, 200, 23};
  std::vector<int64_t> reverse_axes_shape{1};
  std::vector<int32_t> reverse_axes_value{2};
  std::vector<int64_t> output{200, 200, 200, 23};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234562";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "2 1 1 1 1 1250 200 46 0 0 0 0 0 1 0 1 1 1 1 1 1 32 0 0 0 0 0 0 0 0 6 1250 3 1 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_3) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "2}}";

  std::vector<int64_t> input{200, 200, 200, 6400};
  std::vector<int64_t> reverse_axes_shape{1};
  std::vector<int32_t> reverse_axes_value{2};
  std::vector<int64_t> output{200, 200, 200, 6400};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234563";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "3 1 1 1 1 1 1 12800 0 0 0 0 0 0 0 1 1 1 1 32 1250 200 0 0 0 0 0 0 1 0 512 12800 1 3 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_4) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{64, 64, 64, 4};
  std::vector<int64_t> reverse_axes_shape{3};
  std::vector<int64_t> reverse_axes_value{0, 1, 3};
  std::vector<int64_t> output{66, 66, 66, 4};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int64";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 8, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234564";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "11 1 1 1 1 128 64 4 0 0 0 0 1 0 1 1 1 1 1 1 1 32 0 0 0 0 0 0 1 1 60 128 3 1 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_5) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{64, 64, 64, 129};
  std::vector<int64_t> reverse_axes_shape{3};
  std::vector<int64_t> reverse_axes_value{3};
  std::vector<int64_t> output{66, 66, 66, 129};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int64";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 8, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234564";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "5 1 1 1 1 1 8192 129 0 0 0 0 0 0 1 1 1 1 1 1 1 32 0 0 0 0 0 0 0 0 240 8192 2 1 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_tiling_6) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{64, 64, 64, 6400};
  std::vector<int64_t> reverse_axes_shape{3};
  std::vector<int64_t> reverse_axes_value{3};
  std::vector<int64_t> output{66, 66, 66, 6400};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int64";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 8, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234564";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "6 1 1 1 1 1 1 6400 0 0 0 0 0 0 1 1 1 1 1 1 32 8192 0 0 0 0 0 0 0 1 512 6400 1 2 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_only_one_dim) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{2};
  std::vector<int64_t> reverse_axes_shape{1};
  std::vector<int32_t> reverse_axes_value{0};
  std::vector<int64_t> output{2};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "01234561";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 1 1 1 1 1 1 2 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 2 2 1 0 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_only_one_dim_1) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{1};
  std::vector<int64_t> reverse_axes_shape{1};
  std::vector<int32_t> reverse_axes_value{0};
  std::vector<int64_t> output{1};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "11234561";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 ");
}
TEST_F(ReverseV2Tiling, rReverseV2_only_one_dim_scalar) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input;
  std::vector<int64_t> reverse_axes_shape{1};
  std::vector<int32_t> reverse_axes_value{0};
  std::vector<int64_t> output;

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "111234561";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 ");
}
TEST_F(ReverseV2Tiling, rReverseV2_test_1) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "1}}";

  std::vector<int64_t> input{1, 1, 31, 36};
  std::vector<int64_t> reverse_axes_shape{4};
  std::vector<int32_t> reverse_axes_value{0, 1, 2, 3};
  std::vector<int64_t> output{1, 1, 31, 36};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "111234561";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 1 1 1 1 31 1 36 0 0 0 0 1 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 31 31 3 0 ");
}

TEST_F(ReverseV2Tiling, rReverseV2_test_big_first) {
  using namespace optiling;
  std::string op_name = "ReverseV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo =
      "{\"vars\": {\"topk_threshold\": 16, \"max_elements\": 61440, \"max_elements_last_large_size\": 512, "
      "\"core_num\": 32, \"dtype_rate\": "
      "2}}";

  std::vector<int64_t> input{1, 79, 79, 3};
  std::vector<int64_t> reverse_axes_shape{1};
  std::vector<int32_t> reverse_axes_value{3};
  std::vector<int64_t> output{1, 79, 79, 3};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_reverse_axes;
  tensor_reverse_axes.shape = reverse_axes_shape;
  tensor_reverse_axes.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_reverse_axes_arg;
  tensor_reverse_axes_arg.tensor.push_back(tensor_reverse_axes);
  tensor_reverse_axes_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
      (const uint8_t*)reverse_axes_value.data(), reverse_axes_value.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_reverse_axes_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "11123456333";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "11 1 1 1 1 79 3 2 0 0 0 0 0 1 0 1 1 1 1 1 1 79 0 0 0 0 0 0 0 0 79 79 3 1 ");
}

