#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class GatherV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GatherV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GatherV2Tiling TearDown" << std::endl;
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

TEST_F(GatherV2Tiling, gather_v2_tiling_0) {
  using namespace optiling;
  std::string op_name = "GatherV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{87552,};
  std::vector<int64_t> inputB{174,1};
  std::vector<int64_t> inputC{};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{174,1};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "13 1 87552 1 174 0 8 0 21 6 0 32512 21 65024 32512 0 65024 21 0 87552 0 0 0 0 ");
}

TEST_F(GatherV2Tiling, gather_v2_tiling_1) {
  using namespace optiling;
  std::string op_name = "GatherV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{81,6,3};
  std::vector<int64_t> inputB{6,};
  std::vector<int64_t> inputC{};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{81,6,3};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "8 81 6 3 6 0 32 0 6 0 0 19712 6 13136 6576 1 13136 6 0 1458 0 0 2 17 ");
}

TEST_F(GatherV2Tiling, gather_v2_tiling_2) {
  using namespace optiling;
  std::string op_name = "GatherV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{81,6,32};
  std::vector<int64_t> inputB{6,};
  std::vector<int64_t> inputC{};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{81,6,32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "10 81 6 32 6 0 32 0 6 0 0 19712 6 1232 0 16 1232 6 0 15552 0 0 2 17 ");
}

TEST_F(GatherV2Tiling, gather_v2_tiling_3) {
  using namespace optiling;
  std::string op_name = "GatherV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{81,6,32};
  std::vector<int64_t> inputB{6,};
  std::vector<int64_t> inputC{};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{6,6,32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "6 1 81 192 6 0 6 0 1 0 0 19712 1 205 32 96 205 1 0 15552 0 0 0 0 ");
}

TEST_F(GatherV2Tiling, gather_v2_tiling_4) {
  using namespace optiling;
  std::string op_name = "GatherV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 30, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{16,8,16,32};
  std::vector<int64_t> inputB{32,};
  std::vector<int64_t> inputC{};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{32,8,16,32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "3 1 16 4096 32 0 32 0 1 0 0 32512 1 15 7 2167 15 1 0 65536 0 0 0 0 ");
}

TEST_F(GatherV2Tiling, gather_v2_tiling_5) {
  using namespace optiling;
  std::string op_name = "GatherV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 30, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{16,8,16,32};
  std::vector<int64_t> inputB{320,};
  std::vector<int64_t> inputC{};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{320,8,16,32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "7 1 16 4096 320 0 32 0 10 0 0 32512 10 15 7 2167 15 10 0 65536 0 0 0 0 ");
}