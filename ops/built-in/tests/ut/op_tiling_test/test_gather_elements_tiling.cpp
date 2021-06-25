#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class GatherElementsTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GatherElementsTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GatherElementsTiling TearDown" << std::endl;
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

TEST_F(GatherElementsTiling, gather_elements_tiling_0) {
  using namespace optiling;
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2, \"axis\":0}}";

  std::vector<int64_t> inputA{87552,};
  std::vector<int64_t> inputB{174,1};

  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{174,1};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 87552 1 174 1 174 0 0 0 174 87552 0 0 ");
}

TEST_F(GatherElementsTiling, gather_elements_tiling_1) {
  using namespace optiling;
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2, \"axis\":0}}";

  std::vector<int64_t> inputA{81,6,3};
  std::vector<int64_t> inputB{6,};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{6,};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";

  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 1 81 18 6 1 6 0 0 0 6 1458 0 0 ");
}

TEST_F(GatherElementsTiling, gather_elements_tiling_2) {
  using namespace optiling;
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2, \"axis\":0}}";

  std::vector<int64_t> inputA{81,30,32};
  std::vector<int64_t> inputB{30,20};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{30,20};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";

  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "6 1 81 960 600 32 18 24 0 32512 18 77760 0 3 ");
}

TEST_F(GatherElementsTiling, gather_elements_tiling_3) {
  using namespace optiling;
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2, \"axis\":0}}";

  std::vector<int64_t> inputA{81,30,32};
  std::vector<int64_t> inputB{16,16};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{16,16};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";

  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "3 1 81 960 256 32 8 0 0 32512 8 77760 0 0 ");
}

TEST_F(GatherElementsTiling, gather_elements_tiling_4) {
  using namespace optiling;
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 30, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2, \"axis\":0}}";

  std::vector<int64_t> inputA{64,64,12};
  std::vector<int64_t> inputB{16,16};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{16,16};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";

  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 1 64 768 256 32 8 0 0 32512 8 49152 0 0 ");
}

TEST_F(GatherElementsTiling, gather_elements_tiling_5) {
  using namespace optiling;
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 30, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2, \"axis\":0}}";

  std::vector<int64_t> inputA{64,17,17};
  std::vector<int64_t> inputB{16,17};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{16,17};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";

  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "5 1 64 289 272 32 8 16 0 32512 8 18496 0 2 ");
}

TEST_F(GatherElementsTiling, gather_elements_tiling_6) {
  using namespace optiling;
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 30, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2, \"axis\":0}}";

  std::vector<int64_t> inputA{64,64,64};
  std::vector<int64_t> inputB{16,17};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{16,17};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";

  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["axis"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)axis.data(), axis.size() * 4, ge::Tensor());

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "6 1 64 4096 272 32 8 16 0 32512 8 262144 0 2 ");
}