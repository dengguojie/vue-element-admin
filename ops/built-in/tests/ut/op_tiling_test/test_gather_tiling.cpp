#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class GatherTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GatherTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GatherTiling TearDown" << std::endl;
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

TEST_F(GatherTiling, gather_tiling_0) {
  using namespace optiling;
  std::string op_name = "Gather";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Gather");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, "
                            "\"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{87552,};
  std::vector<int64_t> inputB{174,1};
  std::vector<int64_t> output{174,1};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
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
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "gather_tiling_0";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "13 1 87552 1 174 0 8 0 21 6 0 32512 21 65024 "
                                            "32512 0 65024 21 0 87552 0 0 0 0 1 1 0 1 ");
}

TEST_F(GatherTiling, gather_tiling_1) {
  using namespace optiling;
  std::string op_name = "Gather";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("Gather");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
                            "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":1}}";

  std::vector<int64_t> inputA{55, 32, 16};
  std::vector<int64_t> inputB{55, 6};
  std::vector<int64_t> output{55, 6, 16};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
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
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "gather_tiling_1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "29 1 32 16 330 0 32 0 6 138 0 6 0 "
                                            "2464 6 0 0 0 0 512 0 0 0 0 6 1 23 55 ");
}
