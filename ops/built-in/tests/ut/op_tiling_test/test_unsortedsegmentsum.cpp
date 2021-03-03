#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class UnsortedSegmentSumTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UnsortedSegmentSumTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UnsortedSegmentSumTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
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

TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_0) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,3132864};
  std::vector<int64_t> inputB{2};
  std::vector<int64_t> inputC{1};
  std::vector<int64_t> output{1,3132864};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

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
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "5 2 3132864 1 0 0 16384 3520 0 0 1 0 0 16384 3520 0 0 3132864 1 0 0 16384 3520 0 0 1 0 0 16384 3520 0 0 3132864 192 2048 16384 440 3520 2 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_1) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,80};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int64_t> output{300,80};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

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
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 32 2560 1 320 320 2560 2560 32 32 1 320 320 2560 2560 32 32 2560 1 320 320 2560 2560 32 32 1 320 320 2560 2560 32 32 80 1 10 80 10 80 1024 32 1 4 4 32 32 32 1 4 4 32 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}


