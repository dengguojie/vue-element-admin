#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class PadDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "PadDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PadDTiling TearDown" << std::endl;
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

TEST_F(PadDTiling, pad_d_tiling_0) {
  using namespace optiling;
  std::string op_name = "PadD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("PadD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"padding\":[1,2,3,4]}}";

  TeOpParas opParas;
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(PadDTiling, pad_d_tiling_1) {
  using namespace optiling;
  std::string op_name = "PadD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("PadD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"padding\":[[1], [2], [3]], \"length\":3}}";
  std::vector<int64_t> inputA{10, 20, 32};
  std::vector<int64_t> outputA{10, 20, 32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_outputA;
  tensor_outputA.shape = outputA;
  tensor_outputA.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_outputA);
  tensor_arg.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "padtest1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 2 32 7 160 2 1 640 160 4 704 640 64 0 0 640 0 7 9 0 32 10 0 32 64 0 16 64 0 32 640 0 16 640 0 0 0 0 6400 6400 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0 160 160 32 160 160 32 1 5 32 1 5 32 0 0 3 0 0 0 ");
}
TEST_F(PadDTiling, pad_d_tiling_2) {
  using namespace optiling;
  std::string op_name = "PadD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("PadD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"padding\":[[1], [2], [3]], \"length\":3}}";
  std::vector<int64_t> inputA{10, 20, 32};
  std::vector<int64_t> outputA{10, 20, 32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_outputA;
  tensor_outputA.shape = outputA;
  tensor_outputA.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_outputA);
  tensor_arg.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "padtest2";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(PadDTiling, pad_d_tiling_3) {
  using namespace optiling;
  std::string op_name = "PadD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("PadD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"padding\":[[1], [2], [3]], \"length\":3}}";
  std::vector<int64_t> inputA{10, 20, 32};
  std::vector<int64_t> outputA{10, 20, 32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_outputA;
  tensor_outputA.shape = outputA;
  tensor_outputA.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_outputA);
  tensor_arg.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "padtest3";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(PadDTiling, pad_d_tiling_4) {
  using namespace optiling;
  std::string op_name = "PadD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("PadD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"padding\":[[1], [2], [3]], \"length\":3}}";
  std::vector<int64_t> inputA{10, 20, 32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "padtest4";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
