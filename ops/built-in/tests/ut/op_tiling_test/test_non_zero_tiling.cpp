#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class NonZeroTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NonZeroTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NonZeroTiling TearDown" << std::endl;
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

TEST_F(NonZeroTiling, NonZero_tiling_test_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NonZero");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputArg, tensorOutputsArg;
  TeOpParas opParas;

  vector<int64_t> inputx{16, 16};

  tensorInputArg.tensor.clear();
  TeOpTensor tensorInput;
  tensorInput.shape = inputx;
  tensorInput.dtype = "float16";
  tensorInputArg.tensor.push_back(tensorInput);
  tensorInputArg.arg_type = TensorArgType::TA_SINGLE;
  opParas.inputs.push_back(tensorInputArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = inputx;
  tensorOutput.dtype = "int32";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TensorArgType::TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "NonZero";
  std::string compileInfo = "{\"block_dim\": 32, \"workspace\":[4096]}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "NonZero_tiling_test_1";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  // EXPECT_EQ(to_string(runInfo.tiling_data),
  //           "2 12 ");
}

