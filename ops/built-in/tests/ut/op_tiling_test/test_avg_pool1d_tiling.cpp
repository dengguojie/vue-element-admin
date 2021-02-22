#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class AvgPool1DTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPool1DTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPool1DTiling TearDown" << std::endl;
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

TEST_F(AvgPool1DTiling, avgpool1d_tiling_0) {
  using namespace optiling;
  std::string op_name = "AvgPool1DD";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("AvgPool1DD");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"core_num\":32, \"max_w_in_ub\":2730, \"ksize\":3, \"strides\":1,\"pad_l\":0,\"pad_r\":0,\"ceil_mode\":true}";

  std::vector<int64_t> inputA{1,1,1,3,16};
  std::vector<int64_t> inputB{1,1,1,1,16};
  std::vector<int64_t> output{1,1,1,1,16};

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
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

