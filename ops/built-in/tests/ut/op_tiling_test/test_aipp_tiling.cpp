#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class AippTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AippTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AippTiling TearDown" << std::endl;
  }
};

/*
 * be careful of the to_string fuction
 * the type of tiling_data in other ops is int64 while int32 here
 */
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

TEST_F(AippTiling, aipp_tiling_0) {
  using namespace optiling;
  std::string op_name = "Aipp";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Aipp");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

  TeOpTensor tensor_iamges;
  tensor_iamges.shape = {1,3,256,256};
  tensor_iamges.dtype = "uint8";
  TeOpTensor tensor_params;
  tensor_params.shape = {3168};
  tensor_params.dtype = "uint8";
  TeOpTensor tensor_features;
  tensor_features.shape = {1,1,224,224,16};
  tensor_features.dtype = "float16";

  TeOpTensorArg tensor_argImages;
  tensor_argImages.tensor.push_back(tensor_iamges);
  tensor_argImages.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_argParams;
  tensor_argParams.tensor.push_back(tensor_params);
  tensor_argParams.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_features);
  tensor_arg.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argImages);
  opParas.inputs.push_back(tensor_argParams);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "12345666";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 1 224 224 16 1 1 ");
}

