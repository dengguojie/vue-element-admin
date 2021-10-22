#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class ConfusionSoftmaxGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ConfusionSoftmaxGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConfusionSoftmaxGradTiling TearDown" << std::endl;
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

TEST_F(ConfusionSoftmaxGradTiling, ConfusionSoftmaxGradTiling1) {
    using namespace optiling;
    std::string op_name = "ConfusionSoftmaxGrad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = R"({"_pattern": "ConfusionSoftmaxGrad", 
                                  "_vars": {
                                  "10000000": ["dim_0_0", "dim_0_1", "dim_0_2"], 
                                  "10000002": ["dim_0_0", "dim_0_1", "dim_0_2", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"10000000": [], "10000002": []},
                                  "_attr_vars": {"10000000": [], "10000002": []},
                                  "_custom_vars": {"10000000": ["dim_0_0", "dim_0_1", "dim_0_2"],
                                                   "10000002": ["dim_0_0", "dim_0_1", "dim_0_2", "block_factor", "ub_factor"]}})";

    std::vector<int64_t> inputA{1, 2 ,32};
    std::vector<int64_t> inputB{1, 1, 32};
    std::vector<int64_t> outputA{1, 2, 32};

    std::string in_dtype1 = "float32";
    std::string in_dtype2 = "float32";
    std::string dtype = "float32";

    TeOpTensor tensor_inputA;
    tensor_inputA.shape = inputA;
    tensor_inputA.dtype = in_dtype1;
    TeOpTensor tensor_inputB;
    tensor_inputB.shape = inputB;
    tensor_inputB.dtype = in_dtype2;

    TeOpTensor tensor_outputA;
    tensor_outputA.shape = outputA;
    tensor_outputA.dtype = dtype;

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(tensor_inputA);
    tensor_argA.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_argB;
    tensor_argB.tensor.push_back(tensor_inputB);
    tensor_argB.arg_type = TA_SINGLE;

    TeOpTensorArg tensor_arg_outputA;
    tensor_arg_outputA.tensor.push_back(tensor_outputA);
    tensor_arg_outputA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.inputs.push_back(tensor_argB);
    opParas.outputs.push_back(tensor_arg_outputA);
    opParas.op_type = op_name;

    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(runInfo.tiling_key, 10000002);
    EXPECT_EQ(to_string(runInfo.tiling_data), "1 2 32 1 2 ");
    

}