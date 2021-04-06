#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class BNTrainingUpdateGradDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BNTrainingUpdateGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BNTrainingUpdateGradTiling TearDown" << std::endl;
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

TEST_F(BNTrainingUpdateGradDTiling, BNTrainingUpdateGradDTiling1) {
    using namespace optiling;
    std::string op_name = "BNTrainingUpdateGrad";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = R"({"_pattern": "bn_update_grad", 
                                  "common_info": [32, 1, 8, 0], 
                                  "pattern_info": [134], 
                                  "max_ub_count": 13107,
                                  "_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"1013400": []},
                                  "_attr_vars": {"1013400": []},
                                  "_custom_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]}})";

    std::vector<int64_t> inputA{32, 64, 7, 7, 16};
    std::vector<int64_t> inputB{32, 64, 7, 7, 16};
    std::vector<int64_t> inputC{1, 64, 1, 1, 16};
    std::vector<int64_t> inputD{1, 64, 1, 1, 16};
    std::vector<int64_t> outputA{1, 64, 1, 1, 16};
    std::vector<int64_t> outputB{1, 64, 1, 1, 16};

    std::string in_dtype1 = "float32";
    std::string in_dtype2 = "float32";
    std::string dtype = "float32";

    TeOpTensor tensor_inputA;
    tensor_inputA.shape = inputA;
    tensor_inputA.dtype = in_dtype1;
    TeOpTensor tensor_inputB;
    tensor_inputB.shape = inputB;
    tensor_inputB.dtype = in_dtype1;
    TeOpTensor tensor_inputC;
    tensor_inputC.shape = inputC;
    tensor_inputC.dtype = in_dtype2;
    TeOpTensor tensor_inputD;
    tensor_inputD.shape = inputD;
    tensor_inputD.dtype = in_dtype2;

    TeOpTensor tensor_outputA;
    tensor_outputA.shape = outputA;
    tensor_outputA.dtype = dtype;
    TeOpTensor tensor_outputB;
    tensor_outputB.shape = outputB;
    tensor_outputB.dtype = dtype;

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(tensor_inputA);
    tensor_argA.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_argB;
    tensor_argB.tensor.push_back(tensor_inputB);
    tensor_argB.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_argC;
    tensor_argC.tensor.push_back(tensor_inputC);
    tensor_argC.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_argD;
    tensor_argD.tensor.push_back(tensor_inputD);
    tensor_argD.arg_type = TA_SINGLE;

    TeOpTensorArg tensor_arg_outputA;
    tensor_arg_outputA.tensor.push_back(tensor_outputA);
    tensor_arg_outputA.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_arg_outputB;
    tensor_arg_outputB.tensor.push_back(tensor_outputB);
    tensor_arg_outputB.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.inputs.push_back(tensor_argB);
    opParas.inputs.push_back(tensor_argC);
    opParas.inputs.push_back(tensor_argD);
    opParas.outputs.push_back(tensor_arg_outputA);
    opParas.outputs.push_back(tensor_arg_outputB);
    opParas.op_type = op_name;

    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(runInfo.tiling_key, 1013400);
    EXPECT_EQ(to_string(runInfo.tiling_data), "32 64 7 7 2 16 ");
    

}