#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class BNTrainingUpdateGradTiling : public testing::Test {
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

TEST_F(BNTrainingUpdateGradTiling, BNTrainingUpdateGradTiling4) {
    using namespace optiling;
    std::string op_name = "BNTrainingUpdateGrad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = R"({"mode": "original",
                                  "_pattern": "BNTrainingUpdateGrad", 
                                  "common_info": [32, 1, 8, 1], 
                                  "pattern_info": [134], 
                                  "max_ub_count": 7168,
                                  "_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_attr_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_custom_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]}})";

    std::vector<int64_t> inputA{2, 2, 16, 16, 16};
    std::vector<int64_t> inputB{2, 2, 16, 16, 16};
    std::vector<int64_t> inputC{1, 2, 1, 1, 16};
    std::vector<int64_t> inputD{1, 2, 1, 1, 16};
    std::vector<int64_t> outputA{1, 2, 1, 1, 16};
    std::vector<int64_t> outputB{1, 2, 1, 1, 16};

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
    op_compile_info.key = "4";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(runInfo.tiling_key, 5213400);
    EXPECT_EQ(to_string(runInfo.tiling_data), "2 2 16 16 2 1 ");
}

TEST_F(BNTrainingUpdateGradTiling, BNTrainingUpdateGradTiling1) {
    using namespace optiling;
    std::string op_name = "BNTrainingUpdateGrad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = R"({"mode": "original",
                                  "_pattern": "BNTrainingUpdateGrad", 
                                  "common_info": [32, 1, 8, 1], 
                                  "pattern_info": [134], 
                                  "max_ub_count": 7168,
                                  "_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_attr_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_custom_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]}})";

    std::vector<int64_t> inputA{32, 16, 13, 13, 16};
    std::vector<int64_t> inputB{32, 16, 13, 13, 16};
    std::vector<int64_t> inputC{1, 16, 1, 1, 16};
    std::vector<int64_t> inputD{1, 16, 1, 1, 16};
    std::vector<int64_t> outputA{1, 16, 1, 1, 16};
    std::vector<int64_t> outputB{1, 16, 1, 1, 16};

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
    op_compile_info.key = "1";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(runInfo.tiling_key, 13400);
    EXPECT_EQ(to_string(runInfo.tiling_data), "32 16 13 13 1 1 ");
}

TEST_F(BNTrainingUpdateGradTiling, BNTrainingUpdateGradTiling2) {
    using namespace optiling;
    std::string op_name = "BNTrainingUpdateGrad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = R"({"mode": "original",
                                  "_pattern": "BNTrainingUpdateGrad", 
                                  "common_info": [32, 1, 8, 1], 
                                  "pattern_info": [134], 
                                  "max_ub_count": 7168,
                                  "_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_attr_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_custom_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]}})";

    std::vector<int64_t> inputA{256, 32, 28, 28, 16};
    std::vector<int64_t> inputB{256, 32, 28, 28, 16};
    std::vector<int64_t> inputC{1, 32, 1, 1, 16};
    std::vector<int64_t> inputD{1, 32, 1, 1, 16};
    std::vector<int64_t> outputA{1, 32, 1, 1, 16};
    std::vector<int64_t> outputB{1, 32, 1, 1, 16};

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
    op_compile_info.key = "2";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(runInfo.tiling_key, 1213400);
    EXPECT_EQ(to_string(runInfo.tiling_data), "256 32 28 28 1 14 ");
}

TEST_F(BNTrainingUpdateGradTiling, BNTrainingUpdateGradTiling3) {
    using namespace optiling;
    std::string op_name = "BNTrainingUpdateGrad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = R"({"mode": "original",
                                  "_pattern": "BNTrainingUpdateGrad", 
                                  "common_info": [32, 1, 8, 1], 
                                  "pattern_info": [134], 
                                  "max_ub_count": 7168,
                                  "_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]},
                                  "_normal_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_attr_vars": {"1013400": [], "1213400": [], "1313400": [], "4013400": [], "4213400": [], 
                                  "4313400": [], "13400": [], "213400": [], "3213400": [], "2213400": [], "2313400": [], "3313400": []},
                                  "_custom_vars": {"1013400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"], 
                                  "1213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "1313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4013400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4213400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "4313400":["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "13400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2213400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "2313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"],
                                  "3313400": ["dim_0_0", "dim_0_1", "dim_0_2", "dim_0_3", "block_factor", "ub_factor"]}})";

    std::vector<int64_t> inputA{2, 2, 256, 256, 16};
    std::vector<int64_t> inputB{2, 2, 256, 256, 16};
    std::vector<int64_t> inputC{1, 2, 1, 1, 16};
    std::vector<int64_t> inputD{1, 2, 1, 1, 16};
    std::vector<int64_t> outputA{1, 2, 1, 1, 16};
    std::vector<int64_t> outputB{1, 2, 1, 1, 16};

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
    op_compile_info.key = "3";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(runInfo.tiling_key, 2213400);
    EXPECT_EQ(to_string(runInfo.tiling_data), "2 2 256 256 8 1 ");
}