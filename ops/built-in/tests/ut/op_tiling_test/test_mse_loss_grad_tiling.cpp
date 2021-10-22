#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class MseLossGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MseLossGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MseLossGradTiling TearDown" << std::endl;
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

TEST_F(MseLossGradTiling, MseLossGrad_tiling_test_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("MseLossGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {16},
      {16},
      {1},
  };

  vector<string> dtypes = {"float32", "float32", "float32"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  TeOpTensor tensorOutput;
  tensorOutput.shape = {16};
  tensorOutput.dtype = "float32";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "MseLossGrad";
  std::string compileInfo = R"({"_fusion_index": [[0]], "_pattern": "Broadcast", "_outs_uint1": false, "reduce_mean_cof_dtype": "float16", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "200": [32, 2, 42320, 21152]}, "_elewise_vars": {"210000000": [20000, 30000], "210010000": [20000, 30000], "220000000": [10000, 10001, 20000, 30000]}, "push_status": 1, "_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"]}, "_normal_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"210000000": [], "210010000": [], "220000000": []}, "_custom_vars": {"210000000": [], "210010000": [], "220000000": ["cof", "cof_empty"]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456zjcc";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "16 16 16 16 12288 ");
}

TEST_F(MseLossGradTiling, MseLossGrad_tiling_test_2) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("MseLossGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {16},
      {16},
      {1},
  };

  vector<string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  TeOpTensor tensorOutput;
  tensorOutput.shape = {16};
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "MseLossGrad";
  std::string compileInfo = R"({"_fusion_index": [[0]], "_pattern": "Broadcast", "_outs_uint1": false, "reduce_mean_cof_dtype": "float16", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "200": [32, 2, 42320, 21152]}, "_elewise_vars": {"210000000": [20000, 30000], "210010000": [20000, 30000], "220000000": [10000, 10001, 20000, 30000]}, "push_status": 1, "_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"]}, "_normal_vars": {"210000000": ["_block_factor_0", "_ub_factor_0"], "210010000": ["_block_factor_0", "_ub_factor_0"], "220000000": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"210000000": [], "210010000": [], "220000000": []}, "_custom_vars": {"210000000": [], "210010000": [], "220000000": ["cof", "cof_empty"]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456zjcd";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "16 16 16 16 12288 ");
}