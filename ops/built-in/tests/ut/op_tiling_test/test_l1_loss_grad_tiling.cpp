#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class L1LossGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "L1LossGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "L1LossGradTiling TearDown" << std::endl;
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

TEST_F(L1LossGradTiling, L1LossGrad_tiling_test_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("L1LossGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {16, 8, 375},
      {16, 8, 375},
      {16, 8, 375},
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
  tensorOutput.shape = {16, 8, 375};
  tensorOutput.dtype = "float32";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "L1LossGrad";
  std::string compileInfo = R"({"_fusion_index": [[0], [1, 2]],"reduce_mean_cof_dtype": "float32", "_pattern": "Broadcast", "_outs_uint1": false, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 4, 8184, 4088], "210": [32, 4, 10920, 5456]}, "_elewise_vars": {"210000000": [10000, "cof", 20000, 30000], "210010000": [10000, "cof", 20000, 30000], "221000000": [10000, 10001, "cof"], "221000001": [10000, 10001, "cof", 20000, 30000], "221000002": [10000, 10001, "cof", 20000, 30001], "210000004": [10000, 10001, "cof", 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "cof"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "cof"], "221000000": ["_dim_0_0", "_dim_0_1", "cof"], "221000001": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0", "cof"], "221000002": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1", "cof"], "221000004": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1", "cof"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_0_1"], "221000001": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"210000000": [], "210010000": [], "221000000": [], "221000001": [], "221000002": [], "221000004": []}, "_custom_vars": {"210000000": ["cof"], "210010000": ["cof"], "221000000": ["cof"], "221000001": ["cof"], "221000002": ["cof"], "221000004": ["cof"]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456c";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "48000 1536 1536 1536 934200126 ");
}

TEST_F(L1LossGradTiling, L1LossGrad_tiling_test_2) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("L1LossGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {16, 8, 375},
      {16, 8, 375},
      {16, 8, 375},
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
  tensorOutput.shape = {16, 8, 375};
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "L1LossGrad";
  std::string compileInfo = R"({"_fusion_index": [[0], [1, 2]],"reduce_mean_cof_dtype": "float16", "_pattern": "Broadcast", "_outs_uint1": false, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 21840, 10912], "210": [32, 2, 21840, 10912]}, "_elewise_vars": {"210000000": [10000, "cof", "cof_empty", 20000, 30000], "210010000": [10000, "cof", "cof_empty", 20000, 30000], "221000000": [10000, 10001, "cof", "cof_empty"], "221000001": [10000, 10001, "cof", "cof_empty", 20000, 30000], "221000002": [10000, 10001, "cof", "cof_empty", 20000, 30001], "210000004": [10000, 10001, "cof", "cof_empty", 20001, 30001]}, "push_status": 1, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"], "221000000": ["_dim_0_0", "_dim_0_1", "cof", "cof_empty"], "221000001": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0", "cof", "cof_empty"], "221000002": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1", "cof", "cof_empty"], "221000004": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1", "cof", "cof_empty"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_0_1"], "221000001": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"210000000": [], "210010000": [], "221000000": [], "221000001": [], "221000002": [], "221000004": []}, "_custom_vars": {"210000000": ["cof", "cof_empty"], "210010000": ["cof", "cof_empty"], "221000000": ["cof", "cof_empty"], "221000001": ["cof", "cof_empty"], "221000002": ["cof", "cof_empty"], "221000004": ["cof", "cof_empty"]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456d";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "48000 1536 1536 1536 1536 350 ");
}