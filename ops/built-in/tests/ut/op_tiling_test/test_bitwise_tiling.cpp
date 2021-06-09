#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class BitwiseTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BitwiseTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BitwiseTiling TearDown" << std::endl;
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

TEST_F(BitwiseTiling, Bitwise_tiling_test_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("BitwiseAnd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {3, 3, 3},
      {3, 3, 3},
  };

  vector<string> dtypes = {"int16", "int16"};
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
  tensorOutput.shape = {3, 3, 3};
  tensorOutput.dtype = "int16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "BitwiseAnd";
  std::string compileInfo = R"( {"_fusion_index": [[0], [1], [2]], "push_status": 0, "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 2, 32752, 16368]}, "_elewise_vars": {"0": [10000, 10001], "1": [10000, 10001, 20000, 30000], "2": [10000, 10001, 20000, 30001], "3": [10000, 10001, 20000, 30002], "5": [10000, 10001, 20001, 30001], "6": [10000, 10001, 20001, 30002], "9": [10000, 10001, 20002, 30002]}, "_vars": {"0": ["_dim_0_0", "_dim_0_1"], "1": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_block_factor_2", "_ub_factor_2"]}, "_normal_vars": {"0": ["_dim_0_0", "_dim_0_1"], "1": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_block_factor_2", "_ub_factor_2"]}, "_attr_vars": {"0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": []}, "_custom_vars": {"0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": []}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "3 3 ");
}

