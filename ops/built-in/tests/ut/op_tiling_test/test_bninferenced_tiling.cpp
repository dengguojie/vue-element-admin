#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class BNInferenceDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BNInferenceDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BNInferenceDTiling TearDown" << std::endl;
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

TEST_F(BNInferenceDTiling, BNInferenceDTiling_test_1) {

  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("BNInferenceD");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {2, 8, 2, 2},
      {8},
      {8},
      {1},
      {8}
  };

  vector<string> dtypes = {
    "float16",
    "float16",
    "float16",
    "float16",
    "float16"
  };

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
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "BNInferenceD";

  std::string compileInfo = R"({"broadcast_mean_shape": [1, 8, 1, 1], "push_status": 0, "_fusion_index": [[0], [1], [2], [4]], "_pattern": "Broadcast", "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [2, 4, 15872, 7936]}, "_elewise_vars": {"0": [10000, 10200], "1": [10000, 10200, 20000, 30000], "2": [10000, 10200, 20000, 30001], "3": [10000, 10200, 20000, 30002], "4": [10000, 10200, 20000, 30003], "6": [10000, 10200, 20001, 30001], "7": [10000, 10200, 20001, 30002], "8": [10000, 10200, 20001, 30003], "11": [10000, 10200, 20002, 30002], "12": [10000, 10200, 20002, 30003], "16": [10000, 10200, 20003, 30003]}, "_vars": {"0": ["_dim_0_0", "_dim_2_0"], "1": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_2_0", "_block_factor_0", "_ub_factor_3"], "6": ["_dim_0_0", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "7": ["_dim_0_0", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "8": ["_dim_0_0", "_dim_2_0", "_block_factor_1", "_ub_factor_3"], "11": ["_dim_0_0", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "12": ["_dim_0_0", "_dim_2_0", "_block_factor_2", "_ub_factor_3"], "16": ["_dim_0_0", "_dim_2_0", "_block_factor_3", "_ub_factor_3"]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "BNInferenceDTiling_test_1";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "BNInferenceDTiling tiling_data:" << to_string(runInfo.tiling_data) << std::endl;

  EXPECT_EQ(to_string(runInfo.tiling_data), "2 2 ");
}
