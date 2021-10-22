#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class BinaryCrossEntropyTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BinaryCrossEntropyTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BinaryCrossEntropyTiling TearDown" << std::endl;
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

TEST_F(BinaryCrossEntropyTiling, BinaryCrossEntropy_tiling_test_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BinaryCrossEntropy");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {11},
      {11},
      {11},
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
  tensorOutput.shape = {1};
  tensorOutput.dtype = "float32";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "BinaryCrossEntropy";
  std::string compileInfo = R"({"reduction": "mean", "_ori_axis": [0], "reduce_mean_cof_dtype": "float32", "_pattern": "CommReduce", "_zero_ub_factor": 7936, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4], "_ub_info": [8064, 7936], "_ub_info_rf": [8064, 7936], "_idx_before_reduce": 0, "push_status": 1, "_var":{"-1000500": ["_dim_1","_block_factor", "_ub_factor", "cof"], "-1100500": ["_dim_1","_block_factor", "_ub_factor", "cof"], "500": ["_dim_1","_block_factor", "_ub_factor", "cof"], "100500": ["_dim_1","_block_factor", "_ub_factor", "cof"], "2147483647": ["_dim_1", "cof"], "-400": ["_dim_0", "_dim_1","_block_factor", "_ub_factor", "cof"], "-100400": ["_dim_0", "_dim_1","_block_factor", "_ub_factor", "cof"], "1000400": ["_dim_0", "_dim_1","_block_factor", "_ub_factor", "cof"], "1100400": ["_dim_0", "_dim_1","_block_factor", "_ub_factor", "cof"]}, "_normal_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}, "_attr_vars":{"-1000500": [], "-1100500": [], "500": [], "100500": [], "2147483647": [], "-400": [], "-100400": [], "1000400": [], "1100400": []}, "_custom_vars": {"-1000500": ["cof"], "-1100500": ["cof"], "500": ["cof"], "100500": ["cof"], "2147483647": ["cof"], "-400": ["cof"], "-100400": ["cof"], "1000400": ["cof"], "1100400": ["cof"]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 11 1 1 1035611788 ");
}

TEST_F(BinaryCrossEntropyTiling, BinaryCrossEntropy_tiling_test_2) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BinaryCrossEntropy");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {11},
      {11},
      {11},
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
  tensorOutput.shape = {1};
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "BinaryCrossEntropy";
  std::string compileInfo = R"({"reduction": "mean", "_ori_axis": [0], "reduce_mean_cof_dtype": "float16", "_pattern": "CommReduce", "_zero_ub_factor": 16000, "_common_info": [32, 1, 16, 0, 1], "_pattern_info": [5, 4], "_ub_info": [16256, 16000], "_ub_info_rf": [16256, 16000], "_idx_before_reduce": 0, "push_status": 1, "_var":{"-1000500": ["_dim_1","_block_factor", "_ub_factor", "cof", "cof_empty"], "-1100500": ["_dim_1","_block_factor", "_ub_factor", "cof", "cof_empty"], "2147483647": ["_dim_1", "cof", "cof_empty"], "-400": ["_dim_0", "_dim_1","_block_factor", "_ub_factor", "cof", "cof_empty"], "-100400": ["_dim_0", "_dim_1","_block_factor", "_ub_factor", "cof", "cof_empty"]}, "_normal_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}, "_attr_vars":{"-1000500": [], "-1100500": [], "2147483647": [], "-400": [], "-100400": []}, "_custom_vars": {"-1000500": ["cof", "cof_empty"], "-1100500": ["cof", "cof_empty"], "2147483647": ["cof", "cof_empty"], "-400": ["cof", "cof_empty"], "-100400": ["cof", "cof_empty"]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456b";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 11 1 1 11729 ");
}
