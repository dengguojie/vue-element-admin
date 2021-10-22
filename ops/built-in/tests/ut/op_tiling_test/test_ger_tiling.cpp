#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class GerTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GerTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GerTilingTest TearDown" << std::endl;
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

TEST_F(GerTilingTest, Ger_Tiling_Test_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Ger");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
    {1},
    {10}
  };

  vector<string> dtypes = {"float32", "float32"};
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
  tensorOutput.shape = {1, 10};
  tensorOutput.dtype = "float32";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "Ger";
  std::string compileInfo = R"({"_fusion_index":[[0,1]], "_pattern":"Broadcast", "_outs_uint1":false, "_flag_info": [false, false, true, true, true, false, false], "_base_info":{"100":[32,4,21840,10920], "320":[32,4,21840,10920]}, "_vars":{"210000000": ["_block_factor_0","_ub_factor_0"], "210010000": ["_block_factor_0","_ub_factor_0"], "232000000": ["_dim_0_1","_block_factor_0","_ub_factor_0"]}, "_normal_vars": {"210000000": ["_block_factor_0","_ub_factor_0"], "210010000": ["_block_factor_0","_ub_factor_0"], "232000000": ["_dim_0_1","_block_factor_0","_ub_factor_0"]}, "_attr_vars": {"210000000": [], "210010000": [], "232000000": []}, "_custom_vars": {"210000000": [], "210010000": [], "232000000": []}, "_elewise_vars": {"210000000": [20000,30000], "210010000": [20000,30000], "232000000": [10001,20000,30000]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "10 10 10 ");
}

TEST_F(GerTilingTest, Ger_Tiling_Test_2) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Ger");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
    {20},
    {20}
  };

  vector<string> dtypes = {"float32", "float32"};
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
  tensorOutput.shape = {20, 20};
  tensorOutput.dtype = "float32";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "Ger";
  std::string compileInfo = R"({"_fusion_index":[[0],[1]], "_pattern":"Broadcast", "_outs_uint1":false, "_flag_info": [false, false, true, true, true, false, false], "_base_info":{"320":[32,4,21840,10920], "000":[32,4,21840,10920]}, "_vars":{"232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0"], "1":["_dim_0_0","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_block_factor_1","_ub_factor_1"]}, "_normal_vars": {"232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0"], "1":["_dim_0_0","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_block_factor_1","_ub_factor_1"]}, "_attr_vars": {"232000000":[], "0":[], "1":[], "2":[], "4":[]}, "_custom_vars": {"232000000":[], "0":[], "1":[], "2":[], "4":[]}, "_elewise_vars": {"232000000":[10001,20000,30000], "0":[10000], "1":[10000,20000,30000], "2":[10000,20000,30001], "4":[10000,20001,30001]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456b";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "20 ");
}

TEST_F(GerTilingTest, Ger_Tiling_Test_3) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Ger");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
    {20},
    {20}
  };

  vector<string> dtypes = {"float32", "float32"};
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
  tensorOutput.shape = {20, 20};
  tensorOutput.dtype = "float32";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "Ger";
  std::string compileInfo = R"({"_fusion_index":[[0],[1]], "_pattern":"Broadcast", "_outs_uint1":false, "_flag_info": [false, false, true, true, true, false, false], "_base_info":{"100":[32,4,21840,10920], "320":[32,4,21840,10920], "230":[32,4,21840,10920], "000":[32,4,21840,10920]}, "_vars":{"210000000":["_block_factor_0","_ub_factor_0"], "210010000":["_block_factor_0","_ub_factor_0"], "232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "223000000":["_dim_0_0","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0","_dim_1_1"], "1":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_dim_1_1","_block_factor_1","_ub_factor_1"]}, "_normal_vars": {"210000000":["_block_factor_0","_ub_factor_0"], "210010000":["_block_factor_0","_ub_factor_0"], "232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "223000000":["_dim_0_0","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0","_dim_1_1"], "1":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_dim_1_1","_block_factor_1","_ub_factor_1"]}, "_attr_vars": {"210000000":[], "210010000":[], "232000000":[], "223000000":[], "0":[], "1":[], "2":[], "4":[]}, "_custom_vars": {"210000000":[], "210010000":[], "232000000":[], "223000000":[], "0":[], "1":[], "2":[], "4":[]}, "_elewise_vars": {"210000000":[20000,30000], "210010000":[20000,30000], "232000000":[10001,20000,30000], "223000000":[10000,20000,30000], "0":[10000,10101], "1":[10000,10101,20000,30000], "2":[10000,10101,20000,30001], "4":[10000,10101,20001,30001]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456c";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "20 20 ");
}


TEST_F(GerTilingTest, Ger_Tiling_Test_4) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Ger");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
    {1},
    {1}
  };

  vector<string> dtypes = {"float32", "float32"};
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
  tensorOutput.shape = {1, 1};
  tensorOutput.dtype = "float32";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "Ger";
  std::string compileInfo = R"({"_fusion_index":[[0],[1]], "_pattern":"Broadcast", "_outs_uint1":false, "_flag_info": [false, false, true, true, true, false, false], "_base_info":{"100":[32,4,21840,10920], "320":[32,4,21840,10920], "230":[32,4,21840,10920], "000":[32,4,21840,10920]}, "_vars":{"210000000":["_block_factor_0","_ub_factor_0"], "210010000":["_block_factor_0","_ub_factor_0"], "232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "223000000":["_dim_0_0","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0","_dim_1_1"], "1":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_dim_1_1","_block_factor_1","_ub_factor_1"]}, "_normal_vars": {"210000000":["_block_factor_0","_ub_factor_0"], "210010000":["_block_factor_0","_ub_factor_0"], "232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "223000000":["_dim_0_0","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0","_dim_1_1"], "1":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_dim_1_1","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_dim_1_1","_block_factor_1","_ub_factor_1"]}, "_attr_vars": {"210000000":[], "210010000":[], "232000000":[], "223000000":[], "0":[], "1":[], "2":[], "4":[]}, "_custom_vars": {"210000000":[], "210010000":[], "232000000":[], "223000000":[], "0":[], "1":[], "2":[], "4":[]}, "_elewise_vars": {"210000000":[20000,30000], "210010000":[20000,30000], "232000000":[10001,20000,30000], "223000000":[10000,20000,30000], "0":[10000,10101], "1":[10000,10101,20000,30000], "2":[10000,10101,20000,30001], "4":[10000,10101,20001,30001]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456d";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 ");
}

TEST_F(GerTilingTest, SmoothL1LossV2_Tiling_Test_5) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Ger");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
    {1},
    {10}
  };

  vector<string> dtypes = {"float32", "float32"};
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
  tensorOutput.shape = {1, 10};
  tensorOutput.dtype = "float32";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "Ger";
  std::string compileInfo = R"({"_fusion_index":[[0],[1]], "_pattern":"Broadcast", "_outs_uint1":false, "_flag_info": [false, false, true, true, true, false, false], "_base_info":{"320":[32,4,21840,10920], "000":[32,4,21840,10920]}, "_vars":{"232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0"], "1":["_dim_0_0","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_block_factor_1","_ub_factor_1"]}, "_normal_vars": {"232000000":["_dim_0_1","_block_factor_0","_ub_factor_0"], "0":["_dim_0_0"], "1":["_dim_0_0","_block_factor_0","_ub_factor_0"], "2":["_dim_0_0","_block_factor_0","_ub_factor_1"], "4":["_dim_0_0","_block_factor_1","_ub_factor_1"]}, "_attr_vars": {"232000000":[], "0":[], "1":[], "2":[], "4":[]}, "_custom_vars": {"232000000":[], "0":[], "1":[], "2":[], "4":[]}, "_elewise_vars": {"232000000":[10001,20000,30000], "0":[10000], "1":[10000,20000,30000], "2":[10000,20000,30001], "4":[10000,20001,30001]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456e";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "10 10 10 ");
}