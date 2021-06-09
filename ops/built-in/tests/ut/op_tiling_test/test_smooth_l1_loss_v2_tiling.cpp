#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class SmoothL1LossV2TilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SmoothL1LossV2TilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SmoothL1LossV2TilingTest TearDown" << std::endl;
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

TEST_F(SmoothL1LossV2TilingTest, SmoothL1LossV2_Tiling_Test_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SmoothL1LossV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {100, 40},
      {100, 40}
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
  tensorOutput.shape = {100, 40};
  tensorOutput.dtype = "float32";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "SmoothL1LossV2";
  std::string compileInfo = R"({"reduction": "mean","_ori_axis": [0, 1],"reduce_mean_cof_dtype": "float32","_pattern": "CommReduce","_common_info": [32, 1, 16, 0, 1],"_pattern_info": [5, 4],"_ub_info_rf": [10752, 9088],"_ub_info": [10752, 10624],"_idx_before_reduce": 0, "_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor", "cof"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor", "cof"], "2147483647": ["_dim_1", "cof"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "cof"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "cof"]}, "_normal_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}, "_attr_vars": {"-1000500": [], "-1100500": [], "2147483647": [], "-400": [], "-100400": []}, "_custom_vars":{"-1000500":["cof"],"-1100500":["cof"],"2147483647":["cof"],"-400":["cof"],"-100400":["cof"]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 4000 1 1 964891246 ");
}

TEST_F(SmoothL1LossV2TilingTest, SmoothL1LossV2_Tiling_Test_2) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SmoothL1LossV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {100, 40},
      {100, 40}
  };

  vector<string> dtypes = {"float16", "float16"};
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
  tensorOutput.shape = {100, 40};
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "SmoothL1LossV2";
  std::string compileInfo = R"({"reduction": "sum","_ori_axis": [0, 1],"reduce_mean_cof_dtype": "float32","_pattern": "CommReduce","zero_ub_factor": 10624,"_common_info": [32, 1, 16, 0, 1],"_pattern_info": [5, 4],"_ub_info_rf": [10752, 9088],"_ub_info": [10752, 10624],"_idx_before_reduce": 0,"_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]},"_normal_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]},"_attr_vars": {"-1000500": [], "-1100500": [], "2147483647": [], "-400": [], "-100400": []},"_custom_vars":{"-1000500":[],"-1100500":[],"2147483647":[],"-400":[],"-100400":[]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456b";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 4000 1 1 ");
}

TEST_F(SmoothL1LossV2TilingTest, SmoothL1LossV2_Tiling_Test_3) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SmoothL1LossV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {100, 50, 40},
      {100, 50, 40}
  };

  vector<string> dtypes = {"float16", "float16"};
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
  tensorOutput.shape = {100, 50, 40};
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "SmoothL1LossV2";
  std::string compileInfo = R"({"reduction": "none","reduce_mean_cof_dtype": "float32","_pattern": "ElemWise","_outs_uint1": false,"_flag_info": [false, false, false, true, false, false, false],"_base_info": {"100": [261760, 4, 4, 32]},"_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000]},"_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]},"_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]},"_attr_vars": {"210000000": [], "210010000": []},"_custom_vars": {"210000000": [], "210010000": []}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456c";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "200000 200000 200000 ");
}


TEST_F(SmoothL1LossV2TilingTest, SmoothL1LossV2_Tiling_Test_4) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SmoothL1LossV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {100, 40},
      {100, 40}
  };

  vector<string> dtypes = {"float16", "float16"};
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
  tensorOutput.shape = {100, 40};
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "SmoothL1LossV2";
  std::string compileInfo = R"({"reduction": "sum","_ori_axis": [0, 1],"reduce_mean_cof_dtype": "float16","_pattern": "CommReduce","zero_ub_factor": 10624,"_common_info": [2, 1, 256, 0, 1],"_pattern_info": [5, 4],"_ub_info_rf": [17792, 15360],"_ub_info": [17792, 17536],"_idx_before_reduce": 0,"_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]},"_normal_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]},"_attr_vars": {"-1000500": [], "-1100500": [], "2147483647": [], "-400": [], "-100400": []},"_custom_vars":{"-1000500":[],"-1100500":[],"2147483647":[],"-400":[],"-100400":[]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456d";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 4000 1 1 ");
}

TEST_F(SmoothL1LossV2TilingTest, SmoothL1LossV2_Tiling_Test_5) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("SmoothL1LossV2");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {100, 40},
      {100, 40}
  };

  vector<string> dtypes = {"float16", "float16"};
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
  tensorOutput.shape = {100, 40};
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "SmoothL1LossV2";
  std::string compileInfo = R"({"reduction": "mean","_ori_axis": [0, 1],"reduce_mean_cof_dtype": "float16","_pattern": "CommReduce","_common_info": [2, 1, 256, 0, 1],"_pattern_info": [5, 4],"_ub_info_rf": [17792, 15360],"_ub_info": [17792, 17536],"_idx_before_reduce": 0, "_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor", "cof", "cof_empty"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor", "cof", "cof_empty"], "2147483647": ["_dim_1", "cof", "cof_empty"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "cof", "cof_empty"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor", "cof", "cof_empty"]}, "_normal_vars": {"-1000500": ["_dim_1", "_block_factor", "_ub_factor"], "-1100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "-400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "-100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}, "_attr_vars": {"-1000500": [], "-1100500": [], "2147483647": [], "-400": [], "-100400": []}, "_custom_vars":{"-1000500":["cof", "cof_empty"],"-1100500":["cof", "cof_empty"],"2147483647":["cof", "cof_empty"],"-400":["cof", "cof_empty"],"-100400":["cof", "cof_empty"]}})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456e";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 4000 1 1 3097 ");
}