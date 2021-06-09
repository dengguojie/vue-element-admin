#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class TileDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TileDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TileDTiling TearDown" << std::endl;
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

TEST_F(TileDTiling, TileD_tiling1) {
  using namespace optiling;
  std::string op_name = "TileD";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  // dynamic_tile_d_llt_case_1
  std::string compileInfo = R"({ "_pattern": "Broadcast", "push_status": 0, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32768, 16384]}, "_elewise_vars": {"1":[10200, 20000, 30000], "2":[10200, 20000, 30001], "3":[10200, 20000, 30002], "5": [10200, 20001, 30001], "6":[10200, 20001, 30002], "9":[10200, 20002, 30002]}, "_vars": {"1":["_dim_2_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_2_0", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_2_0", "_block_factor_0", "_ub_factor_2"],  "5": ["_dim_2_0", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_2_0", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_2_0", "_block_factor_2", "_ub_factor_2"]}, "tiling_info": [1, 0, 1, 1, -1, 42763, 16, -1]})";

  std::vector<int64_t> inputA{777};
  std::vector<int64_t> output{42763, 16, 777};
  std::string in_dtype = "float32";
  std::string dtype = "float32";

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = in_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = dtype;
  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "TileD_tiling1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "777 1337 2 ");
}
