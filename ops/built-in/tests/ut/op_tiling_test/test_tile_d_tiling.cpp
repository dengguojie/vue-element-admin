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
  std::string compileInfo = R"({ "_pattern": "Broadcast", "_only_const_tiling": false, "_flag_info": [true, false, false, false, 1], "_base_info": {"000": [262144, 4, 2, 32]}, "_elewise_vars": {"0":[1010], "1":[1010,2000,3000], "2":[1010, 2000, 3001], "4": [1010, 2001, 3001]}, "_vars": {"0":["dim_1_0"], "1": ["dim_1_0", "block_factor_0", "ub_factor_0"], "2": ["dim_1_0", "block_factor_0", "ub_factor_1"],  "4": ["dim_1_0", "block_factor_1", "ub_factor_1"]}, "_compile_shape": [-1], "_origin_multiples": [2048]})";

  std::vector<int64_t> inputA{32};
  std::vector<int64_t> output{2048, 32};
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
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 32 64 64 ");
}

