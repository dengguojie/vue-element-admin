#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <register/op_tiling.h>

using namespace std;

class EletwiseTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "EletwiseTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "EletwiseTiling TearDown" << std::endl;
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

TEST_F(EletwiseTiling, Eletwise_tiling1) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingInterf::RegisteredOpInterf().end());

  // dynamic_op_add_267.static_op_add_269
  std::string compileInfo = R"({ "_pattern": "ElemWise", "_fusion_index": [[0], [1]], "_only_const_tiling": false, "_flag_info": [true, true, true, false, 2], "_base_info": { "320": [262144, 4, 3, 32], "000": [262144, 4, 3, 32] }, "_elewise_vars": { "232000000": [101, 200, 300], "0": [110], "1": [110, 200, 300], "2": [110, 200, 301], "4": [110, 201, 301] }, "_vars": { "232000000": ["dim_0_1", "block_factor_0", "ub_factor_0"], "0": ["dim_1_0"], "1": ["dim_1_0", "block_factor_0", "ub_factor_0"], "2": ["dim_1_0", "block_factor_0", "ub_factor_1"], "4": ["dim_1_0", "block_factor_1", "ub_factor_1"] } })";

  std::vector<int64_t> inputA{1, 5824};
  std::vector<int64_t> inputB{100, 1};
  std::vector<int64_t> output{100, 5824};
  std::string in_dtype = "float32";
  std::string dtype = "float32";

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = in_dtype;
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = in_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = dtype;
  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);

  nlohmann::json op_info = nlohmann::json::parse(compileInfo);
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(op_name, opParas, op_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 25);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 5824 4 3 ");
}

TEST_F(EletwiseTiling, Eletwise_tiling2) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingInterf::RegisteredOpInterf().end());

  // dynamic_op_exp_432.static_op_exp_433
  std::string compileInfo = R"({ "_pattern": "ElemWise", "_only_const_tiling": false, "_flag_info": [ false, true, false, false, 1 ], "_base_info": { "100": [ 262144, 4, 2, 32 ] }, "_elewise_vars": { "210000000": [ 100, 200, 300 ] }, "_vars": { "210000000": [ "dim_0_0", "block_factor_0", "ub_factor_0" ] } })";

  std::vector<int64_t> inputA{1, 33, 1089};
  std::vector<int64_t> output{1, 33, 1089};
  std::string dtype = "float32";

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = dtype;
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

  nlohmann::json op_info = nlohmann::json::parse(compileInfo);
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(op_name, opParas, op_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "210000000 35937 1128 1128 ");
}