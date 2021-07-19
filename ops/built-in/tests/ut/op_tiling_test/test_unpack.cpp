#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class UnpackTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UnpackTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UnpackTiling TearDown" << std::endl;
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

TEST_F(UnpackTiling, Unpack_tiling1) {
  using namespace optiling;
  std::string op_name = "Unpack";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  // dynamic_tile_d_llt_case_1
  std::string compileInfo = R"({ "push_status": 0, "compile_vars": {"core_num": 32, "ub_size": 262144, "output_num": 1, "axis": 2, "is_special_tiling": false}, "vars": {"0": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"]}})";

  std::vector<int64_t> inputA{100, 1, 1, 902, 6};
  std::vector<int64_t> output{100, 1, 902, 6};
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
  op_compile_info.key = "Unpack_tiling1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "100 5412 32 16913 16913 ");
}

TEST_F(UnpackTiling, Unpack_tiling2) {
  using namespace optiling;
  std::string op_name = "Unpack";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  // dynamic_tile_d_llt_case_2
  std::string compileInfo = R"({ "push_status": 0, "compile_vars": {"core_num": 32, "ub_size": 262144, "output_num": 2, "axis": 1, "is_special_tiling": false}, "vars": {"1": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "2": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"], "3": ["left_dim", "right_dim", "left_dim_out", "right_dim_in", "split_factor"]}})";

  std::vector<int64_t> inputA{70, 2,  758, 518};
  std::vector<int64_t> output{70,  758, 518};
  std::vector<int64_t> outputB{70,  758, 518};
  std::string in_dtype = "float16";
  std::string dtype = "float16";

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = in_dtype;
  TeOpTensor tensor_output;
  TeOpTensor tensor_outputB;
  tensor_output.shape = output;
  tensor_output.dtype = dtype;
  tensor_outputB.shape = outputB;
  tensor_outputB.dtype = dtype;
  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  TeOpTensorArg tensor_argB;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_argB.tensor.push_back(tensor_outputB);
  tensor_arg.arg_type = TA_SINGLE;
  tensor_argB.arg_type = TA_SINGLE;
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.outputs.push_back(tensor_arg);
  opParas.outputs.push_back(tensor_argB);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "Unpack_tiling2";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "70 392644 32 392644 130876 ");
}