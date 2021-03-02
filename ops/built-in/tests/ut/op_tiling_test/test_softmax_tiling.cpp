#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class SoftmaxTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "SoftmaxTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "SoftmaxTiling TearDown" << std::endl;
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

TEST_F(SoftmaxTiling, Softmax_Tiling1) {
  using namespace optiling;
  std::string op_name = "SoftmaxV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::vector<int64_t> input{128, 128 , 128};
  std::vector<int64_t> output{128, 128 , 128};
  std::string in_dtype = "float32";

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = in_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = in_dtype;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_input);
  tensor_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg_out;
  tensor_arg_out.tensor.push_back(tensor_output);
  tensor_arg_out.arg_type = TA_SINGLE;
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_arg);
  opParas.outputs.push_back(tensor_arg_out);
  opParas.op_type = op_name;

  std::string compileInfo = R"({ "ori_axis": [-1], "common_info": [16256, 32, 1, 1]})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "SOFTMAX_";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(1, 1);
}

TEST_F(SoftmaxTiling, Softmax_Tiling2) {
  using namespace optiling;
  std::string op_name = "SoftmaxV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::vector<int64_t> input{128, 128 , 128};
  std::vector<int64_t> output{128, 128 , 128};
  std::string in_dtype = "float32";

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = in_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = in_dtype;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_input);
  tensor_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg_out;
  tensor_arg_out.tensor.push_back(tensor_output);
  tensor_arg_out.arg_type = TA_SINGLE;
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_arg);
  opParas.outputs.push_back(tensor_arg_out);
  opParas.op_type = op_name;

  std::string compileInfo = R"({ "ori_axis": [1], "common_info": [16256, 32, 1, 1]})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "SOFTMAX_";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(1, 1);
}

TEST_F(SoftmaxTiling, Softmax_Tiling3) {
  using namespace optiling;
  std::string op_name = "SoftmaxV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::vector<int64_t> input{128, 128 , 128};
  std::vector<int64_t> output{128, 128 , 128};
  std::string in_dtype = "float32";

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = in_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = in_dtype;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_input);
  tensor_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg_out;
  tensor_arg_out.tensor.push_back(tensor_output);
  tensor_arg_out.arg_type = TA_SINGLE;
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_arg);
  opParas.outputs.push_back(tensor_arg_out);
  opParas.op_type = op_name;

  std::string compileInfo = R"({ "ori_axis": [0], "common_info": [16256, 32, 1, 1]})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "SOFTMAX_";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(1, 1);
}

TEST_F(SoftmaxTiling, Softmax_Tiling4) {
  using namespace optiling;
  std::string op_name = "SoftmaxV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::vector<int64_t> input{128};
  std::vector<int64_t> output{128};
  std::string in_dtype = "float32";

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = in_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = in_dtype;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_input);
  tensor_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg_out;
  tensor_arg_out.tensor.push_back(tensor_output);
  tensor_arg_out.arg_type = TA_SINGLE;
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_arg);
  opParas.outputs.push_back(tensor_arg_out);
  opParas.op_type = op_name;

  std::string compileInfo = R"({ "ori_axis": [-1], "common_info": [16256, 32, 1, 1]})";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "SOFTMAX_";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(1, 1);
}