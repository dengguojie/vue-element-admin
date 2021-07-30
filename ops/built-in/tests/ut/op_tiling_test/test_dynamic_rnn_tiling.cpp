#include <stdlib.h>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class DynamicRnnTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DynamicRnnTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DynamicRnnTiling TearDown" << std::endl;
  }
};

/*
 * be careful of the to_string fuction
 * the type of tiling_data in other ops is int64 while int32 here
 */
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

TEST_F(DynamicRnnTiling, dynamic_rnn_tiling_0) {
  using namespace optiling;
  std::string op_name = "DynamicRNN";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("DynamicRNN");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[32,8,0]]}}";

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {32,32,8,16,16},
      {40,32,16,16},
      {512},
  };

  vector<std::string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  vector<vector<int64_t>> output_shapes = {
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
  };

  vector<std::string> out_dtypes = {"float16", "float16", "float16", "float16",
                           "float16", "float16", "float16", "float16"};
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputsArg.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = out_dtypes[i];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "32 8 -1 ");
}

TEST_F(DynamicRnnTiling, dynamic_rnn_tiling_1) {
  using namespace optiling;
  std::string op_name = "DynamicRNN";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("DynamicRNN");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[32,8,0]]}}";

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {32,32,8,16,16},
  };

  vector<std::string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  vector<vector<int64_t>> output_shapes = {
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
  };

  vector<std::string> out_dtypes = {"float16", "float16", "float16", "float16",
                           "float16", "float16", "float16", "float16"};
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputsArg.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = out_dtypes[i];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(DynamicRnnTiling, dynamic_rnn_tiling_2) {
  using namespace optiling;
  std::string op_name = "DynamicRNN";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("DynamicRNN");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"tune_shape_list\": [[32,8,0]]}}";

  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;

  vector<vector<int64_t>> input_shapes = {
      {-1,32,-1,16,16},
      {40,32,16,16},
      {512},
  };

  vector<std::string> dtypes = {"float16", "float16", "float16"};
  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  vector<vector<int64_t>> output_shapes = {
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
      {32,8,8,16,16},
  };

  vector<std::string> out_dtypes = {"float16", "float16", "float16", "float16",
                           "float16", "float16", "float16", "float16"};
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputsArg.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = out_dtypes[i];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}