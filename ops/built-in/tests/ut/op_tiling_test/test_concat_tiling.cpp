#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class ConcatTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ConcatTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConcatTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

TEST_F(ConcatTiling, Concat_tiling1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {5, 4, 4, 4},
      {6, 4, 4, 4},
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
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "ConcatD";
  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":0, \"input_size\":3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456a";
// do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 1 384 256 960 3 0 0 256 0 320 256 384 576 ");
}

TEST_F(ConcatTiling, Concat_tiling2) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4, 5, 4, 4},
      {4, 6, 4, 4},
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
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "ConcatD";
  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":1, \"input_size\":3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456ab";
// do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 4 96 64 240 3 0 0 64 0 80 64 96 144 ");
}

TEST_F(ConcatTiling, Concat_tiling3) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {4, 4},
      {4, 5},
      {4, 6},
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
  tensorOutput.shape = input_shapes[0];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "ConcatD";
  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":-1, \"input_size\":3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456abc";
// do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 4 6 4 15 3 0 0 4 0 5 4 6 9 ");
}
