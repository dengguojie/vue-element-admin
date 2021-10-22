#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class DynamicLSTMGradCellTilling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_lstm_grad_cell tilling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_lstm_grad_cell tilling TearDown" << std::endl;
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


TEST_F(DynamicLSTMGradCellTilling, DynamicLSTMGradCellTilling_tilling1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("DynamicLSTMGradCell");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  TeOpParas opParas;
  vector<vector<int64_t>> input_shapes = {
      {1, 1, 4, 16, 16},
      {2, 1, 4, 16, 16},
      {2, 1, 4, 16, 16},
      {1, 1, 4, 16, 16},
      {1, 1, 4, 16, 16},
      {2, 1, 4, 16, 16},
      {2, 1, 4, 16, 16},
      {2, 1, 4, 16, 16},
      {2, 1, 4, 16, 16},
      {2, 1, 4, 16, 16},
      {2, 1, 4, 16, 16},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1, 4, 4, 16, 16},
      {1, 1, 4, 16, 16},
  };
  vector<string> dtypes = {"float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "int32"};


  for (size_t i = 0; i < input_shapes.size(); i++) {
    TeOpTensorArg tensorInputArg;
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputArg.tensor.push_back(tensorInput);
    tensorInputArg.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputArg);
  }

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TeOpTensorArg tensorOutputsArg;
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputsArg.tensor.push_back(tensorOutput);
    tensorOutputsArg.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputsArg);
  }


  opParas.op_type = "DynamicLSTMGradCell";
  std::string compileInfo = "{\"vars\": {\"device_aicore_num\": 32, \"ub_size\":253952, \"mask_input\":1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456cde";
  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "2 256 2 128 0 0 128 128 16 64 4 1024 ");
}
