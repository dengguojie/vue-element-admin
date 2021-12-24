#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class SparseApplyAdadeltaDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "SparseApplyAdadeltaDTiling SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "SparseApplyAdadeltaDTiling TearDown" << endl;
  }
};

static string to_string(const stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += to_string(tmp);
    result += " ";
  }
  return result;
}

TEST_F(SparseApplyAdadeltaDTiling, sparseapplyadadeltad_tiling_0) {
  using namespace optiling;
  string op_name = "SparseApplyAdadeltaD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SparseApplyAdadeltaD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 131072, \"indices_dsize\": 4, \"ub_take_parts\": 1, \"ub_block_num\":4, \"cache_threshold_col\":7}}";

  vector<int64_t> input1{12, 16, 32};
  vector<int64_t> input2{12, 16, 32};
  vector<int64_t> input3{12, 16, 32};
  vector<int64_t> input4{12, 16, 32};
  vector<int64_t> input5{12,};
  vector<int64_t> output1{12, 16, 32};
  vector<int64_t> output2{12, 16, 32};

  TeOpTensor tensor_input1;
  tensor_input1.shape = input1;
  tensor_input1.dtype = "float32";
  TeOpTensor tensor_input2;
  tensor_input2.shape = input2;
  tensor_input2.dtype = "float32";
  TeOpTensor tensor_input3;
  tensor_input3.shape = input3;
  tensor_input3.dtype = "float32";
  TeOpTensor tensor_input4;
  tensor_input4.shape = input4;
  tensor_input4.dtype = "float32";
  TeOpTensor tensor_input5;
  tensor_input5.shape = input5;
  tensor_input5.dtype = "int32";
  TeOpTensor tensor_output1;
  tensor_output1.shape = output1;
  tensor_output1.dtype = "float32";
  TeOpTensor tensor_output2;
  tensor_output2.shape = output2;
  tensor_output2.dtype = "float32";

  TeOpTensorArg tensor_arg1;
  tensor_arg1.tensor.push_back(tensor_input1);
  tensor_arg1.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_arg2;
  tensor_arg2.tensor.push_back(tensor_input2);
  tensor_arg2.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_arg3;
  tensor_arg3.tensor.push_back(tensor_input3);
  tensor_arg3.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_arg4;
  tensor_arg4.tensor.push_back(tensor_input4);
  tensor_arg4.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_arg5;
  tensor_arg5.tensor.push_back(tensor_input5);
  tensor_arg5.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_arg_output1;
  tensor_arg_output1.tensor.push_back(tensor_output1);
  tensor_arg_output1.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_arg_output2;
  tensor_arg_output2.tensor.push_back(tensor_output2);
  tensor_arg_output2.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_arg1);
  opParas.inputs.push_back(tensor_arg2);
  opParas.inputs.push_back(tensor_arg3);
  opParas.inputs.push_back(tensor_arg4);
  opParas.inputs.push_back(tensor_arg4);
  opParas.inputs.push_back(tensor_arg4);
  opParas.inputs.push_back(tensor_arg5);
  opParas.outputs.push_back(tensor_arg_output1);
  opParas.outputs.push_back(tensor_arg_output2);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "aa";
  OpRunInfo runInfo;

  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));

  EXPECT_EQ(to_string(runInfo.tiling_data), "12 12 512 12 12 256 256 ");
}


