#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class IouTilingTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "IouTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "IouTilingTest TearDown" << std::endl;
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

TEST_F(IouTilingTest, iou_tiling_test_1) {
  using namespace optiling;
  std::string op_name = "Iou";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"full_core_num\": 32, \"product\": true}}";
  std::vector<int64_t> input0{32,4};
  std::vector<int64_t> input1{32,4};
  std::vector<int64_t> output{32,32};

  TeOpTensor tensor_input0;
  tensor_input0.shape = input0;
  tensor_input0.dtype = "float16";
  tensor_input0.format = "ND";
  tensor_input0.ori_format = "ND";
  TeOpTensor tensor_input1;
  tensor_input1.shape = input1;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "ND";
  tensor_input1.ori_format = "ND";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_input_arg0;
  tensor_input_arg0.tensor.push_back(tensor_input0);
  tensor_input_arg0.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_arg1;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;
  
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg0);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "iou_32_4_32_4";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 16 0 2 4096 32 32 0 64 0 ");
}

TEST_F(IouTilingTest, iou_tiling_test_2) {
  using namespace optiling;
  std::string op_name = "Iou";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"full_core_num\": 32, \"product\": true}}";
  std::vector<int64_t> input0{32,4};
  std::vector<int64_t> input1{4128,4};
  std::vector<int64_t> output{32,4128};

  TeOpTensor tensor_input0;
  tensor_input0.shape = input0;
  tensor_input0.dtype = "float16";
  tensor_input0.format = "ND";
  tensor_input0.ori_format = "ND";
  TeOpTensor tensor_input1;
  tensor_input1.shape = input1;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "ND";
  tensor_input1.ori_format = "ND";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_input_arg0;
  tensor_input_arg0.tensor.push_back(tensor_input0);
  tensor_input_arg0.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_arg1;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;
  
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg0);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "iou_32_4_4128_4";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 129 0 32 4096 32 4128 0 128 0 ");
}

TEST_F(IouTilingTest, iou_tiling_test_3) {
  using namespace optiling;
  std::string op_name = "Iou";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"full_core_num\": 32, \"product\": true}}";
  std::vector<int64_t> input0{451143,4};
  std::vector<int64_t> input1{10,4};
  std::vector<int64_t> output{451143,10};

  TeOpTensor tensor_input0;
  tensor_input0.shape = input0;
  tensor_input0.dtype = "float16";
  tensor_input0.format = "ND";
  tensor_input0.ori_format = "ND";
  TeOpTensor tensor_input1;
  tensor_input1.shape = input1;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "ND";
  tensor_input1.ori_format = "ND";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_input_arg0;
  tensor_input_arg0.tensor.push_back(tensor_input0);
  tensor_input_arg0.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_arg1;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;
  
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg0);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "iou_451143_4_10_4";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 14112 13671 32 4096 451143 10 3 7296 49152 ");
}

TEST_F(IouTilingTest, iou_tiling_test_4) {
  using namespace optiling;
  std::string op_name = "Iou";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"full_core_num\": 32, \"product\": true}}";
  std::vector<int64_t> input0{1025,4};
  std::vector<int64_t> input1{25,4};
  std::vector<int64_t> output{1025,25};

  TeOpTensor tensor_input0;
  tensor_input0.shape = input0;
  tensor_input0.dtype = "float16";
  tensor_input0.format = "ND";
  tensor_input0.ori_format = "ND";
  TeOpTensor tensor_input1;
  tensor_input1.shape = input1;
  tensor_input1.dtype = "float16";
  tensor_input1.format = "ND";
  tensor_input1.ori_format = "ND";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "ND";
  tensor_output.ori_format = "ND";

  TeOpTensorArg tensor_input_arg0;
  tensor_input_arg0.tensor.push_back(tensor_input0);
  tensor_input_arg0.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_arg1;
  tensor_input_arg1.tensor.push_back(tensor_input1);
  tensor_input_arg1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;
  
  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg0);
  opParas.inputs.push_back(tensor_input_arg1);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "iou_1025_4_25_4";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 48 17 22 4096 1025 25 0 192 0 ");
}