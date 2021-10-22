#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class InTopKTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "InTopKTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "InTopKTiling TearDown" << std::endl;
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

TEST_F(InTopKTiling, InTopK_tiling1) {
  using namespace optiling;
  std::string op_name = "InTopKD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"mini_cloud_core_nums\": 32}}";

  std::vector<int64_t> inputx1{56, 84};
  std::string x1_dtype = "float32";
  std::vector<int64_t> inputx2{56, };
  std::string x2_dtype = "int32";
  std::vector<int64_t> output{56, };
  std::string output_dtype = "bool";

  TeOpTensor tensor_inputx1;
  tensor_inputx1.shape = inputx1;
  tensor_inputx1.dtype = x1_dtype;
  TeOpTensor tensor_inputx2;
  tensor_inputx2.shape = inputx2;
  tensor_inputx2.dtype = x2_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;

  TeOpParas opParas;
  TeOpTensorArg tensor_argx1;
  tensor_argx1.tensor.push_back(tensor_inputx1);
  tensor_argx1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argx2;
  tensor_argx2.tensor.push_back(tensor_inputx2);
  tensor_argx2.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  opParas.inputs.push_back(tensor_argx1);
  opParas.inputs.push_back(tensor_argx2);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "InTopK_tiling1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 2);
  EXPECT_EQ(to_string(runInfo.tiling_data), "56 84 2 ");
}

TEST_F(InTopKTiling, InTopK_tiling2) {
  using namespace optiling;
  std::string op_name = "InTopKD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"mini_cloud_core_nums\": 32}}";

  std::vector<int64_t> inputx1{16, 84};
  std::string x1_dtype = "float32";
  std::vector<int64_t> inputx2{16, };
  std::string x2_dtype = "int32";
  std::vector<int64_t> output{16, };
  std::string output_dtype = "bool";

  TeOpTensor tensor_inputx1;
  tensor_inputx1.shape = inputx1;
  tensor_inputx1.dtype = x1_dtype;
  TeOpTensor tensor_inputx2;
  tensor_inputx2.shape = inputx2;
  tensor_inputx2.dtype = x2_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;

  TeOpParas opParas;
  TeOpTensorArg tensor_argx1;
  tensor_argx1.tensor.push_back(tensor_inputx1);
  tensor_argx1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argx2;
  tensor_argx2.tensor.push_back(tensor_inputx2);
  tensor_argx2.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  opParas.inputs.push_back(tensor_argx1);
  opParas.inputs.push_back(tensor_argx2);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "InTopK_tiling2";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(to_string(runInfo.tiling_data), "16 84 1 ");
}

TEST_F(InTopKTiling, InTopK_tiling3) {
  using namespace optiling;
  std::string op_name = "InTopKD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"mini_cloud_core_nums\": 32}}";

  std::vector<int64_t> inputx1{1156, 844};
  std::string x1_dtype = "float32";
  std::vector<int64_t> inputx2{1156, };
  std::string x2_dtype = "int32";
  std::vector<int64_t> output{1156, };
  std::string output_dtype = "bool";

  TeOpTensor tensor_inputx1;
  tensor_inputx1.shape = inputx1;
  tensor_inputx1.dtype = x1_dtype;
  TeOpTensor tensor_inputx2;
  tensor_inputx2.shape = inputx2;
  tensor_inputx2.dtype = x2_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;

  TeOpParas opParas;
  TeOpTensorArg tensor_argx1;
  tensor_argx1.tensor.push_back(tensor_inputx1);
  tensor_argx1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argx2;
  tensor_argx2.tensor.push_back(tensor_inputx2);
  tensor_argx2.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  opParas.inputs.push_back(tensor_argx1);
  opParas.inputs.push_back(tensor_argx2);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "InTopK_tiling3";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1156 844 32 ");
}

TEST_F(InTopKTiling, InTopK_tiling4) {
  using namespace optiling;
  std::string op_name = "InTopKD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"mini_cloud_core_nums\": 32}}";

  std::vector<int64_t> inputx1{56, 18444};
  std::string x1_dtype = "float32";
  std::vector<int64_t> inputx2{56, };
  std::string x2_dtype = "int32";
  std::vector<int64_t> output{56, };
  std::string output_dtype = "bool";

  TeOpTensor tensor_inputx1;
  tensor_inputx1.shape = inputx1;
  tensor_inputx1.dtype = x1_dtype;
  TeOpTensor tensor_inputx2;
  tensor_inputx2.shape = inputx2;
  tensor_inputx2.dtype = x2_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;

  TeOpParas opParas;
  TeOpTensorArg tensor_argx1;
  tensor_argx1.tensor.push_back(tensor_inputx1);
  tensor_argx1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argx2;
  tensor_argx2.tensor.push_back(tensor_inputx2);
  tensor_argx2.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  opParas.inputs.push_back(tensor_argx1);
  opParas.inputs.push_back(tensor_argx2);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "InTopK_tiling4";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 2);
  EXPECT_EQ(to_string(runInfo.tiling_data), "56 18444 2 ");
}

TEST_F(InTopKTiling, InTopK_tiling5) {
  using namespace optiling;
  std::string op_name = "InTopKD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"mini_cloud_core_nums\": 32}}";

  std::vector<int64_t> inputx1{72, 2048};
  std::string x1_dtype = "float32";
  std::vector<int64_t> inputx2{72, };
  std::string x2_dtype = "int32";
  std::vector<int64_t> output{72, };
  std::string output_dtype = "bool";

  TeOpTensor tensor_inputx1;
  tensor_inputx1.shape = inputx1;
  tensor_inputx1.dtype = x1_dtype;
  TeOpTensor tensor_inputx2;
  tensor_inputx2.shape = inputx2;
  tensor_inputx2.dtype = x2_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;

  TeOpParas opParas;
  TeOpTensorArg tensor_argx1;
  tensor_argx1.tensor.push_back(tensor_inputx1);
  tensor_argx1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argx2;
  tensor_argx2.tensor.push_back(tensor_inputx2);
  tensor_argx2.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  opParas.inputs.push_back(tensor_argx1);
  opParas.inputs.push_back(tensor_argx2);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "InTopK_tiling5";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 3);
  EXPECT_EQ(to_string(runInfo.tiling_data), "72 2048 3 ");
}


TEST_F(InTopKTiling, InTopK_tiling6) {
  using namespace optiling;
  std::string op_name = "InTopKD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"mini_cloud_core_nums\": 32}}";

  std::vector<int64_t> inputx1{795, 35208};
  std::string x1_dtype = "float32";
  std::vector<int64_t> inputx2{795, };
  std::string x2_dtype = "int32";
  std::vector<int64_t> output{795, };
  std::string output_dtype = "bool";

  TeOpTensor tensor_inputx1;
  tensor_inputx1.shape = inputx1;
  tensor_inputx1.dtype = x1_dtype;
  TeOpTensor tensor_inputx2;
  tensor_inputx2.shape = inputx2;
  tensor_inputx2.dtype = x2_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;

  TeOpParas opParas;
  TeOpTensorArg tensor_argx1;
  tensor_argx1.tensor.push_back(tensor_inputx1);
  tensor_argx1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argx2;
  tensor_argx2.tensor.push_back(tensor_inputx2);
  tensor_argx2.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  opParas.inputs.push_back(tensor_argx1);
  opParas.inputs.push_back(tensor_argx2);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "InTopK_tiling6";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 25);
  EXPECT_EQ(to_string(runInfo.tiling_data), "795 35208 25 ");
}

TEST_F(InTopKTiling, InTopK_tiling7) {
  using namespace optiling;
  std::string op_name = "InTopKD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"mini_cloud_core_nums\": 32}}";

  std::vector<int64_t> inputx1{1312, 4};
  std::string x1_dtype = "float32";
  std::vector<int64_t> inputx2{1312, };
  std::string x2_dtype = "int32";
  std::vector<int64_t> output{1312, };
  std::string output_dtype = "bool";

  TeOpTensor tensor_inputx1;
  tensor_inputx1.shape = inputx1;
  tensor_inputx1.dtype = x1_dtype;
  TeOpTensor tensor_inputx2;
  tensor_inputx2.shape = inputx2;
  tensor_inputx2.dtype = x2_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;

  TeOpParas opParas;
  TeOpTensorArg tensor_argx1;
  tensor_argx1.tensor.push_back(tensor_inputx1);
  tensor_argx1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argx2;
  tensor_argx2.tensor.push_back(tensor_inputx2);
  tensor_argx2.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  opParas.inputs.push_back(tensor_argx1);
  opParas.inputs.push_back(tensor_argx2);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "InTopK_tiling7";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1312 4 32 ");
}

TEST_F(InTopKTiling, InTopK_tiling8) {
  using namespace optiling;
  std::string op_name = "InTopKD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"mini_cloud_core_nums\": 32}}";

  std::vector<int64_t> inputx1{958, 9857};
  std::string x1_dtype = "float32";
  std::vector<int64_t> inputx2{958, };
  std::string x2_dtype = "int32";
  std::vector<int64_t> output{958, };
  std::string output_dtype = "bool";

  TeOpTensor tensor_inputx1;
  tensor_inputx1.shape = inputx1;
  tensor_inputx1.dtype = x1_dtype;
  TeOpTensor tensor_inputx2;
  tensor_inputx2.shape = inputx2;
  tensor_inputx2.dtype = x2_dtype;
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = output_dtype;

  TeOpParas opParas;
  TeOpTensorArg tensor_argx1;
  tensor_argx1.tensor.push_back(tensor_inputx1);
  tensor_argx1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argx2;
  tensor_argx2.tensor.push_back(tensor_inputx2);
  tensor_argx2.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  opParas.inputs.push_back(tensor_argx1);
  opParas.inputs.push_back(tensor_argx2);
  opParas.outputs.push_back(tensor_arg);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "InTopK_tiling8";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 30);
  EXPECT_EQ(to_string(runInfo.tiling_data), "958 9857 30 ");
}

