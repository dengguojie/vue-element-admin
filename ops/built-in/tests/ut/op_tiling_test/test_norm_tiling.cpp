#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class NormTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "NormTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "NormTiling TearDown" << std::endl;
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

TEST_F(NormTiling, NormTiling1) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 1, 16080, 16120], "_workspace_info": {"_workspace_type": [1, 0, 0], "_workspace_bytes": [4, 4, 4]}, "_vars": {"400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}})";

  std::vector<int64_t> input{2, 10496, 41};
  std::vector<int64_t> output{2, 10496, 41};
  std::string in_dtype = "float16";

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
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "NormTiling1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "20992 41 656 328 ");
}

TEST_F(NormTiling, NormTiling2) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [2], "_pattern": "Norm", "_common_info": [32, 8, 1, 12896, 12896], "_workspace_info": {"_workspace_type": [0], "_workspace_bytes": [4]}, "_vars": {"100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}})";

  std::vector<int64_t> input{16, 5, 15003};
  std::vector<int64_t> output{16, 5, 15003};
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
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "NormTiling2";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 10);
  EXPECT_EQ(to_string(runInfo.tiling_data), "80 15003 8 7502 ");
}

TEST_F(NormTiling, NormTiling3) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "Norm", "_common_info": [32, 8, 1, 16336, 16360], "_workspace_info": {"_workspace_type": [1, 0, 0], "_workspace_bytes": [4, 4, 4]}, "_vars": {"2100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}})";

  std::vector<int64_t> input{16, 5, 15003};
  std::vector<int64_t> output{16, 5, 15003};
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
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "NormTiling3";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "16 5 15003 7504 2 ");
}

TEST_F(NormTiling, NormTiling4) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "Norm", "_common_info": [32, 8, 1, 16336, 16360], "_workspace_info": {"_workspace_type": [1, 0, 0], "_workspace_bytes": [4, 4, 4]}, "_vars": {"1000500": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}})";

  std::vector<int64_t> input{31, 2400};
  std::vector<int64_t> output{31, 2400};
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
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "NormTiling4";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 30);
  EXPECT_EQ(to_string(runInfo.tiling_data), "31 2400 80 31 ");
}

TEST_F(NormTiling, NormTiling5) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [0, 2], "_pattern": "Norm", "_common_info": [32, 8, 1, 16216, 16248], "_workspace_info": {"_workspace_type": [1, 0, 0], "_workspace_bytes": [4, 4, 4]}, "_vars": {"21001200": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}})";

  std::vector<int64_t> input{1968, 3, 3};
  std::vector<int64_t> output{1968, 3, 3};
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
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "NormTiling5";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(to_string(runInfo.tiling_data), "1968 3 3 3 677 ");
}
