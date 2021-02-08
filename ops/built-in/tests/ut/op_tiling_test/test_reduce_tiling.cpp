#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class ReduceTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "ReduceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "ReduceTiling TearDown" << std::endl;
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

TEST_F(ReduceTiling, ReduceTiling1) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"common_info": [32, 1, 8, 1, 1], "pattern_info": [5], "ub_info": [16256], "_vars": {"-1000500": ["dim_1_0", "block_factor", "ub_factor"]}})";

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};
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
  op_compile_info.key = "REDUCE__COUNTER__";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(to_string(runInfo.tiling_data), "-1000500 1 1 1 ");
}

TEST_F(ReduceTiling, ReduceTiling2) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [2], "_pattern": "CommReduce", "push_status": 0, "zero_ub_factor": 25600, "_vars": {"10": ["dim_1", "ub_factor"]}})";

  std::vector<int64_t> input{2, 39, 0};
  std::vector<int64_t> output{2, 39, 1};
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
  op_compile_info.key = "REDUCE__COUNTER__2";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(to_string(runInfo.tiling_data), "10 78 25600 ");
}

TEST_F(ReduceTiling, ReduceTiling3) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "zero_ub_factor": 32128, "_vars": {"110": ["dim_2", "ub_factor"]}})";

  std::vector<int64_t> input{2, 39, 0};
  std::vector<int64_t> output{};
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
  op_compile_info.key = "REDUCE__COUNTER__3";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(to_string(runInfo.tiling_data), "110 2 128 ");
}

TEST_F(ReduceTiling, ReduceTiling4) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0, "zero_ub_factor": 32512, "common_info": [32,1,8,1,1], "pattern_info": [1], "ub_info":[32512], "reduce_shape_known": true, "const_shape_post": true, "compile_pattern": 1, "_block_dims":{"1":32},
      "_atomic_flags":{"1": true},
      "_vars": {"1": []}})";
  std::vector<int64_t> input{64,64};
  std::vector<int64_t> output{1,64};
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
  op_compile_info.key = "REDUCE__COUNTER__4";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ReduceTiling, ReduceTiling5) {
  using namespace optiling;
  std::string op_name = "AutoTiling";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());


  std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"common_info": [32, 1, 8, 1, 1], "pattern_info": [20000], "ub_info": [16256], "_vars": {"-1000500": ["dim_1_0", "block_factor", "ub_factor"]}})";

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};
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
  op_compile_info.key = "REDUCE__COUNTER__5";
  OpRunInfo runInfo;
  ASSERT_TRUE(!iter->second(opParas, op_compile_info, runInfo));
}
