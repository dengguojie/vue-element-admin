#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class GatherNdTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GatherNdTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GatherNdTiling TearDown" << std::endl;
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

TEST_F(GatherNdTiling, gather_nd_tiling_0) {
  using namespace optiling;
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{87552};
  std::vector<int64_t> inputB{174,1};
  std::vector<int64_t> output{174};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;


  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 2 0 87 0 0 32000 87 64000 32000 0 87 0 1 1 87552 0 0 19 1 0 0 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_1) {
  using namespace optiling;
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{5,10,13,31};
  std::vector<int64_t> inputB{2,4};
  std::vector<int64_t> output{2};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 1 0 2 0 0 4800 2 38400 4800 0 2 0 1 4 20150 0 0 19 4030 403 31 1 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_2) {
  using namespace optiling;
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{7,6,81,6,32};
  std::vector<int64_t> inputB{2,6,3};
  std::vector<int64_t> output{2,6,6,32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;


  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "3 12 0 1 0 0 10666 1 333 10 32 1 0 192 3 653184 0 0 19 93312 15552 192 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_3) {
  using namespace optiling;
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{81,6,32};
  std::vector<int64_t> inputB{1};
  std::vector<int64_t> output{6,32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "5 1 0 1 0 0 19200 1 200 0 96 1 0 192 1 15552 0 0 19 192 0 0 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_4) {
  using namespace optiling;
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{81,600,310};
  std::vector<int64_t> inputB{1};
  std::vector<int64_t> output{600,310};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "9 32 0 1 0 0 0 0 0 0 0 55952 1 186000 1 15066000 0 0 19 186000 0 0 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_5) {
  using namespace optiling;
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{800,600,320};
  std::vector<int64_t> inputB{4,2};
  std::vector<int64_t> output{4,320};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 4 0 1 0 0 16000 1 200 0 80 1 0 320 2 153600000 0 0 19 192000 320 0 0 0 0 0 0 ");
}

TEST_F(GatherNdTiling, gather_nd_tiling_6) {
  using namespace optiling;
  std::string op_name = "GatherNd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GatherNd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":4, \"params_dsize\":2}}";

  std::vector<int64_t> inputA{800,611,1111};
  std::vector<int64_t> inputB{2,2};
  std::vector<int64_t> output{2,1111};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float16";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.ori_shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;

  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "6 2 0 1 0 0 16000 1 57 40 280 1 0 1111 2 543056800 0 0 19 678821 1111 0 0 0 0 0 0 ");
}