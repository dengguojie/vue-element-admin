#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class ScatterAddTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScatterAddTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScatterAddTiling TearDown" << std::endl;
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

TEST_F(ScatterAddTiling, scatter_add_tiling_0) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234561";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 17);
  EXPECT_EQ(to_string(runInfo.tiling_data), "7 2 17 704 0 2112 1486848 0 704 23232 0 23232 176 88 33 1408 0 0 0 0 0 0 ");
}

TEST_F(ScatterAddTiling, scatter_add_tiling_1) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{3,2210};
  std::vector<int64_t> inputB{258,1};
  std::vector<int64_t> inputC{258,1,2210};
  std::vector<int64_t> output{3,2210};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "int32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "int32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 3);
  EXPECT_EQ(to_string(runInfo.tiling_data), "14 1 3 2210 0 258 570180 0 2210 6630 0 2210 0 0 3 0 0 0 0 0 0 0 ");
}

TEST_F(ScatterAddTiling, scatter_add_tiling_2) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{21340,1};
  std::vector<int64_t> inputB{21340,};
  std::vector<int64_t> inputC{21340,1};
  std::vector<int64_t> output{21340,1};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234567";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "16 0 32 1 0 0 0 0 0 0 0 0 0 0 21340 0 667 663 0 667 0 663 ");
}

TEST_F(ScatterAddTiling, scatter_add_tiling_3) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "12345678";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 17);
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 2 17 704 0 2112 1486848 0 704 0 0 0 0 0 33 0 0 0 0 0 0 0 ");
}

TEST_F(ScatterAddTiling, scatter_add_tiling_4) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 22};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456789";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterAddTiling, scatter_add_tiling_5) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{1};
  std::vector<int64_t> inputC{1, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "12345678b";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterAddTiling, scatter_add_tiling_6) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":253952, \"core_num\":32, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{33, 1, 32, 2, 11};
  std::vector<int64_t> inputB{11, 12, 2, 8};
  std::vector<int64_t> inputC{11, 12, 2, 8, 32, 2, 11};
  std::vector<int64_t> output{33, 1, 32, 2, 11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "12345678a";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterAddTiling, scatter_add_tiling_7) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"var_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "a1234561";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterAddTiling, scatter_add_tiling_8) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"var_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "b1234561";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterAddTiling, scatter_add_tiling_9) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "c1234561";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterAddTiling, scatter_add_tiling_10) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "d1234561";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterAddTiling, scatter_add_tiling_11) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\": 4, \"indices_size\": 4}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "e1234561";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterAddTiling, scatter_add_tiling_13) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\": 4, \"indices_size\": 4, \"support_atomic\": 0}}";

  std::vector<int64_t> inputA{};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{33,1,32,2,11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;

  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "f1234561";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterAddTiling, scatter_add_tiling_14) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"var_size\": 4, \"indices_size\": 4, \"support_atomic\": 0}}";

  std::vector<int64_t> inputA{33,1,32,2,11};
  std::vector<int64_t> inputB{11,12,2,8};
  std::vector<int64_t> inputC{11,12,2,8,1,32,2,11};
  std::vector<int64_t> output{};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;


  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "g1234561";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}
TEST_F(ScatterAddTiling, scatter_add_tiling_15) {
  using namespace optiling;
  std::string op_name = "ScatterAdd";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("ScatterAdd");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\":0, \"core_num\":0, \"support_atomic\":1, \"var_size\":4, \"indices_size\":4}}";

  std::vector<int64_t> inputA{21340,1};
  std::vector<int64_t> inputB{21340,};
  std::vector<int64_t> inputC{21340,1};
  std::vector<int64_t> output{21340,1};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "int32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg;
  tensor_arg.tensor.push_back(tensor_output);
  tensor_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "h1234567";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}