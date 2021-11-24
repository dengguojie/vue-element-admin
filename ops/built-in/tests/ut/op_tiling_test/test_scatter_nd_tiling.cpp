#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class ScatterNdTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScatterNdTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScatterNdTiling TearDown" << std::endl;
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

TEST_F(ScatterNdTiling, scatter_nd_tiling_0) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{2,5,7,3};
  std::vector<int64_t> inputB{2,5,7,11,5,7,11};
  std::vector<int64_t> inputC{7};
  std::vector<int32_t> shape{102,5,7,11,5,7,11};
  std::vector<int64_t> output{102,5,7,11,5,7,11};

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
  opParas.const_inputs["shape"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)shape.data(), shape.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "14 112 32 4235 0 210 296450 0 4235 15118950 0 4235 59290 51878 3570 474320 3 35 7 0 0 0 0 0 19 21968 17 10294 0 0 0 0 0 0 ");
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_1) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{2,5,7,3};
  std::vector<int64_t> inputB{2,5,7,11,5,7,11};
  std::vector<int64_t> inputC{7};
  std::vector<int32_t> shape{102,5,7,11,5,7,11};
  std::vector<int64_t> output{102,5,7,11,5,7};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "int32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "float32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
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
  opParas.const_inputs["shape"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)shape.data(), shape.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234567";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_2) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{2,5,7,3};
  std::vector<int64_t> inputB{2,5,6,11,5,7,11};
  std::vector<int64_t> inputC{7};
  std::vector<int32_t> shape{102,5,7,11,5,7,11};
  std::vector<int64_t> output{102,5,7,11,5,7,11};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "int32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "float32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
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
  opParas.const_inputs["shape"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)shape.data(), shape.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "12345678";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_3) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{21340,1};
  std::vector<int64_t> inputB{21340,1};
  std::vector<int64_t> inputC{2};
  std::vector<int32_t> shape{640000,1};
  std::vector<int64_t> output{640000,1};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "int32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "float32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
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
  opParas.const_inputs["shape"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)shape.data(), shape.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "123456789";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "16 0 32 1 0 0 0 0 0 0 0 0 0 0 640000 0 1 0 0 0 0 0 0 0 0 0 0 0 667 663 0 667 0 663 ");
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_4) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{31037,1};
  std::vector<int64_t> inputB{31037,256};
  std::vector<int64_t> inputC{2};
  std::vector<int32_t> shape{300000,256};
  std::vector<int64_t> output{300000,256};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "int32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "float32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
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
  opParas.const_inputs["shape"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)shape.data(), shape.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "12345678K";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "17 9375 32 256 0 0 0 0 0 0 0 0 0 0 300000 0 1 0 0 0 0 0 0 0 0 0 0 0 970 967 0 970 0 967 ");
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_5) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 0, \"core_num\": 0, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":0}}";

  std::vector<int64_t> inputA{2,5,7,3};
  std::vector<int64_t> inputB{2,5,7,11,5,7,11};
  std::vector<int64_t> inputC{7};
  std::vector<int32_t> shape{102,5,7,11,5,7,11};
  std::vector<int64_t> output{102,5,7,11,5,7,11};

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
  opParas.const_inputs["shape"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)shape.data(), shape.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "h123456";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

TEST_F(ScatterNdTiling, scatter_nd_tiling_6) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{279424,1};
  std::vector<int64_t> inputB{279424,1};
  std::vector<int64_t> inputC{2};
  std::vector<int32_t> shape{279424,1};
  std::vector<int64_t> output{279424,1};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "int32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "float32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
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
  opParas.const_inputs["shape"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)shape.data(), shape.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "scatter_nd_tiling_6";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 32);
  EXPECT_EQ(to_string(runInfo.tiling_data), "16 0 32 1 0 0 0 0 0 0 0 0 0 0 279424 0 1 0 0 0 0 0 0 0 0 0 0 0 8732 8732 0 8732 0 8732 ");
}
TEST_F(ScatterNdTiling, scatter_nd_tiling_7) {
  using namespace optiling;
  std::string op_name = "ScatterNd";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ScatterNd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32, \"updates_size\":4, \"indices_size\":4, \"support_atomic\":1}}";

  std::vector<int64_t> inputA{21340,1,2};
  std::vector<int64_t> inputB{21340,1};
  std::vector<int64_t> inputC{2};
  std::vector<int32_t> shape{640000,1};
  std::vector<int64_t> output{640000,1};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "int32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "float32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "int32";
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
  opParas.const_inputs["shape"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)shape.data(), shape.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.outputs.push_back(tensor_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234567890";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(to_string(runInfo.tiling_data), "3 0 1 1 1 10936 21340 0 21340 0 0 0 0 0 640000 0 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}