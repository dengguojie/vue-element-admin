#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class TransDataTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TransDataTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TransDataTiling TearDown" << std::endl;
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

static string to_string_int64(const std::stringstream &tiling_data) {
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

TEST_F(TransDataTiling, TransData_tiling1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {1, 16, 7, 7};
  std::vector<int64_t> output_shape = {1, 1, 7, 7, 16};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo1 = "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo1;
  op_compile_info.key = "123456a";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 68 100 63488 1 784 784 3968 0 0 0 0 784 784 784 784 49 0 16 3968 49 1 3968 0 0 0 0 1 0 0 1 63488 0 0 1 0 1 0 1 0 1 0 1 49 0 1 49 1 1 1 784 1 0 1 0 1 0 1 0 1 49 0 1 49 1 1 1 784 784 784 ");
}

TEST_F(TransDataTiling, TransData_tiling2) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {2, 35, 68, 3};
  std::vector<int64_t> output_shape = {2, 1, 35, 68, 16};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NHWC\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = "123456ab";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 77 101 63488 32 225 1200 3968 0 0 0 0 7140 38080 11904 63488 3968 9443840 16 3968 744 248 3 752 1 744 0 0 1 38080 1 0 0 1 63488 0 0 1 0 1 0 2 0 1 75 1 3 1 75 3 752 1 744 225 1 1 1 1200 1 0 1 0 2 0 1 55 1 3 1 55 3 752 1 744 165 1 1 1 880 ");
}

TEST_F(TransDataTiling, TransData_tiling3) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {2, 17, 10, 11095};
  std::vector<int64_t> output_shape = {2, 17, 694, 1, 16, 16};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_NZ\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = "123456abc";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 77 101 63488 29 384 6144 3968 0 0 0 0 110950 177664 177520 256 3968 63488 16 3968 11095 1 3968 3968 1 3968 0 0 248 256 1 0 0 1 256 0 0 1 0 1 0 34 0 1 10 1 384 10 1 384 384 1 384 384 24 1 1 160 1 0 1 0 34 0 1 10 1 343 10 1 343 352 1 343 343 22 1 1 160 ");
}


//      negative     case               //
TEST_F(TransDataTiling, TransData_tiling4) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {2, 2, 1, 1, 16};
  std::vector<int64_t> output_shape = {2, 1, 1, 31};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"NC1HWC0\", \"dstFormat\": \"NHWC\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = "123456abcd";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "201 63232 1 2 0 0 1 1 1 1 2 1 1 1 1 2 16 256 31 16 8 128 128 4 15 2 1 32 0 0 0 0 0 0 2 1 31 0 0 0 0 0 0 0 ");
}

TEST_F(TransDataTiling, TransData_tiling5) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {100, 3, 7, 16, 16};
  std::vector<int64_t> output_shape = {100, 107, 42};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"FRACTAL_NZ\", \"dstFormat\": \"ND\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = "123456abcde";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "201 63232 1 25 0 0 7 1 4 11 3 7 1 4 11 3 16 256 42 1792 8 14336 128 2 10 100 1 5376 0 0 0 0 0 0 100 1 4494 0 0 0 0 0 0 0 ");
}

TEST_F(TransDataTiling, TransData_tiling6) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {100, 2, 16, 16};
  std::vector<int64_t> output_shape = {19, 5, 1, 5, 63};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"FRACTAL_Z_3D\", \"dstFormat\": \"NDHWC\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = "123456abcdef";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "201 63232 1 25 0 0 2 1 1 3 4 2 1 1 3 4 16 256 1575 2560 8 20480 128 2 15 5 1 512 5 5 10240 0 0 0 5 1 63 5 5 315 0 0 0 1 ");
}