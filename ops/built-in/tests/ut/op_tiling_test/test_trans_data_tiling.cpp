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
  tensorInput.format = "NCHW";
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "NC1HWC0";								  
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo1 = "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo1;
  op_compile_info.key = "123456a";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "100 63232 1 1 0 0 1 1 1 49 0 1 1 1 49 0 240 240 0 49 16 784 784 0 1 1 784 0 0 0 1 1 784 0 0 0 49 1 16 0 0 0 0 3952 ");
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
  tensorInput.format = "NHWC";							  
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "NC1HWC0";							  
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
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
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
  tensorInput.format = "ND";						
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "FRACTAL_NZ";									 
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
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
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
  tensorInput.format = "NC1HWC0";								 
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "NHWC";							   
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
  tensorInput.format = "FRACTAL_NZ";									
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.dtype = dtype;
  tensorOutput.format = "ND";							 
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
  tensorInput.format = "FRACTAL_Z_3D";									  
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "NDHWC";								
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

TEST_F(TransDataTiling, TransData_tiling7) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {2, 2, 49, 49, 16};
  std::vector<int64_t> output_shape = {2, 30, 49, 49};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "NC1HWC0";								 
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "NCHW";							   
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"NC1HWC0\", \"dstFormat\": \"NCHW\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = "123456abcdefg";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "200 63232 0 2 11 0 0 1 2 1 0 0 2 1 2 1 1 0 2 240 1 16 1 38416 2401 72030 38416 38416 14 1 0 0 3952 2 1 76832 1 0 0 2401 1 16 1 0 0 ");
}

TEST_F(TransDataTiling, TransData_tiling8) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {8, 32, 2, 16, 16};
  std::vector<int64_t> output_shape = {8, 2, 2, 16, 16, 16};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "NCDHW";							   
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "NDC1HWC0";								   
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = "123456abcdefgh";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "100 63232 1 8 0 0 3 2 1 32 0 3 2 1 32 0 240 240 0 512 16 8192 4096 0 8 1 16384 0 0 0 8 1 16384 0 0 0 256 1 16 2 256 8192 0 3952 ");
}

TEST_F(TransDataTiling, TransData_tiling9) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {3, 34, 34, 16, 2};
  std::vector<int64_t> output_shape = {3468, 1, 16, 16};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "DHWCN";
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "FRACTAL_Z_3D";
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"DHWCN\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = "123456abcdefghi";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "100 63232 1 32 0 0 1 1 109 2 0 1 1 89 2 0 240 240 0 2 16 32 295936 0 1156 1 32 3 1156 36992 1156 1 256 3 1156 295936 2 1 16 0 0 0 0 3952 ");
}

TEST_F(TransDataTiling, TransData_tiling10) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {79, 79, 23, 31};
  std::vector<int64_t> output_shape = {12482, 2, 16, 16};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "HWCN";							  
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "FRACTAL_Z";									
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"HWCN\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = "23456abcdefghi";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "100 63232 1 32 0 0 1 2 196 31 7 1 2 165 31 7 240 240 0 31 16 496 3195392 7 6241 1 713 0 0 0 6241 1 512 0 0 0 31 1 16 0 0 0 0 3952 ");
}

TEST_F(TransDataTiling, TransData_tiling_NCHW2NHWC) {
  using namespace optiling;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpParas opParas;
  vector<int64_t> inShape = {100, 17, 16, 16};
  vector<int64_t> outShape = {100, 16, 16, 17};

  TeOpTensorArg tensorInputs;
  TeOpTensor tensorInput;
  tensorInput.shape = inShape;
  tensorInput.format = "NCHW";
  tensorInput.dtype = "float16";
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  TeOpTensorArg tensorOutputsArg;
  TeOpTensor tensorOutput;
  tensorOutput.shape = outShape;
  tensorOutput.format = "NHWC";
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);

  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  opParas.op_type = "TransData";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(TransDataTiling, TransData_tiling_NCHW2HWCN) {
  using namespace optiling;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpParas opParas;
  vector<int64_t> inShape = {100, 17, 16, 16};
  vector<int64_t> outShape = {16, 16, 17, 100};

  TeOpTensorArg tensorInputs;
  TeOpTensor tensorInput;
  tensorInput.shape = inShape;
  tensorInput.format = "NCHW";
  tensorInput.dtype = "float16";
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  TeOpTensorArg tensorOutputsArg;
  TeOpTensor tensorOutput;
  tensorOutput.shape = outShape;
  tensorOutput.format = "HWCN";
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);

  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  opParas.op_type = "TransData";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(TransDataTiling, TransData_tiling_NHWC2NCHW) {
  using namespace optiling;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpParas opParas;
  vector<int64_t> inShape = {100, 17, 16, 16};
  vector<int64_t> outShape = {100, 16, 17, 16};

  TeOpTensorArg tensorInputs;
  TeOpTensor tensorInput;
  tensorInput.shape = inShape;
  tensorInput.format = "NHWC";
  tensorInput.dtype = "float16";
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  TeOpTensorArg tensorOutputsArg;
  TeOpTensor tensorOutput;
  tensorOutput.shape = outShape;
  tensorOutput.format = "NCHW";
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);

  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  opParas.op_type = "TransData";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(TransDataTiling, TransData_tiling_NHWC2HWCN) {
  using namespace optiling;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpParas opParas;
  vector<int64_t> inShape = {100, 17, 16, 16};
  vector<int64_t> outShape = {17, 16, 16, 100};

  TeOpTensorArg tensorInputs;
  TeOpTensor tensorInput;
  tensorInput.shape = inShape;
  tensorInput.format = "NHWC";
  tensorInput.dtype = "float16";
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  TeOpTensorArg tensorOutputsArg;
  TeOpTensor tensorOutput;
  tensorOutput.shape = outShape;
  tensorOutput.format = "HWCN";
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);

  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  opParas.op_type = "TransData";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(TransDataTiling, TransData_tiling_HWCN2NCHW) {
  using namespace optiling;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpParas opParas;
  vector<int64_t> inShape = {100, 17, 16, 16};
  vector<int64_t> outShape = {16, 16, 100, 17};

  TeOpTensorArg tensorInputs;
  TeOpTensor tensorInput;
  tensorInput.shape = inShape;
  tensorInput.format = "HWCN";
  tensorInput.dtype = "float16";
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  TeOpTensorArg tensorOutputsArg;
  TeOpTensor tensorOutput;
  tensorOutput.shape = outShape;
  tensorOutput.format = "NCHW";
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);

  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  opParas.op_type = "TransData";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(TransDataTiling, TransData_tiling_HWCN2NHWC) {
  using namespace optiling;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpParas opParas;
  vector<int64_t> inShape = {100, 17, 16, 16};
  vector<int64_t> outShape = {16, 100, 17, 16};

  TeOpTensorArg tensorInputs;
  TeOpTensor tensorInput;
  tensorInput.shape = inShape;
  tensorInput.format = "HWCN";
  tensorInput.dtype = "float16";
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  TeOpTensorArg tensorOutputsArg;
  TeOpTensor tensorOutput;
  tensorOutput.shape = outShape;
  tensorOutput.format = "NHWC";
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);

  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  opParas.op_type = "TransData";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(TransDataTiling, TransData_tiling_CHWN2NCHW) {
  using namespace optiling;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpParas opParas;
  vector<int64_t> inShape = {100, 17, 16, 16};
  vector<int64_t> outShape = {16, 100, 17, 16};

  TeOpTensorArg tensorInputs;
  TeOpTensor tensorInput;
  tensorInput.shape = inShape;
  tensorInput.format = "CHWN";
  tensorInput.dtype = "float16";
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  TeOpTensorArg tensorOutputsArg;
  TeOpTensor tensorOutput;
  tensorOutput.shape = outShape;
  tensorOutput.format = "NCHW";
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);

  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  opParas.op_type = "TransData";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(TransDataTiling, TransData_tiling_CHWN2NHWC) {
  using namespace optiling;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpParas opParas;
  vector<int64_t> inShape = {100, 17, 16, 16};
  vector<int64_t> outShape = {16, 17, 16, 100};

  TeOpTensorArg tensorInputs;
  TeOpTensor tensorInput;
  tensorInput.shape = inShape;
  tensorInput.format = "CHWN";
  tensorInput.dtype = "float16";
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  TeOpTensorArg tensorOutputsArg;
  TeOpTensor tensorOutput;
  tensorOutput.shape = outShape;
  tensorOutput.format = "NHWC";
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);

  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  opParas.op_type = "TransData";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(TransDataTiling, TransData_tiling_CHWN2HWCN) {
  using namespace optiling;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpParas opParas;
  vector<int64_t> inShape = {100, 17, 16, 16};
  vector<int64_t> outShape = {17, 16, 100, 16};

  TeOpTensorArg tensorInputs;
  TeOpTensor tensorInput;
  tensorInput.shape = inShape;
  tensorInput.format = "CHWN";
  tensorInput.dtype = "float16";
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  TeOpTensorArg tensorOutputsArg;
  TeOpTensor tensorOutput;
  tensorOutput.shape = outShape;
  tensorOutput.format = "HWCN";
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);

  std::string compileInfo = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = this->test_info_->name();

  opParas.op_type = "TransData";

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}
