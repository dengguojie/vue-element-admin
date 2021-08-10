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
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "TransDataTilingData: " << to_string_int64(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1000 63232 0 1 784 784 3952 0 16 1 1 1 784 784 1 784 784 49 16 784 784 1 16 49 49 784 1 0 1 0 1 0 1 0 1 0 1 0 1 1 784 1 1 0 49 1 16 1 1 0 ");
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
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NHWC\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1010 63232 2 7140 38080 7140 38080 7140 38080 11856 63232 3 0 3952 1 247 16 3 38080 38080 16 3 3 1 0 1 1 10 157 1 0 1 1 1 1 10 157 1 0 ");
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
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_NZ\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  // EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            // "1 77 101 63488 29 384 6144 3968 0 0 0 0 110950 177664 177520 256 3968 63488 16 3968 11095 1 3968 3968 1 3968 0 0 248 256 1 0 0 1 256 0 0 1 0 1 0 34 0 1 10 1 384 10 1 384 384 1 384 384 24 1 1 160 1 0 1 0 34 0 1 10 1 343 10 1 343 352 1 343 343 22 1 1 160 ");
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1010 63232 17 221900 355328 110950 177664 110950 177664 177520 256 11095 0 3952 1 1 16 3952 63232 256 16 7 3952 2 0 1 1 10 1 3 3191 2 1 1 1 10 1 3 3191 ");
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
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2012 63232 0 1 1 16 64 62 1 1 1 0 0 0 1 1 1 0 0 0 1 16 31 16 31 2 1 16 1 32 32 15 2 32 31 64 62 1 1 16 1 1 0 1 3952 1 ");
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
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2011 63232 0 7 1 16 86016 71904 2 1 1 25 0 0 2 1 1 25 0 4 82 16 42 1312 3444 3 1 1792 1 5376 48 10 16 5376 4494 86016 71904 112 1 16 1 1 0 1 3952 0 ");
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
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2011 63232 0 2 0 16 256 25200 1 1 1 0 0 0 1 1 1 0 0 3 25 0 63 0 1575 4 1 2560 1 10240 64 15 16 16 1575 256 25200 5 1 512 5 5 10240 2 3952 1 ");
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
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2001 63232 0 2 16 76832 72030 1 2 1 0 0 0 1 2 1 0 0 0 2401 1 1 16 1 38416 2401 30 38416 2401 38416 38416 76832 72030 76832 72030 14 1 1 0 1 1 0 2 1 76832 1 1 0 2401 1 16 1 1 0 ");
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
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "1001 63488 0 8 16384 16384 3968 0 16 1 2 1 16384 16384 1 16384 16384 512 16 8192 4096 1 0 512 512 0 1 0 2 0 1 0 1 0 2 0 1 0 8 1 16384 1 1 0 256 1 16 2 256 8192 ");
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
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"DHWCN\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "1000 63488 0 31 3584 0 3968 0 16 2 1 1 32 0 16 512 0 2 16 32 295936 1 16 2 2 32 7 0 1 0 1 0 7 12 1 0 1 0 1156 1 256 3 1156 295936 2 1 16 1 1 0 ");
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
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"HWCN\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "1000 63488 0 31 148304 106496 3968 7 16 1 1 1 713 512 16 11408 8192 31 16 496 3195392 1 16 31 31 496 13 0 2 7 1 0 1 1 2 7 1 0 6241 1 512 1 1 0 31 1 16 1 1 0 ");
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

TEST_F(TransDataTiling, TransData_tiling11) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {2, 7, 11, 11, 16};
  std::vector<int64_t> output_shape = {7, 1, 11, 11, 1, 16, 16};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "NDHWC";
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
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NDHWC\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1011 63232 4 1 3952 0 27104 32 13552 2 3952 3952 247 0 16 30976 30976 16 0 16 1 0 1 0 1 0 1 0 1 106 1 0 121 1 256 7 121 30976 ");
}

TEST_F(TransDataTiling, TransData_tiling12) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {2, 7, 11, 11, 16};
  std::vector<int64_t> output_shape = {2, 7, 1, 11, 11, 16};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "NDHWC";
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
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NDHWC\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1010 63232 1 61952 61952 61952 61952 1936 1936 63232 63232 16 1 3952 2 247 16 16 1936 1936 16 0 16 1 0 2 1 1 121 1 0 1 7 2 1 1 121 1 0 ");
}

TEST_F(TransDataTiling, TransData_tiling13) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {20, 3, 300, 300, 16};
  std::vector<int64_t> output_shape = {270000, 2, 16, 16};
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
  tensorOutput.format = "FRACTAL_Z";							  
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NC1HWC0\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1011 63232 32 1 138320 0 69120000 256 4320000 16 3952 3952 247 0 16 138240000 138240000 16 0 16 2 4 35 0 1 0 2 4 9 29 1 0 270000 1 512 1 270000 138240000 ");
}

TEST_F(TransDataTiling, TransData_tiling_NCDHW2NDC1HWC0) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {2, 7, 11, 11, 16};
  std::vector<int64_t> output_shape = {2, 11, 1, 11, 16, 16};
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
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "TransDataTilingData: " << to_string_int64(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1001 63488 0 2 13552 30976 3968 7 16 1 2 1 13552 30976 1 13552 30976 1936 16 30976 2816 1 0 1936 1936 0 1 0 1 7 1 0 1 0 1 7 1 0 2 1 30976 1 1 0 176 1 16 11 176 2816 ");
}

TEST_F(TransDataTiling, TransData_tiling_NCDHW2NDC1HWC0_C) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {2, 42767, 11, 11, 16};
  std::vector<int64_t> output_shape = {2, 11, 2673, 11, 16, 16};
  std::string dtype = "float32";

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
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"float32\", \"ubSize\": 63488, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "TransDataTilingData: " << to_string_int64(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1000 31744 1 32 2601984 236544 1984 15 16 1 2 1 82796912 82798848 2 165593824 165597696 1936 16 30976 2816 1 0 120 120 0 1 0 84 0 17 16 1 0 69 15 17 16 2 1 82798848 1 1 0 176 1 16 11 176 7527168 ");
}

TEST_F(TransDataTiling, TransData_tiling_NCDHW2NDC1HWC0_Cl) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {42767, 7, 11, 11, 16};
  std::vector<int64_t> output_shape = {42767, 11, 1, 11, 16, 16};
  std::string dtype = "int32";

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
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"int32\", \"ubSize\": 63488, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "TransDataTilingData: " << to_string_int64(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1000 31744 0 32 18213888 41631744 1984 7 16 1 2 1 13552 30976 16 216832 495616 1936 16 30976 2816 1 0 120 120 0 84 0 1 7 17 16 69 15 1 7 17 16 42767 1 30976 1 1 0 176 1 16 11 176 2816 ");
}

TEST_F(TransDataTiling, TransData_tiling_NCDHW2NDC1HWC0_Cr) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {2, 7, 42767, 11, 16};
  std::vector<int64_t> output_shape = {2, 42767, 1, 11, 16, 32};
  std::string dtype = "int8";

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
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"int8\", \"ubSize\": 253952, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "TransDataTilingData: " << to_string_int64(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1001 126976 2 32 238080 0 3968 7 32 1 2 1 52688944 240863744 1 52688944 240863744 7526992 32 240863744 5632 1 0 3968 3968 0 2 0 1 7 60 0 2 0 1 7 37 3664 2 1 240863744 1 1 0 176 1 32 42767 176 5632 ");
}

TEST_F(TransDataTiling, TransData_tiling_HWCN2FRACTALZN) {
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
  tensorOutput.format = "FRACTAL_ZN";									
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"HWCN\", \"dstFormat\": \"FRACTAL_ZN\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "1000 63488 0 31 148304 106496 3968 7 16 1 1 1 713 512 16 11408 8192 31 16 496 3195392 1 16 31 31 496 13 0 2 7 1 0 1 1 2 7 1 0 6241 1 512 1 1 0 31 1 16 1 1 0 ");
}

TEST_F(TransDataTiling, TransData_tiling_ND2FRACTALZ_001) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {79, 23, 13, 71};
  // std::vector<int64_t> output_shape = {1817  2 80 16};
  std::vector<int64_t> output_shape = {1817, 5, 16, 16};
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
  tensorOutput.format = "FRACTAL_Z";									
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "1000 63488 0 29 59072 81920 3968 13 16 1 1 1 923 1280 16 14768 20480 71 16 1136 1280 1 16 71 71 1136 4 0 1 13 1 0 2 9 1 13 1 0 1817 1 1280 1 1 0 71 1 16 1 1 0 ");
}

TEST_F(TransDataTiling, TransData_tiling_ND2FRACTALZN) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {42767};
  std::vector<int64_t> output_shape = {1, 2673, 16, 16};
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
  tensorOutput.format = "FRACTAL_ZN";									
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_ZN\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "1001 63488 2 11 3968 63488 3968 1 16 1 1 1 42767 684288 1 42767 684288 42767 16 684272 684288 1 16 3968 3968 63488 1 0 1 1 1 0 1 0 1 1 1 3087 1 1 684288 1 1 0 42767 1 16 1 1 0 ");
}

TEST_F(TransDataTiling, TransData_tiling_ND2FRACTALZ_002) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {1280, 1280};
  std::vector<int64_t> output_shape = {80, 80, 16, 16};
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
  tensorOutput.format = "FRACTAL_Z";
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "1001 63488 1 27 61440 61440 3968 0 16 1 1 1 1638400 1638400 1 1638400 1638400 1280 16 20480 20480 1 16 1280 1280 20480 1 0 3 0 1 0 1 0 2 0 1 0 1 1 1638400 1 1 0 1280 1 16 1 1 0 ");
}

TEST_F(TransDataTiling, TransData_tiling_NCHW2FRACTALZ) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {1280, 42767, 31, 4};
  std::vector<int64_t> output_shape = {331452, 80, 16, 16};
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
  tensorOutput.format = "FRACTAL_Z";
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "1000 63488 1 32 166656 213319680 3968 15 16 1 1 0 5303108 16 16 84849728 256 124 16 1984 2539520 1 20480 124 124 2539520 80 0 84 0 1 0 80 0 69 15 1 0 1280 1 16 1 1 0 124 1 20480 1 1 0 ");
}

TEST_F(TransDataTiling, TransData_tiling_NCHW2FRACTALZN) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {1280, 42767, 31, 4};
  std::vector<int64_t> output_shape = {331452, 80, 16, 16};
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
  tensorOutput.format = "FRACTAL_ZN";
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"FRACTAL_ZN\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "1000 63488 1 32 166656 213319680 3968 15 16 1 1 0 5303108 16 16 84849728 256 124 16 1984 2539520 1 20480 124 124 2539520 80 0 84 0 1 0 80 0 69 15 1 0 1280 1 16 1 1 0 124 1 20480 1 1 0 ");
}

TEST_F(TransDataTiling, TransData_tiling_NCDHW2FRACTALZ3D) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {1280, 2, 427, 31, 4};
  std::vector<int64_t> output_shape = {52948, 80, 16, 16};
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
  tensorOutput.format = "FRACTAL_Z_3D";
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ubSize\": 126976, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "1001 63488 2 31 1736 0 3968 2 16 1 2 0 105896 16 16 1694336 256 52948 16 847168 2539520 1 0 248 248 0 80 0 1 2 7 0 80 0 1 2 4 124 1280 1 16 1 1 0 124 1 20480 427 124 2539520 ");
}

TEST_F(TransDataTiling, TransData_NDC1HWC02NCDHW) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {3, 4, 5, 17, 17, 16};
  std::vector<int64_t> output_shape = {3, 80, 4, 17, 17};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "NDC1HWC0";								 
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "NCDHW";							   
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"NDC1HWC0\", \"dstFormat\": \"NCDHW\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2001 63232 0 3 16 92480 92480 1 2 1 0 2 0 1 2 1 0 2 0 1156 3 1 0 1 0 1156 80 4624 1156 13872 55488 92480 92480 92480 92480 0 2 1 0 1 1 0 3 1 92480 1 1 0 289 1 16 4 289 23120 ");
}

TEST_F(TransDataTiling, TransData_FRAZ3D2NCDHW) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {5780, 1, 16, 16};
  std::vector<int64_t> output_shape = {3, 80, 4, 17, 17};
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
  tensorOutput.format = "NCDHW";							   
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"FRACTAL_Z_3D\", \"dstFormat\": \"NCDHW\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2001 63232 0 3 16 16 92480 1 2 1 0 2 0 1 2 1 0 2 0 1156 3 1 0 1 0 1156 80 73984 1156 221952 55488 16 92480 16 92480 0 2 1 0 1 0 0 3 1 16 1 1 0 289 1 256 4 289 369920 ");
}

TEST_F(TransDataTiling, TransData_FRAZ2HWCN) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {1445, 1, 16, 16};
  std::vector<int64_t> output_shape = {17, 17, 80, 3};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "FRACTAL_Z";
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "HWCN";							   
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"FRACTAL_Z\", \"dstFormat\": \"HWCN\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2003 63232 0 19 16 4096 3840 1 1 1 0 0 0 1 1 1 0 0 1 3 5 16 16 1 48 3 80 73984 3 369920 240 256 240 4096 3840 0 1 1 0 1 1 240 289 1 256 1 1 0 3 1 16 1 1 0 ");
}

TEST_F(TransDataTiling, TransData_FRAZ3D2DHWCN) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {5780, 1, 16, 16};
  std::vector<int64_t> output_shape = {4, 17, 17, 80, 3};
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
  tensorOutput.format = "DHWCN";							   
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"FRACTAL_Z_3D\", \"dstFormat\": \"DHWCN\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2003 63232 0 25 16 0 11520 1 1 3 0 0 0 1 1 1 0 0 4 3 5 16 16 1 48 3 80 73984 3 369920 240 0 240 0 3840 0 1 2 0 1 1 960 289 1 256 4 289 369920 3 1 16 1 1 0 ");
}

TEST_F(TransDataTiling, TransData_FRAZ2NCHW) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {1445, 1, 16, 16};
  std::vector<int64_t> output_shape = {3, 80, 17, 17};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "FRACTAL_Z";
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
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"FRACTAL_Z\", \"dstFormat\": \"NCHW\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2001 63232 0 2 16 32 46240 1 1 1 0 0 0 1 1 1 0 0 1 289 5 2 256 1 73984 289 80 73984 289 369920 23120 16 23120 32 46240 0 1 1 0 1 0 23120 3 1 16 1 1 0 289 1 256 1 1 0 ");
}


TEST_F(TransDataTiling, TransData_FRAZ2ND) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {40, 3, 16, 16};
  std::vector<int64_t> output_shape = {640, 37};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "FRACTAL_Z";
  tensorInput.dtype = dtype;
  tensorInputsArg.tensor.push_back(tensorInput);
  tensorInputsArg.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputsArg);

  TeOpTensor tensorOutput;
  tensorOutput.shape = output_shape;
  tensorOutput.format = "ND";							   
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"FRACTAL_Z\", \"dstFormat\": \"ND\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2002 63232 1 8 16 3840 2960 1 1 1 0 0 0 1 1 1 0 0 0 37 5 1 16 1 592 37 640 768 37 3840 2960 30720 23680 30720 23680 0 1 1 0 0 1 0 1 1 30720 1 1 0 37 1 16 1 1 0 ");
}

TEST_F(TransDataTiling, TransData_NZ2NC1HWC0) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;

  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {9, 3, 16, 16};
  std::vector<int64_t> output_shape = {35, 1, 3, 3, 16};
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
  tensorOutput.format = "NC1HWC0";								  
  tensorOutput.dtype = dtype;
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  opParas.op_type = "TransData";
  std::string compileInfo1 = "{\"vars\": {\"srcFormat\": \"FRACTAL_NZ\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo1;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "TransDataTilingData: " << to_string_int64(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "2010 63232 0 1 0 16 560 5040 1 1 1 0 0 0 1 1 1 0 0 0 9 0 16 0 144 1 1 6912 1 6912 16 0 35 16 144 560 5040 9 1 768 1 9 6912 2 3952 1 ");
}

TEST_F(TransDataTiling, TransData_NDC1HWC02NDHWC) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {1, 2, 2, 1, 1, 16};
  std::vector<int64_t> output_shape = {1, 2, 1, 1, 31};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "NDC1HWC0";								 
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
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"NDC1HWC0\", \"dstFormat\": \"NDHWC\", \"dType\": \"float16\", \"ubSize\": 126464, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2012 63232 0 1 1 16 64 62 1 1 1 0 0 0 1 1 1 0 0 0 1 16 31 16 31 2 1 16 1 32 32 15 2 32 31 64 62 1 1 16 1 1 0 1 3952 1 ");
}

TEST_F(TransDataTiling, TransData_NZ3D_2_NDHWC) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {1024, 1, 16, 16};
  std::vector<int64_t> output_shape = {10, 8, 16, 8, 11};
  std::string dtype = "float32";

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
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"FRACTAL_Z_3D\", \"dstFormat\": \"NDHWC\", \"dType\": \"float32\", \"ubSize\": 65280, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2011 32640 2 9 0 16 0 1397 1 1 1 0 0 0 1 1 1 8 0 0 127 0 11 0 1397 1 1 32768 1 32768 16 11 10 16 11264 160 112640 128 1 256 8 128 32768 2 2040 0 ");
}

TEST_F(TransDataTiling, TransData_NZ3D_2_NDHWC_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {6250, 1, 16, 16};
  std::vector<int64_t> output_shape = {1, 1, 1, 1, 100000};
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
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"FRACTAL_Z_3D\", \"dstFormat\": \"NDHWC\", \"dType\": \"float16\", \"ubSize\": 130560, \"blockDim\": 32, \"inputSize\": -1, \"hiddenSize\": -1, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data), "2010 65280 1 2 0 16 1044480 65280 1 1 1 0 0 0 1 1 1 0 2170 0 1 0 100000 0 100000 4080 0 256 1 1044480 65280 0 1 16 100000 16 100000 1 1 256 1 1 1600000 2 4080 1 ");
}

TEST_F(TransDataTiling, ND_2_NZ) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {38400, 54};
  std::vector<int64_t> output_shape = {4, 2400, 16, 16};
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
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_NZ\", \"dType\": \"float16\", \"ubSize\": 130560, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1010 65280 20 108864 32256 2073600 2457600 2073600 2457600 54432 16128 54 0 4080 1 63 16 54 2457600 614400 16 6 54 1 1 1 2 0 63 1 0 1 1 1 1 2 33 1 0 ");
}

TEST_F(TransDataTiling, ND_2_NZ_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {2, 10000};
  std::vector<int64_t> output_shape = {625, 1, 16, 16};
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
  std::string compileInfo3 = "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_NZ\", \"dType\": \"float16\", \"ubSize\": 130560, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo3;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1010 65280 3 4080 65280 20000 160000 20000 160000 160000 256 10000 0 4080 1 1 16 4080 65280 256 16 0 4080 1 1 1 1 2 1 1 0 1 1 1 1 2 1 1 1840 ");
}

TEST_F(TransDataTiling, TransData_NDHWC_2_NZ3D) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {1, 1, 1, 1, 47};
  std::vector<int64_t> output_shape = {1, 3, 1, 1, 1, 16, 16};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "NDHWC";
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
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NDHWC\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ubSize\": 130560, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1011 65280 1 0 47 16 47 16 47 1 47 4080 1 0 47 768 256 16 15 47 1 0 1 0 1 0 1 0 1 0 1 0 1 1 256 1 1 768 ");
}

TEST_F(TransDataTiling, TransData_NDHWC_2_NZ3D_1) {
  using namespace optiling;
  optiling::OpRunInfo op_run_info;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputsArg, tensorOutputsArg;
  TeOpParas opParas;
  std::vector<int64_t> input_shape = {100, 1, 1, 1, 47};
  std::vector<int64_t> output_shape = {1, 3, 1, 1, 7, 16, 16};
  std::string dtype = "float16";

  TeOpTensor tensorInput;
  tensorInput.shape = input_shape;
  tensorInput.format = "NDHWC";
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
  std::string compileInfo2 = "{\"vars\": {\"srcFormat\": \"NDHWC\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ubSize\": 130560, \"blockDim\": 32, \"inputSize\": 0, \"hiddenSize\": 0, \"group\": 1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo2;
  op_compile_info.key = this->test_info_->name();

  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string_int64(runInfo.tiling_data),
            "1011 65280 7 0 752 256 752 256 47 16 47 4080 1 0 47 5376 1792 16 15 47 1 0 1 0 1 0 1 4 1 0 1 0 1 1 1792 1 1 5376 ");
}
