#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class ResizeNearestNeighborV2Tiling : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "ResizeNearestNeighborV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResizeNearestNeighborV2Tiling TearDown" << std::endl;
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

TEST_F(ResizeNearestNeighborV2Tiling, resize_nearest_neighbor_tiling_0) {
  using namespace optiling;
  std::string op_name = "ResizeNearestNeighborV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, \"strides_h\": 1, \"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 1000, 1000, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "113000 16 1 1000 1000 1000 1000 16 1 2 ");
}

TEST_F(ResizeNearestNeighborV2Tiling, resize_nearest_neighbor_tiling_2) {
  using namespace optiling;
  std::string op_name = "ResizeNearestNeighborV2";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, \"strides_h\": 1, \"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 999, 999, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_input_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "100000 16 1 1000 1000 999 999 2 4 4 ");
}

