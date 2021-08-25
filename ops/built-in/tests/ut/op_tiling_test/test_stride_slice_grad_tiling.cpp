#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class stried_slice_grad_tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "stried_slice_grad_tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stried_slice_grad_tiling TearDown" << std::endl;
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

TEST_F(stried_slice_grad_tiling, stried_slice_grad_tiling_no_mask) {
  using namespace optiling;
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("StridedSliceGrad");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
  TeOpTensorArg tensorInputs, tensorOutputsArg;
  std::string op_name = "StridedSliceGrad";
  TeOpParas opParas;
  opParas.op_type = op_name;
  vector<vector<int64_t>> input_shapes = {
      {4},
      {4},
      {4},
      {4},
      {4, 4, 4, 4},
  };

  vector<string> dtypes = {"int32", "int32", "int32", "int32", "float16"};

  for (size_t i = 0; i < input_shapes.size(); i++) {
    tensorInputs.tensor.clear();
    TeOpTensor tensorInput;
    tensorInput.shape = input_shapes[i];
    tensorInput.dtype = dtypes[i];
    tensorInputs.tensor.push_back(tensorInput);
    tensorInputs.arg_type = TA_SINGLE;
    opParas.inputs.push_back(tensorInputs);
  }

  TeOpTensor tensorOutput;
  tensorOutput.shape = input_shapes[4];
  tensorOutput.dtype = "float16";
  tensorOutputsArg.tensor.push_back(tensorOutput);
  tensorOutputsArg.arg_type = TA_SINGLE;
  opParas.outputs.push_back(tensorOutputsArg);
  vector<int32_t> shape_value = {3, 3, 3, 3};
  vector<int32_t> begin = {1, 1, 1, 1};
  vector<int32_t> end = {3, 3, 3, 3};
  vector<int32_t> strides = {1, 1, 1, 1};

  opParas.const_inputs["shape"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) shape_value.data(),
                                                      shape_value.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["begin"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) begin.data(),
                                                      begin.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["end"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) end.data(),
                                                      end.size() * sizeof(int32_t), ge::Tensor());
  opParas.const_inputs["strides"] =
      std::tuple<const uint8_t *, size_t, ge::Tensor>((const uint8_t *) strides.data(),
                                                       strides.size() * sizeof(int32_t), ge::Tensor());
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 65536, \"core_num\": 32, \"dtype_rate\": 2, \"begin_mask\": 0, \"end_mask\": 0, \"ellipsis_mask\": 0, \"new_axis_mask\": 0, \"shrink_axis_mask\": 0}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 1 1 1 1 2 2 2 4 0 0 0 0 0 0 0 0 1 0 1 0 1 0 2 0 2 ");
}
